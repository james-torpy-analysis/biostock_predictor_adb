# Databricks notebook source
# MAGIC %md
# MAGIC # Stock Crash Detection Pipeline
# MAGIC Author: James Torpy  
# MAGIC   
# MAGIC Contact: james.torpy@gmail.com 
# MAGIC    
# MAGIC Date: 21/11/24  
# MAGIC   
# MAGIC Description: This notebook downloads NASDAQ stock data, identifies crashes, and generates visualizations to be automatically emailed, informing short term investments.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load packages and set up variables

# COMMAND ----------

# Set Parameters
data_window = 5  # Past days to download
batch_size = 10  # tickers per batch
drop_perc = 10  # Required % drop to identify as a crash
drop_period = 2 # Drop must occur within this number of days to identify as a crash
min_value_today = True  # Determines whether the minimum value of the drop must occur on the latest date
subset_tickers = 500 # Number of tickers to download. If None, all tickers will be downloaded
receiving_email = 'james.torpy@gmail.com'

# Set up paths
out_dir = "/tmp/"

# COMMAND ----------

# Install packages
%pip install yfinance

# COMMAND ----------

# Load packages

# Time/date
import datetime
import time

# Data download
import urllib.request
import yfinance as yf

# General:
import pandas as pd
import matplotlib.pyplot as plt

# Pyspark
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import Row
import requests
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    DoubleType,
    LongType,
    IntegerType,
)


# Emailing
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import io
import smtplib

# Suppress Yahoo Finance error messages
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Define the latest date for which there is close data (i.e. yesterday)
latest_date = datetime.date.today() - datetime.timedelta(days=1)

# Define starting date of crash period
period_start_date = latest_date
weekdays_to_subtract = drop_period

while weekdays_to_subtract > 0:
    period_start_date -= datetime.timedelta(days=1)
    # Check if the current day is a weekday (Monday=0 to Friday=4)
    if period_start_date.weekday() < 5:
        weekdays_to_subtract -= 1


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### a) Retrieve data

# COMMAND ----------

def get_tickers(url="https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"):
    '''
    Downloads NASDAQ tickers and parses into a PySpark DataFrame.

    Args:
        url: The URL of the NASDAQ tickers file.

    Returns:
        A PySpark DataFrame containing NASDAQ tickers.
    '''
    # Download the file content as text
    response = requests.get(url)
    data = response.text

    # Convert the data into an RDD
    rdd = spark.sparkContext.parallelize(data.strip().splitlines())

    # Skip the header and split the data
    rdd_data = rdd.zipWithIndex().filter(lambda x: x[1] > 0).keys()  # Skip the first row (header)
    df = rdd_data.map(lambda line: line.split('|')).toDF(["ticker", "Security Name", "Other Info", "Market", "Test Issue", "Financial Status"])

    # Select only the "ticker" column and return as DataFrame
    tickers_df = df.select("ticker")
    return [row['ticker'] for row in tickers_df.select('ticker').collect()]

def split_stock_data(stock_df):
    '''Splits the stock data output into a df for each ticker.

    Args:
        lst: A list of integers.

    Returns:
        A list of the first indices of concurrent sequences.
    '''

    # Split data
    tickers = list(stock_df['Adj Close'].columns)
    split_data = {ticker: stock_df.xs(ticker, level=1, axis=1) for ticker in tickers}

    # Add date columns
    for key in split_data:
        split_data[key] = split_data[key].copy()
        split_data[key]['date'] = split_data[key].index

    # Convert to pyspark dfs
    pyspark_data = {ticker: spark.createDataFrame(df) for ticker, df in split_data.items()}
    return pyspark_data

def download_stock_data(ticker_list, data_window, batch_size):
    '''Downloads NASDAQ stock data in batches.

    Args:
        ticker_list: A list of NASDAQ tickers to download data for.
        data_window: The number of past months to download stock data for, starting from today.
        batch_size: The number of tickers to download per batch, to avoid blocking from Yahoo.
        out_dir: The output directory to store the data in.

    Returns:
        A dictionary of dfs of stock data, including datetime, adj. close, volume. Each key-value
        pair represents data for a different NASDAQ ticker.
    '''
    out_dict = {}
    for i in range(0, len(ticker_list), batch_size):
        print(f'\nDownloading batch {i}-{i + batch_size} out of {len(ticker_list)} tickers...')
        batch_tickers = ticker_list[i:i + batch_size]
        try:
            batch_data = yf.download(batch_tickers, period=f"{data_window}d")
            out_dict.update(split_stock_data(batch_data))
        except Exception as e:
            print(f"Failed to download batch {i}-{i + batch_size}: {e}")
        print('Sleeping for 2 sec...')
        time.sleep(2)
    return out_dict

# COMMAND ----------

# MAGIC %md
# MAGIC ### b) Detect crashes

# COMMAND ----------

def detect_crashes(ticker, df, latest_date, period_start_date, drop_perc):
    # Filter dates
    filtered_df = df.filter(df['date'].between(period_start_date, latest_date))

    # Get min and max values and dates
    min_value = filtered_df.agg(F.min('Adj Close')).collect()[0][0]
    min_value_date = filtered_df.filter(filtered_df['Adj Close'] == min_value).select('date').collect()[0][0]
    max_value = filtered_df.agg(F.max('Adj Close')).collect()[0][0]
    max_value_date = filtered_df.filter(filtered_df['Adj Close'] == max_value).select('date').collect()[0][0]

    # Determine percentage difference between min and max
    value_diff = max_value - min_value
    diff_perc = (value_diff / max_value)*100

    # Ensure dates are formatted the same
    min_value_date =  min_value_date.date()
    max_value_date =  max_value_date.date()
    
    # Mark min and max values and return if percentage difference >= drop_perc and if max_value_date occurred before 
    # min_value_date
    if diff_perc >= drop_perc and max_value_date < min_value_date:
        if min_value_today:
            #  Mark min and max values and return if min_value_date == latest_date
            if min_value_date == latest_date:
                return df.withColumn('max_value', df['Adj Close'] == max_value).withColumn('min_value', df['Adj Close'] == \
                    min_value).withColumn('ticker', F.lit(ticker))
            else:
                return None
        else:
            return df.withColumn('max_value', df['Adj Close'] == max_value).withColumn('min_value', df['Adj Close'] == \
                min_value).withColumn('ticker', F.lit(ticker))
    else:
        return None


# COMMAND ----------

# MAGIC %md
# MAGIC ### c) Plot crashes

# COMMAND ----------

def plot_crashes(df):
    '''Creates and saves crash plots for each stock with detected crashes.

    Args:
        df: A dataframe of stock prices of companies that have crashed

    '''

    # Convert df to pandas
    df = df.toPandas()
  
    # Plot line/scatter plot of stock price and crashes
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['Adj Close'], label='Adj Close')
    ax.scatter(df[df['min_value']]['date'], df[df['min_value']]['Adj Close'], label='Crash', color='red', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f"{df['ticker'][0]} Crash Detection")
    ax.legend()
    plt.tight_layout()

    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### d) Email data

# COMMAND ----------

def email_crash_data(crash_results, crash_plots, stat_df, email):
    # Set up credentials securely
    email_from = email
    email_password = os.getenv('EMAIL_APP_PASSWORD')  # Retrieve the password from an environment variable
    email_to = email

    # Email setup
    msg = MIMEMultipart()
    msg['From'] = email_from
    msg['To'] = email_to
    msg['Subject'] = 'NASDAQ Crash Updates'

    # Email body
    html_body = """
    <html>
    <head></head>
    <body>
    """

    # Add statistics dataframe as HTML
    table_html = stat_df.toPandas().to_html(index=False)
    html_body += f"<h3> Pipeline statistics</h3>"
    html_body += table_html

    # Add crash dataframes
    for ticker, df in crash_results.items():
        table_html = df.toPandas().to_html(index=False)
        html_body += f"<h3>{ticker} crash data</h3>"
        html_body += table_html

        # Add the plot inline
        html_body += f"<h3>{ticker} crash plot</h3>"
        html_body += f'<img src="cid:plot_{ticker}">'

        # Attach the plot for this ticker
        plot_buffer = io.BytesIO()
        crash_plots[ticker].savefig(plot_buffer, format='png')  # Save the plot to a buffer
        plt.close(crash_plots[ticker])  # Close the plot to release memory
        plot_buffer.seek(0)

        # Attach the image as an inline attachment with a unique Content-ID
        image = MIMEImage(plot_buffer.read(), name=f'{ticker}_crash_plot.png')
        image.add_header('Content-ID', f'<plot_{ticker}>')  # Match the 'cid' in the email body
        msg.attach(image)

    # Close the email body
    html_body += """
    </body>
    </html>
    """
    msg.attach(MIMEText(html_body, 'html'))

    # Send the Email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(email_from, email_password)
            server.sendmail(email_from, email_to, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
    

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Main function

# COMMAND ----------

def main():
    # Create df to catch stats
    schema = StructType([
        StructField("stat", StringType(), True),   # Column 'stat' as String
        StructField("passed", IntegerType(), True),  # Column 'passed' as Integer
        StructField("failed", IntegerType(), True),   # Column 'failed' as Integer
        StructField("total", IntegerType(), True)   # Column 'total' as Integer
    ])
    stat_df = spark.createDataFrame([], schema)

    # Step 1: Download NASDAQ tickers and Stock Data
    nasdaq_tickers = get_tickers()
    if subset_tickers is not None:
        nasdaq_tickers = nasdaq_tickers[0:(subset_tickers)]
    nasdaq_data = download_stock_data(nasdaq_tickers, data_window, batch_size)

    # Record how many failed to download
    fail_count = sum([value.filter(value['Adj Close'].isNotNull()).count() < 2 for value in nasdaq_data.values()])
    print('\n', str(fail_count) + ' out of ' + str(len(nasdaq_data)) + ' failed to download')       
    new_row = [Row(stat="data_download", passed=int(len(nasdaq_data)-fail_count), failed=int(fail_count), \
        total=int(len(nasdaq_data)))]
    stat_df = stat_df.union(spark.createDataFrame(new_row))

    # Remove tickers with 1 or less row with non-null values
    nasdaq_data = {key: value for key, value in nasdaq_data.items() if value.filter(value['Adj Close'].isNotNull()).count() > 1}

    # Step 3: Crash Detection and Visualisation
    crashes = {}
    for i, ticker in enumerate(nasdaq_data.keys()):
        crash_result = detect_crashes(ticker, nasdaq_data[ticker], latest_date, period_start_date, drop_perc)
        if crash_result:
            if crash_result.isEmpty():
                print(f'No crashes detected for {ticker} ({i+1/len(nasdaq_data)}), moving to next ticker.')
            else:
                print(f'Crashes detected for {ticker} ({i+1/len(nasdaq_data)}), adding them to crash results.')
                crashes[ticker] = crash_result
        else:
            print(f'No crashes detected for {ticker} ({i+1/len(nasdaq_data)}), moving to next ticker.')

    crash_count = len(crashes)
    total_count = len(nasdaq_data)
    none_count = total_count - crash_count
    print(f'At least one crash was detected in {crash_count} out of {total_count} tickers')
    new_row = [Row(stat="crashes_detected", passed=int(crash_count), failed=int(none_count), total=int(total_count))]
    stat_df = stat_df.union(spark.createDataFrame(new_row))

    # Step 4: Plot Crashes
    crash_plots = {}
    for ticker in crashes.keys():
        crash_plot = plot_crashes(crashes[ticker])
        if crash_plot:
            crash_plots[ticker] = crash_plot
        plt.close(crash_plot)

    # Step 5: Send Email Notification with Crash Details and Plots
    email_crash_data(crashes, crash_plots, stat_df, email = receiving_email)

# COMMAND ----------

# MAGIC %md
# MAGIC ##4. Execute main function

# COMMAND ----------

if __name__ == "__main__":
    main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Debugging and new features

# COMMAND ----------

# MAGIC %md
# MAGIC ### a) Debugging

# COMMAND ----------

#  # Create df to catch stats
# schema = StructType([
#     StructField("stat", StringType(), True),   # Column 'stat' as String
#     StructField("passed", IntegerType(), True),  # Column 'passed' as Integer
#     StructField("failed", IntegerType(), True),   # Column 'failed' as Integer
#     StructField("total", IntegerType(), True)   # Column 'total' as Integer
# ])
# stat_df = spark.createDataFrame([], schema)

# # Step 1: Download NASDAQ tickers and Stock Data
# nasdaq_tickers = get_tickers()
# if subset_tickers is not None:
#     nasdaq_tickers = nasdaq_tickers[0:(subset_tickers)]
# nasdaq_data = download_stock_data(nasdaq_tickers, data_window, batch_size)

# # Record how many failed to download
# fail_count = sum([value.filter(value['Adj Close'].isNotNull()).count() < 2 for value in nasdaq_data.values()])
# print('\n', str(fail_count) + ' out of ' + str(len(nasdaq_data)) + ' failed to download')       
# new_row = [Row(stat="data_download", passed=int(len(nasdaq_data)-fail_count), failed=int(fail_count), \
#     total=int(len(nasdaq_data)))]
# stat_df = stat_df.union(spark.createDataFrame(new_row))

# # Remove tickers with 1 or less row with non-null values
# nasdaq_data = {key: value for key, value in nasdaq_data.items() if value.filter(value['Adj Close'].isNotNull()).count() > 1}

# # Step 3: Crash Detection and Visualisation
# crashes = {}
# for ticker in nasdaq_data.keys():
#     crash_result = detect_crashes(ticker, nasdaq_data[ticker], latest_date, period_start_date, drop_perc)
#     if crash_result:
#         if crash_result.isEmpty():
#             print(f'No crashes detected for {ticker}, moving to next ticker.')
#         else:
#             print(f'Crashes detected for {ticker}, adding them to crash results.')
#             crashes[ticker] = crash_result
#     else:
#         print(f'No crashes detected for {ticker}, moving to next ticker.')

# crash_count = len(crashes)
# total_count = len(nasdaq_data)
# none_count = total_count - crash_count
# print(f'At least one crash was detected in {crash_count} out of {total_count} tickers')
# new_row = [Row(stat="crashes_detected", passed=int(crash_count), failed=int(none_count), total=int(total_count))]
# stat_df = stat_df.union(spark.createDataFrame(new_row))




# COMMAND ----------

# # Step 4: Plot Crashes
# crash_plots = {}
# for ticker in crashes.keys():
#     crash_plot = plot_crashes(crashes[ticker])
#     if crash_plot:
#         crash_plots[ticker] = crash_plot
#     plt.close(crash_plot)

# crash_plots['AAL']

# # ticker = 'AAL'
# # df = crashes[ticker]

# # # Convert df to pandas
# # df = df.toPandas()

# # # Plot line/scatter plot of stock price and crashes
# # fig, ax = plt.subplots()
# # ax.plot(df['date'], df['Adj Close'], label='Adj Close')
# # ax.scatter(df[df['min_value']]['date'], df[df['min_value']]['Adj Close'], label='Crash', color='red', alpha=0.8)
# # plt.xticks(rotation=45, ha='right')
# # ax.set_title(f"{df['ticker'][0]} Crash Detection")
# # ax.legend()
# # plt.tight_layout()

# # fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### b) New features

# COMMAND ----------

# Parallelise downloading

tickers = ['AAPL', 'GOOG', 'TSLA', 'AMZN']

# Method 1
out_dict1 = {}
for i in range(0, len(tickers), batch_size):
    print(f'\nDownloading batch {i}-{i + batch_size} out of {len(tickers)} tickers...')
    batch_tickers = tickers[i:i + batch_size]
    try:
        batch_data = yf.download(batch_tickers, period=f"{data_window}d")
        out_dict1.update(split_stock_data(batch_data))
    except Exception as e:
        print(f"Failed to download batch {i}-{i + batch_size}: {e}")

# Method 2 (parallel)
from concurrent.futures import ThreadPoolExecutor

def fetch_data(ticker):
    try:
        data = yf.download(ticker, period="{data_window}d")
        print(f"Downloaded data for {ticker}")
        return ticker, data
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return ticker, None

out_dict2 = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    for ticker, data in executor.map(fetch_data, tickers):
        if data is not None:
            out_dict2[ticker] = data

