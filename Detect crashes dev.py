# Databricks notebook source
# MAGIC %md
# MAGIC # Stock Crash Detection Pipeline
# MAGIC Author: James Torpy  
# MAGIC   
# MAGIC Contact: james.torpy@gmail.com 
# MAGIC    
# MAGIC Date: 16/10/24  
# MAGIC   
# MAGIC Description: This notebook downloads NASDAQ stock data, detects anomalies, identifies crashes, and generates visualizations to be automatically emailed, informing short term investments.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load packages and set up variables

# COMMAND ----------

# Install packages
%pip install yfinance

# COMMAND ----------

# Load packages
from azure.storage.filedatalake import DataLakeServiceClient

from datetime import date
import urllib.request
import yfinance as yf
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.functions import abs, col, lit, when, row_number, split
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

import time

# Load emailing packages
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import io
import smtplib

# Suppress Yahoo Finance error messages
import logging

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Set Parameters
data_window = 5  # Past days to download
batch_size = 10  # Symbols per batch
cont_val = 0.1  # Outlier contamination for Isolation Forest
perc_drop_req = 5  # Required % drop to identify as a crash
perc_drop_time = 5  # Days over which drop should occur
subset_symbols = 10 # Number of symbols to download. If None, all symbols will be downloaded
receiving_email = 'james.torpy@gmail.com'

# Set up paths
out_dir = "/tmp/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### a) Retrieve data

# COMMAND ----------

def get_symbols(url="https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"):
    '''
    Downloads NASDAQ symbols and parses into a PySpark DataFrame.

    Args:
        url: The URL of the NASDAQ symbols file.

    Returns:
        A PySpark DataFrame containing NASDAQ symbols.
    '''
    # Download the file content as text
    response = requests.get(url)
    data = response.text

    # Convert the data into an RDD
    rdd = spark.sparkContext.parallelize(data.strip().splitlines())

    # Skip the header and split the data
    rdd_data = rdd.zipWithIndex().filter(lambda x: x[1] > 0).keys()  # Skip the first row (header)
    df = rdd_data.map(lambda line: line.split('|')).toDF(["Symbol", "Security Name", "Other Info", "Market", "Test Issue", "Financial Status"])

    # Select only the "Symbol" column and return as DataFrame
    symbols_df = df.select("Symbol")
    return [row['Symbol'] for row in symbols_df.select('Symbol').collect()]

def download_stock_data(symbol_list, data_window, batch_size):
    '''Downloads NASDAQ stock data in batches.

    Args:
        symbol_list: A list of NASDAQ symbols to download data for.
        data_window: The number of past months to download stock data for, starting from today.
        batch_size: The number of symbols to download per batch, to avoid blocking from Yahoo.
        out_dir: The output directory to store the data in.

    Returns:
        A dictionary of dfs of stock data, including datetime, adj. close, volume. Each key-value
        pair represents data for a different NASDAQ symbol.
    '''
    out_dict = {}
    for i in range(0, len(symbol_list), batch_size):
        print(f'\nDownloading batch {i}-{i + batch_size} out of {len(symbol_list)} symbols...')
        batch_symbols = symbol_list[i:i + batch_size]
        try:
            batch_data = yf.download(batch_symbols, period=f"{data_window}d")
            out_dict.update(split_stock_data(batch_data))
        except Exception as e:
            print(f"Failed to download batch {i}-{i + batch_size}: {e}")
        print('Sleeping for 2 sec...')
        time.sleep(2)
    return out_dict

def split_stock_data(stock_df):
    '''Splits the stock data output into a df for each symbol.

    Args:
        lst: A list of integers.

    Returns:
        A list of the first indices of concurrent sequences.
    '''
    symbols = list(stock_df['Adj Close'].columns)
    split_data = {symbol: stock_df.xs(symbol, level=1, axis=1) for symbol in symbols}
    return split_data




# COMMAND ----------

# MAGIC %md
# MAGIC ### b) Detect anomalies

# COMMAND ----------

def predict_anomalies(symbol, df, cont_val, symbol_count, total_symbol_no, pyspark = True):
    '''Predicts anomalies in NASDAQ stock data dfs.

    Args:
        
        data_df: A NASDAQ stock data df outputted from download_stock_data.
        cont_val: A parameter of the IsolationForest function which controls the detection
                  sensitivity. Input value is an estimation of the amount of contamination 
                  of the data set, i.e. the proportion of outliers.
        acolname: The name of the anomaly output column

    Returns:
        The input df data_df with an additional boolean column indicating detection (True) 
        or no detection (False) of anomalies.
    '''
    try:
        print(f'Predicting anomalies for {symbol} ({symbol_count}/{total_symbol_no} symbols)...')
        df['date'] = df.index
        df['VolumeClose'] = df['Adj Close'] * df['Volume']
        df = df.dropna(subset=['VolumeClose'])
        if_model = IsolationForest(contamination=cont_val, n_jobs = -1)
        if_model.fit(df[['VolumeClose']])
        df['anomaly'] = pd.Series(if_model.predict(df[['VolumeClose']])).map({1: 0, -1: 1}).to_numpy()
        df['date'] = df.index

        # Add weekday-before dates and closes
        df['preanomaly_date'] = df['date'].shift(1)
        df['preanomaly_close'] = df['Adj Close'].shift(1)

        no_predicted = (df['anomaly'] == 1).sum()
        total_no = len(df)
        print(f'{no_predicted} anomalies predicted for {symbol} out of {total_no} dates')

        # Filter anomalies
        df['anomaly'] = df['anomaly'].astype(int)
        df = df[df['anomaly'] == 1].copy()

        if len(df) == 0:
            print(f'No anomalies predicted for {symbol}')
            return None
        else :
            no_anomalies = len(df)
            if pyspark:
                return spark.createDataFrame(df)
            else:
                return df
    
    except:
        print(f'Failed to predict anomalies for {symbol}')
        return None


# COMMAND ----------

# MAGIC %md
# MAGIC ### c) Detect crashes

# COMMAND ----------

def find_crashes(symbol, anomaly_df, orig_df, perc_drop_req, perc_drop_time):
    print(f'Finding crashes for {symbol}...')

    try:
        orig_df = spark.createDataFrame(orig_df)

        # Add preanomaly prices
        anomalies = anomaly_df.filter(anomaly_df['anomaly'] == 1)
        anomalies = anomalies.withColumnRenamed("date", "anomaly_date")
        anomalies = anomalies.withColumnRenamed("Adj Close", "anomaly_adj_close")

        # Calculate price drops
        crashes = anomalies.withColumn("preanomaly_close_diff", F.col("Close") - F.col("preanomaly_close"))
        crashes = crashes.withColumn("preanomaly_close_percent_diff", F.col("preanomaly_close_diff") / F.col("preanomaly_close") * 100)

        # Define crash range and check if drop exceeds threshold within specified period
        crashes = crashes.withColumn("period_end_date", F.date_add(F.col("anomaly_date"), perc_drop_time))
        crashes = crashes.select('anomaly_adj_close', 'anomaly_date', 'preanomaly_date', 'preanomaly_close', 'preanomaly_close_diff', \
            'preanomaly_close_percent_diff', 'period_end_date', 'Volume', 'VolumeClose')

        joined_crashes = crashes.join(
            orig_df,
            (orig_df['date'] >= crashes['preanomaly_date']) & (orig_df['date'] <= crashes['period_end_date']),
            "inner"
        )

        crash_df = joined_crashes.groupBy("anomaly_date").agg(F.collect_list("Adj Close").alias("adj_close_values"),
            (F.min("Adj Close") - F.max("Adj Close")).alias("min_max_diff"),
            (F.col("min_max_diff")/F.max("Adj Close")*100).alias("min_max_percent_diff"))

        crash_stats = crash_df.join(crashes, on = ['anomaly_date'])

        # Add symbol
        crash_stats = crash_stats.withColumn('symbol', lit(symbol))

        # Select and rename columns
        crash_stats = crash_stats.select('symbol', 'anomaly_date', 'anomaly_adj_close', 'Volume', 'VolumeClose', 'preanomaly_date', \
            'preanomaly_close', 'preanomaly_close_diff', 'preanomaly_close_percent_diff', 'period_end_date', 'min_max_diff', 'min_max_percent_diff')
        new_names = ['symbol', 'crash_date', 'crash_close', 'crash_volume', 'crash_volume_close', 'precrash_date', \
            'precrash_close', 'precrash_close_diff', 'precrash_close_percent_diff', 'period_end_date', 'period_max_diff', 'period_percent_max_diff']
        crash_stats = crash_stats.toDF(*new_names)

        # Determine whether the max value occurs before the min value for each crash period
        crash_stats = crash_stats.withColumn('crash', lit(False)) \
          .withColumn('min_value_date', lit(None)) \
          .withColumn('min_close', lit(None))

        crash_rows = crash_stats.collect()
        for i, row in enumerate(crash_rows):
            period_df = orig_df.filter(orig_df['date'].between(row['precrash_date'], row['period_end_date']))
            min_close = period_df.orderBy(col('Adj Close').asc()).first()['Adj Close']
            min_val_date = period_df.orderBy(col('Adj Close').asc()).first()['date']
            max_val_date = period_df.orderBy(col('Adj Close').desc()).first()['date']

            if max_val_date < min_val_date:
                crash_stats = crash_stats.withColumn(
                    "crash",
                    when(crash_stats["crash_date"] == row["crash_date"], lit(True)).otherwise(crash_stats["crash"])
                ).withColumn(
                    "min_value_date",
                    when(crash_stats["crash_date"] == row["crash_date"], lit(min_val_date)).otherwise(crash_stats["min_value_date"])
                ).withColumn(
                    "min_close",
                    when(crash_stats["crash_date"] == row["crash_date"], lit(min_close)).otherwise(crash_stats["min_close"])
                )

        # Filter for crashes where period_percent_max_diff <= -perc_drop_req
        crash_stats = crash_stats.filter((crash_stats['period_percent_max_diff'] <= -perc_drop_req) & (crash_stats['crash']))
        crash_stats = crash_stats.drop('crash')
        
        return crash_stats
    
    except Exception as e:
        print(f'Error identifying crashes for {symbol}: {e}')
        return None
    

# COMMAND ----------

# MAGIC %md
# MAGIC ### d) Plot crashes

# COMMAND ----------

def plot_crashes(df, crashes, symbol):
    '''Creates and saves crash plots for each stock with detected crashes.

    Args:
        df: A dataframe of anomalies for a company outputted from the 
        predict_anomalies function.
        crashes: A dataframe of crashes for a company outputted from the 
        find_crashes function.
        symbol: A NASDAQ symbol string

    '''

    # Copnvert dates and crashes df
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    crashes = crashes.toPandas()
    crashes['min_value_date'] = pd.to_datetime(crashes['min_value_date']).dt.tz_localize(None)

    # Plot line/scatter plot of stock price and crashes
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['Adj Close'], label='Adj Close')
    ax.scatter(crashes['min_value_date'], crashes['min_close'], label='Crash', color='red', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f"{symbol} Crash Detection")
    ax.legend()
    plt.tight_layout()

    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### e) Email data

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
    for symbol, df in crash_results.items():
        table_html = df.toPandas().to_html(index=False)
        html_body += f"<h3>{symbol} crash data</h3>"
        html_body += table_html

        # Add the plot inline
        html_body += f"<h3>{symbol} crash plot</h3>"
        html_body += f'<img src="cid:plot_{symbol}">'

        # Attach the plot for this symbol
        plot_buffer = io.BytesIO()
        crash_plots[symbol].savefig(plot_buffer, format='png')  # Save the plot to a buffer
        plt.close(crash_plots[symbol])  # Close the plot to release memory
        plot_buffer.seek(0)

        # Attach the image as an inline attachment with a unique Content-ID
        image = MIMEImage(plot_buffer.read(), name=f'{symbol}_crash_plot.png')
        image.add_header('Content-ID', f'<plot_{symbol}>')  # Match the 'cid' in the email body
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
# MAGIC ### f) Collate and save crash data

# COMMAND ----------


def collate_crashes(stock_data, crash_data):
    # Add symbols to the stock dfs and bind together
    stock_dfs = []
    for key, df in stock_data.items():
        df["symbol"] = key  # Add the key as a new column
        stock_dfs.append(df)
    stock_df = pd.concat(stock_dfs, ignore_index=True)

    # Add symbols to the crash dfs and bind together
    crash_df = None
    for key, df in crash_data.items():
        df = df.withColumn("symbol", lit(key))  # Add the key as a new column
        if crash_df is None:
            crash_df = df
        else:
            crash_df = crash_df.union(crash_df)
    
    # Join crash and stock data together
    collated_df = spark.createDataFrame(stock_df).join(crash_df, (stock_df["symbol"] == crash_df["symbol"]) & (stock_df \
        ["min_value_date"] == crash_df["min_value_date"]), how="left")
    
    return collated_df

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

    # Step 1: Download NASDAQ Symbols and Stock Data
    nasdaq_symbols = get_symbols()
    if subset_symbols is not None:
        nasdaq_symbols = nasdaq_symbols[0:subset_symbols]
    nasdaq_data = download_stock_data(nasdaq_symbols, data_window, batch_size)
    
    # Record how many failed to download
    status_list = list({key: value.isna().all()['Adj Close'] for key, value in nasdaq_data.items()}.values())
    print('\n', str(sum(status_list)) + ' out of ' + str(len(status_list)) + ' failed to download')       
    new_row = [Row(stat="data_download", passed=int(len(status_list)-sum(status_list)), failed=int(sum(status_list)), \
        total=int(len(status_list)))]
    stat_df = stat_df.union(spark.createDataFrame(new_row))

    # Remove symbols with NaN
    nasdaq_data = {key: value for key, value in nasdaq_data.items() if not value.isna().all()['Adj Close']}

    # Step 2: Anomaly Detection
    anomaly_data = {}
    for i, (symbol, df) in enumerate(nasdaq_data.items()):
        anomalies = predict_anomalies(symbol, df, cont_val, symbol_count = i+1, total_symbol_no = len(nasdaq_data))
        if anomalies:
            if not anomalies.isEmpty():
                anomaly_data[symbol] = anomalies

    total_count = len(nasdaq_data)
    anomaly_count = len(anomaly_data)
    none_count = total_count - anomaly_count
    print(f'At least one anomaly was detected in {anomaly_count} out of {total_count} symbols')
    new_row = [Row(stat="anomalies_detected", passed=int(anomaly_count), failed=int(none_count), total=int(total_count))]
    stat_df = stat_df.union(spark.createDataFrame(new_row))
    
    # Step 3: Crash Detection and Visualisation
    crash_results = {}
    for symbol in anomaly_data.keys():
        crashes = find_crashes(symbol, anomaly_data[symbol], nasdaq_data[symbol], \
        perc_drop_req, perc_drop_time)
        if crashes:
            if crashes.isEmpty():
                print(f'No crashes detected for {symbol}, moving to next symbol.')
            else:
                print(f'Crashes detected for {symbol}, adding them to crash results.')
                crash_results[symbol] = crashes

    crash_count = len(crash_results)
    none_count = total_count - crash_count
    print(f'At least one crash was detected in {crash_count} out of {total_count} symbols')
    new_row = [Row(stat="crashes_detected", passed=int(crash_count), failed=int(none_count), total=int(total_count))]
    stat_df = stat_df.union(spark.createDataFrame(new_row))

    # Step 4: Plot Crashes
    crash_plots = {}
    for symbol in nasdaq_symbols:
        if symbol in crash_results.keys():
            crash_plot = plot_crashes(nasdaq_data[symbol], crash_results[symbol], symbol)
            if crash_plot:
                crash_plots[symbol] = crash_plot
            plt.close(crash_plot)

    # Step 5: Remove empty values
    crash_results = {key: value for key, value in crash_results.items() if value}
    crash_plots = {key: value for key, value in crash_plots.items() if value} 

    # Step 6: Send Email Notification with Crash Details and Plots
    email_crash_data(crash_results, crash_plots, stat_df, email = receiving_email)

    # # Step 7: Collate stock and crash data for PowerBI
    # collated_crashes = collate_crashes(nasdaq_data, crash_results)
    # collated_crashes.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Execute main function

# COMMAND ----------

if __name__ == "__main__":
    main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Debugging

# COMMAND ----------

# Create df to catch stats
schema = StructType([
    StructField("stat", StringType(), True),   # Column 'stat' as String
    StructField("passed", IntegerType(), True),  # Column 'passed' as Integer
    StructField("failed", IntegerType(), True),   # Column 'failed' as Integer
    StructField("total", IntegerType(), True)   # Column 'total' as Integer
])
stat_df = spark.createDataFrame([], schema)

# Step 1: Download NASDAQ Symbols and Stock Data
nasdaq_symbols = ['AKTX', 'AACG']
nasdaq_data = download_stock_data(nasdaq_symbols, data_window, batch_size)

# Record how many failed to download
status_list = list({key: value.isna().all()['Adj Close'] for key, value in nasdaq_data.items()}.values())
print(str(sum(status_list)) + ' out of ' + str(len(status_list)) + ' failed to download')       
new_row = [Row(stat="data_download", passed=int(len(status_list)-sum(status_list)), failed=int(sum(status_list)), \
    total=int(len(status_list)))]
stat_df = stat_df.union(spark.createDataFrame(new_row))

# Remove symbols with NaN
nasdaq_data = {key: value for key, value in nasdaq_data.items() if not value.isna().all()['Adj Close']}

# Step 2: Anomaly Detection
anomaly_data = {}
for i, (symbol, df) in enumerate(nasdaq_data.items()):
    anomalies = predict_anomalies(symbol, df, cont_val, symbol_count=i, total_symbol_no=len(nasdaq_data))
    if anomalies:
        if not anomalies.isEmpty():
            anomaly_data[symbol] = anomalies

total_count = len(nasdaq_data)
anomaly_count = len(anomaly_data)
none_count = total_count - anomaly_count
print(f'At least one anomaly was detected in {anomaly_count} out of {total_count} symbols')
new_row = [Row(stat="anomalies_detected", passed=int(anomaly_count), failed=int(none_count), total=int(total_count))]
stat_df = stat_df.union(spark.createDataFrame(new_row))

# Step 3: Crash Detection and Visualisation
crash_results = {}
for symbol in anomaly_data.keys():
    crashes = find_crashes(symbol, anomaly_data[symbol], nasdaq_data[symbol], \
    perc_drop_req, perc_drop_time)
    if crashes:
        if crashes.isEmpty():
            print(f'No crashes detected for {symbol}, moving to next symbol.')
        else:
            print(f'Crashes detected for {symbol}, adding them to crash results.')
            crash_results[symbol] = crashes

crash_count = len(crash_results)
none_count = total_count - crash_count
print(f'At least one crash was detected in {crash_count} out of {total_count} symbols')
new_row = [Row(stat="crashes_detected", passed=int(crash_count), failed=int(none_count), total=int(total_count))]
stat_df = stat_df.union(spark.createDataFrame(new_row))

# Step 4: Plot Crashes
crash_plots = {}
for symbol in nasdaq_symbols:
    if symbol in crash_results.keys():
        crash_plot = plot_crashes(nasdaq_data[symbol], crash_results[symbol], symbol)
        if crash_plot:
            crash_plots[symbol] = crash_plot
        plt.close(crash_plot)

# Step 5: Remove empty values
crash_results = {key: value for key, value in crash_results.items() if value}
crash_plots = {key: value for key, value in crash_plots.items() if value} 



# COMMAND ----------

# stock_data = nasdaq_data
# crash_data = crash_results

# # Add symbols to the stock dfs and bind together
# stock_dfs = []
# for key, df in stock_data.items():
#     df["symbol"] = key  # Add the key as a new column
#     stock_dfs.append(df)
# stock_df = pd.concat(stock_dfs, ignore_index=True)

# # Add symbols to the crash dfs and bind together
# crash_df = None
# for key, df in crash_data.items():
#     df = df.withColumn("symbol", lit(key))  # Add the key as a new column
#     if crash_df is None:
#         crash_df = df
#     else:
#         crash_df = crash_df.union(crash_df)

# # Convert stock_df to pyspark df
# stock_df = spark.createDataFrame(stock_df)

# # Join crash and stock data together
# collated_df = stock_df.join(crash_df, (stock_df["symbol"] == crash_df["symbol"]) & (stock_df["date"] == crash_df["min_value_date"]), how="left")
# collated_df.columns
# collated_df = collated_df.select('Adj Close')
# collated_df = collated_df.selectExpr()

