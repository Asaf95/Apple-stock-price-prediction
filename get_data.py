import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time

def get_apple_stock_api():
    path_to_file = "C:/API/AAPL/appleapi.txt"
    with open(path_to_file) as f:
        contents = f.read()
    return contents
    print(contents)

def get_apple_stock_data():
    ts = TimeSeries(key=get_apple_stock_api(), output_format='pandas')
    data, meta_data = ts.get_daily(symbol='AAPL', outputsize = 'full')
    print(meta_data)
    data.to_csv('data/apple_daily_data')
    print(data)