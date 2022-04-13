import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time

"""
this module prepare the data for the ML models.
function basic_prepare() -
    basic naming for the columns names and format the values to the right format (string/ int/ date)
    this function also calls the apply_stock_split and return to the main module the 
function apply_stock_split() -
    normalize the values of the stock according to the current value of the stock share


"""

def get_apple_stock_api():
    path_to_file = "C:/API/AAPL/appleapi.txt"
    with open(path_to_file) as f:
        contents = f.read()
        print('a  a')
    return contents


def get_apple_stock_data():
    ts = TimeSeries(key=get_apple_stock_api(), output_format='pandas')
    data, meta_data = ts.get_daily(symbol='AAPL', outputsize = 'full')
    print(meta_data)
    data.to_csv('data/apple_daily_data')
    print(data)
    print('tsting this')