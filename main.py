import pandas as pd
import data_preparation




if __name__ == "__main__":

    #get_data.get_apple_stock_data() # get/ update apple stock data
    df = pd.read_csv('data/apple_daily_data')
    data_preparation.basic_prepare(df)