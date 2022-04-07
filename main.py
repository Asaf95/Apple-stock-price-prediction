import pandas as pd
import data_preparation
import ml_classification_models
import masho



if __name__ == "__main__":

    #get_data.get_apple_stock_data() # get/ update apple stock data
    df = pd.read_csv('data/apple_daily_data')
    df = data_preparation.basic_prepare(df)
    masho.LSTM_prepare_variables(df)
    print(df)
    results = ml_classification_models.classification_models(df)