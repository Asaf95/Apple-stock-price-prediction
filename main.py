import pandas as pd
import data_preparation
#import ml_classification_models
import model_lstm
#import get_data


"""
the main module calls all the other modules and have access to all the functions that are needed 
for building the stock model
the main three modules are:
1. get_data()- using this module for getting the last stock price and the history prices.
2. data_preparation()- using this module to prepare the data for the ML models by fixing the types of columns 
fixing the stock splits issue and more.
3. model_lstm()- predict the stock value and is the main ML model in this project.
"""


def main():
    #get_data.get_apple_stock_data()
    raw_data = pd.read_csv('data/apple_daily_data')
    prepare_data = data_preparation.basic_prepare(raw_data)
    recursive_predictions = model_lstm.LSTM_prepare_variables(prepare_data)
    print(recursive_predictions)
    #results = ml_classification_models.classification_models(prepare_data)
    #print(results)


if __name__ == "__main__":
    main()
