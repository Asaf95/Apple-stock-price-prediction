import pandas as pd
import data_preparation
import ml_classification_models
import masho



if __name__ == "__main__":

    #get_data.get_apple_stock_data() # get/ update apple stock data
    raw_data = pd.read_csv('data/apple_daily_data')
    prepare_data = data_preparation.basic_prepare(raw_data)
    recursive_predictions = masho.LSTM_prepare_variables(prepare_data)
    print(recursive_predictions)
    results = ml_classification_models.classification_models(prepare_data)
    print(results)