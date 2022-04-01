import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import get_data

def data_analysis(df):
    """

    :param df:
    :return:
    """
    df.plot( x='date', y='close')
    plt.show()



if __name__ == "__main__":

    #get_data.get_apple_stock_data() # get/ update apple stock data
    df = pd.read_csv('data/apple_daily_data')
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low',
                            '4. close': 'close','5. volume': 'volume'})
    print(df)
    data_analysis(df)