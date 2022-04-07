import pandas as pd


def apply_stock_split1(df, number, date):
    list_of_columns = ['high', 'low', 'close', 'open']
    list_of_columns1 = ['Volume']
    for item in list_of_columns:
        df['temp'] = df[(df['date'] < date)][item].div(number)
        df['temp2'] = df['temp'].fillna(df[item])
        df[item] = df['temp2']

    df['temp'] = df[(df['date'] < date)]['volume'].div(1 / number)
    df['temp2'] = df['temp'].fillna(df['volume'])
    df['volume'] = df['temp2']
    return df


def basic_prepare(df):
    """

    :param df:
    :return:
    """
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low',
                            '4. close': 'close','5. volume': 'volume'})
    df_org= df.copy()

    df['date'] = df.apply(lambda x: pd.to_datetime(x['date'], format='%Y-%m-%d'), axis=1)


    """
    handle the stock splits problem
    """
    thisdict = {
        "2020-08-31": 4,
        "2014-06-9": 7,
        "2005-02-28": 2,
        "2000-06-21": 2,
        "1987-06-16": 2
    }
    for key, value in thisdict.items():
        df = apply_stock_split1(df,value, key)

    df =df.drop(columns=['temp', 'temp2'])

    return df
