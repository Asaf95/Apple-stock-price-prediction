import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from copy import deepcopy


def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)


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


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n - i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df


def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)


def LSTM_model(q_80,q_90,dates_train, X_train, y_train,dates_val,
               X_val, y_val,dates_test, X_test, y_test):

    model = Sequential([layers.Input((3, 1)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    train_predictions = model.predict(X_train).flatten()

    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.legend(['Training Predictions', 'Training Observations'])
    #plt.show()

    val_predictions = model.predict(X_val).flatten()

    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.legend(['Validation Predictions', 'Validation Observations'])
    #plt.show()

    test_predictions = model.predict(X_test).flatten()

    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Testing Predictions', 'Testing Observations'])
    #plt.show()

    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Training Predictions',
                'Training Observations',
                'Validation Predictions',
                'Validation Observations',
                'Testing Predictions',
                'Testing Observations'])
    #plt.show()

    recursive_predictions = []
    recursive_dates = np.concatenate([dates_val, dates_test])

    for target_date in recursive_dates:
        last_window = deepcopy(X_train[-1])
        next_prediction = model.predict(np.array([last_window])).flatten()
        recursive_predictions.append(next_prediction)
        last_window[-1] = next_prediction

    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.plot(recursive_dates, recursive_predictions)
    plt.legend(['Training Predictions',
                'Training Observations',
                'Validation Predictions',
                'Validation Observations',
                'Testing Predictions',
                'Testing Observations',
                'Recursive Predictions'])
    #plt.show()


def LSTM_prepare_variables(df):

    df_close = df.copy()
    df_close = df_close[['date', 'close']]

    df_close.index = df_close.pop('date')
    df_close = df_close.iloc[::-1]
    windowed_df = df_to_windowed_df(df_close,
                                    '2021-03-25',
                                    '2022-03-23',
                                    n=3)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)

    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    LSTM_model(q_80, q_90, dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test)


def basic_prepare(df):
    """

    :param df:
    :return:
    """
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low',
                            '4. close': 'close','5. volume': 'volume'})
    from datetime import datetime
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
        print(key, ' and ', value)
        df = apply_stock_split1(df,value, key)

    df =df.drop(columns=['temp', 'temp2'])

    #LSTM_prepare_variables(df)