from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold
import pandas as pd; import numpy as np
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns; import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)
score = 'neg_root_mean_squared_error'


def data_analysis_raw_data(x):
    """
    data_analysis_raw_data fun() create data visualizations such as graphs for the independent Variables
    :param x: independent values that according to them we will build the regions
    :return: None
    """
    sns.pairplot(x)
    plt.show()
    sns.heatmap(x.corr(), annot=True); plt.show()
    for column in x.columns:
        #print(column)
        try:
            x[column].value_counts().plot(kind='kde')
            plt.show()
        except:
            continue


def data_analysis_models_output(y_test, predicted) -> None:
    """
    data_analysis_raw_data fun() create data visualizations such as graphs for the independent Variables
    :param y_test: independent values that according to them we will build the regions
    :param predicted: independent values that according to them we will build the regions
    :return: None
     """
    pd.DataFrame({'Error Values': (y_test - predicted)}).plot.kde()
    plt.show()
    # y_test = y_test.sort_values()
    # predicted = predicted.sort_values()
    x = pd.Series(range(0, len(y_test)))
    print(type(y_test))
    print(type(predicted))
    new_series = pd.Series(predicted)
    print(type(new_series))

    new_df = pd.concat([y_test, new_series],axis=1)
    print(new_df)
    new_df = new_df.sort_values(by=['Positive Tests'], ascending=True)
    new_df =new_df.reset_index()
    new_df.fillna(0)

    print(new_df)
    print(len(new_df))
    x = pd.Series(range(0, len(new_df)))
    print(len(x))
    plt.scatter(x,new_df.iloc[:, 2],color = 'blue')
    plt.plot(new_df.iloc[:, 1], color='red')


def data_analysis_compare_models(results_df) -> None:
    """
    data_analysis_compare_models fun() create data visualizations results of the regressions  independent Variables
    :results_df: holds the results of the regressions indicators
    :return: None
     """
    results_df.set_index('Model', inplace=True)
    results_df['R2 Square'].plot(kind='barh', figsize=(12, 8))
    plt.show()


def degree_chooser(x_train, x_test, y_train, y_test) -> int:
    """
     degree_chooser find and describe what is the best degree to use

     :param x_train: This includes your all independent variables,these will be used to train the model
     :param x_test: This is remaining  portion of the independent variables from the data,
            will be used to make predictions to test the accuracy of the model.
     :param y_train: This is your dependent variable which needs to be predicted by this model
     :param y_test: This data has category labels for your test data,
            these labels will be used to test the accuracy between actual and predicted categories.

     :return: pd.DataFrame that hold the indices of regression quality for the Regression
                in addition to the other regressions indices

     credit: I used this page to do this function
     https://stackoverflow.com/questions/47442102/how-to-find-the-best-degree-of-polynomials
     """
    #https://stackoverflow.com/questions/47442102/how-to-find-the-best-degree-of-polynomials
    rmses = []
    degrees = np.arange(1, 10)
    min_rmse, min_deg = 1e10, 0

    for deg in degrees:

        # Train features
        poly_features = PolynomialFeatures(degree=deg, include_bias=False)
        x_poly_train = poly_features.fit_transform(x_train)

        # Linear regression
        poly_reg = LinearRegression()
        poly_reg.fit(x_poly_train, y_train)

        # Compare with test data
        x_poly_test = poly_features.fit_transform(x_test)
        poly_predict = poly_reg.predict(x_poly_test)
        poly_mse = mean_squared_error(y_test, poly_predict)
        poly_rmse = np.sqrt(poly_mse)
        rmses.append(poly_rmse)

        # Cross-validation of degree
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg

    #print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(degrees, rmses)
    ax.set_yscale('log')
    ax.set_xlabel('Degree')
    ax.set_ylabel('RMSE')
    return min_deg


def best_parameters_ridge(x_train, y_train) -> float:
    """this function use the GridSearchCV library tp find the best alpha for the ridge regression"""
    model = Ridge(max_iter=11111)
    alpha = dict()
    alpha['alpha'] = np.arange(0.01, 1, 0.01)
    search= GridSearchCV(estimator=model, param_grid=alpha, scoring=score, cv=cv, n_jobs=-1)
    results = search.fit(x_train, y_train)
    return search.best_params_['alpha']


def best_parameters_lasso(x_train, y_train) -> float:
    """this function use the GridSearchCV library tp find the best alpha for the lasso regression"""
    model = Lasso(max_iter=11111)
    alpha = dict()
    alpha['alpha'] = np.arange(0.001, 1, 0.001)  # jump of 0.001 its the best for ridge
    search= GridSearchCV(estimator=model, param_grid=alpha, scoring=score, cv=cv, n_jobs=-1)
    results = search.fit(x_train, y_train)
    return search.best_params_['alpha']


def cross_val(model) -> float:
    """
     cross_val  Get predictions from each split of cross-validation for diagnostic purposes..

     :param model: the current model that we look for the cross validation for

     :return: mean() of the predictions
     """
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()


def remove_columns_corr(dataset, threshold) -> pd.DataFrame:
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        dataset: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''
    #https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return dataset


def print_evaluate(true, predicted) -> None:
    """
     print evaluate print the values of the indicators to the regression model
     :param true: the ture values (y_test)
     :param predicted: what the model think the results are
     :return: describe what it returns
     """
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__')


def evaluate(true, predicted) -> float:
    """
     evaluate function create evaluates for the model according to predicted results Compared to the real results.

     :param true: what is the real results for the  x_test
     :param predicted: what is the predicted results for the x_test with the regression
     :return: mae, mse, rmse, re_square
        mae: is the easiest to understand, because it's the average error.
        mse: is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
        rmse: is even more popular than MSE, because RMSE is interpretable in the "y" units.
        r2_square: is the proportion of the variation in the dependent variable that is predictable from the independent variable
     """
    "This fucncton gets two parametrs"
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


def backward_elimination(data, target, significance_level=0.05) -> pd.DataFrame:
    """
     backward_elimination is a feature selection technique while building a machine learning model.
        It is used to remove those features that do not have a significant effect on the dependent variable.
     took from https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/

     :param data: describe about parameter p1
     :param target: describe about parameter p2
     :param significance_level: describe about parameter p3
     :return: describe what it returns
     """
    data = pd.DataFrame(data=data)
    features = data.columns.tolist()
    while (len(features) > 0):
        features_with_constant = sm.add_constant(data[features])
        # features_with_constant = features_with_constant.drop(['const'],axis = 1)
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if (max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return data[features]


def linear_reg(x_train, x_test, y_train, y_test) -> pd.DataFrame:
    """
     linear_reg predicted according to his Regression models a target prediction value based on independent variables

     :param x_train: This includes your all independent variables,these will be used to train the model
     :param x_test: This is remaining  portion of the independent variables from the data,
            will be used to make predictions to test the accuracy of the model.
     :param y_train: This is your dependent variable which needs to be predicted by this model
     :param y_test: This data has category labels for your test data,
            these labels will be used to test the accuracy between actual and predicted categories.

     :return: pd.DataFrame that hold the indices of regression quality for the Regression
                in addition to the other regressions indices
     """
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    test_pred = lin_reg.predict(x_test)

    results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred), cross_val(lin_reg)]],
                              columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
    #data_analysis_models_output(y_test,test_pred)
    #print_evaluate(y_test,test_pred)
    return results_df


def poly_reg(x_train, x_test, y_train, y_test, results_df, min_d) -> pd.DataFrame:
    """
     polynomail_reg predicted according to Regression models a target prediction value based on independent variables

     :rtype:
     :param results_df: has indicators for the other regression until now
     :param x_train: This includes your all independent variables,these will be used to train the model
     :param x_test: This is remaining  portion of the independent variables from the data,
            will be used to make predictions to test the accuracy of the model.
     :param y_train: This is your dependent variable which needs to be predicted by this model
     :param y_test: This data has category labels for your test data,
            these labels will be used to test the accuracy between actual and predicted categories.

     :return: pd.DataFrame that hold the indices of regression quality for the Regression
                in addition to the other regressions indices
     """
    poly_reg = PolynomialFeatures(degree=min_d)
    x_train_2_d = poly_reg.fit_transform(x_train)
    x_test_2_d = poly_reg.transform(x_test)
    lin_reg = LinearRegression()
    lin_reg.fit(x_train_2_d, y_train)

    test_pred = lin_reg.predict(x_test_2_d)

    results_df_2 = pd.DataFrame(data=[["Polynomail Regression", *evaluate(y_test, test_pred), cross_val(lin_reg)]],
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
    results_df = results_df.append(results_df_2, ignore_index=True)
    #data_analysis_models_output(y_test,test_pred)
    #print_evaluate(y_test,test_pred)

    return results_df


def ridge_reg(x_train, x_test, y_train, y_test, results_df) -> pd.DataFrame:
    """
     ridge_reg predicted according to Regression models a target prediction value based on independent variables

     :param x_train: This includes your all independent variables,these will be used to train the model
     :param x_test: This is remaining  portion of the independent variables from the data,
            will be used to make predictions to test the accuracy of the model.
     :param y_train: This is your dependent variable which needs to be predicted by this model
     :param y_test: This data has category labels for your test data,
            these labels will be used to test the accuracy between actual and predicted categories.

     :return: pd.DataFrame that hold the indices of regression quality for the Regression
                in addition to the other regressions indices
     """
    best_alpha = best_parameters_ridge(x_train,y_train)
    model = Ridge(alpha=best_alpha, solver='cholesky', tol=0.0001, random_state=2)

    model.fit(x_train, y_train)

    test_pred = model.predict(x_test)

    results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred), cross_val(Ridge())]],
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
    results_df = results_df.append(results_df_2, ignore_index=True)
    #data_analysis_models_output(y_test,test_pred)
    #print_evaluate(y_test,test_pred)
    return results_df


def lasso_reg(x_train, x_test, y_train, y_test, results_df) -> pd.DataFrame:
    """
     lasso_reg predicted according to Regression models a target prediction value based on independent variables

     :param x_train: This includes your all independent variables,these will be used to train the model
     :param x_test: This is remaining  portion of the independent variables from the data,
            will be used to make predictions to test the accuracy of the model.
     :param y_train: This is your dependent variable which needs to be predicted by this model
     :param y_test: This data has category labels for your test data,
            these labels will be used to test the accuracy between actual and predicted categories.

     :return: pd.DataFrame that hold the indices of regression quality for the Regression
                in addition to the other regressions indices
     """
    best_alpha = best_parameters_lasso(x_train, y_train)
    model = Lasso(alpha=best_alpha,
                  precompute=True,
                  positive=True,
                  selection='random',
                  tol=0.0001,
                  max_iter=11111,
                  random_state=2)

    model.fit(x_train, y_train)
    test_pred = model.predict(x_test)
    results_df_2 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, test_pred), cross_val(model)]],
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    results_df = results_df.append(results_df_2, ignore_index=True)
    #data_analysis_models_output(y_test,test_pred)
    #print_evaluate(y_test,test_pred)
    return results_df


def random_forrest(x_train, x_test, y_train, y_test, results_df) -> pd.DataFrame:
    """
     random_forrest predicted according to Regression models a target prediction value based on independent variables

     :param x_train: This includes your all independent variables,these will be used to train the model
     :param x_test: This is remaining  portion of the independent variables from the data,
            will be used to make predictions to test the accuracy of the model.
     :param y_train: This is your dependent variable which needs to be predicted by this model
     :param y_test: This data has category labels for your test data,
            these labels will be used to test the accuracy between actual and predicted categories.

     :return: pd.DataFrame that hold the indices of regression quality for the Regression
                in addition to the other regressions indices
     """
    rf = RandomForestRegressor(random_state=2)
    param_grid = {'max_depth': np.arange(5, 10)}
    search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=score, cv=cv)
    results = search.fit(x_train, y_train)

    best_par = results.best_params_['max_depth']
    rf_reg = RandomForestRegressor(n_estimators=1000, max_depth=best_par)
    rf_reg.fit(x_train, y_train)

    test_pred = rf_reg.predict(x_test)

    results_df_2 = pd.DataFrame(data=[["Random Forest Regressor", *evaluate(y_test, test_pred), cross_val(rf_reg)]],
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
    results_df = results_df.append(results_df_2, ignore_index=True)
    #data_analysis_models_output(y_test,test_pred)
    #print_evaluate(y_test,test_pred)

    return results_df


def kn_reg(x_train, x_test, y_train, y_test, results_df) -> pd.DataFrame:
    """
     kn_reg predicted according to Regression models a target prediction value based on independent variables

     :param x_train: This includes your all independent variables,these will be used to train the model
     :param x_test: This is remaining  portion of the independent variables from the data,
            will be used to make predictions to test the accuracy of the model.
     :param y_train: This is your dependent variable which needs to be predicted by this model
     :param y_test: This data has category labels for your test data,
            these labels will be used to test the accuracy between actual and predicted categories.

     :return: pd.DataFrame that hold the indices of regression quality for the Regression
                in addition to the other regressions indices
     """
    KNC = KNeighborsClassifier()
    grid = {'n_neighbors': np.arange(1, 7)}
    search = GridSearchCV(estimator=KNC, param_grid=grid, scoring=score, cv=cv)
    results = search.fit(x_train, y_train)
    x = results.best_params_
    best_n = x['n_neighbors']
    model = KNeighborsClassifier(n_neighbors=best_n, metric='minkowski', p=2)
    model.fit(x_train, y_train)
    test_pred = model.predict(x_test)

    results_df_2 = pd.DataFrame(data=[["KNeighbors Regressor", *evaluate(y_test, test_pred),'NAN']],
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
    results_df = results_df.append(results_df_2, ignore_index=True)
    #data_analysis_models_output(y_test,test_pred)
    #print_evaluate(y_test,test_pred)

    return results_df


def main(c1, cu_y):
    global y
    global X

    y = cu_y
    c1 = backward_elimination(c1, y)
    c1 = remove_columns_corr(c1, 0.95)
    X = c1


    #data_analysis_raw_data(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #  Linear Regression
    results_df = linear_reg(x_train, x_test, y_train, y_test)

    # Polynomial_Regression.
    min_d = degree_chooser(x_train, x_test, y_train, y_test)
    results_df = poly_reg(x_train, x_test, y_train, y_test, results_df,min_d)

    # Ridge_Reg
    results_df = ridge_reg(x_train, x_test, y_train, y_test, results_df)

    # Lasso_Regression
    results_df = lasso_reg(x_train, x_test, y_train, y_test, results_df)

    # Random Forest Regressor.
    results_df = random_forrest(x_train, x_test, y_train, y_test, results_df)

    # KNeighbors Regressor.
#    results_df = kn_reg(x_train, x_test, y_train, y_test, results_df)

    # Models Comparison
    results_df.set_index('Model', inplace=True)

    return results_df