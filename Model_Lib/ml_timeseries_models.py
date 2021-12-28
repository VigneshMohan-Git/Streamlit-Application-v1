from ml_settings_ import standard_init as std
import ml_subfile_ as var
import pandas as pd
import numpy as np
#import q_model_hp as hp



def get_numbers_from_filename(filename):
    filename = re.sub('[^A-Za-z0-9]+', '', filename)
    filename = re.search(r'\d+', filename).group(0)
    return filename

"""
String Name       
"""
def model_str(_p_model_):
    _p_model_str = int(_p_model_)
    #Default as KNN
    if _p_model_str == 1:
        return 'HOLTS WINTER ES'
    elif _p_model_str == 2:
        return 'ARIMA'
    elif _p_model_str == 3:
        return 'SARIMA'
    elif _p_model_str == 4:
        return 'LSTM'     
    else:
        return 'NA'

"""
ABSOLUTE
"""
def cal(data):
    Accuracy = (100 - abs(data - 100)).mean()
    return Accuracy
   

"""
ABSOLUTE ACCURACY
"""
def absacc_(y_test, y_pred):
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    # Calculate Accuracy Level
    Accuracy = (y_pred/y_test)*100
    F_Accuracy = (100 - abs(Accuracy - 100)).mean()
    
    # Calculate the Root Mean Squared Error
    error = y_test - y_pred
    rmse = np.sqrt(np.mean(error**2))
    
    return round(F_Accuracy,2), round(rmse,2)
    
"""
MAKE NEW DIRECTORIES
"""
def create_():
    import os
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

"""
LOGGING
"""
def log_():
    import logging
    LOG_FILENAME = "logfile.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename= var.std.vOutputFolder + LOG_FILENAME,filemode='w', level=logging.DEBUG, encoding='utf-8')    
    
    
"""
Creates time series features from datetime index
"""
def create_time_features(df, target=None):
    from datetime import datetime as dt
    
    df['date'] = df.index
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    X = df.drop(['date'], axis=1)
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y

    return X, y




"""
Minmax scaler
"""
def scale_(x):
    from sklearn.preprocessing import MinMaxScaler
    # fit scaler
    mms = MinMaxScaler(feature_range=(0,1))
    norm = mms.fit_transform(x)
    return norm


"""
LAG_, Corelation Coefficient
"""
def lag_cor_(X):    
    """
    #Introducing a lag value
    from itertools import product 
    lags = [-3,-2]
    for col,lag  in product(X.columns,lags):
        X[col+'_'+str(lag)] = X[col].shift(lag).fillna(0)
    """
    
    # Selecting highly corelated features
    cor_matrix = X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

    #Dropping highly corelated columns
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

    X = X[X.columns.difference(to_drop)]

    return X



"""
MODEL LOSS CROSS ENTROPY

An important aspect of this is that cross entropy loss 
penalizes heavily the predictions that are confident but
wrong.
"""
def cross_entropy(predictions, targets, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5)))/N
    return ce_loss




"""
MODEL ERROR
"""
def model_eval(y, predictions):

    
    # Import library for metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.metrics import accuracy_score
    
    # Mean absolute error (MAE)
    mae = mean_absolute_error(y, predictions)

    # Mean squared error (MSE)
    mse = mean_squared_error(y, predictions)


    # SMAPE is an alternative for MAPE when there are zeros in the testing data. It
    # scales the absolute percentage by the sum of forecast and observed values
    SMAPE = np.mean(np.abs((y - predictions) / ((y + predictions)/2))) * 100


    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y, predictions))

    # Calculate the Mean Absolute Percentage Error
    # y, predictions = check_array(y, predictions)
    MAPE = np.mean(np.abs((y - predictions) / y)) * 100

    # mean_forecast_error
    mfe = np.mean(y - predictions)

    # NMSE normalizes the obtained MSE after dividing it by the test variance. It
    # is a balanced error measure and is very effective in judging forecast
    # accuracy of a model.

    # normalised_mean_squared_error
    NMSE = mse / (np.sum((y - np.mean(y)) ** 2)/(len(y)-1))
    
    # Accuracy
    Accuracy = (predictions / y) * 100
    
    # theil_u_statistic
    # It is a normalized measure of total forecast error.
    error = y - predictions
    mfe = np.sqrt(np.mean(predictions**2))
    mse = np.sqrt(np.mean(y**2))
    rmse = np.sqrt(np.mean(error**2))
    theil_u_statistic =  rmse / (mfe*mse)
    
    
    return round(mae,3), round(rmse,3), round(MAPE,3), round(mfe,3), round(NMSE,3), cal(Accuracy)# round(theil_u_statistic,3)




"""
SEASONAL DECOMPOSITION
"""
from statsmodels.tsa.seasonal import seasonal_decompose
def combine_seasonal_cols(input_df, seasonal_model_results):
    """Adds inplace new seasonal cols to df given seasonal results

    Args:
        input_df (pandas dataframe)
        seasonal_model_results (statsmodels DecomposeResult object)
    """
    # Add results to original df
    #input_df['observed'] = seasonal_model_results.observed
    input_df['residual'] = seasonal_model_results.resid
    input_df['seasonal'] = seasonal_model_results.seasonal
    input_df['trend'] = seasonal_model_results.trend
   

   
"""
ERROR CALCULATION
"""
def accuracy_(data):
    data['Accuracy'] = (data['Predicted'] / data['Sales']) * 100
    return data


"""
MODEL ERROR CALCULATION
"""
def error_(data):
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error
    
    MAE  = round(mean_absolute_error(data['Sales'], data['Predicted']), 4)
    R2   = round(r2_score(data['Sales'], data['Predicted']), 4)
    MSE  = round(mean_squared_error(data['Sales'], data['Predicted']), 4)
    #AS   = accuracy_score(data['Sales'], np.round(abs(data['Predicted'])), normalize=False)
    #AS = (data['Predicted'] / data['Sales']) * 100
    #CrossEntropy = cross_entropy(data['Sales'], data['Predicted'], epsilon=1e-10)
    
    return MAE, MSE, R2




"""
TRAIN TEST SPLIT
"""
def norsplit_(X):
    #X = X.astype('float32')
    #Y = Y.astype('float32')
    x_train = X[:int(X.shape[0] * var.std.vTEST_RATIO)]
    x_test  = X[int(X.shape[0] * var.std.vTEST_RATIO):]
    #y_train = Y[:int(X.shape[0] * var.std.vTEST_RATIO)]
    #y_test  = Y[int(X.shape[0] * var.std.vTEST_RATIO):]
    
    return x_train, x_test
    
    
def split_(X):
    X = X.astype('float32')
    Y = Y.astype('float32')
    BREAK_DATE = pd.to_datetime(var.std.vDate)
    
    x_train = X[X.index <= BREAK_DATE]
    x_test = X[X.index > BREAK_DATE]
   
    y_train = Y[Y.index <= BREAK_DATE]
    y_test = Y[Y.index > BREAK_DATE]
    
    return x_train, x_test, y_train, y_test
    
    
def train_test_split(data):
    train_set, test_set = np.split(data, [int(var.std.vTEST_RATIO *len(data))])
    return train_set, test_set    
 
    
    
"""
DEFINE FUTURE DATAFRAME
"""
def future_df_(x_train, y_train, FORECASTING_STEPS_AHEAD, YEARS, MONTHS):
    
    # Future dates
    from datetime import timedelta
    import calendar
    periods = FORECASTING_STEPS_AHEAD
    start_date = y_train.index[-1]
    freq='m'
    days_in_month = calendar.monthrange(start_date.year, start_date.month)[1]
    start_date = start_date + timedelta(days=days_in_month)
    dates = pd.date_range(
            start=start_date,
            periods=periods + 1,  # An extra in case we include start
            freq=freq)
    dates = dates[dates > start_date]  # Drop start if equals last_date
    dates = dates[:periods]
    dates = pd.Series(dates)
    dates = dates.apply(lambda dt: dt.replace(day=15))
    dates = pd.DataFrame(dates,columns=['key'])
    
    #Future values
    df = pd.DataFrame(columns= x_train.columns)
    data = pd.DataFrame(columns=x_train.columns)
    years = YEARS
    months = MONTHS
    x_years = iter(years)
    x_months = iter(months)
    for i in years:
        for j in months:
            this_column = df.columns
            df = df.append(x_train.loc[(x_train.index >= pd.datetime(i,j,15)) & (x_train.index <= pd.datetime(i,j,15))])

    for h in months:
        data = data.append(df.loc[(df.index.month == h)].mean(),ignore_index=True)
        
    # Concatination 
    future_df = pd.concat([dates, data],axis=1)
    future_df = future_df.set_index(pd.DatetimeIndex(future_df['key']))
    future_df.drop(['key'],axis=1,inplace=True)
    
    return future_df

"""
******************************** MODELS *************************************
"""
"""
RandomForestRegressor
"""
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
def random_forest_(x_train, x_test, y_train, y_test, X,Y):
    # fit model
    model = RandomForestRegressor(n_estimators=100, min_samples_split=2, n_jobs=1).fit(x_train, y_train)
    
    #Model Score
    #score = cross_val_score(model, x_test, y_test, scoring='accuracy', cv=2, n_jobs=-1)
    # make a one-step prediction
    yhat = model.predict(x_test)
    
    # DataFrame
    predicted_df = pd.DataFrame(yhat, columns= ['Predicted'])
    predicted_df.index = y_test.index
    
    
    return y_test, predicted_df #, score


"""
KNeighborsRegressor
"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
def knn_(x_train, x_test, y_train, y_test,X,Y):
    leaf_size = list(range(1,50))
    n_neighbors = list(range(1,30))
    p=[1,2]
    
    #hyperparameters
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    
    #Create new KNN object
    knn = KNeighborsRegressor()
    
    #Use GridSearch
    clf = GridSearchCV(knn, hyperparameters, cv=10)
    # fit model
    grid_result = clf.fit(x_train, y_train)
    
    #best params
    model = KNeighborsRegressor(**grid_result.best_params_).fit(x_train, y_train)
    
    #Model_score
    #score = cross_val_score(model, x_test, y_test, scoring='accuracy', cv=2, n_jobs=-1)
    
    # make a one-step prediction
    yhat = model.predict(x_test)
    
    # DataFrame
    predicted_df = pd.DataFrame(yhat, columns= ['Predicted'])
    predicted_df.index = y_test.index
    
    
    return y_test, predicted_df#, score


"""
ExtraTreesRegressor
"""
from sklearn.ensemble import ExtraTreesRegressor
def extratree_(x_train, x_test, y_train, y_test,X,Y):
    n_estimators = list(range(1,2))
    n_jobs = list(range(1,2))
    min_samples_split = list(range(2,3))
    min_samples_leaf = list(range(2,3))
    
    #hyperparameters
    hyperparameters = dict(n_estimators= n_estimators,
                           n_jobs= n_jobs,
                           min_samples_split= min_samples_split,
                           min_samples_leaf= min_samples_leaf)
    
    #Create new ETR object
    ETR = ExtraTreesRegressor()
    #Use GridSearch
    clf = GridSearchCV(ETR, hyperparameters,scoring='r2', cv=2)
    # fit model
    grid_result = clf.fit(x_train, y_train)
    
    #best params
    model = ExtraTreesRegressor(**grid_result.best_params_).fit(x_train, y_train)
    #score = cross_val_score(model, x_test, y_test, scoring='accuracy', cv=2, n_jobs=-1)
    # make a one-step prediction
    yhat = model.predict(x_test)
    
    # DataFrame
    predicted_df = pd.DataFrame(yhat, columns= ['Predicted'])
    predicted_df.index = y_test.index
    
    
    return y_test, predicted_df#, score

"""
SVM
"""
from sklearn import svm
def svm_(x_train, x_test, y_train, y_test,X,Y):
    
    #Create new SVM object
    model = svm.SVR(kernel ='sigmoid', C=3, degree=1, gamma = 'auto').fit(x_train, y_train)
    
    #Score
    #score = cross_val_score(model, x_test, y_test, scoring='accuracy', cv=2, n_jobs=-1)
    
    # make a one-step prediction
    yhat = model.predict(x_test)
    
    # DataFrame
    predicted_df = pd.DataFrame(yhat, columns= ['Predicted'])
    predicted_df.index = y_test.index
 
    
    return y_test, predicted_df#, score


"""
Holts Winter ExponentialSmoothing
"""
from statsmodels.tsa.api import ExponentialSmoothing
def holts_winter_(x_train, x_test):
    #Guessed Parameter
    span = 12
    alpha= 0.1 #2/(span+1) #0.15
    beta = 0.4
    gamma= 0.1
    #Actual_df = Y

    #Model
    from statsmodels.tsa.api import ExponentialSmoothing
    model = ExponentialSmoothing(x_train, trend='mul',seasonal='mul',
                                seasonal_periods=12)
    model.exog = x_train
    model_fit = model.fit(smoothing_level=alpha,smoothing_slope=beta,
                          smoothing_seasonal=gamma) 
    #smoothing_level=alpha,smoothing_slope=beta,smoothing_seasonal=gamma
    #model_fit.exog = x_test
   
    #score = cross_val_score(model, x_test, y_test, scoring='accuracy', cv=2, n_jobs=-1)
    
    yhat = model_fit.forecast(len(x_test))
    predicted_df = pd.DataFrame(yhat, columns=['Predicted'])
    predicted_df.index = x_test.index

    
    return x_test, predicted_df


"""
AUTO ARIMA
"""
import pmdarima as pm
def auto_arima_(x_train, x_test):
    print("...")
    print("AUTO ARIMA")
    #Auto Arima with Seasonality
    import pmdarima as pm
    from pmdarima import model_selection
    smodel = pm.auto_arima(x_train, 
                             start_p=0, start_q=0, d=1,
                             max_p=10, max_q=10, max_d=10,
                             test='adf',
                             trace=True,
                             stationary=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True)

    #smodel.summary()
    smodel_fit = smodel.fit(x_train)
    
    """
    # Cross Validation
    cv = model_selection.SlidingWindowForecastCV(window_size=100, step=24, h=1)
    score = model_selection.cross_val_score(smodel_fit,y_test, scoring='smape', cv=cv, verbose=2)
    """
    prediction_values = smodel_fit.predict(n_periods=len(x_test))

    predicted_df = pd.DataFrame(prediction_values, columns=['Predicted'])
    predicted_df.index = x_test.index
    
    
    return x_test, predicted_df#, score


"""
SARIMAX
"""
import pmdarima as pm
def auto_sarima_(x_train, x_test):
    print("...")
    print("AUTO ARIMA WITH SEASONALITY")
    #Auto Arima with Seasonality
    import pmdarima as pm
    smodel = pm.auto_arima(x_train,  
                             start_p=0, start_q=0, d=1,
                             start_P=0, start_Q=0, D=1,
                             max_p=3, max_q=3, max_d=3,
                             max_P=3, max_Q=3, max_D=3,
                             test='adf', #kpss #adf
                             stationary=True,
                             m=12,
                             alpha=0.7,
                             seasonal=True, 
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True)

    #smodel.summary()
    smodel_fit = smodel.fit(x_train)
    prediction_values = smodel_fit.predict(n_periods=len(x_test))

    predicted_df = pd.DataFrame(prediction_values, columns=['Predicted'])
    predicted_df.index = x_test.index
    
    
    return x_test, predicted_df


"""
PROPHET
"""
from fbprophet import Prophet
def prophet_(x_train, x_test, y_train, y_test,X,Y):
    df0_copy = pd.concat([X,Y],axis=1)
    df0_copy = df0_copy.reset_index()
    df_copy = df0_copy.rename(columns={'key':'ds','Sales':'y'})

    BREAK_DATE = pd.to_datetime(var.std.vDate)
    df_train = df_copy.loc[df_copy["ds"] <= BREAK_DATE]
    df_test  = df_copy.loc[df_copy["ds"] > BREAK_DATE]
    
    #df_train = df_copy[:int(df_copy.shape[0] * var.std.vTEST_RATIO)]
    #df_test  = df_copy[int(df_copy.shape[0] * var.std.vTEST_RATIO):]
    
    #print('LEN TRAIN_DATA:',len(df_train))
    #print('LEN TEST_DATA:',len(df_test))
    
    feature_df = pd.read_csv(var.std.vFeatureFile)
    ext_var = feature_df.Specs.dropna()
    
    # Looping and fitting the models with external variables
    print("PROPHET MODEL TRAINING...")
    for i in ext_var:
        model = Prophet(weekly_seasonality=True, daily_seasonality=True)
        model.add_country_holidays(country_name= var.std.vCountryCode)
        #model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_regressor(i)
        model.fit(df_train)

    pred = model.predict(df_test.drop(columns='y'))
    #pred_df = pred[['ds','yhat','yhat_lower','yhat_upper']]
    #model.plot(pred_df)
    
    pred_df = pred[['ds','yhat']]
    pred_df = pred_df.rename(columns={'ds':'key','yhat':'Predicted'})
    
    Actual_df = df_test[['ds','y']]
    Actual_df = Actual_df.reset_index(drop=True)
    Actual_df = Actual_df.rename(columns={'ds':'key','y':'Sales'})
    Actual_df = Actual_df['Sales']
    
    #merged_df = pd.merge(Actual_df, pred_df, on='ds')
    #merged_df = merged_df[['ds','y','yhat']]
    #merged_df = merged_df.rename(columns={'ds':'key', 'y':'Sales','yhat':'Predicted'})
    #merged_df['Accuracy'] = (merged_df['Predicted'] / merged_df['Sales']) * 100
    #merged_df = merged_df[['key','Sales','Predicted','Accuracy']]
    
    
    return Actual_df, pred_df


"""
LSTM
"""
def lstm_(x_train, x_test):

    import math
    import torch.optim as OPTIM
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, Dropout,BatchNormalization
    from keras.layers import LSTM
    
    x_train2 = x_train.values.reshape(x_train.shape + (1,))
    #y_train2 = y_train.values.reshape(y_train.shape + (1,))
    x_test2 = x_test.values.reshape(x_test.shape + (1,))
    #y_test2 = y_test.values.reshape(y_test.shape + (1,))
    
    
    ''' Fitting the data in LSTM Deep Learning model '''
    model = Sequential()
    model.add(LSTM(100, input_shape=(x_train2.shape[1], x_train2.shape[2])))
    model.add(Dense(100,activation='relu',input_dim=3))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(1))
    model.add(Dropout(0.9))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(x_train2 ,epochs=1000, batch_size=50, validation_data=(x_test2), verbose=1, shuffle=False)
    
    #model.compile(optimizer='adam', loss='mae', metrics='accuracy')
    #history = model.fit(x_train, y_train, epochs=20, batch_size=100, verbose=2)
    
    # Training Phase
    model.summary()
    
    # make a prediction on train data
    yhat = model.predict(x_test2)

    predicted_df = pd.DataFrame(yhat, columns=['Predicted'])
    predicted_df.index = x_test.index
    
    return y_test, predicted_df




"""
ALG_LIST
"""
def model_(_p_model_no):
    _p_model_no = int(_p_model_no)
    #Default as KNN
    if _p_model_no==1:
        return holts_winter_
    elif _p_model_no==2:
        return auto_arima_
    elif _p_model_no==3:
        return auto_sarima_
    elif _p_model_no==4:
        return lstm_
    else:
        print('Error Running The Model')

