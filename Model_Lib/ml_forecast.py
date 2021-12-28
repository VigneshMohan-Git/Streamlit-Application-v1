from ml_settings_ import standard_init as std
import ml_subfile_ as var
import pandas as pd
import numpy as np



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
        
        
        

"""
Holts Winter ExponentialSmoothing
"""
from statsmodels.tsa.api import ExponentialSmoothing
def holts_winter_(x_train, Forecastrange):
    #Guessed Parameter
    span = 12
    alpha= 0.1 #2/(span+1) #0.15
    beta = 0.4
    gamma= 0.1

    #Model
    from statsmodels.tsa.api import ExponentialSmoothing
    model = ExponentialSmoothing(x_train, trend='mul',seasonal='mul',seasonal_periods=12)
    model.exog = x_train
    model_fit = model.fit(smoothing_level=alpha,smoothing_slope=beta,smoothing_seasonal=gamma)
    
    #smoothing_level=alpha,smoothing_slope=beta,smoothing_seasonal=gamma
    #model_fit.exog = x_test
    
    yhat = model_fit.forecast(Forecastrange)
    yhat_df = pd.DataFrame(yhat, columns=['ForecastedValues'])
    
    return yhat_df



    
"""
AUTO ARIMA
"""
import pmdarima as pm
def auto_arima_(x_train ,Forecastrange):
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
    
    yhat = smodel_fit.predict(n_periods= Forecastrange)
    yhat_df = pd.DataFrame(yhat, columns=['ForecastedValues'])
    
    return yhat_df


"""
SARIMAX
"""
import pmdarima as pm
def auto_sarima_(x_train, Forecastrange):
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
    yhat = smodel_fit.predict(n_periods= Forecastrange)
    yhat_df = pd.DataFrame(yhat, columns=['ForecastedValues'])
    
    return yhat_df




#Read different data files
def read_(data_path):
    import os
    ext = os.path.splitext(data_path.name)[-1].lower()
    if ext == '.xlsx':
        df = pd.read_excel(data_path)
    elif ext == '.csv':
        df = pd.read_csv(data_path)
    else:
        print("Error Loading The Data")
    return df


def timeframe(dataframe, period):
    #DateSeries
    period_ = period
    date = dataframe.Month.iloc[-1:]
    futuredate = pd.Series(pd.date_range(date, freq="M", periods=period_))
    return futuredate