from ml_settings_ import standard_init as std
import ml_timeseries_models as models
import ml_subfile_ as var
import pandas as pd
import numpy as np
  
   
   
"""
PRE PROCESS DATA
"""
def pre_process_(vInput):
    
    data = pd.read_csv(vInput)
    
    df_ = data.set_index(pd.DatetimeIndex(data['Month']))
    df_.drop(['Month'], axis=1, inplace=True)
    df_ = df_.sort_index()
    
    ####Introducing time as parameter
    # X & Y SPLIT
    #X, Y = timeseries_models.create_time_features(data, 'Sales')
    
    
    # Seasonal Decompose
    #sd = seasonal_decompose(Y, period=12)
    
      
    return df_
    
 