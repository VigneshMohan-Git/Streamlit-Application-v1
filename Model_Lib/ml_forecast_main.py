import ml_forecast as forecastmodels
import ml_settings_ as std
import ml_subfile_ as var
import ml_PreprocessData_ as preprocess
import ml_timeseries_models as models
#import ml_feature_selection as fs

import os
import pandas as pd
import numpy as np
import warnings
import multiprocessing as mp
os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from warnings import simplefilter
# ignore all warnings
simplefilter(action='ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Loading Environment Variables
vPath = os.path.dirname(os.getcwd()) + '/' + 'WORKFILES' #os.getcwd()
var.define(vPath)
vInputData = var.std.vInputFile


#Defining Dataframe
merged_dataframe    =  pd.DataFrame()
forecasted_df       =  pd.DataFrame()

# Reading DataFrame
fml = std.vBestModel


preprocessed_data   = preprocess.pre_process_(vInputData)


for i in fml:
    forecasted_df = forecastmodels.model_(i)(preprocessed_data, int(std.vForecastRange))
    
    print("MODEL RUNNING:", i)
    
    merged_dataframe = pd.concat([preprocessed_data, forecasted_df])
    merged_dataframe = merged_dataframe.rename({"ForecastedValues": "Forecasted" + '_' + str(models.model_str(i))}, axis='columns',inplace=False)
    
    
    #Save File
    merged_dataframe.to_csv(var.std.vOutputFolder + 'FORECASTED_DF' + '/' + "FORECASTED_VALUES" + "_" + str(i) + "_" + str(int(std.vForecastRange)) + ".csv")

