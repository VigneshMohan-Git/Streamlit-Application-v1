import pandas as pd
import numpy as np
import ml_settings_ as std
import ml_subfile_ as var
import ml_timeseries_models as models
import ml_PreprocessData_ as preprocess
#import ml_feature_selection as fs
import os
import warnings
import multiprocessing as mp
os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from warnings import simplefilter
# ignore all warnings
simplefilter(action='ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


#q.create_(var.std.vOutputFolder + "Scenario")
        
vPath = os.path.dirname(os.getcwd()) + '/' + 'WORKFILES' #os.getcwd()
#********************************
# Loading Environment Variables
#********************************
var.define(vPath)
vInputData = var.std.vInputFile

# Loading Various Scenario's
#vSceData   = pd.read_csv(var.std.vScenarioFile)
#vSceData   = vSceData.drop(['Scenario'],axis=1, inplace=False)

# Algorithm List
Al = var.std.vAlg_List

# List
#sce_list  = var.std.vScenario_List
Model_Error = pd.DataFrame()
Best_Model = pd.DataFrame()
Accuracy_level = var.std.vAccuracy_Level

"""
Feature Selection
"""
#col_importance = fs.feature_selection_(vInputData)


"""
Preprocessing
"""
preprocessed_df = preprocess.pre_process_(vInputData)
x_train, x_test = models.train_test_split(preprocessed_df)


#Merged_df
merged_df = pd.DataFrame()

for i in (iter(Al)):
    print("Model_Running:",i)
    x_test, y_pred = models.model_(i)(x_train, x_test)
    
    # Model_Error
    MAE, RMSE, MAPE, MFE, NMSE, Accuracy = models.model_eval(x_test['Monthly beer production'], y_pred['Predicted'])
    Model_Error = Model_Error.append({"MAE":MAE,"RMSE":RMSE,"MFE":MFE,"NMSE":NMSE,"Accuracy":Accuracy ,"Models":str(i)}, ignore_index=True)
    
    #dd = pd.concat([y_test, y_pred, x_para], axis=1)
    dd = pd.concat([x_test, y_pred], axis=1)
    
    #Merging all dataframes together
    dd = dd.rename({"Predicted": "Predicted" + '_' + str(models.model_str(i))}, axis='columns',inplace=False)
    merged_df = pd.concat([merged_df, dd], axis=1)
    merged_df = merged_df.T.drop_duplicates().T
    
    #Best_Model 
    if (Accuracy_level < Accuracy):
        Best_Model = Best_Model.append({'Best_Model':Accuracy, "RMSE":RMSE, "MAE":MAE, "Models":str(i)}, ignore_index=True)
    
    #Saving files to Output folder
    Model_Error.to_csv(var.std.vOutputFolder + "MODEL_ERROR" + ".csv")    
    dd.to_csv(var.std.vOutputFolder + var.std.vOutputPrefix + str(i) + "_" + ".csv")
    
merged_df.to_csv(var.std.vOutputFolder + "MERGED_DATAFRAME" + ".csv")
Best_Model.to_csv(var.std.vOutputFolder + "BEST_MODEL" + ".csv")

