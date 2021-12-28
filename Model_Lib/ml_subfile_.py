import ml_settings_ as std
import os
import pandas as pd

def define(vPath):
	
    std.vPath                 =  vPath  #vPath
    std.vInputPrefix          = "Data.csv"
    std.vConfigPrefix         = "CONFIG_.xlsx"
    std.vOutputPrefix         = "OUTPUT_"
    std.vScenarioPrefix       = "SIM.csv"
    std.vBMPrefix             = "BEST_MODEL.csv"
    std.vAlg_listPrefix       = "Alg_list.csv"
    std.vBestAccuracyRe       = "BEST_ACCURACY_RESULT_.csv"
    std.vForecastRangePrefix  = "FORECAST_RANGE.csv"
    std.vBestModelPrefix      = "BEST_ACCURACY_RESULT_.csv"
    #std.vFeaturePrefix       = "FEATURE.csv"
    
    
    std.vInputFolderName      = "Input_Data"
    std.vOutputFolderName     = "Output"
    std.vConfigFolderName     = "Config"
    
    std.vInputFile            = vPath + '/' + std.vInputFolderName + '/' + std.vInputPrefix
    std.vConfigFile           = vPath + '/' + std.vConfigFolderName + '/' + std.vConfigPrefix
    std.vScenarioFile         = vPath + '/' + std.vConfigFolderName + '/' + std.vScenarioPrefix
    std.vBMFile               = vPath + '/' + std.vOutputFolderName + '/' + std.vBMPrefix
    std.vOutputFolder         = vPath + '/' + std.vOutputFolderName + '/' 
    std.vAlg_list             = vPath + '/' + std.vConfigFolderName + '/' + std.vAlg_listPrefix
    std.vforecastAlg_list     = vPath + '/' + std.vOutputFolderName + '/' + std.vBestAccuracyRe
    std.vForecastRange_list   = vPath + '/' + std.vConfigFolderName + '/' + std.vForecastRangePrefix
    std.vBestModelFile        = vPath + '/' + std.vConfigFolderName + '/' + std.vBestModelPrefix
    #std.vFeatureFile         = std.vOutputFolder + '/' + std.vFeaturePrefix
    

    # std.vConfig          = pd.read_excel(std.vConfigFile, sheet_name = 'config_main')
    # std.vMax_Train_Data = '2021-02-15'
    
    # Train Test Split
    # [0.95]test_size = 4 Months
    # [0.83]test_size = 12 Months
    std.vFORECAST_AHEAD_ = "4"
    std.vTEST_RATIO      = 0.95  
    
    # Data Split
    std.vDate = '2021-02-15'
    
    
    # [Knn=1, Random_Forest=2, ExtraTreeReg=3, Svm=4, HoltsWinter=5, Arima=6, Sarimax=7, Prophet=8, Lstm =9 ]
    vAlg_list_df = pd.read_csv(std.vAlg_list)
    std.vAlg_List        = vAlg_list_df.Alg_list.dropna() #[2,3]  
    
    
    #ForecastRange
    vForecastRange = pd.read_csv(std.vForecastRange_list)
    std.vForecastRange = vForecastRange.ForecastRange.dropna()
    
    #BestModel #based on Accuracy
    vBestModel     = pd.read_csv(std.vBestModelFile)
    std.vBestModel = vBestModel.Models.dropna()
    
    
    # Model Accuracy 
    std.vAccuracy_Level = 70 
    
    
    # Scenario1 = 1, .... , Scenario5 = 5
    std.vScenario_List   = [1,2,3,4,5]
    
    # If 'True' Scenario will be executed 
    # If != 'True' Scenario will not be executed
    # 'True' / 'False'
    std.vScenario        = 'True'
    
    # Country_Holidays
    # To include Country holidays into dataframe
    std.vCountryCode     = 'UK' 
    