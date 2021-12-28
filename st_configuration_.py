from st_global_ import standard_init as std
import glob
import streamlit as st

#Input_Path / Output_Path
std.vPath               = "C:/Users/Poovendran/VIGNESH_/WORKFILES"  #os.getcwd()


std.ImagePrefix         = 'Yoda_Logo.png'
std.HtmlPrefix          = 'View.html'
std.ModelErrorPrefix    = 'MODEL_ERROR.csv'
std.ModelAccuracyPrefix = "BEST_MODEL.csv"
std.MergedDFPrefix      = "MERGED_DATAFRAME.csv"
std.BestModelPrefix     = "BEST_ACCURACY_RESULT_.csv"


std.InputFolder  = 'Input_Data'
std.HtmlFolder   = 'Html'
std.ConfigFolder = 'Config'
std.OutputFolder = 'Output'
std.ImageFolder  = 'Image'


std.vDataSavePath       = std.vPath + '/' + std.InputFolder + '/'
std.vImage_path         = std.vPath + '/' + std.ImageFolder + '/' + std.ImagePrefix
std.vall_files          = glob.glob(std.vPath + "/*.csv")


std.vOutputPath         = std.vPath + '/' + std.OutputFolder + '/'
std.vHtmlSavePath       = std.vPath + '/' + std.HtmlFolder + '/' 
std.vConfigPath         = std.vPath + '/' + std.ConfigFolder + '/'
std.vModelErrorFile     = std.vPath + '/' + std.OutputFolder + '/' + std.ModelErrorPrefix
std.ModelAccuracyFile   = std.vPath + '/' + std.OutputFolder + '/' + std.ModelAccuracyPrefix
std.vHtmlFile           = std.vPath + '/' + std.HtmlFolder + '/' + std.HtmlPrefix
std.vMergedDF_File      = std.vPath + '/' + std.OutputFolder + '/' + std.MergedDFPrefix
std.vBestModel_File     = std.vPath + '/' + std.ConfigFolder + '/' + std.BestModelPrefix