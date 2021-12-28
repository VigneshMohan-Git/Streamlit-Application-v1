import st_functions_ as fun
import st_configuration_
from st_global_ import standard_init as std


import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
from streamlit_metrics import metric, metric_row
import os
from os import listdir
from os.path import isfile, join
import fnmatch
import qgrid
import streamlit.components.v1 as components
import webbrowser
from tempfile import NamedTemporaryFile
from IPython.display import HTML
from st_aggrid import AgGrid
from bokeh.models.widgets import Div
import pandas as pd
import csvtotable
import datetime
import pybase64
from pivottablejs import pivot_ui 
import pyautogui
import tabloo

def main():
    
    #==================================== Page Configuration =======================================================
    #Page Configuration
    st.set_page_config(layout='wide') #initial_sidebar_state="expanded"
    #Buttons Configuration
    primaryColor = st.get_option("theme.primaryColor")
    #txt_clr = st.get_option("theme.textColor")
    
    
    
    #Buttons
    s = f"""
        <style>
        div.stButton > button:first-child {{ border: .5px solid {primaryColor}; border-radius:10px 10px 10px 10px; }}
        <style>
    """
    st.markdown(s, unsafe_allow_html=True)


    #Page Configuration
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
    
    
    #==============================================================================================================
    page_container = st.container()
    with page_container:
        #Logo
        title_container = st.container()
        
        with title_container:
            col1, col2, _ = st.columns(3)
            with col1:
                from PIL import Image
                image = Image.open(std.vImage_path)
                st.image(image, width=75)
            
            with col2:
                st.header('YODA ANALYTICS')
           
        st.markdown(
            '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
            unsafe_allow_html=True,
        )
        query_params = st.experimental_get_query_params()
        tabs = ["HOME", "DATA", "MODEL", "FORECAST"]
        if "tab" in query_params:
            active_tab = query_params["tab"][0]
        else:
            active_tab = "Home"

        if active_tab not in tabs:
            st.experimental_set_query_params(tab="Home")
            active_tab = "Home"

        li_items = "".join(
            f"""
            
            <li class="nav-item">
                <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}">{t}</a>
            </li>
            """
            for t in tabs
        )
        
        tabs_html = f"""
            
            <ul class="nav nav-tabs">
            {li_items}
            </ul>
            
        """
        
            
        st.markdown(tabs_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        
        
        #================================ Home =====================================
        
        if active_tab == "HOME":
            #HTML STYLE
            st.markdown('#')
            fun.html(fun.card_begin_str("WELCOME !!!"))
            st.markdown('#')
        
        
        
        #================================ Data Upload =====================================
        
        elif active_tab == "DATA":
        
            dataview = st.selectbox("Select ... ",["...", "DataUploadPage", "DataView", "DataExploratory"])
            
            if dataview == "DataUploadPage":
                #HTML STYLE
                fun.html(fun.card_begin_str("Data Upload Page"))
                st.markdown('#')
                
                with st.form('my_form'):
                    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
                    
                    
                    for files in uploaded_files:
                        name = str(files.name)
                        df = fun.read_(files)
                        df = fun.dfpreprocess(df)
                        
                    save_button = st.form_submit_button('Save')
                    if save_button:
                        fun.save_uploaded_file(df, name)
                    
                    
        #=============================== Data View ====================================
        
            elif dataview == "DataView":
                #HTML STYLE
                fun.html(fun.card_begin_str("Data View"))
                st.markdown('#')
                
                mypath = std.vDataSavePath
                onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
                #Filter based on specific File_Names
                #list_ = fnmatch.filter(onlyfiles, 'SALES_*.csv')
                
                if len(onlyfiles) > 0:
                
                    container_df = st.container()
                    with container_df:
                        _, col2, col3, col4, col5 = st.columns((.2,1,.2,.2,.2))
                        
                        #fun.local_css("style.css")
                          
                        #Data List View
                        with col2:
                            for c2 in onlyfiles:  
                                base = fun.remove_extention(c2)
                                st.write(base)
                            
                        #View Data
                        with col3:
                            for c3 in onlyfiles:
                                button_c3 = st.button('View', key=[c3])
                                if button_c3:
                                    df = pd.read_csv(mypath + c3)
                                    pivot_ui(df, outfile_path= std.vHtmlSavePath + 'View.html')
                                    webbrowser.open(std.vHtmlFile, new=1, autoraise=True)
                                    
                                    
                        #Data Download            
                        with col4:
                            for c4 in onlyfiles:
                                df = pd.read_csv(mypath + c4)
                                csv = fun.save_csv(df)
                                st.download_button('Download', csv, "Data.csv","text/csv", key=[c4])
                          
                        #Data Delete 
                        with col5:
                            for c5 in onlyfiles:
                                button_c5 = st.button('Delete', key=[c5])
                                if button_c5:
                                   os.remove(mypath + c5)
                                   #Page Refresh
                                   pyautogui.hotkey('f5')
                              
                                
                else:
                    st.info("No files uploaded")
                    
                    
            #=============================== Data Exploratory ============================================        
            elif dataview == "DataExploratory":
                #HTML STYLE
                fun.html(fun.card_begin_str("Data Exploratory Analysis"))
                st.markdown('#')

                container = st.container()
                with container:
                    mypath = std.vDataSavePath
                    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
                    
                    if len(onlyfiles) > 0: 
                        for files in onlyfiles:
                            df_file = st.selectbox('Select', [files])
                            if df_file:
                                selected_df = pd.read_csv(mypath + df_file)
                                preprocessed_df = fun.dfpreprocess(selected_df)
                                
                                
                                #Tabulur View --------------------------------------------------------------------------
                                fun.html(fun.card_begin_str("TABULAR VIEW"))
                                AgGrid_= AgGrid(preprocessed_df, editable=True)
                                    
                                #Graph View ---------------------------------------------------------------------------
                                st.markdown('#')
                                fun.html(fun.card_begin_str("GRAPH VIEW"))
                                fig = fun.plotly_(preprocessed_df, preprocessed_df.iloc[:,0], preprocessed_df.iloc[:,1])
                                st.plotly_chart(fig)
               
               
            else: 
                st.info('Select from options')        
        
        
        #========================================= MODEl TRIGGER ================================================
        elif active_tab == "MODEL":
            
            #SHOW EXISTING MODEL PERFORMANCE
            SelectBox = st.selectbox("Select ...",[' ','EXECUTE NEW MODELS', 'EXISTING MODEL PERFORMANCE'])
            
            if SelectBox == 'EXECUTE NEW MODELS':
            
                #HTML STYLE
                fun.html(fun.card_begin_str("Model Execution"))
                st.markdown('#')
                vPath = os.getcwd()
                
                
                _, col1, _ = st.columns((.1,2,.1))
                
                with col1:
                    Input_alg_list_ = st.multiselect('SELECT MODELS:',['HOLTS WINTER ES','ARIMA','SARIMA','LSTM']) 
                        
                    if len(Input_alg_list_) > 0: 
                        st.write('YOU HAVE SELECTED ',len(Input_alg_list_),'MODEL') 
                    else:
                        st.caption('NO MODELS SELECTED...')
                            
                    df_ = pd.DataFrame(columns=['Alg_list'])
                    for list_ in Input_alg_list_:
                        df_ = df_.append({'Alg_list': fun.model_(list_)}, ignore_index=True)
                        fun.save_Alg_list(df_)
                
                
                _, col2, _ = st.columns((.1,2,.1))
                with col2:
                    #Default Selection
                    submit_button = st.button(label='Submit')
                    #st.caption("Click to Execute Command!")
                    
                    
                    if submit_button:
                        with st.spinner("Model Getting Executed.. This might take some time."):
                            #st.write('Models Running: {}'.format(Input_alg_list_))
                            os.system('python Model_Lib/ml_model_main_.py')
                            st.success('Model Running...Done!')
                            
                  
                            #Model Result View
                            ModelNameDF = pd.DataFrame()
                            ModelAccuracyDataFrame = pd.read_csv(std.ModelAccuracyFile)
                            ModelAccuracyDataFrame = ModelAccuracyDataFrame[['Models','Best_Model']]
                            for m in ModelAccuracyDataFrame['Models']:
                                ModelNameDF = ModelNameDF.append({"ModelName": fun.model_str(m)},ignore_index=True)
                                
                            Concat_df = pd.concat([ModelAccuracyDataFrame, ModelNameDF],axis=1).reset_index(drop=True)
                            Concat_df = Concat_df.rename(columns={'Best_Model': 'Model_Accuracy'})
                            BestModelResult = Concat_df[['ModelName','Models','Model_Accuracy']]
                            
                            
                            BestModelResult = BestModelResult[BestModelResult['Model_Accuracy'] == BestModelResult['Model_Accuracy'].max()]
                            BestModelResult.to_csv(std.vConfigPath + 'BEST_ACCURACY_RESULT_' + '.csv')
                        
                            #Table View
                            st.table(Concat_df)
                            
                            
            elif SelectBox == 'EXISTING MODEL PERFORMANCE':
                
                #HTML STYLE -------------------------------------------------------------------------------------------
                st.markdown('#')
                fun.html(fun.card_begin_str("COMBINED VALUES - MODEL ERROR RATE"))
                
                with st.spinner("Please wait ..."):
                    #Model_Error
                    ModelNameDF = pd.DataFrame()
                    ModelErrorDataFrame = pd.read_csv(std.vModelErrorFile)
                    for ml in ModelErrorDataFrame['Models']:
                        ModelNameDF = ModelNameDF.append({'ModelName': fun.model_str(ml)},ignore_index=True)
                        
                    MergedErrorDataFrame = pd.concat([ModelErrorDataFrame, ModelNameDF],axis=1).reset_index(drop=True)
                    
                    #Table View
                    MergedErrorDataFrame = MergedErrorDataFrame[['ModelName','Accuracy','MAE','RMSE','NMSE']]
                    MergedErrorDataFrame = MergedErrorDataFrame.round(2)
                    st.table(MergedErrorDataFrame)
                
                
                #HTML STYLE --------------------------------------------------------------------------------------------
                st.markdown('#')
                fun.html(fun.card_begin_str("COMBINED VALUES - DATAFRAME"))
                with st.spinner("Please wait ..."):
                    #VIEW COMBINED DATAFRAME
                    Combined_PDF = pd.read_csv(std.vMergedDF_File)
                    Combined_PDF = Combined_PDF.round(2)
                    AgGrid_3 = AgGrid(Combined_PDF, editable=True)
            
            
                #HTML STYLE --------------------------------------------------------------------------------------------
                st.markdown('#')
                fun.html(fun.card_begin_str("COMBINED VALUES - GRAPH VIEW"))
                with st.spinner("Please wait ..."):
                    Combined_PDF_fig = fun.plotly_(Combined_PDF, Combined_PDF.iloc[:,0], Combined_PDF.columns[1:])
                    st.plotly_chart(Combined_PDF_fig)
            
                    
                    
            else:
                st.info("Select from options to view Existing model performance or to Execute new models")            
                        
                          
                    
                        
                            
        #============================================ FORECAST =======================================================        
        elif active_tab == "FORECAST":
            #HTML STYLE
            fun.html(fun.card_begin_str("FORECAST"))
            st.markdown('#')
            
            #SelectBox
            forecast_selectbox = st.selectbox('Select',['...','Execute New Models', 'Existing Results'])
            
            #------------------------------------------------------------------------------------------------------------
            if forecast_selectbox == 'Execute New Models':
                
                #Date Slider
                ForecastRange_df = pd.DataFrame(columns=['ForecastRange'])
                
                ForecastRange = st.slider('Forecast Range:', 3,12)
                ForecastRange_df = ForecastRange_df.append({"ForecastRange": ForecastRange}, ignore_index=True)
                st.write("YOU HAVE SELECTED",int(ForecastRange),"FORECAST RANGE")
                ForecastRange_df.to_csv(std.vConfigPath + "FORECAST_RANGE" + ".csv")
                
                #BestModel Selected [Best Accuracy Rate]
                BestModel = pd.read_csv(std.vBestModel_File)
                BestModel_Value = BestModel.ModelName
                BestModelNumber = BestModel.Models.dropna()
                
                #Default Selection
                submit_button = st.button(label='Submit')       
                
                if submit_button:
                    with st.spinner("Model Getting Executed.. This might take some time."):
                        st.info("Using Best Model for Prediction.")
                        st.write("Model Selected:", BestModel_Value)
                        os.system('python Model_Lib/ml_forecast_main.py')
                        st.success('Model Running...Done!')
                        
                        
                        mypath_f = std.vOutputPath + "FORECASTED_DF" + '/'
                        onlyfiles = [f for f in listdir(mypath_f) if isfile(join(mypath_f, f))]
                        #Filter based on specific File_Names
                        filename = 'FORECASTED_VALUES_' + str(int(BestModelNumber)) + '_' + str(ForecastRange)
                        list_ = fnmatch.filter(onlyfiles,  filename + '*.csv')
                        
                        #Tabulur View --------------------------------------------------------------------------
                        fun.html(fun.card_begin_str("TABULAR VIEW"))
                        with st.spinner("Please wait ..."):
                            Forecasted_DataFrame = pd.read_csv(mypath_f + list_[0])
                            AgGrid_= AgGrid(Forecasted_DataFrame, editable=True)
                                    
                        
            #------------------------------------------------------------------------------------------------------------
            elif forecast_selectbox == 'Existing Results':
                
                mypath = std.vOutputPath + "FORECASTED_DF" + '/'
                onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
                #Filter based on specific File_Names
                list_ = fnmatch.filter(onlyfiles, 'FORECASTED_VALUES_*.csv')
                
                if len(onlyfiles) > 0:
                
                    container_df = st.container()
                    with container_df:
                        _, col2, col3, col4, col5 = st.columns((.2,1,.2,.2,.2))
                        
                        #fun.local_css("style.css")
                          
                        #Data List View
                        with col2:
                            for c2 in onlyfiles:  
                                base = fun.remove_extention(c2)
                                st.write(base)
                            
                        #View Data
                        with col3:
                            for c3 in onlyfiles:
                                button_c3 = st.button('View', key=[c3])
                                if button_c3:
                                    df = pd.read_csv(mypath + c3)
                                    pivot_ui(df, outfile_path= std.vHtmlSavePath + 'View.html')
                                    webbrowser.open(std.vHtmlFile, new=1, autoraise=True)
                                    
                                    
                        #Data Download            
                        with col4:
                            for c4 in onlyfiles:
                                df = pd.read_csv(mypath + c4)
                                csv = fun.save_csv(df)
                                st.download_button('Download', csv, "Data.csv","text/csv", key=[c4])
                          
                        #Data Delete 
                        with col5:
                            for c5 in onlyfiles:
                                button_c5 = st.button('Delete', key=[c5])
                                if button_c5:
                                   os.remove(mypath + c5)
                                   #Page Refresh
                                   pyautogui.hotkey('f5')
                              
                                
                else:
                    st.info("No files uploaded")
                    
            #------------------------------------------------------------------------------------------------------------
            else:
                st.info("Select from options to View or Execute new forecast models.")
            


if __name__ == '__main__':
    main()
