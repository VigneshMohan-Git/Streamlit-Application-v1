import pandas as pd
import streamlit as st
from typing import Dict
import st_configuration_
from st_global_ import standard_init as std

#Function
@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}



def my_func(x):
    try:
        res = x.iloc[:, 0]*2 + x.iloc[:, 1]*4
    except IndexError:
        res = x.iloc[:, 0]*2
    return res  
    
#FUNCTION
def model_(_p_model_):
    _p_model_str = str(_p_model_)
    #Default as KNN
    
    if _p_model_str =='HOLTS WINTER ES':
        return 1
    if _p_model_str =='ARIMA':
        return 2
    elif _p_model_str =='SARIMA':
        return 3
    elif _p_model_str =='LSTM':
        return 4      
    else:
        return 1
      
#String Name       
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


#Read Function
def read_(data_path):
    import os
    ext = os.path.splitext(data_path)[-1].lower()
    if ext == '.xlsx':
        df = pd.read_excel(data_path)
    elif ext == '.csv':
        df = pd.read_csv(data_path)
    else:
        print("Error Loading The Data")
    return df
        

#Remove Extention
def remove_extention(df):
    import os
    base = os.path.splitext(df)[0]
    return base
    

    
    
#Line Chart
def plotly_(df, xaxis, yaxis):
    import plotly.express as px
    import datetime
    
    fig = px.line(df, x=xaxis, y=yaxis)
    
    #Styles
    #fig.update_yaxes( tickprefix="$", showgrid=True)
    fig.update_layout(xaxis_tickformat = '%B<br>%Y')
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(autosize=False,width=1175,height=500)
    fig.update_layout(
        #title='GRAPH VIEW', title_x=0.5,
        xaxis=dict(
        title='DATE',
        titlefont_size=16,
        tickfont_size=12,
        ),
        yaxis=dict(
        title='VALUES',
        titlefont_size=16,
        tickfont_size=12,
        ),
    )
    
    return fig
   

#Matplotlib
def matplotlib(dataframe, xaxis, yaxis):
    import numpy as np
    import matplotlib.pyplot as plt
    y = yaxis 
    x = xaxis
    plt.plot(x,y)
    fig = plot.show()
    
    return fig
    
   
def plotly_pie(labels, values):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    return fig.show()
    
    

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file for Visualization...', filenames)
    return os.path.join(folder_path, selected_filename)


def dfpreprocess(df):
    #df['uid'] = df['Business Unit_code'].astype(str) + df['Country_Code'].astype(str) + df['Magnitude_Product_Group_Name'].astype(str) + df['Customer Business Sector'].astype(str)
    #df = df.set_index(pd.DatetimeIndex(df['key']))
    df_ = df.sort_index()
    #df.drop(['key'],axis=1,inplace=True)
    return df_

"""
PANDAS PROFILING
"""
@st.cache
def pandas_profile(df):
    import pandas_profiling
    from pandas_profiling import ProfileReport
    import streamlit.components.v1 as components
    profile = ProfileReport(df)
    output = profile.to_file(std.vHtmlPath + "Analysis.html", silent=False) 
    return output
    
    
#Sort DataFrame
def sortindex_(df):
    df = df.sort_index()
    return df
    
    
    
 
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

def strip_no():
    numbers = []
    for word in a_string.split():
       if word.isdigit():
          numbers.append(int(word))
#Save to CSV
@st.cache
def save_csv(df):
    return df.to_csv().encode('utf-8')
    
#Save Uploaded Files    
def save_uploaded_file(uploadedfile, name):
    import os
    #name = os.path.splitext(uploadedfile)[0]
    open(std.vDataSavePath + str(name),'w+').write(uploadedfile.to_csv(index=False, line_terminator='\n'))
    return st.success("File Saved")  
    
    
#Save Algorithm list
def save_Alg_list(df):
    open(std.vConfigPath + 'ALG_LIST.csv','w+').write(df.to_csv(index=False, line_terminator='\n'))
    return

def save_TimeFrame(df):
    open(std.vConfigPath + 'TIMEFRAME.csv','w+').write(df.to_csv(index=False, line_terminator='\n'))
    return


def df_window(df):
    import webbrowser
    html = df.to_html()
    
    text_file = open(std.vHtmlPath + "Dataframe.html", "w+")
    text_file.write(html)
    text_file.close()
    
    return webbrowser.open(html)
    
    
def header(url):
    st.markdown(f'<p style="font-family:serif; \
                text-align: center; \
                background-color: #FFFFFF;  \
                color:#000000; \
                font-size:30px; \
                border-radius:5%;">{url}</p>', \
                unsafe_allow_html=True)


def View(df):
    css = """<style>
    table { border-collapse: collapse; border: 3px solid #eee; }
    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
    table thead th { background-color: #eee; color: #000; }
    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
    padding: 3px; font-family: monospace; font-size: 10px }</style>
    """
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + (df.to_html() + css).replace("\n",'\\') + '\';'
    s += '</script>'
    return(HTML(s+css))


"""
ABSOLUTE
"""
def cal(data):
    Accuracy = (100 - abs(data - 100)).mean()
    return round(Accuracy)


"""
MODEL ERROR
"""
def model_eval(y, predictions):

    
    # Import library for metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.metrics import accuracy_score
    import numpy as np
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
    
    
    return round(mae), round(rmse), cal(Accuracy)# round(theil_u_statistic,3)


"""
Get Number From String
"""
def isdigit_(my_str):
    import re
    from itertools import groupby
    l = [int(''.join(i)) for is_digit, i in groupby(my_str, str.isdigit) if is_digit]
    return l

"""
HTML TAGS
"""
def html(body):
    st.markdown(body,unsafe_allow_html=True)
    

def br(n):
    html(n * "<br>")

def div(body):
    f"<div>{body}</div>"    
    
def card_end_str():
    return "</div></div>"
   
   
def card_begin_str(header):
    return (
        "<style>div.card{background-color:#FAFAFA;border-radius: 10px;box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);transition: 0.3s;}</style>"
        '<div class="card">'
        '<div class="container">'
        '<div class="col d-flex justify-content-center">'
        f"{header}"
    )

def card(header, body):
    lines = [card_begin_str(header), f"<p>{body}</p>", card_end_str()]
    html("".join(lines))



def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)