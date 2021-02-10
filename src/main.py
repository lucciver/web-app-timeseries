import streamlit as st
import pathlib
import os
import pandas as pd

import auxx as aux
import train as tr
import evaluate as ev


#-----------------------------------------------------------------------------#
APP_PATH = str(pathlib.Path(__file__).parent.resolve())
DATADIR = APP_PATH  + "/data/"
IMPLEMENTED_ALGORITHMS = {'KNR': {'n_neighbors': ("inputbox",0,10,"integer"), 'metric':("selectbox",["minkowski"])},
                          'SVR':{'paramter 1': ("slider",0,10)},
                          'NNR':{'paramter 2': ("slider",0,10)},
                          'Embeding':{'no parameter': ("slider",0,10)}}
COMPARACION_ALGORITHMS = ["RangePredictor", "MovilMeanPredictor"]
UI_MODULES = ["Data", "Train", "Evaluate"]
UPDATED = 'Last Updated: February 2020'
AUTHOR = 'Under construction. Created by Juan Tzintzun (https://juantztz.github.io/)'
#-----------------------------------------------------------------------------#
def write_info_data():
    attributes = []
    my_expander_data_info = st.beta_expander("Information about implemented datasets")
    dataset = my_expander_data_info.selectbox("Select from available datasets", os.listdir(DATADIR))
    if dataset in os.listdir(DATADIR):
        my_expander_data_info.write("For detailled information about the dataset see the following http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset")
        dataPreview = pd.read_csv(APP_PATH + "/data/" + dataset)
        attributes = dataPreview.columns
    return attributes

def write_data_vis(attributes):
    expander_visual = st.beta_expander("Data visualization and preparation")
    selectbox_dataset = expander_visual.selectbox('Choose Dataset', available_data_sets)
    expander_visual.header("Data trasnformation")
    selectbox_frame = expander_visual.selectbox('time series data to supervised learning data ', implemented_framing)
    selectbox_attr = expander_visual.multiselect('Select attribute for shifting', attributes)
    print(selectbox_attr)
    #selectbox_attr = expander_visual.selectbox('Select attribute for shifting', attributes)
    submit_frame = expander_visual.button('Transform')
    if submit_frame:
        dataPreview = pd.read_csv(APP_PATH + "/data/" + selectbox_dataset)
        for attr in selectbox_attr:
            dataPreview['lag_'+attr] = dataPreview[attr].shift()
        dataPreview.to_csv(APP_PATH + "/data/" + selectbox_dataset +"_supervised" + ".csv")
    expander_visual.header("Data visualization")
    c1, c2 = expander_visual.beta_columns((2, 2))
    #Visualiation
    selectbox_users_plot = c1.selectbox('Visualization', ['users/time', 'registered/time', 'casual/time'])
    #Cleansing
    selectbox_plot = c2.selectbox('Cleansing', ['Outliers', 'PCA'])

def write_train():
    alg_params = {}
    train_visual = st.beta_expander("Select Training Settings")
    alg_params["algorithms"] = train_visual.multiselect('Select algorithms', list(IMPLEMENTED_ALGORITHMS.keys())) + COMPARACION_ALGORITHMS
    c1, c2 = train_visual.beta_columns((2, 2))
    selectbox_dataset = c1.selectbox('Choose Dataset', os.listdir(DATADIR))
    alg_params["numberRuns"] = int(c1.text_input('Number runs',1))
    alg_params["timeUnit"] = c1.selectbox("Time unit", ["hours","days"])
    alg_params["forecastHorizon"] = c1.slider("Forecast Horizon (time units)", 1, 100, 12, 1)
    alg_params["train/test partition"] = c1.slider("train/test partition", 10, 100, 70, 10)
    alg_params["numberParams"] = int(c1.text_input('Number hyperparameters',3))
    for val in alg_params["algorithms"]:
        print(val)
        if val != "Embeding":
            c2.write(val + " parameters")
            for key in IMPLEMENTED_ALGORITHMS[val].keys():
                if IMPLEMENTED_ALGORITHMS[val][key][0] == "inputbox":
                    c2.write(val + " parameters")
                    alg_params[val] = c2.text_input(('Input value for parameter {0} ({1})').format(key,IMPLEMENTED_ALGORITHMS[val][key][3]))
                elif IMPLEMENTED_ALGORITHMS[val][key][0] == "selectbox":
                    alg_params[val]  = c2.selectbox(('Input value for parameter {0}').format(key), IMPLEMENTED_ALGORITHMS[val][key][1])
                elif IMPLEMENTED_ALGORITHMS[val][key][0] == "slider":
                    c2.slider(('Input value for parameter {0}').format(key), 10, 100, 70, 10)
                else:
                    print("hola")
    submit = st.button('Train model with selected parameters')
    if submit:
        data = pd.read_csv(APP_PATH + "/data/" + selectbox_dataset).to_numpy()
        predicted, expected = tr.main(alg_params, data)
        print(predicted, expected)

def write_eval():
    eval_visual = st.beta_expander("Select Evaluation Settings")
    alg = eval_visual.multiselect('Select algorithms', COMPARACION_ALGORITHMS + list(IMPLEMENTED_ALGORITHMS.keys()))
    c1, c2 = eval_visual.beta_columns((2, 2))
    selectbox_dataset = c1.selectbox('Choose Dataset', os.listdir(DATADIR))
    timeUnit = c1.selectbox("Time unit", ["hours","days"])
    forecastHorizon = c1.slider("Forecast Horizon (time units)", 1, 100, 12, 1)
    submit = st.button('Evaluate model with selected parameters')
    if submit:
        data = pd.read_csv(APP_PATH + "/data/" + selectbox_dataset).to_numpy()
        predicted, expected = tr.main(alg_params)


if __name__ == '__main__':
    #Main Menu
    st.set_page_config(layout="wide")
    st.title('Web app for time series analysis and forecasting')
    st.write(UPDATED)
    st.write(AUTHOR)
    name_file = fname_to_run = st.sidebar.selectbox('Select from Menu', UI_MODULES)
    if name_file == "Data":
        #Info about the data
        attributes = write_info_data()
        #Data visualization
        #write_data_vis(attributes)
    elif name_file == "Train":
        #Train the model
        write_train()
    else:
        write_eval()
    st.header("References:")
    st.write("")
