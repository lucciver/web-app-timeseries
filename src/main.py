import streamlit as st
import pathlib
import os
import pandas as pd

import auxx as aux
import train as tr


#-----------------------------------------------------------------------------#
APP_PATH = str(pathlib.Path(__file__).parent.resolve())
DATADIR = APP_PATH  + "/data/"

IMPLEMENTED_ALGORITHMS = {'KNR': {'n_neighbors': ("inputbox",0,10,"integer"), 'metric':("selectbox",["minkowski"])},
                          'SVR':{'paramter 1': ("slider",0,10)},
                          'NNR':{'paramter 2': ("slider",0,10)},
                          'Embeding':{'no parameter': ("slider",0,10)}}

COMPARACION_ALGORITHMS = ["RangePredictor", "MovilMeanPredictor"]
UI_MODULES = ["Data and Analysis", "Training and Evaluation"]
FOOTER = "© Copyright 2021. Powered by Streamlit with Python. Hosted by Amazon Web Serives. Photos are my own. Last updated: February 10, 2021."
HEADER = 'Under construction. Created by Juan Tzintzun (https://juantztz.github.io/)'
#-----------------------------------------------------------------------------#


def write_train():
    alg_params = {}
    train_visual = st.sidebar.beta_expander("Select Training Settings")
    alg_params["algorithms"] = train_visual.multiselect('Select algorithms', list(IMPLEMENTED_ALGORITHMS.keys()))
    alg_params["nameModel"] = train_visual.text_input('Name of trained model',"")
    selectbox_dataset = train_visual.selectbox('Choose Dataset', os.listdir(DATADIR))
    c1, c2 = train_visual.beta_columns((2, 2))
    alg_params["numberRuns"] = int(c1.text_input('Number runs',1))
    alg_params["timeUnit"] = c1.selectbox("Time unit", ["hours","days"])
    alg_params["forecastHorizon"] = c1.slider("Forecast Horizon (time units)", 1, 100, 12, 1)
    alg_params["train/test partition"] = c1.slider("train/test partition", 10, 100, 70, 10)
    alg_params["numberParams"] = int(c1.text_input('Number hyperparameters',3))
    for val in alg_params["algorithms"]:
        if val != "Embeding":
            c2.write(val + " parameters")
            for key in IMPLEMENTED_ALGORITHMS[val].keys():
                if IMPLEMENTED_ALGORITHMS[val][key][0] == "inputbox":
                    alg_params[val] = c2.text_input(('Input value for parameter {0} ({1})').format(key,IMPLEMENTED_ALGORITHMS[val][key][3]))
                elif IMPLEMENTED_ALGORITHMS[val][key][0] == "selectbox":
                    alg_params[val]  = c2.selectbox(('Input value for parameter {0}').format(key), IMPLEMENTED_ALGORITHMS[val][key][1])
                elif IMPLEMENTED_ALGORITHMS[val][key][0] == "slider":
                    c2.slider(('Input value for parameter {0}').format(key), 10, 100, 70, 10)
                else:
                    print("hola")
    st.markdown("### Training model")
    submit = st.button('Train model with selected parameters')
    if submit:
        data = pd.read_csv(APP_PATH + "/data/" + selectbox_dataset).to_numpy()
        trainned_model, = tr.main(alg_params, data)
        print(predicted, expected)
    progress_bar = st.progress(0)
    st.markdown("### Evaluating model")
    aux.write_eval(alg_params["algorithms"], selectbox_dataset,COMPARACION_ALGORITHMS)


if __name__ == '__main__':
    #Main Menu
    st.set_page_config(layout="wide")
    st.markdown(HEADER)
    st.title('Web app for time series analysis and forecasting using machine learning')
    name_file = st.sidebar.selectbox('Select from Menu', UI_MODULES)
    if name_file == "Data and Analysis":
        attributes = aux.write_info_data(DATADIR, APP_PATH)
        #write_data_vis(attributes)
    elif name_file == "Training and Evaluation":
        write_train()
    else:
        Print("Not implemented module")
    st.markdown(FOOTER)
