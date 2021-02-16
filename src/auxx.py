import numpy as np
import streamlit as st
import os
import pandas as pd


def write_info_data(DATADIR, APP_PATH):
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

def write_eval(algo, datasetName, COMPARACION_ALGORITHMS):
    for alg in algo + COMPARACION_ALGORITHMS :
        st.checkbox(alg, False, key=1)
    submit = st.button('Predict')
    if submit:
        data = pd.read_csv(APP_PATH + "/data/" + datasetName).to_numpy()

def altair_chart(err_name, st, data):
    st.subheader('Comparision of infection growth')
    total_cases_graph  = alt.Chart(data).transform_filter(alt.datum.total_cases > 0
    ).mark_line().encode(x=alt.X('date', type='nominal', title = err_name),
   y=alt.Y('sum(total_cases):Q',  title='Confirmed cases'),
   color='Country',
   tooltip = 'sum(total_cases)',
   ).properties(
    width=1500,
    height=600
    ).configure_axis(
    labelFontSize=17,
    titleFontSize=20
    )
    st.altair_chart(total_cases_graph)
