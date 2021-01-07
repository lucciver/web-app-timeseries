import streamlit as st
import os

import auxx as aux
import algorithms as alg


#-----------------------------------------------------------------------------#
this_file = os.path.abspath(__file__)
folder =  this_file.replace("/main.py","") # path to the src folder /home/juantztz/Documentos/Bike-sharing-system/src
implemented_algorithms = {'KNN.py': {'n_neighbors': ("slider",0,10), 'metric':("selectbox",["minkowski"])}, 'SVR':10,'NNR':5, 'Embeding':87}
implemented_number_data_splits = (500, 100)
available_data_sets = ['None', 'bike_sharing_dataset_hour', 'bike_sharing_dataset_day']
ui_modules = ["Data", "Train", "Evaluate"]
implemented_data_partition = [(70,30),(50,50),(80,20),(60,40)]

#-----------------------------------------------------------------------------#
def write_info_data():
    my_expander_data_info = st.beta_expander("Information about implemented datasets")
    dataset = my_expander_data_info.selectbox("Select from available datasets", available_data_sets)
    if dataset == "Bike_Sharing_DataSet_Day" or dataset =="Bike_Sharing_DataSet_Hour":
        my_expander_data_info.write("For detailled information about the dataset see the following http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset")

def write_data_vis():
    expander_visual = st.beta_expander("Data visualization")
    expander_visual.header("Data visualization")
    selectbox_dataset = expander_visual.selectbox('Choose Dataset', available_data_sets)
    c1, c2 = expander_visual.beta_columns((2, 2))
    #Visualiation
    selectbox_users_plot = c1.selectbox('Visualization', ['users/time', 'registered/time', 'casual/time'])
    #Cleansing
    selectbox_plot = c2.selectbox('Cleansing', ['Outliers', 'PCA'])

def write_train():
    train_visual = st.beta_expander("Training")
    train_visual.title('Training the model')
    c1, c2 = train_visual.beta_columns((2, 2))
    alg_params = {}
    selectbox_dataset = c1.selectbox('Choose Dataset', available_data_sets)
    alg_params["algorithms"] = c1.selectbox('Choose Algorithm(s)', list(implemented_algorithms.keys()))
    alg_params["n_splits"] = int(c1.text_input('Input number of time series splits (integer value):',720))
    alg_params["train/test"] = c1.selectbox('Split data for test and training', implemented_data_partition)
    print(alg_params["train/test"], type(alg_params["train/test"][0]), type(alg_params["train/test"][1]))
    for val in implemented_algorithms[alg_params["algorithms"]]:
        if implemented_algorithms[alg_params["algorithms"]][val][0] == "slider":
            alg_params[val] = c2.slider(val, implemented_algorithms[alg_params["algorithms"]][val][1], implemented_algorithms[alg_params["algorithms"]][val][2], 25, 1)
        elif implemented_algorithms[alg_params["algorithms"]][val][0] == "selectbox":
            alg_params[val]  = c2.selectbox(val, implemented_algorithms[alg_params["algorithms"]][val][1])
        else:
            print("hola")
    submit = st.button('Train model with selected parameters')
    if submit:
        algt = alg.load(alg_params["algorithms"], folder + "/algorithms/"+alg_params["algorithms"]) #carga funciones especificas del scenario
        data = hour = aux.loadcsv(folder + "/data/" + selectbox_dataset+".csv")
        err_mae, err_mape, err_rmse = algt.main(data,alg_params)
        st.altair_chart(err_name, st)


if __name__ == '__main__':
    #Main Menu
    st.set_page_config(layout="wide")
    st.title('Web app for time series analysis and forecasting')
    st.write('Last Updated: January 2020')
    st.write('Created by Juan Tzintzun (https://juantztz.github.io/)')
    name_file = fname_to_run = st.sidebar.selectbox('Select from Menu', ui_modules)
    if name_file == "Data":
        #Info about the data
        write_info_data()
        #Data visualization
        write_data_vis()
    elif name_file == "Train":
        #Train the model
        write_train()
    else:
        st.write("EVALUATION NOT IMPLEMENTED YET")
    st.header("References:")
    st.write("[1] Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg")
