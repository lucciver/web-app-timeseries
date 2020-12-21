import streamlit as st
import os

import auxx as aux
import algorithms as alg


#-----------------------------------------------------------------------------#
this_file = os.path.abspath(__file__)
folder =  this_file.replace("/main.py","") # path to the src folder /home/juantztz/Documentos/Bike-sharing-system/src
implemented_algorithms = {'KNN.py': {'n_neighbors': ("slider",0,10), 'metric':("selectbox",["minkowski"])}, 'SVR':10,'NNR':5, 'Embeding':87}
implemented_number_data_splits = (500,100)
available_data_sets = ['hour']
ui_modules = ["Data", "Train", "Evaluate"]

#-----------------------------------------------------------------------------#
def write_info_data():
    my_expander_data_info = st.beta_expander("Information about the data set")
    my_expander_data_info.write("For detailled information about the dataset see the following http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset")
    my_expander_data_info.header("Data Attributes:")
    my_expander_data_info.write("instant: record index")
    my_expander_data_info.write("dteday : date")
    my_expander_data_info.write("season : season (1:winter, 2:spring, 3:summer, 4:fall")
    my_expander_data_info.write("yr : year (0: 2011, 1:2012")
    my_expander_data_info.write("mnth : month ( 1 to 12)")
    my_expander_data_info.write("hr : hour (0 to 23)")
    my_expander_data_info.write("holiday : weather day is holiday or not (extracted from [Web Link])")
    my_expander_data_info.write("weekday : day of the week")
    my_expander_data_info.write("workingday : if day is neither weekend nor holiday is 1, otherwise is 0")
    my_expander_data_info.write("+ weathersit :")
    my_expander_data_info.write("1: Clear, Few clouds, Partly cloudy, Partly cloudy")
    my_expander_data_info.write("2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist")
    my_expander_data_info.write("3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds")
    my_expander_data_info.write("4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog")
    my_expander_data_info.write("temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)")
    my_expander_data_info.write("atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)")
    my_expander_data_info.write("hum: Normalized humidity. The values are divided to 100 (max)")
    my_expander_data_info.write("windspeed: Normalized wind speed. The values are divided to 67 (max)")
    my_expander_data_info.write("casual: count of casual users")
    my_expander_data_info.write("registered: count of registered users")
    my_expander_data_info.write("cnt: count of total rental bikes including both casual and registered")

def write_data_vis():
    expander_visual = st.beta_expander("Data visualization")
    expander_visual.header("Data visualization")
    selectbox_dataset = expander_visual.selectbox('Choose Dataset', ['day', 'hour'])
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
    for val in implemented_algorithms[alg_params["algorithms"]]:
        if implemented_algorithms[alg_params["algorithms"]][val][0] == "slider":
            alg_params[val] = c2.slider(val, implemented_algorithms[alg_params["algorithms"]][val][1], implemented_algorithms[alg_params["algorithms"]][val][2], 25, 1)
        elif implemented_algorithms[alg_params["algorithms"]][val][0] == "selectbox":
            alg_params[val]  = c2.selectbox(val, implemented_algorithms[alg_params["algorithms"]][val][1])
        else:
            print("hola")
    print(alg_params)
    submit = st.button('Train model with selected parameters')
    if submit:
        algt = alg.load(alg_params["algorithms"], folder + "/algorithms/"+alg_params["algorithms"]) #carga funciones especificas del scenario
        data = hour = aux.loadcsv(folder + "/data/" + selectbox_dataset+".csv")
        err_mae, err_mape, err_rmse = algt.main(data,alg_params)


if __name__ == '__main__':
    #Main Menu
    st.set_page_config(layout="wide")
    st.title('Predictions on the number of users at  Capital Bike Sharing')
    st.write('Last Updated: December 2020')
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
        st.write("NOT IMPLEMENTED YET")
    st.header("References:")
    st.write("[1] Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg")
