from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


import time
import numpy as np
from math import sqrt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

######################################## Datasets ###################################################
def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset
####################################### Utilities ########################################################


def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



######################################### Models ######################################################

def KNN(X_train,X_test,y_train,y_test,metric):
    # k-Nearest Neighbor
    print("k-Nearest Neighbor Regressor")
    # fit a k-nearest neighbor model to the data
    startTime = time.time()
    model = KNeighborsRegressor(n_neighbors=14,metric=metric)
    print(model)
    model.fit(X_train, y_train)
    running=time.time() - startTime
    print("--- %s seconds ---" % (time.time() - startTime))
    # make predictions
    expected = y_test
    predicted = model.predict(X_test)
    return expected,predicted,running

def SVCR(X_train,X_test,y_train,y_test):
     # Support Vector Regression
    print("Support Vector Regression")
    startTime = time.time()
    model = SVR(C=14,kernel='poly')
    print(model)
    model.fit(X_train, y_train)
    running=time.time() - startTime
    print("--- %s seconds ---" % (time.time() - startTime))
    # make predictions
    expected = y_test
    predicted = model.predict(X_test)
    return expected,predicted,running

def NNR(X_train,X_test,y_train,y_test):
    # Neural Network Regressor
    print("Neural Network Regressor")
    # fit a k-nearest neighbor model to the data
    startTime = time.time()
    model = MLPRegressor()
    print(model)
    model.fit(X_train, y_train)
    running=time.time() - startTime
    print("--- %s seconds ---" % (time.time() - startTime))
    # make predictions
    expected = y_test
    predicted = model.predict(X_test)
    return expected,predicted,running



############################### MAIN ######################################

#-------------------------------------Data Load-------------------------------------------------------------------------------------------------------------------------------
filename = 'datasets/hour.csv'
hour = loadcsv(filename)
X_t=hour[:17380,:14]
y_t=hour[:17380,14:]

X_t_1Y=hour[:8690,:14]
y_t_1Y=hour[:8690,14:]

X_t_2Y=hour[8666:,:14]
y_t_2Y=hour[8666:,14:]
#-------------------------------------- Global Definitions----------------------------------------------------------------------------------------------------------------


#-------------------------------Time series Data Split-----------------------------------------------------------------------------------------------------------

#Data Split for the first year
err=[]
run=[]
pred=[]
tscv = TimeSeriesSplit(n_splits=720)
for train_index, test_index in tscv.split(X_t):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_t_2Y[train_index],X_t_2Y[test_index]
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train, y_test = y_t_2Y[train_index], y_t_2Y[test_index]
    y_train=np.ravel(y_train)
    y_test=np.ravel(y_test)
    print("X train",X_train.shape,"y train",y_train.shape,"X test",X_test.shape,"y test",y_test.shape)

    Hidden=32
    Epochs=100
    expected_NNR,predicted_NNR,running_NNR=NNR(X_train,X_test,y_train,y_test)
    pred.append(predicted_NNR)
    run.append(running_NNR)
    MAE_NNR=mean_absolute_error(expected_NNR, predicted_NNR)
    err.append(MAE_NNR)
    MAPE_NNR=mean_absolute_percentage_error(expected_NNR, predicted_NNR)
    err.append(MAPE_NNR)
    RMSE_NNR=sqrt(mean_squared_error(expected_NNR, predicted_NNR))
    err.append(RMSE_NNR)
    print(MAE_NNR,MAPE_NNR,RMSE_NNR)
