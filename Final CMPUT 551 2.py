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
    model = SVR(C=14,kernel='linear')
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
hour= loadcsv(filename)    
X_t=hour[:9660,:14]
y_t=hour[:9660,14:]

X_t_1Y=hour[:8690,:14]
y_t_1Y=hour[:8690,14:]

X_t_2Y=hour[8690:,:14]
y_t_2Y=hour[8690:,14:]
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
    
    #KNN
    """
    #metricKNR='canberra'  
    metricKNR='braycurtis'  
    #metricKNR='manhattan'    
    centerKNR=14    
    expected_KNR,predicted_KNR,running_KNR=KNN(X_train,X_test,y_train,y_test,metricKNR) 
    pred.append(predicted_KNR)
    run.append(running_KNR)
    MAE_KNR=mean_absolute_error(expected_KNR, predicted_KNR)
    err.append(MAE_KNR)
    MAPE_KNR=mean_absolute_percentage_error(expected_KNR, predicted_KNR)
    err.append(MAPE_KNR)
    RMSE_KNR=sqrt(mean_squared_error(expected_KNR, predicted_KNR))
    err.append(RMSE_KNR)
    print(MAE_KNR,MAPE_KNR,RMSE_KNR)
          
    """

     
    expected_SVR,predicted_SVR,running_SVR=SVCR(X_train,X_test,y_train,y_test) 
    pred.append(predicted_SVR)
    run.append(running_SVR)
    MAE_SVR=mean_absolute_error(expected_SVR, predicted_SVR)
    err.append(MAE_SVR)
    MAPE_SVR=mean_absolute_percentage_error(expected_SVR, predicted_SVR)
    err.append(MAPE_SVR)
    RMSE_SVR=sqrt(mean_squared_error(expected_SVR, predicted_SVR))
    err.append(RMSE_SVR)
    print(MAE_SVR,MAPE_SVR,RMSE_SVR)   
    
          
#------------------------------------KNeighborsRegressor--------------------------------------------------------------------------------------------------------           
    
    X_train_1Y, X_test_1Y, y_train_1Y, y_test_1Y = train_test_split(X_t_1Y,y_t_1Y,test_size=0.3)
    X_train2, X_val, y_train2, y_val = train_test_split(X_train_1Y, y_train_1Y, test_size=0.3, random_state=0)
    y_train2=np.ravel(y_train2)
    y_val=np.ravel(y_val)  
    """
    #metricKNR='canberra'  
    metricKNR='braycurtis'  
    #metricKNR='manhattan'
    err=[]
    centerKNR=14    
    expected_KNR,predicted_KNR,running_KNR=KNN(X_train2,X_val,y_train2,y_val,metricKNR)     
    MAE_KNR=mean_absolute_error(expected_KNR, predicted_KNR)
    err.append(MAE_KNR)
    MAPE_KNR=mean_absolute_percentage_error(expected_KNR, predicted_KNR)
    err.append(MAPE_KNR)
    RMSE_KNR=sqrt(mean_squared_error(expected_KNR, predicted_KNR))
    err.append(RMSE_KNR)    
    print(MAE_KNR,MAPE_KNR,RMSE_KNR)
    """
#-----------------------------------Support Vector Regression--------------------------------------------------------------------------------------------------    
    
    #kernel='linear'
    #kernel="poly"
    #kernel='rbf'

    
    expected_SVR,predicted_SVR,running_SVR=SVCR(X_train2,X_val,y_train2,y_val) 
    pred.append(predicted_SVR)
    run.append(running_SVR)
    MAE_SVR=mean_absolute_error(expected_SVR, predicted_SVR)
    err.append(MAE_SVR)
    MAPE_SVR=mean_absolute_percentage_error(expected_SVR, predicted_SVR)
    err.append(MAE_SVR)
    RMSE_SVR=sqrt(mean_squared_error(expected_SVR, predicted_SVR))
    err.append(RMSE_SVR)
    print(MAE_SVR,MAPE_SVR,RMSE_SVR)   
    
#----------------------------------Neural Network Regressor----------------------------------------------------------------------------------------------------
    """
    Hidden=2
    Epochs=100    
    expected_NNR,predicted_NNR,running=NNR(X_train2,X_test,y_train2,y_test) 
    MAE_NNR=mean_absolute_error(expected_NNR, predicted_NNR)
    MAPE_NNR=mean_absolute_percentage_error(expected_NNR, predicted_NNR)
    RMSE_NNR=sqrt(mean_squared_error(expected_NNR, predicted_NNR))
    print(MAE_NNR,MAPE_NNR,RMSE_NNR)    
    """
           
        

