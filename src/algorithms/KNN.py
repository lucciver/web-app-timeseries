from sklearn.model_selection import train_test_split,TimeSeriesSplit
import time
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def KNN(X_train,X_test,y_train,y_test, metric, n_neighbors):
    # k-Nearest Neighbor
    print("k-Nearest Neighbor Regressor")
    # fit a k-nearest neighbor model to the data
    startTime = time.time()
    model = KNeighborsRegressor(n_neighbors =14,metric=metric)
    print(model)
    model.fit(X_train, y_train)
    running = time.time() - startTime
    print("--- %s seconds ---" % (time.time() - startTime))
    # make predictions
    expected = y_test
    predicted = model.predict(X_test)

    return expected, predicted, running


def main(data, params):
    """
    data()

    params[n_splits] = 720

    """

    X_t=data[:17380,:14]
    y_t=data[:17380,14:]

    X_t_1Y=data[:8690,:14]
    y_t_1Y=data[:8690,14:]

    X_t_2Y=data[8666:,:14]
    y_t_2Y=data[8666:,14:]

    err_mae =[]
    err_mape = []
    err_rmse = []
    run=[]
    pred=[]
    tscv = TimeSeriesSplit(params["n_splits"])
    for train_index, test_index in tscv.split(X_t):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_t_2Y[train_index],X_t_2Y[test_index]
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train, y_test = y_t_2Y[train_index], y_t_2Y[test_index]
        y_train=np.ravel(y_train)
        y_test=np.ravel(y_test)
        print("X train",X_train.shape,"y train",y_train.shape,"X test",X_test.shape,"y test",y_test.shape)

        expected ,predicted ,running = KNN(X_train,X_test,y_train,y_test, params["metric"], params["n_neighbors"])
        pred.append(predicted)
        run.append(running)
        #Metricas
        MAE = mean_absolute_error(expected, predicted)
        err_mae.append(MAE)
        MAPE = mean_absolute_percentage_error(expected, predicted)
        err_mape.append(MAPE)
        RMSE = sqrt(mean_squared_error(expected, predicted))
        err_rmse.append(RMSE)

    return err_mae, err_mape, err_rmse
