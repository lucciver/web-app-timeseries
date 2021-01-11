from sklearn.model_selection import train_test_split,TimeSeriesSplit
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
    model = KNeighborsRegressor(n_neighbors =14,metric=metric)
    print(model)
    model.fit(X_train, y_train)
    # make predictions
    expected = y_test
    predicted = model.predict(X_test)

    return expected, predicted


def main(data, params):
    """
    data()

    params[n_splits] = 720

    """
    index_split = int((data.shape[0])*(params["train/test"][0]/100))
    print("Index", index_split)

    X_t=data[:data.shape[0]+1,:data.shape[1]-1]
    Y_t=data[:data.shape[0]+1,data.shape[1]-1:]

    X_split = np.split(X_t,[index_split])
    Y_split = np.split(Y_t,[index_split])

    X_t_train = X_split[0]
    y_t_train = Y_split[0]
    X_t_test = X_split[1]
    y_t_test = Y_split[1]
    print("X train", X_t_train.shape,"y train",y_t_train.shape,"X test",X_t_test.shape,"y test",y_t_test.shape)

    err_mae =[]
    err_mape = []
    err_rmse = []
    run = []
    pred = []
    tscv = TimeSeriesSplit(params["n_splits"])

    for train_index, test_index in tscv.split(X_t_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_t_train[train_index], X_t_train[test_index]
        y_train, y_test = y_t_train[train_index], y_t_train[test_index]
        print("X train",X_train.shape,"y train",y_train.shape,"X test",X_test.shape,"y test",y_test.shape)
        expected ,predicted  = KNN(X_train,X_test,y_train,y_test, params["metric"], params["n_neighbors"])

        pred.append(predicted)
        #Metricas
        MAE = mean_absolute_error(expected, predicted)
        err_mae.append(MAE)
        MAPE = mean_absolute_percentage_error(expected, predicted)
        err_mape.append(MAPE)
        RMSE = sqrt(mean_squared_error(expected, predicted))
        err_rmse.append(RMSE)

    return err_mae, err_mape, err_rmse
