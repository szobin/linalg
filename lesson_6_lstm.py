import warnings
import os
import tensorflow as tf

import requests
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import math


warnings.simplefilter(action="ignore", category=Warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)


def get_silso():
    r = requests.get("http://www.sidc.be/silso/INFO/snmtotcsv.php")
    if r.status_code != 200:
        print('get data error {}'.format(r.status_code))
        return -1
    csv = "year;month;time;activity;o1;o2;o3\n" + r.content.decode("utf-8")  
    df = pd.read_csv(io.StringIO(csv), delimiter=';')
    df1 = df[["time", "activity"]]
    return df1


def create_dataset(data_set, look_back):
    data_x, data_y = [], []
    for i in range(len(data_set)-look_back-1):
        a = data_set[i:(i+look_back)]
        data_x.append(a)
        data_y.append(data_set[i+look_back])
    return np.array(data_x), np.array(data_y)


def main():
    # 1. preparacion de los datos
    # 1.1 importar los datos de fuente
    df1 = get_silso()
    yy = df1['activity'].values
    # x = df1['time'].values

    # 1.2 normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    ym = scaler.fit_transform(yy.reshape((-1, 1)))

    # 1.3 dividir para training y testing set
    train_size = int(len(ym)*2/3)
    # test_size = len(ym)-train_size

    train, test = ym[0:train_size], ym[train_size+1:]

    # 1.4 crear dataset
    look_back = 24
    n_features = 1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], n_features))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], n_features))

    # 2. Creating of model LSTM

    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, n_features)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer='adam')

    # 3. Ensena de la red
    model.fit(train_x, train_y, epochs=2, batch_size=1, verbose=2)

    # 4. prediccion
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    # print(train_y)
    # print(train_predict[:, 0])
    # 5. Model statistic's
    train_score = math.sqrt(mean_squared_error(train_y[:, 0], train_predict[:, 0]))
    test_score = math.sqrt(mean_squared_error(test_y[:, 0], test_predict[:, 0]))
    print("Train score ={0:.3f}".format(train_score))
    print("Test score ={0:.3f}".format(test_score))

    # 6. draw results
    train_predict = scaler.inverse_transform(train_predict)
    train_predict_plot = np.empty_like(ym)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    # print(train_predict_plot)

    test_predict = scaler.inverse_transform(test_predict)
    test_predict_plot = np.empty_like(ym)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + look_back * 2+2:len(ym) - 1] = test_predict

    # print(test_predict_plot)

    plt.plot(scaler.inverse_transform(ym))
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    # plt.show()
    plt.savefig("lstm.png")


main()
