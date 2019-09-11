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


def get_silso():
    r = requests.get("http://www.sidc.be/silso/INFO/snmtotcsv.php")
    if r.status_code != 200:
        print('get data error {}'.format(r.status_code))
        return -1
    csv = "year;month;time;activity;o1;o2;o3\n" + r.content.decode("utf-8")  # ???
    df = pd.read_csv(io.StringIO(csv), delimiter=';')
    df1 = df[["time", "activity"]]
    return df1


def create_dataset(data_set, lookback):
    data_x, data_y = [], []
    for i in range(len(data_set)-lookback-1):
        a = data_set[i:(i+lookback), 0]
        data_x.append(a)
        data_y.append(data_set[i+lookback, 0])
    return np.array(data_x), np.array(data_y)



def main():
    df1 = get_silso()
    yy = df1['activity'].values
    x = df1['time'].values
    y = np.array([(y1, x1) for y1, x1 in zip(yy, x)])

    # нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    ym = scaler.fit_transform(y)

    #разбиваем на training and testing set
    train_size = int(len(ym)*2/3)
    test_size = len(ym)-train_size

    train, test = ym[0:train_size], ym[train_size+1:]

    lookback = 1
    train_x, train_y = create_dataset(train, lookback)
    test_x, test_y = create_dataset(test, lookback)
    print(train_x.shape)

    train_x = np.reshape(train_x, (train_x.shape[0]), 1, train_x.shape[1])
    test_x = np.reshape(test_x, (test_x.shape[0]), 1, test_x.shape[1])

    # Создание модели LSTM

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, lookback)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer='adam')

    model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2)

    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Статистика модели
    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    print("Train score ={.2f}".format(train_score))
    print("Test score ={.2f}".format(test_score))

    # рисуем


    train_predict_plot = np.empty_like(y)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[len(train_predict) + lookback * 2 + 1:len(y) - 1, :] = train_predict

    test_predict_plot = np.empty_like(y)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(test_predict) + lookback * 2 + 1:len(y) - 1, :] = test_predict

    plt.plot(scaler.inverse_transform(y))
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show()


main()