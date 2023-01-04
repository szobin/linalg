# linear prediction

import warnings
warnings.simplefilter(action="ignore", category=Warning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # WARN

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense


def create_dataset(data_set, lookback):
    data_x, data_y = [], []
    for i in range(len(data_set)-lookback-1):
        a = data_set[i:(i+lookback)]
        data_x.append(a)
        data_y.append(data_set[i+lookback])
    return np.array(data_x), np.array(data_y)


def show_predict(model, data, n_steps=3, n_features=1):
    x_input = np.array(data)
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(data, yhat[0])


# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
# choose a number of time steps
n_steps = 3
n_features = 1
# split into samples
X, y = create_dataset(raw_seq, n_steps)
# summarize the data
for i in range(len(X)):
    print(X[i], y[i])


# define model 1 layer
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# define model 2 layers
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# define model bidirectional
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# fit model
model.fit(X, y, epochs=150, batch_size=30, verbose=0)

print("predictions")
show_predict(model, [50, 60, 70])
show_predict(model, [70, 80, 90])
show_predict(model, [80, 90, 100])
show_predict(model, [85, 95, 105])

