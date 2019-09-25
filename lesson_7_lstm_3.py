# quadratic prediction

import warnings
warnings.simplefilter(action="ignore", category=Warning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # WARN

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


A = -9.8
B = 30


def show_predict(model, data, n_features=1):
    x_input = np.array(data)
    x_input = x_input.reshape((1, len(data), n_features))
    y_hat = model.predict(x_input, verbose=0)
    print(data, func(*data), y_hat[0])


# quadratic eq
def func(x, a, b, c):
    return x*x*a + x*b + c


# get data for n times (x) and m values of c
def train_data_seria(n, m, a=A, b=B):
    x = [(i, a, b, j*100) for j in range(m) for i in range(n)]
    return np.array(x), np.array([func(*xi) for xi in x])


n_steps = 4
n_features = 1

# split into samples
x, y = train_data_seria(10, 10)

# print the data
for i in range(len(x)):
    print(x[i], y[i])


# define simple model
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

x = x.reshape((x.shape[0], x.shape[1], n_features))

# fit model
model.fit(x, y, epochs=150, batch_size=30, verbose=2)

print("predictions")
show_predict(model, [2, A, B, 100])
show_predict(model, [3, A, B, 200])
show_predict(model, [4, A, B, 300])
show_predict(model, [5, A, B, 400])

