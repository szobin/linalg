# quadratic prediction
import math

import warnings
warnings.simplefilter(action="ignore", category=Warning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # WARN

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense


def show_predict(model, data, n_features=1):
    x_input = np.array(data)
    x_input = x_input.reshape((1, len(data), n_features))
    y_hat = model.predict(x_input, verbose=0)
    print(data, func(*data), y_hat[0])


# quadratic eq
def func(a, b, c):
    if a == 0:
        if b == 0:
            return -1000, -1000
        return -c/b, -c/b

    d = b*b - 4*a*c
    if d < 0:
        return -1000, -1000
    qd = math.sqrt(d)
    return (-b-qd)/(2*a), (-b+qd)/(2*a)


# get data for n times (x) and m values of c
def train_data_seria(n, m, l):
    x = [(a, b, c) for c in range(l) for b in range(m) for a in range(-5, n-5)]
    return np.array(x), np.array([func(*xi) for xi in x])


n_steps = 3
n_features = 1

# split into samples
x, y = train_data_seria(10, 10, 10)

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
model.add(Bidirectional(LSTM(500, activation='relu', input_shape=(n_steps, n_features), return_sequences=True)))
model.add(LSTM(500, activation='relu'))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')

x = x.reshape((x.shape[0], x.shape[1], n_features))

# fit model
model.fit(x, y, epochs=150, batch_size=30, verbose=2)

print("predictions")
show_predict(model, [-4, 1, 1])
show_predict(model, [-3, 2, 2])
show_predict(model, [-2, 1, 1])
show_predict(model, [-1, 1, 0])

