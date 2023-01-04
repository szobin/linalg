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

 
# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
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
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
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
model.fit(X, y, epochs=200, verbose=0)

x_input = np.array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)