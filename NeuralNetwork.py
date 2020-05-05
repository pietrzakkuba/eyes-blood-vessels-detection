import random

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, GRU, Conv1D, MaxPooling1D, Conv2D


class NeuralNetwork:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3),  input_shape=(5, 5, 1), activation='relu'))

        # self.model.add(Flatten())
        # self.model.add(Conv2D(32, [3, 3], activation='relu', input_shape=(400, 5, 5, 1)))
        # self.model.add(Dense(units=64, input_dim=500,
        # self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, positive, negative, n_split=3):
        data = positive + negative
        random.shuffle(data)
        x = np.array([sample.segment for sample in data])
        y = np.array([sample.center for sample in data])
        np.random.seed(0)
        indices = np.random.rand(len(data)) < 0.8
        x_train = x[indices]
        x_test = x[~indices]
        y_train = y[indices]
        y_test = y[~indices]
        # self.model.fit(x_train, y_train, epochs=20)
        # print('Model evaluation ', self.model.evaluate(x_test, y_test))
