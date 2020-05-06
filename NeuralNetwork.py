import random

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, GRU, Conv1D, MaxPooling1D, Conv2D
from tensorflow.keras import datasets, layers, models
import sys
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, size):
        self.size = size
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.size, self.size, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1024, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.summary()
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def train(self, data, n_split=3):
        random.shuffle(data)
        x=np.array([seg.segment for seg in data])
        y=np.array([seg.label for seg in data])
        random.shuffle(data)

        np.random.seed(0)
        indices = np.random.rand(len(data)) < 0.8  # 80% train
        train_images = x[indices]
        test_images = x[~indices]
        train_labels = y[indices]
        test_labels = y[~indices]
        #######################################
        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.size, self.size, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_images, train_labels, epochs=10,
                            validation_data=(test_images, test_labels))

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(test_acc)

    def train2(self, data, n_split=3):
        random.shuffle(data)
        x = np.array([seg.segment for seg in data])
        y = np.array([seg.label for seg in data])
        random.shuffle(data)


        for train_index, test_index in KFold(n_split).split(x):
            print("KFOLD!")
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(x_train, y_train, epochs=10)

            print('Model evaluation ', self.model.evaluate(x_test, y_test))
