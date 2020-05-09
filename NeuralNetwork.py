import random

import cv2
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, GRU, Conv1D, MaxPooling1D, Conv2D
from tensorflow.keras import datasets, layers, models
import sys
import tensorflow as tf

from datetime import datetime

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt



class NeuralNetwork:
    def __init__(self, size):
        self.size = size
        self.model = models.Sequential()

        self.model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.size, self.size, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        # self.model.summary()
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train2(self, data, epoch=10, n_split=3):
        random.shuffle(data)
        x = np.array([seg.segment for seg in data])
        y = np.array([seg.label for seg in data])

        x = np.expand_dims(x, -1)
        y = np.expand_dims(y, -1)

        for train_index, test_index in KFold(n_split).split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test))

            print('Model evaluation ', self.model.evaluate(x_test, y_test))
        self.model.save('.\\saved_model\\my_model2')

    def load_model(self, name):
        self.model = tf.keras.models.load_model('saved_model\\' + name)

    def predictImage(self, picture, size, name):
        image = picture.original_image

        h, w = image.shape[:2]

        fovmask = picture.fovmask
        # blueChannel = image[:, :, 0]
        # greenChannel = image[:, :, 1]
        # redChannel = image[:, :, 2]
        newColor = np.average(image)
        # newColor = [np.average(blueChannel), np.average(greenChannel), np.average(redChannel)]
        image[fovmask == 0] = newColor
        padding = size // 2

        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, newColor)

        network_input = []
        predicted_values = []
        counter = 0
        counter2 = 0
        pixels = w * h
        for i in range(h):
            for j in range(w):
                counter += 1
                x = i + padding
                y = j + padding

                network_input.append(image[x - padding:x + padding + 1, y - padding:y + padding + 1] / 255.0)

                if not counter % 1000:
                    counter2 += 1
                    network_input = np.array(network_input)
                    network_input = np.expand_dims(network_input, -1)
                    predicted_values.append(self.model.predict(network_input))
                    network_input = []
                    if not counter2 % 10:
                        print('progress: {} out of {}'.format(counter, pixels))

        if len(network_input):
            network_input = np.array(network_input)
            network_input = np.expand_dims(network_input, -1)
            predicted_values.append(self.model.predict(network_input))

        predicted_values = np.concatenate(predicted_values)

        predicted_values[predicted_values >= 0.7] = 1
        predicted_values[predicted_values < 0.7] = 0

        result_image = np.reshape(predicted_values, (h, w))
        result_image = result_image * 255
        cv2.imwrite('result\\' + name + '.jpg', result_image)
        return result_image
        # Picture.test_image(result_image, 'result', 600)
