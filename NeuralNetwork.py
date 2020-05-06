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

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from Picture import Picture


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
        x = np.array([seg.segment for seg in data])
        y = np.array([seg.label for seg in data])
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
        model.save('.\\saved_model\\my_model1')

    def train2(self, data,epoch=10,  n_split=3):
        random.shuffle(data)
        x = np.array([seg.segment for seg in data])
        y = np.array([seg.label for seg in data])
        random.shuffle(data)

        for train_index, test_index in KFold(n_split).split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test))

            print('Model evaluation ', self.model.evaluate(x_test, y_test))
        self.model.save('.\\saved_model\\my_model2')

    def load_model(self, name):
        self.model = tf.keras.models.load_model('saved_model\\' +name)

    def predictImage(self, picture, size):
        image = picture.original_image

        h, w = image.shape[:2]

        Picture.test_image(image, 'test1', 600)

        fovmask = picture.fovmask
        blueChannel = image[:, :, 0]
        greenChannel = image[:, :, 1]
        redChannel = image[:, :, 2]
        newColor = [np.average(blueChannel), np.average(greenChannel), np.average(redChannel)]
        image[fovmask == 0] = newColor
        padding = size // 2

        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, newColor)


        network_input=[]
        predicted_values=[]
        counter=0

        for i in range(h):
            # print(i, counter)
            for j in range(w):
                counter+=1
                x = i + padding
                y = j + padding

                network_input.append(image[x - padding:x + padding + 1, y - padding:y + padding + 1]/255.0)

                if not counter%1000:
                    network_input=np.array(network_input)
                    predicted_values.append(self.model.predict(network_input))
                    network_input=[]
                    print('predicted up to:', i, 'rows', counter,'segments')


        print(predicted_values.shape)
        print(predicted_values[:10])

        result_image=np.reshape(predicted_values, (h, w))

        cv2.imshow('result', result_image)

        cv2.waitKey(0)
