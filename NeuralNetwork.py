import random

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, GRU, Conv1D, MaxPooling1D, Conv2D


class NeuralNetwork:
    def __init__(self):
        self.model = Sequential()  # sequential = sieć jako lista warstw, dodajemy warstwy metodą .add() (jak w standardowej liście)

        self.model.add(Flatten())

        self.model.add(Conv2D(32, [3, 3], activation='relu', input_shape=(400, 5, 5, 1)))

        # self.model.add(Dense(units=64, input_dim=500,
        #                 activation='relu'))  # dodajemy warstwę Dense (gęstą). Dense oznacza, że wszystkie wejścia (w tym przypadku 100) połączone są z neuronami warstwy w sposób każdy z każdym (każdy neuron z poprzedniej warstwy połączony z każdym neuronem warstwy następnej, tak jak to robiliśmy na poprzednich laboratoriach)
        self.model.add(Dense(units=1,
                        activation='sigmoid'))  # rozmiar wejścia zdefiniować musimy tylko w pierwszej warstwie (definiujemy ile jest cech na wejściu). Ponieważ model wie jakie są rozmiary poprzednich warstw - może w sposób automatyczny odkryć, że opprzednia warstwa generuje 64 wyjścia

        self.model.compile(loss='binary_crossentropy',
                           # budujemy model! ustawiamy funkcję kosztu - mamy klasyfikację z dwiema etykietami, więc stosujemy 'binary_crossentropy'
                           optimizer='adam',  # wybieramy w jaki sposób sieć ma się uczyć
                           metrics=['accuracy'])  # i wybieramy jaka miara oceny nas interesuje

    def train(self, positive, negative, n_split=3):
        X = positive + negative
        Y = [1 for x in positive] + [0 for x in negative]
        # Y = [x.target for x in X]

        order = [x for x in range(len(X))]
        random.shuffle(order)
        X = [X[i] for i in order]
        Y = [Y[i] for i in order]

        X = [x.segment for x in X]
        X = np.array(X)
        Y = np.array(Y)
        print(X[0])

        for train_index, test_index in KFold(n_split).split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            self.model.fit(x_train, y_train, epochs=20)

            print('Model evaluation ', self.model.evaluate(x_test, y_test))
