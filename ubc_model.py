# -*- coding: utf-8 -*-


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
from keras.utils import to_categorical
import keras

# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed

set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os

from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)

data = pd.read_csv("./gdrive/My Drive/activities_data2.csv").fillna(0)
data_sensors = pd.read_csv("./gdrive/My Drive/sensors.csv")

values = data.values

data_sensors_value = data_sensors.values[:, 0]

sensor_values = []
goals = []
goals.append(5)
sensor_values.append(values[0])
i = 1
for i in range(1, 295):
    sensor_values.append(values[5 * i])
    goals.append(values[5 * i - 1][0])

for sensor in goals:
    print(sensor)

for i in range(len(sensor_values)):
    for j in range(len(sensor_values[i])):
        k = 0

        while (k < len(original_sensors_value)):
            if (sensor_values[i][j] == original_sensors_value[k]):
                print(sensor_values[i][j])

                sensor_values[i][j] = int(new_sensors_value[k])

            k = k + 1

# print(sensor_values)

labels = to_categorical(goals, num_classes=None)


def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    return predictors, max_sequence_len


predictors, max_sequence_len = generate_padded_sequences(sensor_values)

for predictor in predictors:
    print(predictor)

total_sensors = 155


def create_model(max_sequence_len, total_sensors):
    input_len = 155
    model = Sequential()

    # Add Input Embedding Layer
    model.add(Embedding(total_sensors, 10, input_length=input_len))

    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(max_sequence_len, return_sequences=True))

    model.add(LSTM(max_sequence_len))

    # Add Output Layer

    adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.add(Dense(labels.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=adam)

    return model


model = create_model(max_sequence_len, total_sensors)
model.summary()

model.fit(predictors, labels, epochs=1000, verbose=True)

model.save("./gdrive/My Drive/UBC_model.hdf5")
