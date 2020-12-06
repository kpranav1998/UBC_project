# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

MAIN = "./"
LABEL_MAIN = "./"
data_files = [" accelerometer_chest_with_sax.csv", " accelerometer_forearm_with_sax.csv",
              " accelerometer_head_with_sax.csv", " accelerometer_shin_with_sax.csv",
              " accelerometer_thigh_with_sax.csv", " accelerometer_upperarm_with_sax.csv",
              " accelerometer_waist_with_sax.csv"]

permutations = pd.read_csv(os.path.join(MAIN, " label 4 .csv"))
permutations = permutations.iloc[:, 1:8]
values = permutations.values
# print(data2)
data_labels = []
for data_file in data_files:
    d1 = pd.read_csv(os.path.join(LABEL_MAIN, data_file))
    d1 = d1.loc[d1['f'] == 4]
    d1 = d1.iloc[:, 1:4]
    data_labels.append(d1)

print(data_labels[1])

X = []
for row in range(values.shape[0]):
    a = np.zeros(shape=(3, 7))
    for j in range(7):
        label_val = data_labels[j].values  ## h1
        row_number = values[row][j]

        temp = np.expand_dims(label_val[row_number - 1], 1)
        a[0, j], a[1, j], a[2, j] = temp[0], temp[1], temp[2]

    X.append(a)

y = np.asarray(X)

np.save('label4_input', y)

