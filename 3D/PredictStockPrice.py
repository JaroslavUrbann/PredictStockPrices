import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K
import sys


def process_stock_data_1(name, n_adj_close_i):
    # <editor-fold desc="Reading csv File">
    stock = pd.read_csv("../Companies2/" + name + ".csv",
                        parse_dates=[0],
                        usecols=["open", "high", "low", "adjusted_close", "MACD", "MACD_Hist", "MACD_Signal", "SlowD", "SlowK", "RSI", "ADX", "Aroon Down", "Aroon Up", "Real Lower Band", "Real Middle Band", "Real Upper Band", "ATR", "Chaikin A/D"])[::-1]
    df = pd.DataFrame(stock)
    y = pd.DataFrame()
    # </editor-fold>

    # <editor-fold desc="Reshaping Input Data">
    df = df.apply(pd.to_numeric)
    df = df.apply(zscore)
    x = pd.DataFrame()
    x["StockPrice"] = df.values.tolist()
    # </editor-fold>

    # <editor-fold desc="Adding Shifted Columns">
    x.insert(1, "StockPrice+1", x.StockPrice.shift(-1))
    for i in range(1, n_adj_close_i):
        x.insert(0, "StockPrice-" + str(i), x.StockPrice.shift(i))
    x = x.dropna()
    y["StockPrice+1"] = x.loc[:, "StockPrice+1"]
    x.drop('StockPrice+1', axis=1, inplace=True)
    # </editor-fold>

    # <editor-fold desc="Reshaping DataFrames to Arrays">
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    for i, row in y.iterrows():
        y.set_value(i, "StockPrice+1", row[0][3])
    x = array(x.values.tolist())
    # x = x.reshape(x.shape[0], x.shape[1], 1)
    y = array(y.values.tolist())
    # </editor-fold>

    return x, y


def build_model1(input_shape, output_shape):
    dropout = 0.2
    model = Sequential()
    model.add(LSTM(30, batch_input_shape=(None, input_shape[1], input_shape[2]), return_sequences=True))
    model.add(Dense(30, input_dim=30, activation="linear"))
    # model.add(LSTM(16, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(output_shape[1], activation="linear"))
    model.add(Flatten())
    # model.add(Dense(16, activation="linear"))
    # model.add(Dense(output_shape, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


def custom_mse(y_true, y_pred):
    mask = K.max((abs(y_true) - abs(y_pred)) * 10000000000000, 0) + 1
    return K.square(y_true - y_pred) * mask


bestmodel_path = "weights3.hdf5"
x, y = process_stock_data_1("AAL", 0)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, shuffle=False)
model = build_model1(train_x.shape, train_y.shape)
checkpoint = ModelCheckpoint(bestmodel_path, monitor='val_loss', verbose=2, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
callbacks_list = [checkpoint, reduce_lr]

h = []
for a in range(1):
    if a > 0:
        time.sleep(30)
    history = model.fit(train_x,
                        train_y,
                        epochs=30,
                        verbose=2,
                        validation_split=0.2)
                        # callbacks=callbacks_list)
    h = h + history.history["val_loss"]
# best_model = load_model(bestmodel_path)
# predict_y = best_model.predict(test_x)
# csv_predict = pd.DataFrame(predict_y)
# csv_predict.to_csv("csv_predict3.csv")
predict_y2 = model.predict_classes(test_x)

true_array = []
predicted_array = []
predicted_array2 = []
list_test_y = test_y.tolist()
for i in range(len(list_test_y)):
    true_array.append(list_test_y[i][0])
    # predicted_array.append(predict_y[i][0])
    predicted_array2.append(predict_y2[i])

print(list_test_y)
print(predict_y2)
plt.plot(true_array, color="red", label="truth")
# plt.plot(predicted_array, color="blue", label="prediction_best")
plt.plot(predicted_array2, color="green", label="prediction_new")
# plt.plot(np.zeros(325,), color="black", label="0")
plt.legend(loc='upper left')
plt.show()
plt.plot(h)
plt.show()
