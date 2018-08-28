import numpy as np
from numpy import array
import matplotlib.pyplot as plt
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
from keras.callbacks import ModelCheckpoint


def process_stock_data_1(name, n_adj_close_i, n_adj_close_o):
    # <editor-fold desc="Reading csv File">
    stock = pd.read_csv("Companies/" + name + ".csv",
                        parse_dates=[0],
                        usecols=["timestamp", "adjusted_close"])
    x = pd.DataFrame(stock)
    y = pd.DataFrame()
    # </editor-fold>

    # <editor-fold desc="Adding Shifted Columns">
    tempdf = x.copy(True)
    for i in range(0, n_adj_close_i + n_adj_close_o):
        if i > n_adj_close_o:
            x.insert(1, "Adjusted Close " + str(n_adj_close_o-i), tempdf["adjusted_close"])
        if i == n_adj_close_o:
            x["adjusted_close"] = tempdf["adjusted_close"]
            x["timestamp"] = tempdf["timestamp"]
        if i < n_adj_close_o:
            y.insert(0, "Adjusted Close +" + str(n_adj_close_o-i), tempdf["adjusted_close"])
            x["timestamp"] = tempdf["timestamp"]
        tempdf = tempdf.iloc[1:]
        tempdf.reset_index(drop=True, inplace=True)

    x.drop(x.index[len(x)-n_adj_close_i-n_adj_close_i+3:len(x)], inplace=True)
    y.drop(y.index[len(y)-n_adj_close_i-n_adj_close_i+3:len(y)], inplace=True)

    for i in range(1, 15):
        y.drop("Adjusted Close +" + str(i), axis=1, inplace=True)
    # </editor-fold>

    # <editor-fold desc="Encoding Timestamp">
    datecol = x["timestamp"]
    x.insert(0, "week_day_sin", np.sin(datecol.dt.weekday*(2.*np.pi/7)))
    x.insert(0, "week_day_cos", np.cos(datecol.dt.weekday*(2.*np.pi/7)))
    x.insert(0, "day_of_year_sin", np.sin(datecol.dt.dayofyear*(2.*np.pi/366)))
    x.insert(0, "day_of_year_cos", np.sin(datecol.dt.dayofyear*(2.*np.pi/366)))
    x.insert(0, "year", datecol.dt.year)
    x.drop('timestamp', axis=1, inplace=True)
    # </editor-fold>

    # <editor-fold desc="Adding Sector Columns">
    x.insert(0, "Energy", 0)
    x.insert(0, "Finance", 0)
    x.insert(0, "Health Care", 0)
    x.insert(0, "Transportation", 0)
    with open("1500 companies.csv", "r") as companies:
        companieslist = pd.DataFrame(pd.read_csv(companies, header=None, names=["ticker", "company_name", "sector"]))
        row = companieslist.loc[companieslist['ticker'] == name]
        x[row.iloc[0]["sector"]] = 1
    # </editor-fold>

    # <editor-fold desc="Reversing DataFrame by Row">
    x = x.reindex(index=x.index[::-1])
    x.reset_index(drop=True, inplace=True)
    y = y.reindex(index=y.index[::-1])
    y.reset_index(drop=True, inplace=True)
    # </editor-fold>

    # <editor-fold desc="Reshaping DataFrames to Arrays">
    x = array(x.values.tolist())
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = array(y.values.tolist())
    # </editor-fold>

    return x, y


def build_model1(input_shape, output_shape):
    dropout = 0.2
    model = Sequential()
    # model.add(LSTM(128, input_shape=(input_shape[1], 1), return_sequences=True))
    # model.add(Dropout(dropout))
    model.add(LSTM(30, input_shape=(input_shape[1], 1), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(24, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(output_shape, kernel_initializer="uniform", activation="relu"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


n_close_x, n_close_y = 20, 16
bestmodel_path = "weights2.hdf5"

x, y = process_stock_data_1("AAWW", n_close_x, n_close_y)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, shuffle=False)

model = build_model1(train_x.shape, 2)
checkpoint = ModelCheckpoint(bestmodel_path, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# <editor-fold desc="Fitting model">
for a in range(1):
    if a > 0:
        time.sleep(60)
    model.fit(train_x,
              train_y,
              epochs=75,
              verbose=2,
              validation_split=0.2,
              callbacks=callbacks_list)
# </editor-fold>

best_model = load_model("weights2.hdf5")
predict_y = best_model.predict(test_x)
csv_predict = pd.DataFrame(predict_y)
csv_predict.to_csv("csv_predict.csv")

true_array = []
predicted_array = []
list_test_y = test_y.tolist()
for i in range(len(list_test_y)):
    # true_array.append(list_test_y[50][i])
    # predicted_array.append(predict_y[50][i])
    true_array.append(list_test_y[i][0])
    predicted_array.append(predict_y[i][0])

print(true_array)
print(predicted_array)
plt.plot(true_array, color="red", label="truth")
plt.plot(predicted_array, color="blue", label="prediction")
plt.legend(loc='upper left')
plt.show()
