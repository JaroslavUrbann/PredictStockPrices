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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM


def process_stock_data_1(name, n_adj_close_i, n_adj_close_o):
    stock = pd.read_csv("Companies/" + name + ".csv",
                        parse_dates=[0],
                        usecols=["timestamp", "adjusted_close"])
    x = pd.DataFrame(stock)
    tempdf = x.copy(True)
    y = pd.DataFrame()
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
    datecol = x["timestamp"]
    x.insert(0, "week_day_sin", np.sin(datecol.dt.weekday*(2.*np.pi/7)))
    x.insert(0, "week_day_cos", np.cos(datecol.dt.weekday*(2.*np.pi/7)))
    x.insert(0, "day_of_year_sin", np.sin(datecol.dt.dayofyear*(2.*np.pi/366)))
    x.insert(0, "day_of_year_cos", np.sin(datecol.dt.dayofyear*(2.*np.pi/366)))
    x.insert(0, "year", datecol.dt.year)
    x.insert(0, "Energy", 0)
    x.insert(0, "Finance", 0)
    x.insert(0, "Health Care", 0)
    x.insert(0, "Transportation", 0)
    with open("1500 companies.csv", "r") as companies:
        companieslist = pd.DataFrame(pd.read_csv(companies, header=None, names=["ticker", "company_name", "sector"]))
        row = companieslist.loc[companieslist['ticker'] == name]
        x[row.iloc[0]["sector"]] = 1
    x.drop('timestamp', axis=1, inplace=True)
    x.drop(x.index[len(x)-n_adj_close_i-n_adj_close_i+3:len(x)], inplace=True)
    y.drop(y.index[len(y)-n_adj_close_i-n_adj_close_i+3:len(y)], inplace=True)
    x = array(x.values.tolist())
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = array(y.values.tolist())
    # y = y.reshape(y.shape[0], y.shape[1], 1)
    return x, y


def build_model1(input_shape):
    dropout = 0.2
    model = Sequential()
    # model.add(LSTM(128, input_shape=(input_shape[1], 1), return_sequences=True))
    # model.add(Dropout(dropout))
    model.add(LSTM(128, input_shape=(input_shape[1], 1), return_sequences=False))
    # model.add(Dropout(dropout))
    model.add(Dense(128, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(16, kernel_initializer="uniform", activation="relu"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


x, y = process_stock_data_1("AAWW", 20, 16)
test_x, train_x, test_y, train_y = train_test_split(x, y, test_size=0.90, shuffle=False)
model = build_model1(train_x.shape)

model.fit(train_x,
          train_y,
          epochs=50,
          verbose=2,
          validation_split=0.2)

p = model.predict(test_x)
k = [0]
pr = [0]
ree = test_y.tolist()
for i in range(len(ree)):
    k.append(ree[i][0])
    pr.append(p[i][0])

plt.plot(k, color="red", label="truth")
plt.plot(pr, color="blue", label="prediction")
plt.legend(loc='upper left')
plt.show()
