import numpy as np
import math
import matplotlib.pyplot as plt

def get_change(first, second):
    if second == first:
        return 1.0
    try:
        return (second - first) / first + 1
    except ZeroDivisionError:
        return 0

def getStockDataVec(key):
    vec = []
    lines = open("../companies/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))
    vec.reverse()

    return vec

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(get_change(block[i], block[i + 1]) - 1))

    return np.array([res])
