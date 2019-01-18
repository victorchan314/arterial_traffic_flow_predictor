import numpy as np
import pandas as pd
import matplotlib as pyplot
import datetime as dt

def mape(data, test, replace_zeros=True, epsilon=0.01):
    if replace_zeros:
        data = data.replace(0, epsilon)

    ape = np.sum(np.abs(data - test) / data)
    return 1 / data.shape[0] * ape

def mase(data, test, seasonal_freq):
    s = seasonal_freq
    T = data.shape[0]

    if s >= T:
        raise ValueError("Seasonality {} is larger than data size".format(s, T))

    mae = 1 / (T - s) * np.sum(np.abs(data[s:].values - data[:T-s].values))
    return 1 / T * np.sum(np.abs(data - test)) / mae

