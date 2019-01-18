import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose

import visualization

def reindex_with_nans(data, data_freq):
    time = data.index.to_pydatetime()
    full_time = min(time) + np.arange((max(time) - min(time)) / data_freq + 1)*data_freq

    data_unimputed = data.reindex(full_time, fill_value=np.nan)

    return data_unimputed

def impute_seasonal_data(data, data_freq, seasonal_freq, graph=False, lower_bound=None, upper_bound=None, title=None, ylabel=None, figsize=None):
    """Imputes seasonal time series data for a dataframe with a Datetime
    Index with missing values."""

    data_unimputed = reindex_with_nans(data, data_freq)

    # Perform linear interpolation so we can use seasonal_decompose
    data = data_unimputed.interpolate()

    sd = seasonal_decompose(data, freq=seasonal_freq)

    if graph:
        visualization.plot_data_over_time(sd.observed, title="{} Original".format(title), ylabel=ylabel, figsize=figsize)
        visualization.plot_data_over_time(sd.trend, title="{} Trend".format(title), ylabel=ylabel, figsize=figsize)
        visualization.plot_data_over_time(sd.resid, title="{} Resid".format(title), ylabel=ylabel, figsize=figsize)
        visualization.plot_data_over_time(sd.seasonal, title="{} Seasonal".format(title), ylabel=ylabel, figsize=figsize)

    sd.trend[data_unimputed.isnull()] = np.nan
    sd.resid[data_unimputed.isnull()] = np.nan

    if graph:
        visualization.plot_data_over_time(sd.trend, title="{} Trend Broken".format(title), ylabel=ylabel, figsize=figsize)
        visualization.plot_data_over_time(sd.resid, title="{} Resid Broken".format(title), ylabel=ylabel, figsize=figsize)

    sd.trend = sd.trend.interpolate()
    sd.resid = sd.resid.interpolate()

    if graph:
        visualization.plot_data_over_time(sd.trend, title="{} Trend Interpolated".format(title), ylabel=ylabel, figsize=figsize)
        visualization.plot_data_over_time(sd.resid, title="{} Resid Interpolated".format(title), ylabel=ylabel, figsize=figsize)

    data_imputed = (sd.trend + sd.resid + sd.seasonal).clip(lower=lower_bound, upper=upper_bound)
    data_imputed.fillna(data_unimputed, inplace=True)

    if graph:
        visualization.plot_data_over_time(data_imputed, title="{} Imputed".format(title), ylabel=ylabel, figsize=figsize)

    return data_imputed

def get_longest_continuous_stretch_of_data(data, freq):
    data = data.dropna()
    index = data.index

    break_indices = np.argwhere((index[1:] - index[:-1]) != freq).flatten() + 1
    stretch_starts = np.concatenate(([0], break_indices, [data.shape[0]]))
    stretch_lengths = stretch_starts[1:] - stretch_starts[:-1]
    longest_stretch_start_index = stretch_lengths.argmax()

    stretch_start = stretch_starts[longest_stretch_start_index]
    stretch_end = stretch_starts[longest_stretch_start_index + 1]

    indices = (stretch_start, stretch_end)
    stretch = data.iloc[stretch_start:stretch_end]

    return indices, stretch

def get_longest_missing_data_stretch_length(data, freq):
    data = data.dropna()
    index = data.index

    break_ends = np.argwhere((index[1:] - index[:-1]) != freq).flatten()
    break_lengths = index[break_ends + 1] - index[break_ends]
    longest_break_length = max(break_lengths) - freq

    return longest_break_length

def test_seasonal_decomposition_imputation(data):
    pass
