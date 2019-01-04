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
