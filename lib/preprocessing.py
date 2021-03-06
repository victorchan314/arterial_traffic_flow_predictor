import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose

from lib import data_utils, visualization


def reindex_with_nans(data, data_freq):
    time = data.index.to_pydatetime()
    full_time = min(time) + np.arange((max(time) - min(time)) / data_freq + 1)*data_freq

    data_unimputed = data.reindex(full_time, fill_value=np.nan)

    return data_unimputed

def fillna_with_means(data, data_freq):
    data = reindex_with_nans(data, data_freq)
    na_mask = data.isnull().values.flatten()

    break_starts = np.argwhere(na_mask[1:] & ~na_mask[:-1]).flatten() + 1
    break_ends = np.argwhere(~na_mask[1:] & na_mask[:-1]).flatten() + 1

    if len(break_starts) == 0:
        return data

    means = (data.iloc[break_ends].values - data.iloc[break_starts - 1].values).flatten() / 2

    for i in range(len(break_starts)):
        data.iloc[break_starts[i]:break_ends[i] + 1] = means[i]

    return data

def fillna_with_data_mean(data, data_freq):
    data = reindex_with_nans(data, data_freq)
    mean = data.mean()

    data[data.isnull()] = mean.iloc[0]

    return data

def seasonal_decomposition_interpolation_imputation(data, data_freq, seasonal_freq, method="linear", lower_bound=None, upper_bound=None, graph=False, title=None, ylabel=None, figsize=None):
    """Imputes seasonal time series data for a dataframe with a Datetime
    Index with missing values."""

    data_unimputed = reindex_with_nans(data, data_freq)

    # Perform interpolation so we can use seasonal_decompose
    if method == "mean":
        data = fillna_with_means(data_unimputed, data_freq)
    elif method == "data_mean":
        data = fillna_with_data_mean(data_unimputed, data_freq)
    else:
        data = data_unimputed.interpolate(method=method)

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

    if len(break_ends) == 0:
        return dt.timedelta()

    break_lengths = index[break_ends + 1] - index[break_ends]
    longest_break_length = max(break_lengths) - freq

    return longest_break_length

def test_imputation_random(data, data_freq, seasonal_freq, imputation_function, imputation_length=None, imputation_params={}, k=10, error="mape", error_params={}, graph=False, figsize=None, graph_zoom=False, graph_naive_forecast=False):
    if imputation_length is None:
        stretch_length = int(get_longest_missing_data_stretch_length(data, data_freq) / data_freq)
    else:
        # 1 week is the default stretch length
        stretch_length = int(imputation_length / data_freq)

    if stretch_length == 0:
        return data
    elif data.shape[0] < 2 * stretch_length + 2:
        # 2 * stretch_length + N; Not sure if N = 1 or N = 2
        # Using N = 2 means no false positives
        # If N = 1, then should be np.random.randint(stretch_length - 1...
        # Not a huge deal if data.shape[0] >> stretch_length
        raise ValueError("Missing data longer than half of the existing continuous data to test")

    _, data = get_longest_continuous_stretch_of_data(data, data_freq)

    test_indices = np.random.randint(stretch_length, data.shape[0] - stretch_length - 1, k)
    errors = []

    if error == "mape":
        error_function = data_utils.mape
    elif error == "mase":
        error_function = data_utils.mase
    else:
        raise ValueError("Invalid error type")

    for i in range(k):
        test_index = test_indices[i]
        test_stretch = data.iloc[test_index:test_index + stretch_length]
        current_data = data.drop(data.index[test_index:test_index + stretch_length])

        imputed_data = imputation_function(current_data, data_freq, seasonal_freq, **imputation_params)
        imputed_stretch = imputed_data.iloc[test_index:test_index + stretch_length]
        current_error = error_function(test_stretch, imputed_stretch, **error_params)[0]
        errors.append(current_error)

        if graph:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title("{}: {}".format(error, current_error))

            if graph_zoom:
                plt.plot(data[max(0, test_index - stretch_length):min(data.shape[0], test_index + stretch_length + stretch_length)], label="Original")
            else:
                plt.plot(data, label="Original")

            plt.plot(imputed_stretch, label="Imputed")

            if graph_naive_forecast:
                plt.plot(imputed_stretch.index, data[test_index - seasonal_freq:test_index + stretch_length - seasonal_freq].values, label="Naive Forecast")

            ax.legend()

            plt.show()

    average_error = sum(errors) / len(errors)

    return average_error, errors

def test_imputation_sequential(data, data_freq, seasonal_freq, imputation_function, imputation_length=None, imputation_params={}, k=10, error="mape", error_params={}, graph=False, figsize=None, graph_zoom=False, graph_naive_forecast=False):
    if imputation_length is None:
        stretch_length = int(get_longest_missing_data_stretch_length(data, data_freq) / data_freq)
    else:
        # 1 week is the default stretch length
        stretch_length = int(imputation_length / data_freq)

    if stretch_length == 0:
        return data
    elif data.shape[0] < 2 * stretch_length + 2:
        # 2 * stretch_length + N; Not sure if N = 1 or N = 2
        # Using N = 2 means no false positives
        # If N = 1, then should be np.random.randint(stretch_length - 1...
        # Not a huge deal if data.shape[0] >> stretch_length
        raise ValueError("Missing data longer than half of the existing continuous data to test")

    _, data = get_longest_continuous_stretch_of_data(data, data_freq)

    test_indices = np.random.randint(stretch_length, data.shape[0] - stretch_length - 1, k)
    errors = []

    # if error == "mape":
    #     error_function = utils.mape
    # elif error == "mase":
    #     error_function = utils.mase
    # else:
    #     raise ValueError("Invalid error type")

    # Default error function is MAPE
    error_function = data_utils.mape

    for i in range(k):
        test_index = test_indices[i]
        test_stretch = data.iloc[test_index:test_index + stretch_length]
        current_data = data.drop(data.index[test_index:test_index + stretch_length])

        imputed_data = imputation_function(current_data, data_freq, seasonal_freq, **imputation_params)
        imputed_stretch = imputed_data.iloc[test_index:test_index + stretch_length]
        current_error = error_function(test_stretch, imputed_stretch, **error_params)[0]
        errors.append(current_error)

        if graph:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title("{}: {}".format(error, current_error))

            if graph_zoom:
                plt.plot(data[max(0, test_index - stretch_length):min(data.shape[0], test_index + stretch_length + stretch_length)], label="Original")
            else:
                plt.plot(data, label="Original")

            plt.plot(imputed_stretch, label="Imputed")

            if graph_naive_forecast:
                plt.plot(imputed_stretch.index, data[test_index - seasonal_freq:test_index + stretch_length - seasonal_freq].values, label="Naive Forecast")

            ax.legend()

            plt.show()

    average_error = sum(errors) / len(errors)

    return average_error, errors
