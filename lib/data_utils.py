import numpy as np
import pandas as pd

# Metrics

def mse(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))

def rmse(y, y_hat):
    return np.sqrt(mse(y, y_hat))

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def mape(y, y_hat, replace_zeros=False, masked=True, epsilon=1e-8):
    if masked:
        mask = np.abs(y) < epsilon
        y_hat = y_hat[~mask]
        y = y[~mask]
    elif replace_zeros:
        y[np.abs(y) < epsilon] = epsilon

    return np.mean(np.abs(y - y_hat) / y)

def mase(data, test, seasonal_freq=1):
    s = seasonal_freq
    T = data.shape[0]

    if s >= T:
        raise ValueError("Seasonality {} is larger than data size".format(s, T))

    mae = 1 / (T - s) * np.sum(np.abs(data[s:].values - data[:T-s].values))
    return 1 / T * np.sum(np.abs(data - test)) / mae

def get_standard_errors(y, y_hat, suffix=None):
    suffix = "_" + suffix if suffix else ""

    return {
        "mse" + suffix: mse(y, y_hat),
        "rmse" + suffix: rmse(y, y_hat),
        "mae" + suffix: mae(y, y_hat),
        "mape" + suffix: mape(y, y_hat)
    }


def compare_timedeltas(operation, timedelta1, timedelta2):
    if operation == "==":
        return pd.to_timedelta(timedelta1) == pd.to_timedelta(timedelta2)
    elif operation == "!=":
        return pd.to_timedelta(timedelta1) != pd.to_timedelta(timedelta2)
    elif operation == ">":
        return pd.to_timedelta(timedelta1) > pd.to_timedelta(timedelta2)
    elif operation == "<":
        return pd.to_timedelta(timedelta1) < pd.to_timedelta(timedelta2)
    elif operation == ">=":
        return pd.to_timedelta(timedelta1) >= pd.to_timedelta(timedelta2)
    elif operation == "<=":
        return pd.to_timedelta(timedelta1) <= pd.to_timedelta(timedelta2)
    else:
        raise Exception("Operation {} not recognized".format(operation))

def convert_to_fourier_day(x, num_fourier_terms=1):
    days = pd.Series(x).dt.dayofweek / 7
    terms = []

    for i in range(1, num_fourier_terms + 1):
        angles = 2 * i * np.pi * days
        terms.append(np.sin(angles))
        terms.append(np.cos(angles))

    fourier_terms = np.column_stack(terms)

    return fourier_terms

# Flattens a matrix composed of sections of rolling values into one flat array
# The first 2 dimensions of matrix should have the rolling values
def flatten_circulant_like_matrix_by_stretches(matrix, stretches):
    flattened = np.empty((0,) + matrix.shape[2:], dtype=matrix.dtype)
    for start, end in stretches:
        flattened = np.concatenate((flattened, matrix[start:end, 0, ...], matrix[end - 1, 1:, ...]), axis=0)

    return flattened

def get_groundtruth_from_y(y):
    if y.shape[-1] == 2:
        return np.transpose(y[:, :, :, 1], axes=(1, 0, 2))
    else:
        return np.transpose(y[:, :, :, 1:], axes=(1, 0, 2, 3))

def get_stretches(datetimes, frequency):
    break_indices = np.argwhere(compare_timedeltas("!=", datetimes[1:] - datetimes[:-1], frequency)).flatten() + 1
    stretch_starts = np.concatenate(([0], break_indices))
    stretch_ends = np.concatenate((stretch_starts, [datetimes.shape[0]]))[1:]
    stretches = np.vstack((stretch_starts, stretch_ends)).T

    return stretches

# Appends filler data to an array until its axis has at least length limit
def pad_array(array, limit, axis=0, filler="zeros"):
    padding = limit - array.shape[axis]
    if padding > 0:
        filler_shape = array.shape[:axis] + (padding,) + array.shape[(axis + 1):]

        if filler == "zeros":
            filler_array = np.zeros(filler_shape)
        else:
            raise ValueError("filler type {} is invalid".format(filler))

        padded_array = np.concatenate((array, filler_array), axis=axis)

        return padded_array
    else:
        return array

def zero_out_detectors(detectors, detector_list):
    indices = [True if detector in detectors else False for detector in detector_list]
    def function(data, timestamps):
        data[..., indices, 1:] = 0

        return data

    return function

def zero_out_days(proportion, seed):
    def function(data, timestamps):
        timestamps_shape = timestamps.shape
        dates = pd.to_datetime(timestamps.flatten()).date.reshape(timestamps_shape)
        unique_dates = np.unique(dates)

        random_state = np.random.get_state()
        np.random.seed(seed)
        zero_dates = np.random.choice(unique_dates, size=(int(proportion * len(unique_dates)),), replace=False)
        np.random.set_state(random_state)

        data[np.isin(dates, zero_dates), ..., 1:] = 0

        return data

    return function
