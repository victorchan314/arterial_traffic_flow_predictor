import os

import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX

from models.model import Model
from lib import data_utils

class SARIMAX(Model):
    """Model that uses SARIMAX to predict future values of training data per sensor"""
    def __init__(self, *args, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), use_exog=True, online=False,
                 train_file=None, ts_dir=None, num_fourier_terms=2, **kwargs):
        super(SARIMAX, self).__init__(*args, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.online = online

        train_npz = np.load(train_file)
        self.train_data = train_npz["data"]

        self.use_exog = use_exog

        if self.use_exog and ts_dir is None:
            raise ValueError("Timestamps provided for train, but not for val or test")

        if self.use_exog:
            self.num_fourier_terms = num_fourier_terms

            self.train_ts = train_npz["timestamps"]

            train_ts_file = os.path.join(ts_dir, "train.npz")
            val_ts_file = os.path.join(ts_dir, "val.npz")
            test_ts_file = os.path.join(ts_dir, "test.npz")

            train_ts = np.load(train_ts_file)
            val_ts = np.load(val_ts_file)
            test_ts = np.load(test_ts_file)

            self.train_ts_x = train_ts["timestamps_x"]
            self.train_ts_y = train_ts["timestamps_y"]
            self.val_ts_x = val_ts["timestamps_x"]
            self.val_ts_y = val_ts["timestamps_y"]
            self.test_ts_x = test_ts["timestamps_x"]
            self.test_ts_y = test_ts["timestamps_y"]
        else:
            self.train_ts_x = None
            self.train_ts_y = None
            self.val_ts_x = None
            self.val_ts_y = None
            self.test_ts_x = None
            self.test_ts_y = None

    def _train(self):
        if self.use_exog:
            self.exog_train = data_utils.convert_to_fourier_day(self.train_ts, self.num_fourier_terms)
        else:
            self.exog_train = None

        if self.online:
            pass
        else:
            self.model = SARIMAX(self.train_data, exog=self.exog_train,
                                 order=self.order, seasonal_order=self.seasonal_order)
            self.results = self.model.fit()
            self.params = self.results.params

        self.train_y_groundtruth = data_utils.get_groundtruth_from_y(self.train_y)
        self.val_y_groundtruth = data_utils.get_groundtruth_from_y(self.val_y)
        self.test_y_groundtruth = data_utils.get_groundtruth_from_y(self.test_y)

        self.train_y_pred = self.predict(self.train_x, ts_x=self.train_ts_x, ts_y=self.train_ts_y)
        self.errors["train"] = data_utils.get_standard_errors(self.train_y_groundtruth, self.train_y_pred)

        self.val_y_pred = self.predict(self.val_x, ts_x=self.val_ts_x, ts_y=self.val_ts_y)
        self.errors["val"] = data_utils.get_standard_errors(self.val_y_groundtruth, self.val_y_pred)

        self.predictions = self.predict(self.test_x, ts_x=self.test_ts_x, ts_y=self.test_ts_y)
        self.errors["test"] = data_utils.get_standard_errors(self.test_y_groundtruth, self.predictions)

    def predict(self, x, ts_x=None, ts_y=None):
        if self.online:
            predict_func = self._predict_online_armax
        else:
            predict_func = self._predict_sarimax

        return predict_func(x, ts_x=ts_x, ts_y=ts_y)

    def _predict_general_sarimax(self, x, ts_x=None, ts_y=None, params=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0)):
        predictions = np.empty((x.shape[0], *self.train_y.shape[1:]))
        horizon = self.train_y.shape[1]

        for i in range(x.shape[0]):
            if self.use_exog:
                exog_x = data_utils.convert_to_fourier_day(ts_x[i, :])
                exog_y = data_utils.convert_to_fourier_day(ts_y[i, :])
            else:
                exog_x = None
                exog_y = None

            if not params is None:
                model = SARIMAX(x[i, :], exog=exog_x, order=order, seasonal_order=seasonal_order)
                results = model.smooth(params)
            else:
                model = SARIMAX(x[i, :], exog=exog_x, order=order)
                results = model.fit()

            predictions[i, :] = results.forecast(horizon, exog=exog_y)

        return predictions

    def _predict_sarimax(self, x, ts_x=None, ts_y=None):
        return self._predict_general_sarimax(x, ts_x=ts_x, ts_y=ts_y, params=self.params,
                                             order=self.order, seasonal_order=self.seasonal_order)

    def _predict_online_armax(self, x, ts_x=None, ts_y=None):
        return self._predict_general_sarimax(x, ts_x=ts_x, ts_y=ts_y, order=self.order)
