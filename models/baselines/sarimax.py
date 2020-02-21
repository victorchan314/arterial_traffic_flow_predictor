import os
import pickle
import warnings

import numpy as np

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX as sarimax

from models.model import Model
from lib import data_utils
from lib import utils

class SARIMAX(Model):
    """Model that uses SARIMAX to predict future values of training data per sensor"""
    def __init__(self, *args, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), use_exog=True, online=False,
                 is_trained=False, base_dir=None, train_file=None, ts_dir=None, num_fourier_terms=1,
                 calculate_train_error=False, calculate_val_error=False, verbose=0, **kwargs):
        super(SARIMAX, self).__init__(*args, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.online = online
        self._is_trained = is_trained
        self.calculate_train_error = calculate_train_error
        self.calculate_val_error = calculate_val_error
        self.verbose = verbose

        train_npz = np.load(train_file)
        self.train_data = train_npz["data"]
        self.num_detectors = self.train_data.shape[1]
        self.num_sensors = self.train_data.shape[2] - 1

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

        # Logging and model saving
        if base_dir is None:
            self.params_pkl_file = None
        else:
            self.params_pkl_file = os.path.join(base_dir, "params.pkl")

    def train(self):
        self._train()

    def _train(self):
        if self.verbose:
            print("Beginning ARMAX training")

        if self.online:
            self._is_trained = True
        elif self.is_trained:
            self._load_params()
        else:
            self._is_trained = True
            if self.use_exog:
                self.exog_train = data_utils.convert_to_fourier_day(self.train_ts, self.num_fourier_terms)
                if self.verbose:
                    print("Train exog created")
            else:
                self.exog_train = None

            self.params = []

            for d in range(self.num_detectors):
                params =[]

                for s in range(1, self.num_sensors + 1):
                    model = sarimax(self.train_data[:, d, s], exog=self.exog_train,
                                    order=self.order, seasonal_order=self.seasonal_order)
                    model_results = model.fit(disp=2*self.verbose)
                    model_params = model_results.params

                    # Done to conserve memory, since we have 2GB * num_detectors * num_sensors worth of models
                    del model
                    del model_results

                    params.append(model_params)

                    if self.verbose > 2:
                        print("Sensor {} model created and trained".format(s))

                self.params.append(params)

                if self.verbose:
                    print("Detector {} models created and trained".format(d))

            if not self.params_pkl_file is None:
                self._log(self.params_pkl_file, self.params)
                if self.verbose:
                    print("Params saved to {}".format(self.params_pkl_file))

        if self.calculate_train_error:
            self.train_y_groundtruth = data_utils.get_groundtruth_from_y(self.train_y)
            self.train_y_pred = self.predict(self.train_x, ts_x=self.train_ts_x, ts_y=self.train_ts_y)
            self.errors["train"] = data_utils.get_standard_errors(self.train_y_groundtruth, self.train_y_pred)

        if self.calculate_val_error:
            self.val_y_groundtruth = data_utils.get_groundtruth_from_y(self.val_y)
            self.val_y_pred = self.predict(self.val_x, ts_x=self.val_ts_x, ts_y=self.val_ts_y)
            self.errors["val"] = data_utils.get_standard_errors(self.val_y_groundtruth, self.val_y_pred)

        self.test_y_groundtruth = data_utils.get_groundtruth_from_y(self.test_y)
        self.predictions = self.predict(self.test_x, ts_x=self.test_ts_x, ts_y=self.test_ts_y)
        self.errors["test"] = data_utils.get_standard_errors(self.test_y_groundtruth, self.predictions)

    def predict(self, x, ts_x=None, ts_y=None):
        predictions = []

        for d in range(self.num_detectors):
            if self.verbose:
                print("Working on detector {} predictions".format(d))
            detector_predictions = []

            for s in range(1, self.num_sensors + 1):
                if self.verbose > 2:
                    print("Working on sensor {} predictions".format(s + 1))

                if self.online:
                    pred = self._predict_online_armax(x[:, :, d, s], ts_x=ts_x, ts_y=ts_y)
                else:
                    pred = self._predict_sarimax(x[:, :, d, s], self.params[d][s-1], ts_x=ts_x, ts_y=ts_y)

                detector_predictions.append(pred)

            if len(detector_predictions) == 1:
                detector_predictions = detector_predictions[0]
            else:
                detector_predictions = np.stack(detector_predictions, axis=-1)

            predictions.append(detector_predictions)

        reshaped_predictions = np.transpose(np.stack(predictions, axis=2), axes=(1, 0) + tuple(range(2, predictions[0].ndim + 1)))
        if self.verbose:
            print("Predictions finished; shape: {}".format(reshaped_predictions.shape))

        return reshaped_predictions

    def _predict_general_sarimax(self, x, ts_x=None, ts_y=None, params=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0)):
        predictions = np.empty((x.shape[0], self.train_y.shape[1]))
        horizon = self.train_y.shape[1]

        for i in range(x.shape[0]):
            if self.use_exog:
                exog_x = data_utils.convert_to_fourier_day(ts_x[i, :])
                exog_y = data_utils.convert_to_fourier_day(ts_y[i, :])
            else:
                exog_x = None
                exog_y = None

            if not params is None:
                model = sarimax(x[i, :], exog=exog_x, order=order, seasonal_order=seasonal_order)
                results = model.smooth(params)
            else:
                try:
                    model = sarimax(x[i, :], exog=exog_x, order=order, seasonal_order=seasonal_order)
                    results = model.fit(disp=self.verbose // 4)
                except ConvergenceWarning:
                    if self.verbose > 1:
                        print("Model for data at index {} did not converge".format(i))

                    warnings.filterwarnings("ignore")
                    model = sarimax(x[i, :], exog=exog_x, order=order, seasonal_order=seasonal_order)
                    results = model.fit(disp=self.verbose // 4, maxiter=200)
                    warnings.filterwarnings("error")
                except np.linalg.linalg.LinAlgError:
                    if self.verbose > 1:
                        print("Model for data at index {} produced non-positive-definite covariance matrix, simple_differencing set to True".format(i))

                    model = sarimax(x[i, :], exog=exog_x, order=order, seasonal_order=seasonal_order,
                                    simple_differencing=True)
                    results = model.fit(disp=self.verbose // 4)
                except Warning:
                    warnings.filterwarnings("ignore")
                    try:
                        model = sarimax(x[i, :], exog=exog_x, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
                        results = model.fit(disp=self.verbose // 4)
                    except np.linalg.linalg.LinAlgError:
                        model = sarimax(x[i, :] + np.random.randn(*x[i, :].shape), exog=exog_x, order=order, seasonal_order=seasonal_order)

                    warnings.filterwarnings("error")

            predictions[i, :] = results.forecast(horizon, exog=exog_y)

            # Memory conservation
            del model
            del results

        return predictions

    def _predict_sarimax(self, x, params, ts_x=None, ts_y=None):
        return self._predict_general_sarimax(x, ts_x=ts_x, ts_y=ts_y, params=params,
                                             order=self.order, seasonal_order=self.seasonal_order)

    def _predict_online_armax(self, x, ts_x=None, ts_y=None):
        warnings.filterwarnings("error")
        predictions = self._predict_general_sarimax(x, ts_x=ts_x, ts_y=ts_y, order=self.order)
        warnings.filterwarnings("default")

        return predictions

    def _log(self, filename, data):
        dir = os.path.dirname(filename)
        utils.verify_or_create_path(dir)

        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.DEFAULT_PROTOCOL)

    def _load_params(self):
        with open(self.params_pkl_file, "rb") as f:
            self.params = pickle.load(f)

    def save_predictions(self, path):
        np.savez_compressed(path,
                            groundtruth=self.test_y_groundtruth,
                            predictions=self.predictions
                            )
