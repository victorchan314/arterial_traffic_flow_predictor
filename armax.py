import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa import arima_model

class armax:
    """
    A wrapper class for statsmodels.tsa.arima_model.ARMA.

    Attributes:
        endog (np.array): Endogenous variable for ARMA model
        exog (np.array): Optional exogenous variable for ARMA model
        dates (np.array): Optional dates corresponding to endog and/or exog
        preprocessed (bool): Whether the data has been preprocessed or not
        preprocessing_method (str): Which preprocessing method has been used
        preprocessed_endog (np.array): Preprocessed endog
        preprocessed_exog (np.array): Preprocessed exog
        w (int): Order of preprocessing
        _armax_models (dict): Dictionary of all the armax models fit
        self.has_fit (bool): Whether a model has been fit or not
        best_model (arima_model.ARMAResults): Results of the best model fit
        best_model_order (tuple): Model order with the lowest llf
        best_model_exog (bool): 
    """

    PREPROCESSING_SMOOTHING = "smoothing"
    PREPROCESSING_AGGREGATING = "aggregating"
    PREPROCESSING_METHODS = [PREPROCESSING_SMOOTHING, PREPROCESSING_AGGREGATING]

    def __init__(self, endog, exog=None, dates=None, frequency=None):
        self.endog = endog
        self.exog = exog
        self.dates = dates
        self.frequency = frequency
        self.preprocessed = False
        self.preprocessing_method = None
        self.preprocessed_endog = None
        self.preprocessed_exog = None
        self.w = 1
        self._armax_models = {}
        self.has_fit = False
        self.best_model = None
        self.best_model_order = (-1, -1)
        self.best_model_exog = False

        if dates is None:
            try:
                self.dates = endog.index.to_pydatetime()
            except Exception:
                warnings.warn("Date not passed, but endog is not pandas dataframe. This can cause trouble in the future.", UserWarning, stacklevel=2)

    def get_endog(self):
        if self.preprocessed:
            return self.preprocessed_endog
        else:
            return self.endog

    def get_exog(self):
        if self.preprocessed:
            return self.preprocessed_endog
        else:
            return self.exog
    
    def get_dates(self):
        return self.dates
    
    def get_frequency(self):
        return self.frequency
    
    def get_preprocessing_method(self):
        return self.preprocessing_method

    def get_preprocessed_endog(self):
        return self.preprocessed_endog

    def get_preprocessed_exog(self):
        return self.preprocessed_exog
    
    def get_preprocessed_dates(self):
        if self.preprocessing_method == self.PREPROCESSING_SMOOTHING:
            return self.dates[self.w-1:]
        elif self.preprocessing_method == self.PREPROCESSING_AGGREGATING:
            return self.dates[self.w-1::self.w]
        else:
            return None
    
    def get_preprocessing_order(self):
        return self.w

    def get_best_model(self):
        if self.has_fit:
            return self._armax_models[self.get_best_model_order()]
        else:
            return None

    def get_best_model_order(self):
        return self.best_model_order

    def get_best_model_ar_order(self):
        return self.best_model_order[0]

    def get_best_model_ma_order(self):
        return self.best_model_order[1]

    def get_best_model_exog(self):
        return self.best_model_exog

    def preprocess(self, method, w):
        if method == self.PREPROCESSING_SMOOTHING:
            self.w = w
            self.preprocessed_endog = self._smooth(self.endog, w)
            self.preprocessed_exog = self._smooth(self.exog, w) if self.exog else None
            self.preprocessed = True
        elif method == self.PREPROCESSING_AGGREGATING:
            self.w = w
            self.preprocessed_endog = self._aggregate(self.endog, w)
            self.preprocessed_exog = self._aggregate(self.exog, w) if self.exog else None
            self.preprocessed = True
        else:
            raise ValueError("Illegal method passed: must be one of '{}'".format(self.PREPROCESSING_METHODS))
    
    def _smooth(self, data):
        smoothed_data = self.data[self.w-1:]

        for i in range(1, self.w):
            smoothed_data += self.data[self.w-1-i:-i]

        smoothed_data /= self.w

        return smoothed_data
    
    def _aggregate(self, data):
        aggregated_data = self.data[self.w-1::self.w]

        for i in range(1, self.w):
            aggregated_data += self.data[self.w-1-i::self.w]

        return aggregated_data

    def fit(self, ar_max=3, ma_max=3, method="css-mle", cross_validate=False, folds="monthly", verbose=False):
        self.grid_search(ar_max, ma_max, method=method, cross_validate=cross_validate, folds=folds, verbose=verbose)
        self.has_fit = True
        print("Done fitting ARMA model; best order: {}".format(self.get_best_model_order()))

    def grid_search(self, ar_max=3, ma_max=3, method="css-mle", cross_validate=False, folds="monthly", verbose=False):
        min_order = (0, 0)
        min_sse = np.inf

        for ar in range(1, ar_max + 1):
            for ma in range(1, ma_max + 1):
                order = (ar, ma)
                results = self.fit_to_order(order, method=method, cross_validate=cross_validate, folds=folds, verbose=verbose)
                sse = results.sse

                aic = results.aic
                self._armax_models[order] = results

                if verbose:
                    print("Order {} sse: {}".format(order, sse))

                if sse < min_sse:
                    min_sse = sse
                    min_order = order

        self.best_model_order = min_order

    def fit_to_order(self, order, method="css-mle", cross_validate=False, folds="monthly", verbose=False):
        if verbose:
            print("Fitting order {}".format(order))

        if not cross_validate:
            model = arima_model.ARMA(self.get_endog(), order, self.get_exog(), dates=self.get_dates())
            results = model.fit(method=method)
            results.sse = np.sum(np.power(results.resid, 2))
        else:
            if self.get_dates() is None:
                raise ValueError("Cannot cross validate time series without dates")
            results, sse = self._cross_validate_model(order, folds=folds, method=method, verbose=verbose)
            results.sse = sse

        return results

    def _cross_validate_model(self, order, folds="monthly", method="css-mle", verbose=False):
        if folds == "monthly":
            results, sse = self._cross_validate_model_monthly(order, method=method, verbose=verbose)
        else:
            raise ValueError("Invalid cross validation folding method argument: {}".format(folds))

        if verbose:
            print("Finished cross validation with order {}; average sse {}".format(order, sse))

        return results, sse

    def _cross_validate_model_monthly(self, order, method="css-mle", verbose=False):
        endog = self.get_endog()
        exog = self.get_exog()
        dates = self.get_dates()
        frequency = self.get_frequency()
        folds = self._split_by_date(dates, period="month")
        total_sse = 0

        if verbose:
            print("Cross validating monthly results in {} folds".format(len(folds) - 1))

        for i in range(len(folds) - 1):
            training_fold = folds[i] + 1
            validation_fold = folds[i+1]

            training_endog = endog[0:training_fold]
            training_dates = dates[0:training_fold]
            validation_endog = endog[training_fold:validation_fold+1]
            validation_dates = dates[training_fold:validation_fold+1]

            model = arima_model.ARMA(training_endog, order, exog=exog, dates=training_dates, freq=frequency)
            result = model.fit(method=method)

            start_date = dates[training_fold]
            end_date = dates[validation_fold]

            predictions = result.predict(start=start_date, end=end_date, exog=exog, dynamic=True)
            prediction_dates = start_date + np.arange(len(predictions))*frequency
            filtered_predictions = predictions[np.isin(prediction_dates, validation_dates)]

            prediction_residuals = validation_endog.subtract(filtered_predictions, axis=0)
            sse = prediction_residuals.applymap(lambda x: x**2).sum()[0] / prediction_residuals.shape[0]

            if verbose:
                print("Trained month {}; sse {}".format(i, sse))

            total_sse += sse

        average_sse = total_sse / len(folds)

        model = arima_model.ARMA(endog, order, exog=exog, dates=dates, freq=frequency)
        results = model.fit(method=method)

        return results, average_sse

    def _split_by_date(self, dates, period="month"):
        """Returns a list of indices of all the datetime objects in dates
        that are the last datetime for that period."""

        if period == "month":
            folds = [i for i in range(len(dates)) if i == len(dates) - 1 or dates[i].month != dates[i+1].month]

            return folds
        else:
            raise ValueError("Internal period argument error with _split_by_date")
