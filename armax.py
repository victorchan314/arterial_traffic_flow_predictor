import numpy as np
from statsmodels.tsa import arima_model

class armax:

    PREPROCESSING_SMOOTHING = "smoothing"
    PREPROCESSING_AGGREGATING = "aggregating"
    PREPROCESSING_METHODS = [PREPROCESSING_SMOOTHING, PREPROCESSING_AGGREGATING]

    def __init__(self, endog, exog=None, dates=None):
        self.endog = endog
        self.exog = exog
        self.dates = dates
        self.preprocessing_method = None
        self.preprocessed_endog = None
        self.preprocessed_exog = None
        self.w = 1
        self._armax_models = {}
        self.best_model = None
        self.best_model_order = (-1, -1)
        self.best_model_exog = False

    def get_endog(self):
        return self.endog

    def get_exog(self):
        return self.exog
    
    def get_dates(self):
        return self.dates
    
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
        return self.best_model

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
            self.smoothed_endog = self._smooth(self.endog, w)
            self.smoothed_exog = self._smooth(self.exog, w) if self.exog else None
        elif method == self.PREPROCESSING_AGGREGATING:
            self.w = w
            self.smoothed_endog = self._aggregate(self.endog, w)
            self.smoothed_exog = self._aggregate(self.exog, w) if self.exog else None
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

    def fit(self, ar_max=5, ma_max=5):
        self.grid_search(ar_max, ma_max)

    def grid_search(self, ar_max=5, ma_max=5):
        for ar in range(ar_max):
            for ma in range(ma_max):
                order = (ar, ma)
                self._armax_models[order] = arima_model.ARMA(self.endog, order, self.exog)
