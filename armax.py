import numpy as np
from statsmodels.tsa import arima_model

class armax:

    def __init__(self, endog, exog=None, dates=None):
        self.endog = endog
        self.exog = exog
        self.dates = dates
        self.smoothed_endog = None
        self.smoothed_exog = None
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

    def get_smoothed_endog(self):
        return self.smoothed_endog

    def get_smoothed_exog(self):
        return self.smoothed_exog
    
    def get_smoothed_dates(self):
        return self.dates[self.w-1:]
    
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
        if method == "smoothing":
            self.w = w
            self.smoothed_endog = self._smooth(self.endog, w)
            self.smoothed_exog = self._smooth(self.exog, w)
        elif method == "aggregation":
            self.w = w
            self.smoothed_endog = self._aggregate(self.endog, w)
            self.smoothed_exog = self._aggregate(self.exog, w)
        else:
            raise ValueError("Illegal method passed: must be 'smoothing' or 'aggregation'")
    
    def smooth(self, data, w):
        return smoothed_data
    
    def aggregate(self, data, w):
        pass

    def fit(self, ar_max=5, ma_max=5):
        self.grid_search(ar_max, ma_max)

    def grid_search(self, ar_max=5, ma_max=5):
        for ar in range(ar_max):
            for ma in range(ma_max):
                order = (ar, ma)
                self._armax_models[order] = arima_model.ARMA(self.endog, order, self.exog)
