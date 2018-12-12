import numpy as np
from statsmodels.tsa import arima_model

class armax:

    def __init__(self, endog, exog=None):
        self.endog = data
        self.exog = exog
        self._armax_models = {}
        self.best_model = None
        self.best_model_order = (-1, -1)
        self.best_model_exog = False

    def get_endog(self):
        return self.endog

    def get_exog(self):
        return self.exog

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

    def fit(self, ar_max=5, ma_max=5):
        self.grid_search(ar_max, ma_max)

    def grid_search(self, ar_max=5, ma_max=5):
        for ar in range(ar_max):
            for ma in range(ma_max):
                pass
