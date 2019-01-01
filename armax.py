import numpy as np
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

    def __init__(self, endog, exog=None, dates=None):
        self.endog = endog
        self.exog = exog
        self.dates = dates
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
            return self._armax_models[self.get_best_model_order]
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

    def fit(self, ar_max=5, ma_max=5):
        self.grid_search(ar_max, ma_max)
        self.fit = True

    def grid_search(self, ar_max=5, ma_max=5):
        min_order = (0, 0)
        min_llf = np.inf

        for ar in range(ar_max):
            for ma in range(ma_max):
                order = (ar, ma)
                model = arima_model.ARMA(self.get_endog(), order, self.get_exog())

                results = model.fit()
                llf = results.llf
                self._armax_models[order] = results

                if llf <= min_llf:
                    min_llf = llf
                    min_order = order

        self.best_model_order = min_order
