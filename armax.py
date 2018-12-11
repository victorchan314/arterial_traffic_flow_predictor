import numpy as np
from statsmodels.tsa import arima_model

class armax:

    def __init__(self, data):
        self.data = data
