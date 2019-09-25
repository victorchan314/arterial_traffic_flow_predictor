import numpy as np

from models.model import Model
from lib import data_utils

class SeasonalNaive(Model):
    """Model that predicts value as the last seasonal occurrence of that value"""
    def __init__(self, *args, seasonality=1, **kwargs):
        super(SeasonalNaive, self).__init__(*args, **kwargs)
        self.seasonality = seasonality
        self.index = 0

    def _train(self):
        self.means = self._get_means(self.train_y)
        self.train_y_pred = self.predict(self.train_x)
        self.errors["train"] = data_utils.get_standard_errors(self.train_y, self.train_y_pred)

        self.val_y_pred = self.predict(self.val_x)
        self.errors["val"] = data_utils.get_standard_errors(self.val_y, self.val_y_pred)

        self.predictions = self.predict(self.test_x)
        self.errors["test"] = data_utils.get_standard_errors(self.test_y, self.predictions)

    def predict(self, x):
        num_data = x.shape[0]
        indices = self._get_indices(num_data)
        self.index = (self.index + num_data) % self.seasonality

        return self.means[indices, ...]

    def _get_means(self, train_y):
        return np.stack([np.mean(train_y[i::self.seasonality, ...], axis=0) for i in range(self.seasonality)], axis=0)

    def _get_indices(self, n):
        base = list(range(self.seasonality))

        start = base[self.index:self.index + n]
        num_reps = max(0, (n - (self.seasonality - self.index)) // self.seasonality)
        end = base[:max(0, n - num_reps * self.seasonality - len(start))]
        indices = start + base * num_reps + end

        return indices
