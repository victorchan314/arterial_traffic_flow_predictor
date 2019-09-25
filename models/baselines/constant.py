import numpy as np

from models.model import Model
from lib import data_utils

class Constant(Model):
    """Model that returns a constant value across all axes"""
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)

    def _train(self):
        self.mean = np.mean(self.train_y)
        self.train_y_pred = self.predict(self.train_x)
        self.errors["train"] = data_utils.get_standard_errors(self.train_y, self.train_y_pred)

        self.val_y_pred = self.predict(self.val_x)
        self.errors["val"] = data_utils.get_standard_errors(self.val_y, self.val_y_pred)

        self.predictions = self.predict(self.test_x)
        self.errors["test"] = data_utils.get_standard_errors(self.test_y, self.predictions)

    def predict(self, x):
        return np.tile(self.mean, x.shape[:-1] + (x.shape[-1] - 1,))
