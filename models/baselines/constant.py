import numpy as np

from models.model import Model
from lib import data_utils

class Constant(Model):
    """Model that returns a constant value across all axes"""
    def __init__(self, *args, base_dir=None, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)

    def _train(self):
        self.mean = np.squeeze(np.mean(self.train_y[..., 1:], axis=0))

        self.train_y_groundtruth = data_utils.get_groundtruth_from_y(self.train_y)
        self.val_y_groundtruth = data_utils.get_groundtruth_from_y(self.val_y)
        self.test_y_groundtruth = data_utils.get_groundtruth_from_y(self.test_y)

        self.train_y_pred = self.predict(self.train_x)
        self.errors["train"] = data_utils.get_standard_errors(self.train_y_groundtruth, self.train_y_pred)

        self.val_y_pred = self.predict(self.val_x)
        self.errors["val"] = data_utils.get_standard_errors(self.val_y_groundtruth, self.val_y_pred)

        self.predictions = self.predict(self.test_x)
        self.errors["test"] = data_utils.get_standard_errors(self.test_y_groundtruth, self.predictions)

    def predict(self, x):
        predictions = np.tile(self.mean, (x.shape[0],) + (1,) * self.mean.ndim)
        reshaped_predictions = np.transpose(predictions, axes=(1, 0)  + tuple(range(2, predictions.ndim)))

        return reshaped_predictions

    def save_predictions(self, path):
        np.savez_compressed(path,
                            groundtruth=self.test_y_groundtruth,
                            predictions=self.predictions
                            )
