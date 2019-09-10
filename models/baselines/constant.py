import os
import sys

parent_dir = os.path.abspath("/Users/victorchan/Desktop/UC Berkeley/Research/Code")
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import datetime as dt

from models.model import Model
import utils



class Constant(Model):
    """Model that returns a constant value across all axes"""
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        self.train_shape = self.train_y.shape
        self.val_shape = self.val_y.shape
        self.test_shape = self.test_y.shape

    def _train(self):
        self.mean = np.mean(self.train_y)
        self.train_y_pred = np.repeat(self.mean, self.train_shape)
        self.errors["train"] = utils.get_standard_errors(self.train_y, self.train_y_pred)

        self.val_y_pred = np.repeat(self.mean, self.val_shape)
        self.errors["val"] = utils.get_standard_errors(self.val_y, self.val_y_pred)

        self.predictions = np.repeat(self.mean, self.test_shape)
        self.errors["test"] = utils.get_standard_errors(self.test_y, self.predictions)
