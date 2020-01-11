import os
import sys
import yaml

import numpy as np



WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def get_subdir(intersection, plan_name, x_offset, y_offset, start_time_buffer=0, end_time_buffer=0, weekday=None):
    subdir = "{}_{}".format(intersection, plan_name)

    if weekday is not None:
        subdir += "_{}".format(WEEKDAYS[weekday])

    subdir += "_o{}_h{}".format(x_offset, y_offset)

    if start_time_buffer != 0:
        subdir += "_sb{}".format(start_time_buffer)
    if end_time_buffer != 0:
        subdir += "_eb{}".format(end_time_buffer)

    subdir += "_sensor_data"

    return subdir

def load_predictions(path):
    predictions_file = np.load(path)
    groundtruth = predictions_file["groundtruth"]
    predictions = predictions_file["predictions"]

    return groundtruth, predictions

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def verify_or_create_path(path):
    os.makedirs(path, exist_ok=True)

class Tee(object):
    def __init__(self, path=None):
        self.file = open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        self.close()

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.flush()

    def flush(self):
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()
