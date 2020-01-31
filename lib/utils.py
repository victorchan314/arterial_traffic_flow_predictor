import builtins
import functools
import itertools
import os
import shutil
import sys
import yaml

import numpy as np


WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def dictionary_product(d):
    split_dict = [[{k: v1} for v1 in v0] for k, v0 in d.items()]
    split_dict_product = list(itertools.product(*split_dict))
    dict_product = list(map(lambda d0: functools.reduce(lambda d1, d2: {**d1, **d2}, d0), split_dict_product))

    return dict_product

def get_subdir(plan_name, x_offset, y_offset, intersection=None, start_time_buffer=0, end_time_buffer=0, weekday=None):
    components = [plan_name, "o{}".format(x_offset), "h{}".format(y_offset)]

    if not weekday is None:
        components.insert(1, WEEKDAYS[weekday])
    if not intersection is None:
        components.insert(0, intersection)
    if start_time_buffer != 0:
        components.append("sb{}".format(start_time_buffer))
    if end_time_buffer != 0:
        components.append("eb{}".format(end_time_buffer))

    subdir = "_".join(map(str, components))

    return subdir

def get_type_by_name(type_name):
    return getattr(builtins, type_name)

def load_predictions(path):
    predictions_file = np.load(path)
    groundtruth = predictions_file["groundtruth"]
    predictions = predictions_file["predictions"]

    return groundtruth, predictions

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def ordered_tuple_product(dict_of_tuples, key_order):
    list_of_tuples = [dict_of_tuples[key] for key in key_order]
    tuples_product = list(itertools.product(*list_of_tuples))

    return tuples_product

def proxy_print(old_string, new_string):
    old_string += new_string
    old_string += "\n"

    return old_string

def run_process_list_parallel(processes):
    for p in processes:
        p.start()

    for p in processes:
        p.join()

def run_process_list_sequential(processes):
    for p in processes:
        p.start()
        p.join()

def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

def verify_or_create_path(path):
    os.makedirs(path, exist_ok=True)


class Namespace(object):
    def __init__(self, _dict):
        self.__dict__.update(_dict)

    def __getattr__(self, item):
        return False

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
