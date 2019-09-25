import argparse
import os
import sys
import yaml

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

import numpy as np

from models import models

DATA_CATEGORIES = ["train", "val", "test"]

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(data_directory, verbose=0):
    data = {}
    if verbose:
        print("Loading data from directory {}".format(data_directory))

    for category in DATA_CATEGORIES:
        path = "{}/{}.npz".format(data_directory, category)
        if verbose > 1:
            print("Loading datafile at path {}".format(path))

        with open(path, "rb") as f:
            npz = np.load(f)
            data["{}_{}".format(category, "x")] = npz["x"]
            data["{}_{}".format(category, "y")] = npz["y"]

    return data

def load_models(model_names):
    return [getattr(models, model_name) for model_name in model_names]

def run_models(data, model_configs, model_order=None, verbose=0):
    train_x = data["train_x"]
    train_y = data["train_y"]
    val_x = data["val_x"]
    val_y = data["val_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    model_names = model_order or list(model_configs.keys())
    models = load_models(model_names)

    if verbose:
        print("Starting to run models...")

    if verbose > 2:
        print("Max train_y: {}".format(np.max(train_y)))
        print("Min train_y: {}".format(np.min(train_y)))
        print("Max val_y: {}".format(np.max(val_y)))
        print("Min val_y: {}".format(np.min(val_y)))
        print("Max test_y: {}".format(np.max(test_y)))
        print("Min test_y: {}".format(np.min(test_y)))

    for i in range(len(models)):
        model_name = model_names[i]
        model_class = models[i]
        kwargs = model_configs[model_name]
        model = model_class(train_x, train_y, val_x, val_y, test_x, test_y, **kwargs)
        if verbose:
            print("Created {} model".format(model_name))

        model.train()

        if verbose:
            print("Trained {} model".format(model_name))

        errors = model.get_errors()

        if verbose > 1:
            for category in DATA_CATEGORIES:
                for key, value in errors[category].items():
                    print("{} {}: {}".format(category, key, value))



def main(args):
    verbose = args.verbose
    config = load_config(args.config)

    data_directory = config["data_directory"]
    data = load_data(data_directory, verbose=verbose)
    model_configs = config["models"]
    model_order = config.get("model_order")

    run_models(data, model_configs, model_order=model_order, verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file to specify data and model")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")
    args = parser.parse_args()

    main(args)
