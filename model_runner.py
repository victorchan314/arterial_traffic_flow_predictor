import argparse
import os
import sys
import yaml

parent_dir = os.path.abspath("/Users/victorchan/Desktop/UC Berkeley/Research/Code")
sys.path.append(parent_dir)

import numpy as np

from models import models

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(data_directory):
    data = {}

    for category in ["train", "val", "test"]:
        path = "{}/{}.npz".format(data_directory, category)
        with open(path, "rb") as f:
            npz = np.load(f)
            data["{}_{}".format(category, "x")] = npz["x"]
            data["{}_{}".format(category, "y")] = npz["y"]

    return data

def load_models(model_names):
    return [getattr(models, model_name) for model_name in model_names]

def run_models(data, model_configs, model_order=None):
    train_x = data["train_x"]
    train_y = data["train_y"]
    val_x = data["val_x"]
    val_y = data["val_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    model_names = model_order or list(model_configs.keys())
    models = load_models(model_names)

    print(np.max(train_y))
    print(np.min(train_y))
    print(np.max(val_y))
    print(np.min(val_y))
    print(np.max(test_y))
    print(np.min(test_y))

    for i in range(len(models)):
        model_name = model_names[i]
        model_class = models[i]
        kwargs = model_configs[model_name]
        model = model_class(train_x, train_y, val_x, val_y, test_x, test_y, **kwargs)
        model.train()
        errors = model.get_errors()
        print(errors["train"])
        print(errors["val"])
        print(errors["test"])



def main(args):
    config = load_config(args.config)

    data_directory = config["data_directory"]
    data = load_data(data_directory)
    model_configs = config["models"]
    model_order = config.get("model_order")

    run_models(data, model_configs, model_order=model_order)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="cnofig file to specify data and model")
    args = parser.parse_args()

    main(args)
