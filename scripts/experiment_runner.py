import argparse
import datetime as dt
from multiprocessing import Process
import os
import shutil
import subprocess
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

import numpy as np

from models import models
from lib import utils
from scripts.generate_graph_connections import main as generate_graph_connections

DATA_CATEGORIES = ["train", "val", "test"]
PLANS = ["P1", "P2", "P3"]

def create_experiment_structure(base_dir, experiment_name, config_path):
    timestamp = dt.datetime.today().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(base_dir, "{}_{}".format(experiment_name, timestamp))

    utils.verify_or_create_path(experiment_dir)
    utils.verify_or_create_path(os.path.join(experiment_dir, "inputs", "model"))
    utils.verify_or_create_path(os.path.join(experiment_dir, "inputs", "sensor_data"))
    utils.verify_or_create_path(os.path.join(experiment_dir, "experiments", "dcrnn"))

    shutil.copy(config_path, os.path.join(experiment_dir, "inputs", "{}.yaml".format(experiment_name)))

    utils.verify_or_create_path(os.path.join(experiment_dir, "experiments", "baselines"))
    utils.verify_or_create_path(os.path.join(experiment_dir, "experiments", "baselines", "constant"))
    utils.verify_or_create_path(os.path.join(experiment_dir, "experiments", "baselines", "seasonal_naive"))
    utils.verify_or_create_path(os.path.join(experiment_dir, "experiments", "baselines", "arimax"))
    utils.verify_or_create_path(os.path.join(experiment_dir, "experiments", "baselines", "online_arimax"))
    utils.verify_or_create_path(os.path.join(experiment_dir, "experiments", "baselines", "rnn"))

    return experiment_dir

def populate_inputs(experiment_dir, detector_list):
    inputs_dir = os.path.join(experiment_dir, "inputs")

    # Save detector list
    detector_list_path = os.path.join(inputs_dir, "model", "detector_list.txt")
    with open(detector_list_path, "w") as f:
        f.write(",".join(detector_list))

    # Generate distances/graph connections
    for plan in PLANS:
        distances_path = os.path.join(inputs_dir, "model", "distances_{}.csv".format(plan))
        generate_graph_connections_args_dict = {
            "plan_name": plan,
            "detector_list": detector_list,
            "distances_path": distances_path
        }
        generate_graph_connections_args = argparse.Namespace(**generate_graph_connections_args_dict)
        generate_graph_connections(generate_graph_connections_args)

        subprocess.run(["python3", "DCRNN/scripts/gen_adj_mx.py"] +
                       "--sensor_ids_filename {} --distances_filename {} --output_pkl_filename {}".format(
                           detector_list_path, distances_path,
                           os.path.join(inputs_dir, "model", "adjacency_matrix_{}.pkl".format(plan))).split())


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

            if verbose > 2:
                print("{}_{} shape: {}".format(category, "x", npz["x"].shape))
                print("{}_{} shape: {}".format(category, "y", npz["y"].shape))

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

    model_names = list(model_configs.keys()) if model_order is None else model_order
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

        if isinstance(model_configs[model_name], list):
            model_kwargs = model_configs[model_name]
        else:
            model_kwargs = [model_configs[model_name]]

        for kwargs in model_kwargs:
            model = model_class(train_x, train_y, val_x, val_y, test_x, test_y,
                                verbose=verbose, **kwargs)
            if verbose:
                print("Created {} model".format(model_name))

            model.train()

            if verbose:
                print("Trained {} model".format(model_name))

            errors = model.get_errors()

            if verbose > 1:
                for category in DATA_CATEGORIES:
                    if category in errors:
                        for key, value in errors[category].items():
                            print("{} {}: {}".format(category, key, value))

            base_dir = kwargs.get("base_dir", None)
            if not base_dir is None:
                utils.verify_or_create_path(base_dir)
                path = os.path.join(base_dir, "predictions.npz")
                model.save_predictions(path)

            model.close()



def run_config(config, verbose=0):
    data_directory = config["data_directory"]
    data = load_data(data_directory, verbose=verbose)
    model_configs = config["models"]
    model_order = config.get("model_order")

    run_models(data, model_configs, model_order=model_order, verbose=verbose)

def main(args):
    verbose = args.verbose
    config_path = args.config

    config = utils.load_yaml(config_path)

    base_dir = config["base_dir"]
    experiment_name = config["experiment_name"]
    detector_list = list(map(str, config["detector_list"]))

    experiment_dir = create_experiment_structure(base_dir, experiment_name, config_path)
    populate_inputs(experiment_dir, detector_list)
    print(1/0)

    for plan in ["P1", "P2", "P3"]:
        for offset in [3, 6, 12, 24]:
            data_directory = "data/inputs/5083/5083_{}_o{}_h6_sb{}_sensor_data".format(plan, offset, offset)
            data = load_data(data_directory, verbose=verbose)
            model_configs = config["models"]
            model_order = config.get("model_order")

            model_configs["DCRNN"]["base_dir"] = "data/test_diag/5083_{}_o{}_h6_sb{}"\
                .format(plan, offset, offset)

            #run_models(data, model_configs, model_order=model_order, verbose=verbose)
            p = Process(target=run_models,
                        args=(data, model_configs),
                        kwargs={"model_order": model_order, "verbose": verbose})
            p.start()

    run_config(config, verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file to specify detector list and parameters")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")
    args = parser.parse_args()

    main(args)
