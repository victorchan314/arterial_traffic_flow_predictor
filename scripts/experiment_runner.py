import argparse
import copy
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
from scripts.generate_training_data import main as generate_training_data

DATA_CATEGORIES = ["train", "val", "test"]
PLANS = ["P1", "P2", "P3"]
OFFSETS = [3, 6, 12, 24]
Y_OFFSET = 6

SARIMAX_ORDER = ([2, 1, 0], [0, 0, 0, 0])

def create_experiment_structure(base_dir, experiment_name, config_path):
    timestamp = dt.datetime.today().strftime("%Y%m%d-%H%M%S")
    experiment_dir = "{}_{}".format(experiment_name, timestamp)
    experiment_path = os.path.join(base_dir, experiment_dir)

    utils.verify_or_create_path(experiment_path)
    utils.verify_or_create_path(os.path.join(experiment_path, "inputs", "model"))
    utils.verify_or_create_path(os.path.join(experiment_path, "inputs", "sensor_data"))
    shutil.copy(config_path, os.path.join(experiment_path, "inputs", "{}.yaml".format(experiment_name)))

    utils.verify_or_create_path(os.path.join(experiment_path, "experiments", "dcrnn"))
    utils.verify_or_create_path(os.path.join(experiment_path, "experiments", "baselines"))
    utils.verify_or_create_path(os.path.join(experiment_path, "experiments", "baselines", "constant"))
    utils.verify_or_create_path(os.path.join(experiment_path, "experiments", "baselines", "seasonal_naive"))
    utils.verify_or_create_path(os.path.join(experiment_path, "experiments", "baselines", "arimax"))
    utils.verify_or_create_path(os.path.join(experiment_path, "experiments", "baselines", "online_arimax"))
    utils.verify_or_create_path(os.path.join(experiment_path, "experiments", "baselines", "rnn"))

    return experiment_dir

def populate_inputs(experiment_path, detector_list):
    inputs_dir = os.path.join(experiment_path, "inputs")

    # Save detector list
    detector_list_path = os.path.join(inputs_dir, "model", "detector_list.txt")
    with open(detector_list_path, "w") as f:
        f.write(",".join(detector_list))

    # Generate distances/graph connections
    for plan in PLANS:
        distances_path = os.path.join(inputs_dir, "model", "distances_{}.csv".format(plan))
        args_dict = {
            "plan_name": plan,
            "detector_list": detector_list,
            "distances_path": distances_path
        }
        args = utils.Namespace(args_dict)
        generate_graph_connections(args)

        subprocess.run(["python3", "DCRNN/scripts/gen_adj_mx.py"] +
                       "--sensor_ids_filename {} --distances_filename {} --output_pkl_filename {}".format(
                           detector_list_path, distances_path,
                           os.path.join(inputs_dir, "model", "adjacency_matrix_{}.pkl".format(plan))).split())

    return detector_list_path

def create_training_data(experiment_path, detector_list_path, verbose=0):
    processes = []
    for plan in PLANS:
        for offset in OFFSETS:
            sensor_data_dir = os.path.join(experiment_path, "inputs", "sensor_data")
            args_dict = {
                "detector_list_path": detector_list_path,
                "plan_name": plan,
                "start_time_buffer": offset,
                "x_offset": offset,
                "y_offset": Y_OFFSET,
                "output_dir": sensor_data_dir,
                "timestamps_dir": sensor_data_dir,
                "verbose": verbose // 2
            }
            args = utils.Namespace(args_dict)

            # Data for ARIMAX
            args_ts_dict = args_dict.copy()
            del args_ts_dict["start_time_buffer"]
            args_ts_dict["timeseries"] = True
            args_ts = utils.Namespace(args_ts_dict)

            def f():
                generate_training_data(args)
                generate_training_data(args_ts)

            p = Process(target=f)
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

def copy_dcrnn_configs(experiment_path, detector_list):
    default_config = utils.load_yaml("config/default/dcrnn_config.yaml")

    rnn_experiment_path = os.path.join(experiment_path, "experiments", "baselines", "rnn")
    dcrnn_experiment_path = os.path.join(experiment_path, "experiments", "dcrnn")
    sensor_data_dir = os.path.join(experiment_path, "inputs", "sensor_data")
    adjacency_matrix_path = os.path.join(experiment_path, "inputs", "model", "adjacency_matrix_{}.pkl")

    for plan in PLANS:
        plan_adjacency_matrix_path = adjacency_matrix_path.format(plan)
        for offset in OFFSETS:
            config = copy.deepcopy(default_config)
            subdir = utils.get_subdir(plan, offset, Y_OFFSET, start_time_buffer=offset)
            rnn_base_dir = os.path.join(rnn_experiment_path, subdir)
            dcrnn_base_dir = os.path.join(dcrnn_experiment_path, subdir)
            utils.verify_or_create_path(rnn_base_dir)
            utils.verify_or_create_path(dcrnn_base_dir)

            config["data"]["dataset_dir"] = os.path.join(sensor_data_dir, subdir + "_sensor_data")
            config["model"]["horizon"] = Y_OFFSET
            config["model"]["num_nodes"] = len(detector_list)
            config["model"]["seq_len"] = offset

            rnn_config = copy.deepcopy(config)
            dcrnn_config = copy.deepcopy(config)

            rnn_config["base_dir"] = rnn_base_dir
            rnn_config["model"]["use_gc_for_ru"] = False

            dcrnn_config["base_dir"] = dcrnn_base_dir
            dcrnn_config["data"]["graph_pkl_filename"] = plan_adjacency_matrix_path
            dcrnn_config["model"]["filter_type"] = "dual_random_walk"
            dcrnn_config["model"]["max_diffusion_step"] = 2
            dcrnn_config["model"]["use_gc_for_ru"] = True

            utils.save_yaml(rnn_config, os.path.join(rnn_base_dir, "{}.yaml".format(subdir)))
            utils.save_yaml(dcrnn_config, os.path.join(dcrnn_base_dir, "{}.yaml".format(subdir)))

def copy_model_runner_config(experiment_path, experiment_name):
    default_config = utils.load_yaml("config/default/model_runner_config.yaml")
    config = copy.deepcopy(default_config)

    dcrnn_base_dir = os.path.join(experiment_path, "experiments", "dcrnn")
    baselines_base_dir = os.path.join(experiment_path, "experiments", "baselines")

    subdir_template = utils.get_subdir("${plan}", "${offset}", 6, start_time_buffer="${offset}")
    ts_subdir_template = utils.get_subdir("${plan}", "${offset}", 6)

    sensor_data_path = os.path.join(experiment_path, "inputs", "sensor_data")
    sensor_data_template = os.path.join(sensor_data_path, subdir_template + "_sensor_data")

    config["data_directory"] = sensor_data_template
    config["models"]["DCRNN"]["base_dir"] = os.path.join(dcrnn_base_dir, subdir_template)
    config["models"]["RNN"]["base_dir"] = os.path.join(baselines_base_dir, "rnn", subdir_template)
    config["models"]["SeasonalNaive"]["seasonality"] = None

    sarimax_config = config["models"]["SARIMAX"]["SARIMAX"]
    online_sarimax_config = config["models"]["SARIMAX"]["OnlineSARIMAX"]

    sarimax_config["order"] = SARIMAX_ORDER[0][:]
    sarimax_config["seasonal_order"] = SARIMAX_ORDER[1][:]
    online_sarimax_config["order"] = SARIMAX_ORDER[0][:]
    online_sarimax_config["seasonal_order"] = SARIMAX_ORDER[1][:]

    sarimax_config["base_dir"] = os.path.join(baselines_base_dir, "arimax", subdir_template)
    sarimax_config["train_file"] = os.path.join(sensor_data_path, ts_subdir_template, "train_ts.npz")
    sarimax_config["ts_dir"] = sensor_data_template
    online_sarimax_config["base_dir"] = os.path.join(baselines_base_dir, "online_arimax", subdir_template)
    online_sarimax_config["train_file"] = os.path.join(sensor_data_path, ts_subdir_template, "train_ts.npz")
    online_sarimax_config["ts_dir"] = sensor_data_template

    utils.save_yaml(config, os.path.join(experiment_path, "inputs",
                                         "model_runner_config_{}.yaml".format(experiment_name)))

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
    experiment_path = os.path.join(base_dir, experiment_dir)

    detector_list_path = populate_inputs(experiment_path, detector_list)
    create_training_data(experiment_path, detector_list_path, verbose=verbose)
    copy_dcrnn_configs(experiment_path, detector_list)
    copy_model_runner_config(experiment_path, experiment_name)
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
