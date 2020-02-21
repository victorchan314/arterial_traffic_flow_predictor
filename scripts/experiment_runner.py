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

from lib import utils
from scripts.generate_graph_connections import main as generate_graph_connections
from scripts.generate_training_data import main as generate_training_data
from scripts.model_runner import main as model_runner


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

def create_training_data(experiment_path, detector_list_path, override=None, verbose=0):
    processes = []
    for plan in PLANS:
        for offset in OFFSETS:
            sensor_data_dir = os.path.join(experiment_path, "inputs", "sensor_data")
            args_dict = {
                "detector_list_path": detector_list_path,
                "plan_name": plan,
                "features": ["Flow"],
                "start_time_buffer": offset,
                "x_offset": offset,
                "y_offset": Y_OFFSET,
                "output_dir": sensor_data_dir,
                "timestamps_dir": sensor_data_dir,
                "verbose": verbose // 2
            }

            if override:
                args_dict.update(override)

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

def copy_dcrnn_configs(experiment_path, detector_list, rnn_config_override=None, dcrnn_config_override=None):
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
            del rnn_config["data"]["graph_pkl_filename"]
            del rnn_config["model"]["filter_type"]
            del rnn_config["model"]["max_diffusion_step"]
            rnn_config["model"]["use_gc_for_ru"] = False

            dcrnn_config["base_dir"] = dcrnn_base_dir
            dcrnn_config["data"]["graph_pkl_filename"] = plan_adjacency_matrix_path
            dcrnn_config["model"]["filter_type"] = "dual_random_walk"
            dcrnn_config["model"]["max_diffusion_step"] = 2
            dcrnn_config["model"]["use_gc_for_ru"] = True

            if rnn_config_override:
                rnn_config = utils.update_config_with_dict_override(rnn_config, rnn_config_override)
            if dcrnn_config_override:
                dcrnn_config = utils.update_config_with_dict_override(dcrnn_config, dcrnn_config_override)

            utils.save_yaml(rnn_config, os.path.join(rnn_base_dir, "{}.yaml".format(subdir)))
            utils.save_yaml(dcrnn_config, os.path.join(dcrnn_base_dir, "{}.yaml".format(subdir)))

def copy_model_runner_config(experiment_path, experiment_name):
    default_config = utils.load_yaml("config/default/model_runner_config.yaml")
    config = copy.deepcopy(default_config)

    dcrnn_base_dir = os.path.join(experiment_path, "experiments", "dcrnn")
    baselines_base_dir = os.path.join(experiment_path, "experiments", "baselines")

    subdir_template = utils.get_subdir("${plan}", "${offset}", 6, start_time_buffer="${offset}")
    sensor_data_path = os.path.join(experiment_path, "inputs", "sensor_data")
    sensor_data_template = os.path.join(sensor_data_path, subdir_template + "_sensor_data")
    ts_sensor_data_subdir = utils.get_subdir("${plan}", "${offset}", 6) + "_sensor_data"

    config["data_directory"] = sensor_data_template
    config["models"]["Constant"]["base_dir"] = os.path.join(baselines_base_dir, "constant", subdir_template)
    config["models"]["SeasonalNaive"]["base_dir"] = os.path.join(baselines_base_dir, "seasonal_naive", subdir_template)
    config["models"]["DCRNN"]["DCRNN"]["base_dir"] = os.path.join(dcrnn_base_dir, subdir_template)
    config["models"]["DCRNN"]["RNN"]["base_dir"] = os.path.join(baselines_base_dir, "rnn", subdir_template)

    sarimax_config = config["models"]["SARIMAX"]["SARIMAX"]
    online_sarimax_config = config["models"]["SARIMAX"]["OnlineSARIMAX"]

    sarimax_config["order"] = SARIMAX_ORDER[0][:]
    sarimax_config["seasonal_order"] = SARIMAX_ORDER[1][:]
    online_sarimax_config["order"] = SARIMAX_ORDER[0][:]
    online_sarimax_config["seasonal_order"] = SARIMAX_ORDER[1][:]

    sarimax_config["base_dir"] = os.path.join(baselines_base_dir, "arimax", subdir_template)
    sarimax_config["train_file"] = os.path.join(sensor_data_path, ts_sensor_data_subdir, "train_ts.npz")
    sarimax_config["ts_dir"] = sensor_data_template
    online_sarimax_config["base_dir"] = os.path.join(baselines_base_dir, "online_arimax", subdir_template)
    online_sarimax_config["train_file"] = os.path.join(sensor_data_path, ts_sensor_data_subdir, "train_ts.npz")
    online_sarimax_config["ts_dir"] = sensor_data_template

    model_runner_config_path = os.path.join(experiment_path, "inputs",
                                            "model_runner_config_{}.yaml".format(experiment_name))
    utils.save_yaml(config, model_runner_config_path)

    return model_runner_config_path

def run_models(model_runner_config_path, debug=False, verbose=0):
    args_dict = {
        "config": model_runner_config_path,
        "debug": debug,
        "verbose": verbose
    }
    args = utils.Namespace(args_dict)

    model_runner(args)



def main(args):
    verbose = args.verbose
    config_path = args.config
    debug = args.debug

    if not os.path.isdir(config_path):
        config = utils.load_yaml(config_path)

        base_dir = config["base_dir"]
        experiment_name = config["experiment_name"]
        detector_list = list(map(str, config["detector_list"]))

        override = config.get("override", {})
        generate_training_data_override = override.get("generate_training_data", {})
        rnn_config_override = override.get("rnn_config", {})
        dcrnn_config_override = override.get("dcrnn_config", {})

        if "features" in config:
            features = config["features"]
            generate_training_data_override["features"] = features
            rnn_config_override["model/input_dim"] = len(features) + 1
            rnn_config_override["model/output_dim"] = len(features) + 1
            dcrnn_config_override["model/input_dim"] = len(features) + 1
            dcrnn_config_override["model/output_dim"] = len(features) + 1

        experiment_dir = create_experiment_structure(base_dir, experiment_name, config_path)
        experiment_path = os.path.join(base_dir, experiment_dir)

        detector_list_path = populate_inputs(experiment_path, detector_list)
        create_training_data(experiment_path, detector_list_path, override=generate_training_data_override, verbose=verbose)
        copy_dcrnn_configs(experiment_path, detector_list,
                           rnn_config_override=rnn_config_override, dcrnn_config_override=dcrnn_config_override)
        model_runner_config_path = copy_model_runner_config(experiment_path, experiment_name)
    else:
        experiment_path = config_path
        inputs_dir = os.path.join(experiment_path, "inputs")
        model_runner_config_path = os.path.join(inputs_dir, [f for f in os.listdir(inputs_dir)
                                                             if f.startswith("model_runner_config")][0])

        config = utils.load_yaml(model_runner_config_path)
        config["overwrite"] = False
        utils.save_yaml(config, model_runner_config_path)

    run_models(model_runner_config_path, debug=debug, verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file to specify detector list and parameters")
    parser.add_argument("-d", "--debug", action="store_true", help="Execute experiment runner without running models")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")
    args = parser.parse_args()

    main(args)
