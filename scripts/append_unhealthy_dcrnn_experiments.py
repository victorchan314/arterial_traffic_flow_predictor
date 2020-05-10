import argparse
import copy
import os
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

from lib import utils
from scripts import model_runner


def get_dcrnn_logdir(path):
    dcrnn_log_dirs = list(filter(lambda x: x.startswith("dcrnn"), os.listdir(path)))
    if len(dcrnn_log_dirs) != 1:
        raise ValueError("More than one DCRNN log dir at path {}".format(path))

    return dcrnn_log_dirs[0]

def get_latest_dcrnn_config_path(dcrnn_log_path):
    dcrnn_configs = list(filter(lambda x: x.startswith("config"), os.listdir(dcrnn_log_path)))
    latest_config_file = max(dcrnn_configs, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1]))
    latest_config_path = os.path.join(dcrnn_log_path, latest_config_file)

    return latest_config_path

def adapt_dcrnn_configs_to_unhealthy_dcrnn_folder(dcrnn_path, unhealthy_dcrnn_path, model_dcrnn_path):
    for subdir in os.listdir(dcrnn_path):
        dcrnn_subdir = os.path.join(dcrnn_path, subdir)
        unhealthy_subdir = os.path.join(unhealthy_dcrnn_path, subdir)
        utils.verify_or_create_path(unhealthy_subdir)
        model_subdir = os.path.join(model_dcrnn_path, subdir)

        dcrnn_logdir = get_dcrnn_logdir(dcrnn_subdir)
        model_logdir = get_dcrnn_logdir(model_subdir)
        dcrnn_log_path = os.path.join(dcrnn_subdir, dcrnn_logdir)
        model_log_path = os.path.join(model_subdir, model_logdir)
        latest_dcrnn_config_path = get_latest_dcrnn_config_path(dcrnn_log_path)
        latest_model_config_path = get_latest_dcrnn_config_path(model_log_path)

        dcrnn_config = utils.load_yaml(latest_dcrnn_config_path)
        model_config = utils.load_yaml(latest_model_config_path)

        dcrnn_config["base_dir"] = unhealthy_subdir
        dcrnn_config["train"]["model_filename"] = model_config["train"]["model_filename"]
        unhealthy_dcrnn_log_dir = os.path.join(unhealthy_subdir, dcrnn_logdir)
        utils.verify_or_create_path(unhealthy_dcrnn_log_dir)
        dcrnn_config["train"]["log_dir"] = unhealthy_dcrnn_log_dir

        unhealthy_dcrnn_config_path = os.path.join(unhealthy_subdir, "{}.yaml".format(subdir))
        utils.save_yaml(dcrnn_config, unhealthy_dcrnn_config_path)

def update_model_runner_config_for_unhealthy_dcrnn(model_runner_config):
    model_runner_config = copy.deepcopy(model_runner_config)

    del model_runner_config["models"]["Constant"]
    del model_runner_config["models"]["SeasonalNaive"]
    del model_runner_config["models"]["SARIMAX"]
    del model_runner_config["models"]["DCRNN"]["RNN"]
    del model_runner_config["models"]["DCRNN"]["__named_models__"]
    del model_runner_config["model_order"]
    del model_runner_config["loop"]["exclude"]
    del model_runner_config["loop"]["substitute"]

    model_runner_config["models"]["DCRNN"] = model_runner_config["models"]["DCRNN"]["DCRNN"]
    dcrnn_base_dir = model_runner_config["models"]["DCRNN"]["base_dir"]
    split_dcrnn_base_dir = os.path.normpath(dcrnn_base_dir).split(os.sep)
    split_dcrnn_base_dir[3] = "unhealthy_dcrnn"
    model_runner_config["models"]["DCRNN"]["base_dir"] = os.path.join(*split_dcrnn_base_dir)
    model_runner_config["models"]["DCRNN"]["is_trained"] = True
    model_runner_config["models"]["DCRNN"]["calculate_train_and_val_errors"] = False
    model_runner_config["loop"]["keys"] = ["data_directory", "models/DCRNN/base_dir"]

    return model_runner_config

def main(args):
    experiment_folder = args.experiment_folder
    model_experiment = args.model_experiment
    debug = args.debug

    dcrnn_path = os.path.join(experiment_folder, "experiments", "dcrnn")
    unhealthy_dcrnn_path = os.path.join(experiment_folder, "experiments", "unhealthy_dcrnn")
    model_dcrnn_path = os.path.join(model_experiment, "experiments", "dcrnn")

    adapt_dcrnn_configs_to_unhealthy_dcrnn_folder(dcrnn_path, unhealthy_dcrnn_path, model_dcrnn_path)

    experiment_name = os.path.normpath(experiment_folder).split(os.sep)[1].split("_")[0]
    model_runner_config_path = os.path.join(experiment_folder, "inputs", "model_runner_config_{}.yaml".format(experiment_name))
    model_runner_config = utils.load_yaml(model_runner_config_path)
    model_runner_config = update_model_runner_config_for_unhealthy_dcrnn(model_runner_config)

    unhealthy_model_runner_config_path = os.path.join(unhealthy_dcrnn_path, "unhealthy_model_runner_config_{}.yaml".format(experiment_name))
    utils.save_yaml(model_runner_config, unhealthy_model_runner_config_path)

    model_runner.loop_config(model_runner_config, debug=debug, verbose=2 * int(debug))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder", help="folder of experiment to re-predict")
    parser.add_argument("model_experiment", help="experiment root folder containing desired models")
    parser.add_argument("-d", "--debug", action="store_true", help="Execute model runner without running models")
    args = parser.parse_args()

    main(args)
