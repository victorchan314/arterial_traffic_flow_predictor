import argparse
import os
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

from lib import utils
from scripts import model_runner


def adapt_dcrnn_configs_to_unhealthy_dcrnn_folder(dcrnn_path, unhealthy_dcrnn_path, adjacency_matrix_folder):
    for subdir in os.listdir(dcrnn_path):
        dcrnn_subdir = os.path.join(dcrnn_path, subdir)
        plan = subdir.split("_")[0]
        adjacency_matrix_path = os.path.join(adjacency_matrix_folder, "adjacency_matrix_{}.pkl".format(plan))

        unhealthy_subdir = os.path.join(unhealthy_dcrnn_path, subdir)
        utils.verify_or_create_path(unhealthy_subdir)

        dcrnn_log_dirs =  list(filter(lambda x: x.startswith("dcrnn"), os.listdir(dcrnn_subdir)))
        if len(dcrnn_log_dirs) != 1:
            raise ValueError("More than one DCRNN log dir at path {}".format(dcrnn_subdir))

        dcrnn_log_dir = dcrnn_log_dirs[0]
        dcrnn_log_path = os.path.join(dcrnn_subdir, dcrnn_log_dir)
        dcrnn_configs = list(filter(lambda x: x.startswith("config"), os.listdir(dcrnn_log_path)))

        latest_config_file = max(dcrnn_configs, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1]))
        latest_config_path = os.path.join(dcrnn_log_path, latest_config_file)

        dcrnn_config = utils.load_yaml(latest_config_path)
        dcrnn_config["base_dir"] = unhealthy_subdir
        dcrnn_config["data"]["graph_pkl_filename"] = adjacency_matrix_path
        unhealthy_dcrnn_log_dir = os.path.join(unhealthy_subdir, dcrnn_log_dir)
        utils.verify_or_create_path(unhealthy_dcrnn_log_dir)
        dcrnn_config["train"]["log_dir"] = unhealthy_dcrnn_log_dir

        unhealthy_dcrnn_config_path = os.path.join(unhealthy_subdir, "{}.yaml".format(subdir))
        utils.save_yaml(dcrnn_config, unhealthy_dcrnn_config_path)

def update_model_runner_config_for_unhealthy_dcrnn(model_runner_config):
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

def main(args):
    experiment_folder = args.experiment_folder
    adjacency_matrix_folder = args.adjacency_matrix_folder
    debug = args.debug

    dcrnn_path = os.path.join(experiment_folder, "experiments", "dcrnn")
    unhealthy_dcrnn_path = os.path.join(experiment_folder, "experiments", "unhealthy_dcrnn")

    adapt_dcrnn_configs_to_unhealthy_dcrnn_folder(dcrnn_path, unhealthy_dcrnn_path, adjacency_matrix_folder)

    experiment_name = os.path.normpath(experiment_folder).split(os.sep)[1].split("_")[0]
    model_runner_config_path = os.path.join(experiment_folder, "inputs", "model_runner_config_{}.yaml".format(experiment_name))
    model_runner_config = utils.load_yaml(model_runner_config_path)
    update_model_runner_config_for_unhealthy_dcrnn(model_runner_config)

    unhealthy_model_runner_config_path = os.path.join(unhealthy_dcrnn_path, "unhealthy_model_runner_config_{}.yaml".format(experiment_name))
    utils.save_yaml(model_runner_config, unhealthy_model_runner_config_path)

    model_runner.loop_config(model_runner_config, debug=debug, verbose=2 * int(debug))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder", help="folder of experiment to re-predict")
    parser.add_argument("adjacency_matrix_folder", help="folder containing desired adjacency_matrix_{plan}.pkl")
    parser.add_argument("-d", "--debug", action="store_true", help="Execute model runner without running models")
    args = parser.parse_args()

    main(args)
