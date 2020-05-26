import argparse
import copy
import os
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

from lib import utils
from scripts.append_unhealthy_dcrnn_experiments import main as repredict


def main(args):
    base_dir = args.base_dir
    model_experiment = args.model_experiment
    debug = args.debug

    unhealthy_experiments = filter(lambda x: x.startswith("unhealthy"), os.listdir(base_dir))

    for experiment in unhealthy_experiments:
        args_dict = {
            "experiment_folder": os.path.join(base_dir, experiment),
            "model_experiment": model_experiment,
            "debug": debug
        }
        repredict_args = utils.Namespace(args_dict)

        repredict(repredict_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="base directory with folders of all experiments to re-predict")
    parser.add_argument("model_experiment", help="experiment root folder containing desired models")
    parser.add_argument("-d", "--debug", action="store_true", help="Execute model runner without running models")
    args = parser.parse_args()

    main(args)
