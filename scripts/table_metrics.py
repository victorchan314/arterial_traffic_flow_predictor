import argparse
import os
import re
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

from lib import utils
from scripts.predictions_metrics import main as predictions_metrics



def get_weekday_experiment_dir(base_dir, experiment_name):
    experiments = os.listdir(base_dir)

    dirs = list(filter(lambda x: x.split("_")[0] == experiment_name or
                                 x.split("_")[0] == "{}-weekday".format(experiment_name), experiments))
    if len(dirs) != 1:
        raise ValueError("Incorrect number of dirs. {} experiments found for {}".format(len(dirs), experiment_name))

    return os.path.join(base_dir, dirs[0])

def build_metrics_table(base_dir, experiments, args_dict):
    tables = {}

    for experiment_name, repredict in experiments:
        experiment_dir = get_weekday_experiment_dir(base_dir, experiment_name)
        dcrnn_dir = os.path.join(experiment_dir, "experiments", "dcrnn" if repredict else "unhealthy_dcrnn")

        detector_list_path = os.path.join(experiment_dir, "inputs", "model", "detector_list.txt")

        args_dict["logdir"] = dcrnn_dir
        args_dict["detector_list_path"] = detector_list_path
        args = utils.Namespace(args_dict)

        output = predictions_metrics(args)

        split_output = re.split("Plan (P[\d], feature [\d][\r\n]+)", output)[1:]
        for i in range(0, len(split_output), 2):
            plan = split_output[i][:-1]
            table = split_output[i + 1]

            if not plan in tables:
                tables[plan] = ""

            tables[plan] += table
            tables[plan] += "\\hline\n"

    return tables



def main(args):
    base_dir = args.base_dir
    id = args.table
    detectors = args.detectors
    horizons = args.horizons
    precision = args.round

    args_dict = {
        "detectors": detectors,
        "horizons": horizons,
        "round": precision,
        "do_not_print": True,
        "not_latex": False,
        "features": None,
        "no_plan_ts_dir": None,
        "phase_plans_csv": None
    }

    if id == 1:
        experiments = [["full-information", True],
                       ["no-upstream", True],
                       ["no-downstream", True],
                       ["full-information-FlOcc", True],
                       ["no-upstream-FlOcc", True],
                       ["no-downstream-FlOcc", True]]
    else:
        raise ValueError("Invalid id {}".format(id))

    tables = build_metrics_table(base_dir, experiments, args_dict)
    for plan, table in tables.items():
        print(plan)
        print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="Location of base directory")
    parser.add_argument("table", type=int, help="id of table to generate")
    parser.add_argument("--detectors", "-d", type=int, nargs="+", default=[508302, 508306], help="Detectors to calculate error for")
    parser.add_argument("--horizons", "--h", action="append", default=[1, 3, 6], help="Horizon(s) to include")
    parser.add_argument("--round", "-r", type=int, default=2, help="Rounding precision")
    args = parser.parse_args()

    main(args)
