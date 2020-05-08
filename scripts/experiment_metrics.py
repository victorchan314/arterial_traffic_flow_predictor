import argparse
import os
import re
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

from lib import utils
from scripts.predictions_metrics import main as predictions_metrics



def build_metrics_table(experiment_dir, args_dict):
    dcrnn_dir = os.path.join(experiment_dir, "experiments", "dcrnn")
    baselines_dir = os.path.join(experiment_dir, "experiments", "baselines")

    logdirs = [
        ["DCRNN", dcrnn_dir],
        ["Constant Mean", os.path.join(baselines_dir, "constant")],
        ["Seasonal Naive", os.path.join(baselines_dir, "seasonal_naive")],
        ["ARIMAX", os.path.join(baselines_dir, "arimax")],
        ["Online ARIMAX", os.path.join(baselines_dir, "online_arimax")],
        ["RNN", os.path.join(baselines_dir, "rnn")]
    ]

    tables = {}

    for name, logdir in logdirs:
        args_dict["logdir"] = logdir
        args = utils.Namespace(args_dict)

        output = predictions_metrics(args)
        split_output = re.split("(P[\d][\r\n]+)", output)[1:]
        for i in range(0, len(split_output), 2):
            plan = split_output[i][:-1]
            table = split_output[i + 1][:-1]
            table_rows = table.splitlines()

            if name == "Online ARIMAX":
                for j in range(len(table_rows)):
                    table_row = table_rows[j]
                    split_table_row = table_row.split("&")
                    for _ in range(3):
                        split_table_row.insert(2, " ")

                    table_rows[j] = "&".join(split_table_row)

            split_name = name.split()
            for j in range(len(split_name)):
                table_rows[j] = "{} {}".format(split_name[j], table_rows[j])

            named_table = "\n".join(table_rows) + "\n"

            if not plan in tables:
                tables[plan] = ""

            tables[plan] += named_table
            tables[plan] += "\\hline\n"

    return tables



def main(args):
    experiment_dir = args.experiment_dir
    detectors = args.detectors
    horizons = args.horizons
    precision = args.round

    detector_list_path = os.path.join(experiment_dir, "inputs", "model", "detector_list.txt")

    args_dict = {
        "detectors": detectors,
        "detector_list_path": detector_list_path,
        "horizons": horizons,
        "round": precision,
        "do_not_print": True,
        "not_latex": False,
        "features": None,
        "no_plan_ts_dir": None,
        "phase_plans_csv": None
    }

    tables = build_metrics_table(experiment_dir, args_dict)
    for plan, table in tables.items():
        print(plan)
        print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str, help="Location of experiment directory")
    parser.add_argument("--detectors", "-d", type=int, nargs="+", help="Detectors to calculate error for")
    parser.add_argument("--horizons", "--h", action="append", default=[1, 3, 6], help="Horizon(s) to include")
    parser.add_argument("--round", "-r", type=int, default=2, help="Rounding precision")
    args = parser.parse_args()

    main(args)
