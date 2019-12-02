import argparse
import os
import re
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

from lib import data_utils
from lib import utils



PREDICTIONS_FILENAME = "predictions.npz"

def print_errors(logdir, horizons, precision):
    dirs = sorted(os.listdir(logdir))

    for dir in dirs:
        predictions_path = os.path.join(logdir, dir, PREDICTIONS_FILENAME)
        groundtruth, predictions = utils.load_predictions(predictions_path)

        horizon = predictions.shape[0]
        horizons = range(1, horizon + 1) if horizons is None else sorted([int(h) for h in horizons])

        print(dir)

        for h in horizons:
            errors = data_utils.get_standard_errors(groundtruth[h-1, ...], predictions[h-1, ...])

            print("Horizon {}:".format(h), end="")

            for k, v in errors.items():
                error = round(v, precision)
                print(" {}: {};".format(k, error), end="")

            print()

def print_errors_latex(logdir, horizons, precision):
    dirs = sorted(os.listdir(logdir))

    error_types = ["mse", "rmse", "mae", "mape"]
    errors = {}

    for dir in dirs:
        plan = dir.split("_")[1]
        offset_matches = re.findall("_o\d+_", dir)
        if len(offset_matches) != 1:
            raise ValueError("Experiment should have exactly 1 offset")

        offset = int(offset_matches[0][2:-1])

        if not plan in errors:
            errors[plan] = {e: {} for e in error_types}

        for v in errors[plan].values():
            v[offset] = {}

        predictions_path = os.path.join(logdir, dir, PREDICTIONS_FILENAME)
        groundtruth, predictions = utils.load_predictions(predictions_path)

        horizon = predictions.shape[0]
        horizons = range(1, horizon + 1) if horizons is None else sorted([int(h) for h in horizons])

        for h in horizons:
            prediction_errors = data_utils.get_standard_errors(groundtruth[h-1, ...], predictions[h-1, ...])

            for k, v in prediction_errors.items():
                if v >= 1000:
                    error = int(v)
                elif k == "mape":
                    error = round(100 * v, precision)
                else:
                    error = round(v, precision)

                errors[plan][k][offset][h] = error

    for plan in sorted(errors.keys()):
        print(plan)

        for error_type in error_types:
            table_line_errors = errors[plan][error_type]
            table_line = error_type.upper()
            for offset in sorted(table_line_errors.keys()):
                for horizon in sorted(table_line_errors[offset].keys()):
                    error = table_line_errors[offset][horizon]
                    if error_type == "mape":
                        table_line += " & {}\\%".format(error)
                    else:
                        table_line += " & {}".format(error)

            table_line += " \\\\"

            print(table_line)



def main(args):
    logdir = args.logdir
    horizons = args.horizons
    latex = args.latex
    precision = args.round

    if latex:
        print_errors_latex(logdir, horizons, precision)
    else:
        print_errors(logdir, horizons, precision)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Location of experiment directories")
    parser.add_argument("--horizons", "--h", action="append", help="Horizon(s) to include")
    parser.add_argument("--latex", "-l", action="store_true", help="Print for LaTeX table")
    parser.add_argument("--round", "-r", type=int, default=4, help="Rounding precision")
    args = parser.parse_args()

    main(args)
