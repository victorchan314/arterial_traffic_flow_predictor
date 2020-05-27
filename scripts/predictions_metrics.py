import argparse
import os
import re
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

import numpy as np
import pandas as pd

from lib import data_utils
from lib import utils



PREDICTIONS_FILENAME = "predictions.npz"



def load_predictions(predictions_path, detectors=None, detector_list=None, predictions_key="predictions", groundtruth_key="groundtruth"):
    groundtruth, predictions = utils.load_predictions(predictions_path, predictions_key=predictions_key, groundtruth_key=groundtruth_key)

    if not detectors is None:
        detectors_mask = [True if detector in detectors else False for detector in detector_list]
        groundtruth = groundtruth[:, :, detectors_mask, ...]
        predictions = predictions[:, :, detectors_mask, ...]

    return groundtruth, predictions

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

def print_errors_latex(logdir, horizons, features, precision, detectors=None, detector_list=None,
                       ts_dir=None, phase_plans=None, gfsd=False):
    dirs = sorted(os.listdir(logdir))

    error_types = ["mse", "rmse", "mae", "mape"]
    errors = {}

    for dir in dirs:
        predictions_path = os.path.join(logdir, dir, PREDICTIONS_FILENAME)
        if not os.path.isdir(os.path.join(logdir, dir)) or not os.path.exists(predictions_path):
            continue

        offset_matches = re.findall("_o\d+_", dir)
        if len(offset_matches) != 1:
            raise ValueError("Experiment should have exactly 1 offset")

        offset = int(offset_matches[0][2:-1])

        groundtruth, predictions = load_predictions(predictions_path, detectors=detectors, detector_list=detector_list)
        if gfsd:
            index = [i for i, d in enumerate(os.path.normpath(logdir).split(os.sep)) if d == "experiments"][1]
            experiment_dir = os.sep.join(os.path.normpath(logdir).split(os.sep)[:index])
            sensor_data_path = os.path.join(experiment_dir, "inputs", "sensor_data", "{}_sensor_data".format(dir), "test.npz")
            sensor_groundtruth, _ = load_predictions(sensor_data_path, detectors=detectors, detector_list=detector_list,
                                                     predictions_key="y", groundtruth_key="y")
            groundtruth = np.transpose(sensor_groundtruth[..., 1:], axes=(1, 0, 2, 3))

        dir_horizon = predictions.shape[0]
        dir_horizons = range(1, dir_horizon + 1) if horizons is None else sorted([int(h) for h in horizons])
        dir_num_features = 1 if len(predictions.shape) < 4 else predictions.shape[3]
        dir_features = range(dir_num_features) if features is None else sorted([int(f) - 1 for f in features])
        batch_size = predictions.shape[1]

        if not ts_dir is None:
            ts_file = os.path.join(ts_dir, dir + "_sensor_data", "test.npz")
            timestamps = np.load(ts_file)["timestamps_y"].astype(int) / 1e9 % 86400

            plans = phase_plans["PlanName"].unique()
            all_groundtruths = {}
            all_predictions = {}

            for plan in plans:
                relevant_phase_plans = phase_plans[phase_plans["PlanName"] == plan]
                intervals = relevant_phase_plans.loc[:, ["StartTime", "EndTime"]].values * 3600

                timestamps_mask = np.logical_or.reduce([(timestamps >= start_time).all(axis=1) & (timestamps < end_time).all(axis=1) for start_time, end_time in intervals])
                all_groundtruths[plan] = groundtruth[:, timestamps_mask, :]
                all_predictions[plan] = predictions[:, timestamps_mask, :]
        else:
            plan = dir.split("_")[0]
            plans = [plan]
            all_groundtruths = {plan: groundtruth}
            all_predictions = {plan: predictions}

        for plan in plans:
            if not plan in errors:
                errors[plan] = {f: {} for f in dir_features}

            for feature in errors[plan]:
                if not errors[plan][feature]:
                    errors[plan][feature] = {e: {} for e in error_types}

                for v in errors[plan][feature].values():
                    if not offset in v:
                        v[offset] = {}

        for plan in plans:
            groundtruth = all_groundtruths[plan]
            predictions = all_predictions[plan]

            for f in dir_features:
                for h in dir_horizons:
                    prediction_errors = data_utils.get_standard_errors(groundtruth[h-1, :, :, f], predictions[h-1, :, :, f])

                    for k, v in prediction_errors.items():
                        error = v
                        if h in errors[plan][f][k][offset]:
                            previous_error = errors[plan][f][k][offset][h][0]
                            previous_batch_size = errors[plan][f][k][offset][h][1]
                            total_batch_size = batch_size + previous_batch_size
                            average_error = (error * batch_size + previous_error * previous_batch_size) / total_batch_size

                            errors[plan][f][k][offset][h] = [average_error, total_batch_size]
                        else:
                            errors[plan][f][k][offset][h] = [error, batch_size]

    for plan, v1 in errors.items():
        for feature, v2 in v1.items():
            for error_type, v3 in v2.items():
                for offset, v4 in v3.items():
                    for horizon, v5 in v4.items():
                        v = v5[0]
                        if v >= 1000:
                            error = int(v)
                        elif error_type == "mape":
                            error = round(100 * v, precision)
                        else:
                            error = round(v, precision)

                        errors[plan][feature][error_type][offset][horizon] = error

    output = ""
    for plan in sorted(errors.keys()):
        for feature in sorted(errors[plan].keys()):
            output = utils.proxy_print(output, "Plan {}, feature {}".format(plan, feature))

            for error_type in error_types:
                table_line_errors = errors[plan][feature][error_type]
                table_line = "& " + error_type.upper()
                for offset in sorted(table_line_errors.keys()):
                    for horizon in sorted(table_line_errors[offset].keys()):
                        error = table_line_errors[offset][horizon]
                        if len(str(error)) > precision + 5:
                            error = ("{:." + str(precision) + "E}").format(error)

                        if error_type == "mape":
                            table_line += " & {}\\%".format(error)
                        else:
                            table_line += " & {}".format(error)

                table_line += " \\\\"

                output = utils.proxy_print(output, table_line)

    return output



def main(args):
    logdir = args.logdir
    detectors = args.detectors
    detector_list_path = args.detector_list_path
    horizons = args.horizons
    features = args.features
    groundtruth_from_sensor_data = args.groundtruth_from_sensor_data
    not_latex = args.not_latex
    precision = args.round
    ts_dir = args.no_plan_ts_dir
    phase_plans_csv = args.phase_plans_csv
    do_not_print = args.do_not_print

    if not ts_dir is None and phase_plans_csv is None:
        raise ValueError("phase_plans_csv must be supplied if ts_npz is supplied")

    if detector_list_path:
        with open(detector_list_path, "r") as f:
            detector_list_file = f.read()

        detector_list = list(map(int, detector_list_file.split(",")))
    else:
        detector_list = None

    phase_plans = pd.read_csv(phase_plans_csv) if phase_plans_csv else None

    if not_latex:
        output = print_errors(logdir, horizons, precision)
        if do_not_print:
            return output
        else:
            print(output)
    else:
        return print_errors_latex(logdir, horizons, features, precision, detectors=detectors,
                                  detector_list=detector_list, ts_dir=ts_dir, phase_plans=phase_plans,
                                  gfsd=groundtruth_from_sensor_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Location of experiment directories")
    parser.add_argument("--detectors", "-d", type=int, nargs="+", help="Detectors to calculate error for")
    parser.add_argument("--detector_list_path", "--dl", type=str, help="Order of detectors in predictions")
    parser.add_argument("--horizons", "--h", nargs="+", help="Horizon(s) to include")
    parser.add_argument("--features", "--f", nargs="+", help="Feature(s) to include")
    parser.add_argument("--groundtruth_from_sensor_data", "-g", action="store_true", help="Use groundtruth from sensor_data and not predictions")
    parser.add_argument("--not_latex", "-l", action="store_true", help="Don't print for LaTeX table")
    parser.add_argument("--round", "-r", type=int, default=2, help="Rounding precision")
    parser.add_argument("--do_not_print", action="store_true", help="Flag to suppress printing")
    parser.add_argument("--no_plan_ts_dir", type=str, help="Location of timestamps to calculate error across plans")
    parser.add_argument("--phase_plans_csv", type=str, help="Location of csv with phase plan times")
    args = parser.parse_args()

    output = main(args)
    print(output)
