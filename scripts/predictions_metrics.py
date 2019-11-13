import argparse
import os
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

import numpy as np

from lib import data_utils



PREDICTIONS_FILENAME = "predictions.npz"

def main(args):
    logdir = args.logdir
    horizons = args.horizons

    dirs = sorted(os.listdir(logdir))

    for dir in dirs:
        predictions_file = os.path.join(logdir, dir, PREDICTIONS_FILENAME)
        predictions_contents = np.load(predictions_file)
        groundtruth = predictions_contents["groundtruth"]
        predictions = predictions_contents["predictions"]

        horizon = predictions.shape[0]
        horizons = range(1, horizon + 1) if horizons is None else sorted([int(h) for h in horizons])

        print(dir)

        for h in horizons:
            errors = data_utils.get_standard_errors(groundtruth[h-1, ...], predictions[h-1, ...])

            print("Horizon {}:".format(h), end="")

            for k, v in errors.items():
                error = round(v, 4)
                print(" {}: {};".format(k, error), end="")

            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Location of experiment directories")
    parser.add_argument("--horizons", "--h", action="append", help="Horizon(s) to include")
    args = parser.parse_args()

    main(args)
