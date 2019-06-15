import argparse
import os
import sys

parent_dir = os.path.abspath("/Users/victorchan/Dropbox/UndergradResearch/Victor/Code")
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import datetime as dt

import mysql_utils
import utils



PHASE_PLANS_PATH = "data/model/phase_plans{}.csv"
DETECTOR_LIST_PATH = "data/model/sensors_advanced{}.txt"

DETECTOR_DATA_QUERY = "SELECT DetectorID, Year, Month, Day, Time, Volume AS Flow FROM detector_data_processed_2017 WHERE ({})"
DETECTOR_DATA_FREQUENCY = dt.timedelta(minutes=5)

DUMMY_DATA_TYPES = ["zeroes", "ones", "linear", "square"]



def get_sensors_list(path):
    with open(path, "r") as f:
        sensors = f.read()
        f.close()

    sensors_list = sensors[:-1].split(",")

    return sensors_list

def get_detector_data(detector_list, intersection="", plan=None, limit=np.inf, date_limit=dt.date.max):
    query = DETECTOR_DATA_QUERY.format(" OR ".join(["DetectorID = {}".format(d) for d in detector_list]))
    if plan:
        phase_plans = pd.read_csv(PHASE_PLANS_PATH.format(intersection))
        relevant_phase_plans = phase_plans[phase_plans["PlanName"] == plan]
        intervals = relevant_phase_plans.loc[:, ["StartTime", "EndTime"]].values * 3600
        query += " AND ({})".format(" OR ".join(["(Time >= {} AND Time < {})".format(interval[0], interval[1]) for interval in intervals]))

    query_results = mysql_utils.execute_query(query)
    
    results = []

    for row in query_results:
        if len(results) >= limit:
            break

        row_date = dt.date(row[1], row[2], row[3])
        if row_date <= date_limit:
            row_datetime = dt.datetime(row[1], row[2], row[3], row[4] // 3600, (row[4] % 3600) // 60, row[4] % 60)
            results.append([row[0], row_date, row_datetime, row[4] / 86400, row[5]])

    detector_data = pd.DataFrame(results, columns=["DetectorID", "Date", "Time", "Seconds", "Flow"])

    return detector_data

def clean_detector_data(detector_data):
    return detector_data

def filter_detector_data_by_num_detectors(detector_data, detector_list):
    detector_list_sorted = sorted([x for x in detector_list])
    detector_data_filtered = detector_data.groupby(["Time"]).filter(lambda x: detector_list_sorted == sorted(list(x["DetectorID"])))
    timestamps = np.sort(detector_data_filtered["Time"].unique())

    return detector_data_filtered, timestamps

def create_4d_detector_data_array(detector_data, timestamps, detector_list, stretch_length, verbose=False):
    stretches = get_long_enough_stretch_indices(timestamps, stretch_length)
    detector_data_grouped_by_time = detector_data.groupby("Time")

    detector_datum_shape = (stretch_length, len(detector_list), detector_data.shape[1] - 3)
    detector_data_array = np.empty((0, *detector_datum_shape))

    for start, end in stretches:
        if verbose:
            print("Working on stretch ({}, {})".format(start, end))

        time_stretch = timestamps[start:end]
        detector_datum = np.zeros((end - start - stretch_length + 1, *detector_datum_shape))

        for i, timestamp in enumerate(time_stretch):
            detector_data_at_timestamp = detector_data_grouped_by_time.get_group(timestamp)
            detector_data_array_at_timestamp = detector_data_at_timestamp.set_index("DetectorID").loc[detector_list].iloc[:, 2:].values

            # Populate the diagonal corresponding to the current timestamp to take care of offsets
            for y in range(min(i, end - start - stretch_length), max(-1, i - stretch_length), -1):
                x = i - y
                detector_datum[y, x, :, :] = np.array(detector_data_array_at_timestamp)

        detector_data_array = np.vstack((detector_data_array, detector_datum))

    return detector_data_array

def get_long_enough_stretch_indices(timestamps, stretch_length):
    break_indices = np.argwhere(utils.compare_timedeltas("!=", timestamps[1:] - timestamps[:-1], DETECTOR_DATA_FREQUENCY)).flatten() + 1
    stretch_starts = np.concatenate(([0], break_indices))
    stretch_ends = np.concatenate((stretch_starts, [timestamps.shape[0]]))[1:]

    long_enough_indices = np.argwhere(stretch_ends - stretch_starts > stretch_length).flatten()
    stretch_starts, stretch_ends = stretch_starts[long_enough_indices], stretch_ends[long_enough_indices]
    stretches = zip(stretch_starts, stretch_ends)

    return stretches

def process_detector_data(detector_data, detector_list, stretch_length, verbose=False):
    if verbose:
        print("Detector data original shape: {}".format(detector_data.shape))

    detector_data_clean = clean_detector_data(detector_data)
    if verbose:
        print("Clean detector data shape: {}".format(detector_data_clean.shape))

    # Keep only timestamps that have data for all len(detector_list) detectors
    detector_data_filtered_by_num_detectors, timestamps = filter_detector_data_by_num_detectors(detector_data_clean, detector_list)
    if verbose:
        print("Filtered detector data shape: {}".format(detector_data_filtered_by_num_detectors.shape))
        print("Number of timestamps: {}".format(len(timestamps)))

    detector_data_array = create_4d_detector_data_array(detector_data_filtered_by_num_detectors, timestamps, detector_list, stretch_length, verbose=verbose)
    if verbose:
        print("Processed detector data shape: {}".format(detector_data_array.shape))

    return detector_data_array

def generate_splits(detector_data_processed, x_offset, y_offset, verbose=False):
    x_offsets = np.arange(-x_offset + 1, 1, 1)
    y_offsets = np.arange(1, y_offset + 1, 1)
    x = detector_data_processed[:, :x_offset, :, :]
    y = detector_data_processed[:, x_offset:, :, :]
    if verbose:
        print("x shape: {}".format(x.shape))
        print("y shape: {}".format(y.shape))

    # Write the data into npz file
    # 7/10 training, 1/10 validation, 2/10 test
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train+num_val], y[num_train:num_train+num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        if verbose:
            print("{} x: {}, y: {}".format(cat, _x.shape, _y.shape))
#        np.savez_compressed(
#            os.path.join(args.output_dir, "%s.npz" % cat),
#            x=_x,
#            y=_y,
#            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
#            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
#        )



def main(args):
    if args.dummy and args.dummy in DUMMY_DATA_TYPES:
        return

    intersection = "_{}".format(args.intersection) if args.intersection else ""
    output_path = args.output_dir or "data"
    plan_name = args.plan_name
    x_offset = int(args.x_offset or "12")
    y_offset = int(args.y_offset or "12")

    detector_list = [int(x) for x in get_sensors_list(DETECTOR_LIST_PATH.format(intersection))]
    detector_data = get_detector_data(detector_list, intersection=intersection, plan=plan_name, date_limit=dt.date(2017, 1, 9))
    detector_data_processed = process_detector_data(detector_data, detector_list, x_offset + y_offset)
    print(detector_data_processed.shape)
    generate_splits(detector_data_processed, x_offset, y_offset, verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", help="intersection to focus on. Assumes all relevant data/model/ files have proper suffix.")
    parser.add_argument("--output_dir", help="output directory for")
    parser.add_argument("--plan_name", help="name of plan: E, P1, P2, or P3")
    parser.add_argument("--x_offset", help="number of time steps to use for training")
    parser.add_argument("--y_offset", help="number of time steps to predict ahead")
    parser.add_argument("--dummy", help="Overrides other arguments. Generate dummy training data with the specified pattern: {}".format(DUMMY_DATA_TYPES))
    args = parser.parse_args()

    main(args)
