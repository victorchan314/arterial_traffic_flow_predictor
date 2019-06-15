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

def get_long_enough_stretch_indices(timestamps, stretch_length):
    break_indices = np.argwhere(utils.compare_timedeltas("!=", timestamps[1:] - timestamps[:-1], DETECTOR_DATA_FREQUENCY)).flatten() + 1
    stretch_starts = np.concatenate(([0], break_indices))
    stretch_ends = np.concatenate((stretch_starts, [timestamps.shape[0]]))[1:]

    long_enough_indices = np.argwhere(stretch_ends - stretch_starts > stretch_length).flatten()
    stretch_starts, stretch_ends = stretch_starts[long_enough_indices], stretch_ends[long_enough_indices]
    stretches = zip(stretch_starts, stretch_ends)

    return stretches

def break_data_into_stretches(data, freq):
    index = data.index

    break_indices = np.argwhere((index[1:] - index[:-1]) != freq).flatten() + 1
    stretch_starts = np.concatenate(([0], break_indices, [data.shape[0]]))
    stretch_lengths = stretch_starts[1:] - stretch_starts[:-1]
    longest_stretch_start_index = stretch_lengths.argmax()

    stretch_start = stretch_starts[longest_stretch_start_index]
    stretch_end = stretch_starts[longest_stretch_start_index + 1]

    indices = (stretch_start, stretch_end)
    stretch = data.iloc[stretch_start:stretch_end]

def consolidate_detector_data(detector_data, x_offset, y_offset):
    #data = break_data_into_stretches(detector_data, DETECTOR_DATA_FREQUENCY)
    dates = np.sort(detector_data["Date"].unique())
    data = []
    print(dates)
    print(1/0)
    for date in dates:
        temp = detector_data[detector_data["Date"] == date]
        if temp.shape[0] == 576:
            #dates.append(date)
            data.append(detector_data[detector_data["Date"] == date])

    print(data[0][data[0]["DetectorID"] == 508302])
    print(data[0].shape)
    #print(len(data))

def process_detector_data(detector_data, detector_list, stretch_length, verbose=False):
    detector_data_clean = clean_detector_data(detector_data)
    if verbose:
        print("Clean detector data shape: {}".format(detector_data_clean.shape))

    # Keep only timestamps that have data for all len(detector_list) detectors
    detector_data_filtered_by_num_detectors, timestamps = filter_detector_data_by_num_detectors(detector_data_clean, detector_list)
    if verbose:
        print("Filtered detector data: {}".format(detector_data_filtered_by_num_detectors.shape))
        print("Number of timestamps: {}".format(len(timestamps)))

    # Get indices of timestamps that have at least stretch_length 
    stretches = get_long_enough_stretch_indices(timestamps, stretch_length)
    detector_data_grouped_by_time = detector_data_filtered_by_num_detectors.groupby("Time")

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

    print(detector_data_array.shape)
    print(1/0)

    return detector_data_array

def generate_splits():
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )



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
    detector_data_processed = process_detector_data(detector_data, detector_list, x_offset + y_offset, verbose=True)

    #generate_splits(args)

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
