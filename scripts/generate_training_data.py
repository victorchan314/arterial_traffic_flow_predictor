import argparse
import os
import sys
import ast

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import datetime as dt

from lib import data_utils, mysql_utils, utils

PHASE_PLANS_PATH = "data/inputs/model/phase_plans.csv"

DETECTOR_DATA_QUERY = \
    "SELECT DetectorID, Year, Month, Day, Time, Volume AS Flow, Occupancy, Speed, Health\
    FROM detector_data_processed_2017 NATURAL JOIN detector_health\
    WHERE ({}) AND Health = 1;"
DETECTOR_DATA_FREQUENCY = dt.timedelta(minutes=5)

DUMMY_DATA_TYPES = ["zeros", "ones", "linear_integers", "linear_floats"]

BASE_HIGH = 1000
INCREMENT_HIGH = 10



def generate_dummy_data(dummy_data_type, dummy_shape, verbose=0):
    if dummy_data_type == "zeros":
        return np.zeros(dummy_shape)
    elif dummy_data_type == "ones":
        return np.ones(dummy_shape)
    elif dummy_data_type.startswith("linear_"):
        base_shape = (dummy_shape[0], 1, *dummy_shape[2:])

        if dummy_data_type[7:] == "integers":
            base = np.random.randint(0, BASE_HIGH + 1, base_shape)
            increment = np.random.randint(0, INCREMENT_HIGH + 1, base_shape)
        elif dummy_data_type[7:] == "floats":
            base = BASE_HIGH * np.random.random_sample(base_shape)
            increment = INCREMENT_HIGH * np.random.random_sample(base_shape)

        dummy_data = np.hstack((base + i * increment for i in range(dummy_shape[1])))

        return dummy_data

def get_sensors_list(path):
    with open(path, "r") as f:
        sensors = f.read()

    sensors_list = sensors.split(",")
    sensors_list = list(map(int, sensors_list))

    return sensors_list

def get_detector_data(detector_list, plan=None, limit=np.inf, date_limit=dt.date.max, features=[],
                      weekday_only=False, start_time_buffer=0, end_time_buffer=0):
    query = DETECTOR_DATA_QUERY.format(" OR ".join(["DetectorID = {}".format(d) for d in detector_list]))
    if plan:
        phase_plans = pd.read_csv(PHASE_PLANS_PATH)
        relevant_phase_plans = phase_plans[(phase_plans["Intersection"] == phase_plans.loc[1, "Intersection"])
                                           & (phase_plans["PlanName"] == plan)]
        intervals = relevant_phase_plans.loc[:, ["StartTime", "EndTime"]].values * 3600
        buffered_intervals = [[interval[0] - (DETECTOR_DATA_FREQUENCY * start_time_buffer).total_seconds(),
                               interval[1] + (DETECTOR_DATA_FREQUENCY * end_time_buffer).total_seconds()] for interval in intervals]
        query = query[:-1] + " AND ({})".format(" OR ".join(["(Time >= {} AND Time < {})".format(
                interval[0], interval[1]) for interval in buffered_intervals])) + query[-1]

    query_results = mysql_utils.execute_query(query)

    columns = ["DetectorID", "Date", "Time", "Seconds"]
    if "Flow" in features:
        columns.append("Flow")
    if "Occupancy" in features:
        columns.append("Occupancy")
    if "Speed" in features:
        columns.append("Speed")

    results = []

    for row in query_results:
        if len(results) >= limit:
            break

        row_date = dt.date(row[1], row[2], row[3])
        if row_date <= date_limit and (not weekday_only or row_date.weekday() < 5):
            row_datetime = dt.datetime(row[1], row[2], row[3], row[4] // 3600, (row[4] % 3600) // 60, row[4] % 60)
            seconds_fraction = row[4] / 86400
            flow = row[5]
            occupancy = row[6]
            speed = row[7]

            data_row = [row[0], row_date, row_datetime, seconds_fraction]

            if "Flow" in features:
                data_row.append(flow)
            if "Occupancy" in features:
                data_row.append(occupancy)
            if "Speed" in features:
                data_row.append(speed)

            results.append(data_row)

    detector_data = pd.DataFrame(results, columns=columns)

    return detector_data

def clean_detector_data(detector_data):
    return detector_data

def filter_detector_data_by_num_detectors(detector_data, detector_list):
    detector_list_sorted = sorted([x for x in detector_list])
    detector_data_filtered = detector_data.groupby(["Time"]).filter(lambda x: detector_list_sorted == sorted(list(x["DetectorID"])))
    timestamps = np.sort(detector_data_filtered["Time"].unique())

    return detector_data_filtered, timestamps

def create_4d_detector_data_array(detector_data, timestamps, detector_list, stretch_length, verbose=0):
    stretches = get_long_enough_stretch_indices(timestamps, stretch_length)
    detector_data_grouped_by_time = detector_data.groupby("Time")

    detector_datum_shape = (stretch_length, len(detector_list), detector_data.shape[1] - 3)
    detector_data_array = np.zeros((0, *detector_datum_shape))
    timestamps_array = np.empty((0, stretch_length), dtype=timestamps.dtype)

    for start, end in stretches:
        if verbose > 1:
            print("Working on stretch ({}, {})".format(start, end))

        time_stretch = timestamps[start:end]
        detector_datum = np.empty((end - start - stretch_length + 1, *detector_datum_shape))
        timestamps_stretch = np.empty((end - start - stretch_length + 1, stretch_length), dtype=timestamps.dtype)

        for i, timestamp in enumerate(time_stretch):
            detector_data_at_timestamp = detector_data_grouped_by_time.get_group(timestamp)
            detector_data_array_at_timestamp = detector_data_at_timestamp.set_index("DetectorID").loc[detector_list].iloc[:, 2:].values
            if verbose > 2:
                print("Working on detector data at timestamp {}".format(timestamp))

            # Populate the diagonal corresponding to the current timestamp to take care of offsets
            for y in range(min(i, end - start - stretch_length), max(-1, i - stretch_length), -1):
                x = i - y
                detector_datum[y, x, :, :] = np.array(detector_data_array_at_timestamp)
                timestamps_stretch[y, x] = timestamp

        detector_data_array = np.vstack((detector_data_array, detector_datum))
        timestamps_array = np.vstack((timestamps_array, timestamps_stretch))

    return detector_data_array, timestamps_array

def create_3d_detector_data_array(detector_data, timestamps, detector_list, stretch_length, verbose=0):
    stretches = get_long_enough_stretch_indices(timestamps, stretch_length)
    detector_data_grouped_by_time = detector_data.groupby("Time")

    detector_datum_shape = (len(detector_list), detector_data.shape[1] - 3)
    detector_data_array = np.zeros((0, *detector_datum_shape))
    filtered_timestamps = np.array([], dtype=timestamps.dtype)

    for start, end in stretches:
        if verbose > 1:
            print("Working on stretch ({}, {})".format(start, end))

        time_stretch = timestamps[start:end]
        filtered_timestamps = np.concatenate((filtered_timestamps, time_stretch))
        detector_datum = np.empty((end - start, *detector_datum_shape))

        for i, timestamp in enumerate(time_stretch):
            detector_data_at_timestamp = detector_data_grouped_by_time.get_group(timestamp)
            detector_data_array_at_timestamp = detector_data_at_timestamp.set_index("DetectorID").loc[detector_list].iloc[:, 2:].values
            if verbose > 2:
                print("Working on detector data at timestamp {}".format(timestamp))

            # Populate the diagonal corresponding to the current timestamp to take care of offsets
            detector_datum[i, :, :] = np.array(detector_data_array_at_timestamp)

        detector_data_array = np.vstack((detector_data_array, detector_datum))

    return detector_data_array, filtered_timestamps

def get_long_enough_stretch_indices(timestamps, stretch_length):
    stretches = data_utils.get_stretches(timestamps, DETECTOR_DATA_FREQUENCY)
    long_enough_indices = np.argwhere(stretches[:, 1] - stretches[:, 0] > stretch_length).flatten()
    stretches = stretches[long_enough_indices]

    return stretches

def process_detector_data(detector_data, detector_list, verbose=0):
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

    return detector_data_filtered_by_num_detectors, timestamps

def process_timeseries_detector_data(detector_data, detector_list, stretch_length, verbose=0):
    detector_data_filtered_by_num_detectors, timestamps = process_detector_data(detector_data, detector_list, verbose=verbose)

    detector_data_array, filtered_timestamps = create_3d_detector_data_array(detector_data_filtered_by_num_detectors, timestamps, detector_list, stretch_length, verbose=verbose)
    if verbose:
        print("Processed detector data shape: {}".format(detector_data_array.shape))

    return detector_data_array, timestamps, filtered_timestamps


def process_array_detector_data(detector_data, detector_list, stretch_length, verbose=0):
    detector_data_filtered_by_num_detectors, timestamps = process_detector_data(detector_data, detector_list, verbose=verbose)

    detector_data_array, timestamps_array = create_4d_detector_data_array(detector_data_filtered_by_num_detectors, timestamps, detector_list, stretch_length, verbose=verbose)
    if verbose:
        print("Processed detector data shape: {}".format(detector_data_array.shape))
        print("Timestamps array shape: {}".format(timestamps_array.shape))

    return detector_data_array, timestamps, timestamps_array

def generate_array_splits(detector_data_processed, x_offset, y_offset, output_path, timestamps=None, verbose=0):
    x_offsets = np.arange(-x_offset + 1, 1, 1)
    y_offsets = np.arange(1, y_offset + 1, 1)
    x = detector_data_processed[:, :x_offset, ...]
    y = detector_data_processed[:, x_offset:, ...]

    if not timestamps is None:
        timestamps_x = timestamps[:, :x_offset, ...]
        timestamps_y = timestamps[:, x_offset:, ...]

    if verbose:
        print("Detector data shape: {}".format(detector_data_processed.shape))
        print("x shape: {}".format(x.shape))
        print("y shape: {}".format(y.shape))

        if not timestamps is None:
            print("Timestamps array shape: {}".format(timestamps.shape))
            print("x shape: {}".format(timestamps_x.shape))
            print("y shape: {}".format(timestamps_y.shape))

    # Write the data into npz file
    # 7/10 training, 1/10 validation, 2/10 test
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train+num_val], y[num_train:num_train+num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    if not timestamps is None:
        timestamps_x_train, timestamps_y_train = timestamps_x[:num_train], timestamps_y[:num_train]
        timestamps_x_val, timestamps_y_val = timestamps_x[num_train:num_train+num_val], timestamps_y[num_train:num_train+num_val]
        timestamps_x_test, timestamps_y_test = timestamps_x[-num_test:], timestamps_y[-num_test:]

    for s in ["train", "val", "test"]:
        _x, _y = locals()["x_" + s], locals()["y_" + s]
        if not timestamps is None:
            _timestamps_x, _timestamps_y = locals()["timestamps_x_" + s], locals()["timestamps_y_" + s]

        if verbose:
            print("{} x: {}, y: {}".format(s, _x.shape, _y.shape))
            if not timestamps is None:
                print("{} timestamps x: {}, timestamps y: {}".format(s, _timestamps_x.shape, _timestamps_y.shape))

        save_dict = {
                "x": _x,
                "y": _y,
                "x_offsets": x_offsets.reshape(list(x_offsets.shape) + [1]),
                "y_offsets": y_offsets.reshape(list(y_offsets.shape) + [1])
                }

        if not timestamps is None:
            save_dict.update({
                "timestamps_x": _timestamps_x,
                "timestamps_y": _timestamps_y,
            })

        utils.verify_or_create_path(output_path)
        file_name = "{}.npz".format(s)
        np.savez_compressed(os.path.join(output_path, file_name), **save_dict)

        if verbose:
            print("Saved {} to {}".format(file_name, output_path))

def generate_timeseries_splits(detector_data_processed, output_path, timestamps=None, verbose=0):
    if verbose:
        print("Detector data shape: {}".format(detector_data_processed.shape))
        if not timestamps is None:
            print("Timestamps array shape: {}".format(timestamps.shape))

    # Write the data into npz file
    # 7/10 training, 1/10 validation, 2/10 test
    num_samples = detector_data_processed.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    train = detector_data_processed[:num_train]
    val = detector_data_processed[num_train:num_train+num_val]
    test = detector_data_processed[-num_test:]

    if not timestamps is None:
        timestamps_train = timestamps[:num_train]
        timestamps_val = timestamps[num_train:num_train+num_val]
        timestamps_test = timestamps[-num_test:]

    for s in ["train", "val", "test"]:
        _detector_data = locals()[s]
        if not timestamps is None:
            _timestamps = locals()["timestamps_" + s]

        if verbose:
            print("{}: {}".format(s, _detector_data.shape))
            if not timestamps is None:
                print("{} timestamps: {}".format(s, _timestamps.shape))

        save_dict = {"data": _detector_data}

        if not timestamps is None:
            save_dict.update({"timestamps": _timestamps})

        utils.verify_or_create_path(output_path)
        file_name = "{}_ts.npz".format(s)
        np.savez_compressed(os.path.join(output_path, file_name), **save_dict)

        if verbose:
            print("Saved {} to {}".format(file_name, output_path))

def save_timestamps(timestamps, output_path, timeseries=False):
    utils.verify_or_create_path(output_path)
    if timeseries:
        file_name = "timestamps_ts.npz"
    else:
        file_name = "timestamps.npz"

    np.savez_compressed(os.path.join(output_path, file_name), timestamps=timestamps)


def main(args):
    features = args.features
    x_offset = args.x_offset
    y_offset = args.y_offset
    verbose = args.verbose or 0

    if args.dummy:
        if not args.dummy in DUMMY_DATA_TYPES:
            raise Exception("{} is not a supported dummy data type".format(args.dummy))
            return

        dummy_shape = ast.literal_eval(args.dummy_shape or "(4706, 0, 16, 2)")
        dummy_shape = (dummy_shape[0], x_offset + y_offset, *dummy_shape[2:])
        dummy_data = generate_dummy_data(args.dummy, dummy_shape, verbose=verbose)
        if args.output_dir:
            generate_array_splits(dummy_data, x_offset, y_offset, args.output_dir, verbose=verbose)
    else:
        detector_list_path = args.detector_list_path
        plan_name = args.plan_name
        split_by_day = args.split_by_day
        weekday_only = args.weekday_only

        is_timeseries = args.timeseries

        if is_timeseries:
            stretch_length = x_offset + y_offset
            start_time_buffer = 0
            end_time_buffer = 0
        else:
            start_time_buffer = args.start_time_buffer
            end_time_buffer = args.end_time_buffer

        detector_list = get_sensors_list(detector_list_path)
        detector_data = get_detector_data(detector_list, plan=plan_name, features=features, weekday_only=weekday_only,
                                          start_time_buffer=start_time_buffer, end_time_buffer=end_time_buffer)

        if split_by_day:
            detector_data_array = [detector_data[detector_data["Time"].dt.dayofweek == i] for i in range(7)]
            weekdays = range(7)
        else:
            detector_data_array = [detector_data]
            weekdays = [None]

        if is_timeseries:
            process_detector_data_func = lambda i:\
                process_timeseries_detector_data(detector_data_array[i], detector_list, stretch_length, verbose=verbose)
            generate_splits_func = lambda detector_data_processed, output_dir, timestamps:\
                generate_timeseries_splits(detector_data_processed, output_dir, timestamps=timestamps, verbose=verbose)
        else:
            process_detector_data_func = lambda i:\
                process_array_detector_data(detector_data_array[i], detector_list, x_offset + y_offset, verbose=verbose)
            generate_splits_func = lambda detector_data_processed, output_dir, timestamps:\
                generate_array_splits(detector_data_processed, x_offset, y_offset, output_dir, timestamps=timestamps,
                              verbose=verbose)

        for i in range(len(detector_data_array)):
            detector_data_processed, _, timestamps = process_detector_data_func(i)

            subdir = utils.get_subdir(plan_name, x_offset, y_offset, start_time_buffer=start_time_buffer,
                                      end_time_buffer=end_time_buffer, weekday=weekdays[i])
            subdir += "_sensor_data"

            if args.output_dir:
                output_dir = os.path.join(args.output_dir, subdir)
                generate_splits_func(detector_data_processed, output_dir, timestamps)

            if args.timestamps_dir:
                timestamps_dir = os.path.join(args.timestamps_dir, subdir)
                save_timestamps(timestamps, timestamps_dir, timeseries=is_timeseries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_list_path", "--dl", help="list of sensors to generate data for")
    parser.add_argument("--plan_name", help="name of plan: E, P1, P2, or P3; or each to loop over all plans")
    parser.add_argument("--features", "-f", nargs="+", default=["Flow"], help="Flow, occupancy, or speed")
    parser.add_argument("--split_by_day", action="store_true", help="split by day of the week")
    parser.add_argument("--weekday_only", action="store_true", help="remove weekend data")
    parser.add_argument("--start_time_buffer", type=int, default=0, help="extra time steps before plan starts to include")
    parser.add_argument("--end_time_buffer", type=int, default=0, help="extra time steps after plan ends to include")
    parser.add_argument("--x_offset", type=int, default=12, help="number of time steps to use for training")
    parser.add_argument("--y_offset", type=int, default=12, help="number of time steps to predict ahead")
    parser.add_argument("--output_dir", help="output directory for npz data files")
    parser.add_argument("--timestamps_dir", help="output directory for npz timestamps")
    parser.add_argument("--timeseries", action="store_true", help="output data as time series instead of arrays. If specified, buffers are ignored, but x_offset and y_offset are used to get stretch length.")
    parser.add_argument("--dummy", help="overrides other arguments. Generate dummy training data with the specified pattern: {}".format(DUMMY_DATA_TYPES))
    parser.add_argument("--dummy_shape", help="the shape of generated dummy data, if dummy is specified. Second value is 0, since it will be replaced by x_offset and y_offset.".format(DUMMY_DATA_TYPES))
    parser.add_argument("-v", "--verbose", action="count", help="verbosity of script")
    args = parser.parse_args()

    plan_names = [args.plan_name] if args.plan_name != "each" else ["P1", "P2", "P3", "E"]

    for plan_name in plan_names:
        args.plan_name = plan_name
        main(args)
