import argparse
import os
import sys

parent_dir = os.path.abspath("/Users/victorchan/Dropbox/UndergradResearch/Victor/Code")
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import datetime as dt

import mysql_utils



PHASE_PLANS_PATH = "data/model/phase_plans{}.csv"
DETECTOR_LIST_PATH = "data/model/sensors_advanced{}.txt"
DETECTOR_DATA_QUERY = "SELECT DetectorID, Year, Month, Day, Time / 86400, Volume AS Flow FROM detector_data_processed_2017 WHERE ({})"



def get_sensors_list(path):
    with open(path, "r") as f:
        sensors = f.read()
        f.close()

    sensors_list = sensors[:-1].split(",")

    return sensors_list

def get_detector_data(detector_list, intersection="", plan=None, limit=np.inf):
    query = DETECTOR_DATA_QUERY.format(" OR ".join(["DetectorID = {}".format(d) for d in detector_list]))
    if plan:
        phase_plans = pd.read_csv(PHASE_PLANS_PATH.format(intersection))
        relevant_phase_plans = phase_plans[phase_plans["PlanName"] == plan]
        intervals = relevant_phase_plans.loc[:, ["StartTime", "EndTime"]].values * 3600
        query += " AND ({})".format(" OR ".join(["(Time >= {} AND Time < {})".format(interval[0], interval[1]) for interval in intervals]))

    query_results = mysql_utils.execute_query(query)
    
    results = []

    for row in query_results:
        results.append([row[0], dt.date(row[1], row[2], row[3]), row[4], row[5]])

        if len(results) >= limit:
            break

    detector_data = pd.DataFrame(results)#, columns=["DetectorID", "Date", "Time", "Flow"])

    return detector_data

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
    intersection = "_{}".format(args.intersection) if args.intersection else ""
    output_path = args.output_dir or "data"
    plan_name = args.plan_name

    detector_list = get_sensors_list(DETECTOR_LIST_PATH.format(intersection))
    detector_data = get_detector_data(detector_list, intersection=intersection, plan=plan_name, limit=50)

    print(detector_data.iloc[-1])

    #generate_splits(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", help="intersection to focus on. Assumes all relevant data/model/ files have proper suffix.")
    parser.add_argument("--output_dir", help="output directory for")
    parser.add_argument("--plan_name", help="name of plan: E, P1, P2, or P3")
    args = parser.parse_args()

    main(args)
