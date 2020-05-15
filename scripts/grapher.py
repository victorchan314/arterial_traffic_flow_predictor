import argparse
import os
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

from lib import data_utils
from lib import utils
from scripts.generate_training_data import get_detector_data


sns_colors = sns.color_palette("colorblind")
cycle = [sns_colors[i] for i in [2, 9, 6, 4, 8, 0, 3, 1, 7, 5]]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=cycle)
colors = cycle

DETECTOR_DATA_FREQUENCY = dt.timedelta(minutes=5)


def get_one_detector_data(detector):
    data_P1 = get_detector_data([detector], plan="P1", features=["Flow", "Occupancy", "Speed"])
    data_P2 = get_detector_data([detector], plan="P2", features=["Flow", "Occupancy", "Speed"])
    data_P3 = get_detector_data([detector], plan="P3", features=["Flow", "Occupancy", "Speed"])

    return data_P1, data_P2, data_P3

def plot_data_over_time(ax, x, y, title=None, xlabel="Date", ylabel=None, color=None):
    ax.set_title(title)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.plot(x, y, c=color)

def plot_fundamental_diagram_color_by_plan(detector_id):
    data_P1, data_P2, data_P3 = get_one_detector_data(detector_id)
    flow_P1 = data_P1["Flow"]
    flow_P2 = data_P2["Flow"]
    flow_P3 = data_P3["Flow"]
    occupancy_P1 = data_P1["Occupancy"] / 36
    occupancy_P2 = data_P2["Occupancy"] / 36
    occupancy_P3 = data_P3["Occupancy"] / 36

    sns.scatterplot(occupancy_P2, flow_P2, linewidth=0, s=10, label="Morning peak")
    sns.scatterplot(occupancy_P1, flow_P1, linewidth=0, s=10, label="Off peak")
    sns.scatterplot(occupancy_P3, flow_P3, linewidth=0, s=10, label="Evening peak")

    plt.title("Detector {} Fundamental Diagram".format(detector_id))
    plt.xlabel("Occupancy (%)")
    plt.ylabel("Flow (vph)")
    plt.xlim(0, 100)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()

def graph_detector_data_time_series(detector_id, xticks_datetime_precision="D", num_xticks=12):
    data_P1, data_P2, data_P3 = get_one_detector_data(detector_id)

    data_P1 = data_P1[data_P1["Time"].dt.month == 8]
    data_P2 = data_P2[data_P2["Time"].dt.month == 8]
    data_P3 = data_P3[data_P3["Time"].dt.month == 8]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.autofmt_xdate()
    fig.suptitle("Detector {} Data in the Month of August".format(detector_id))
    features = ["Flow", "Occupancy"]
    ylabels = ["Flow (vph)", "Occupancy (%)"]
    scales = [1, 36]

    for ax, feature, ylabel, scale in zip(axes, features, ylabels, scales):
        full_data = pd.concat((data_P1, data_P2, data_P3), ignore_index=True)
        full_data.sort_values("Time", inplace=True)

        x = full_data["Time"]
        y = full_data[feature] / scale

        xticks = np.arange(x.shape[0])
        xticks_labels = np.datetime_as_string(x, unit=xticks_datetime_precision)
        xticks_locs = [xticks[x.shape[0] // num_xticks * i] for i in range(num_xticks)] + [xticks[-1]]
        xticks_spaced_labels = [xticks_labels[x.shape[0] // num_xticks * i] for i in range(num_xticks)] + [xticks_labels[-1]]

        p2_starts = ((x.dt.hour == 6) & (x.dt.minute == 0)).values
        p11_starts = ((x.dt.hour == 9) & (x.dt.minute == 0)).values
        p3_starts = ((x.dt.hour == 15) & (x.dt.minute == 30)).values
        p12_starts = ((x.dt.hour == 19) & (x.dt.minute == 0)).values
        starts = xticks[p2_starts | p11_starts | p3_starts | p12_starts]
        starts = np.append(starts, x.shape[0] - 1)

        for i in range(starts.shape[0] - 1):
            if p2_starts[starts[i]]:
                color = 0
            elif p3_starts[starts[i]]:
                color = 1
            else:
                color = 2

            ax.plot(xticks[starts[i]:starts[i + 1] + 1], y[starts[i]:starts[i + 1] + 1], c=colors[color])

        ax.set(xlabel="Time", ylabel=ylabel)
        ax.set_xticks(xticks_locs)
        ax.set_xticklabels(xticks_spaced_labels)

        legend_elements = [Line2D([0], [0], color=colors[0], label="Morning peak"),
                           Line2D([0], [0], color=colors[1], label="Evening peak"),
                           Line2D([0], [0], color=colors[2], label="Off peak")]
        box = ax.get_position()
        ax.set_position([box.x0 - box.width * 0.05, box.y0, box.width, box.height])
        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(0.95, 0.5))

    plt.show()

def extract_flat_data(timestamps_array, groundtruth_array):
    stretches = data_utils.get_stretches(timestamps_array[:, 0], DETECTOR_DATA_FREQUENCY)
    timestamps = data_utils.flatten_circulant_like_matrix_by_stretches(timestamps_array, stretches)
    groundtruth_array_transposed = np.swapaxes(groundtruth_array, 0, 1)
    groundtruth = data_utils.flatten_circulant_like_matrix_by_stretches(groundtruth_array_transposed, stretches)

    return timestamps, groundtruth

def fit_dates_to_timestamps(full_x, original_x, new_x):
    return new_x[np.isin(full_x, original_x)]

def graph_predictions(y, y_hat, x, x_array, sensor, step=4, title=None, num_xticks=12, xticks_datetime_precision="D"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))
    fig.suptitle(title)

    xticks = np.arange(x.shape[0])
    xticks_labels = np.datetime_as_string(x, unit=xticks_datetime_precision)
    xticks_locs = [xticks[x.shape[0] // num_xticks * i] for i in range(num_xticks)] + [xticks[-1]]
    xticks_spaced_labels = [xticks_labels[x.shape[0] // num_xticks * i] for i in range(num_xticks)] + [xticks_labels[-1]]

    for ax in (ax1, ax2):
        ax.set(xlabel="Time", ylabel="Flow (vph)")
        ax.set_xticks(xticks_locs)
        ax.set_xticklabels(xticks_spaced_labels)

        ax.plot(xticks, y[:, sensor], label="Ground Truth")

    cmap = plt.get_cmap("jet")
    reordered_colors =[6, 5, 2]

    for i, h in enumerate([1, 3, 6]):
        stretches = data_utils.get_stretches(x_array[:, h - 1], DETECTOR_DATA_FREQUENCY)
        color = colors[reordered_colors[i]]

        for start, end in stretches:
            x_stretch = x_array[start:end, h - 1]
            y_hat_stretch = y_hat[h - 1, start:end, sensor]
            x_stretch_range = fit_dates_to_timestamps(x, x_stretch, xticks)

            if start == 0:
                label = "{} min predictions".format(h * 5)
            else:
                label = None

            ax1.plot(x_stretch_range, y_hat_stretch, label=label, c=color, alpha=0.6)

    ax1.legend()
    ax1.set_xlim(xticks.shape[0] * 4 / 7 - 13, xticks.shape[0])

    ax2.legend()
    stretches = data_utils.get_stretches(x_array[:, 0], DETECTOR_DATA_FREQUENCY)
    for i, (start, end) in enumerate(stretches):
        for t in range(start, end, step):
            color = cmap((t - start) / (end - start - 1))
            x_stretch = x_array[t, :]
            y_hat_stretch = y_hat[:, t, sensor]
            x_stretch_range = fit_dates_to_timestamps(x, x_stretch, xticks)

            if start == 0:
                label = "Time {} predictions".format(t)
            else:
                label = None

            ax2.plot(x_stretch_range, y_hat_stretch, label=label, c=color, alpha=0.6)

    ax2.set_xlim(xticks.shape[0] * 4 / 8, xticks.shape[0] * 6 / 8)

    plt.show()

def graph_predictions_and_baselines(y, x, x_array, y_hats, sensor, step=4, title=None, num_xticks=12, xticks_datetime_precision="D"):
    plt.figure(figsize=(16, 5))
    plt.title(title)

    xticks = np.arange(x.shape[0])
    xticks_labels = np.datetime_as_string(x, unit=xticks_datetime_precision)
    xticks_locs = [xticks[x.shape[0] // num_xticks * i] for i in range(num_xticks)] + [xticks[-1]]
    xticks_spaced_labels = [xticks_labels[x.shape[0] // num_xticks * i] for i in range(num_xticks)] + [xticks_labels[-1]]

    plt.xlabel("Time")
    plt.ylabel("Flow (vph)")
    plt.xticks(xticks_locs, xticks_spaced_labels)

    plt.plot(xticks, y[:, sensor], label="Ground Truth")

    reordered_colors =[6, 5, 2]

    #for i, h in enumerate([1, 3, 6]):
    h = 6
    for i, (y_hat, model_name) in enumerate(zip(y_hats, ["DCRNN", "GRU", "ARIMAX"])):
        stretches = data_utils.get_stretches(x_array[:, h - 1], DETECTOR_DATA_FREQUENCY)
        color = colors[reordered_colors[i]]
        #color = colors[i + 1]

        for start, end in stretches:
            x_stretch = x_array[start:end, h - 1]
            y_hat_stretch = y_hat[h - 1, start:end, sensor]
            x_stretch_range = fit_dates_to_timestamps(x, x_stretch, xticks)

            if start == 0:
                label = "{}".format(model_name)
            else:
                label = None

            plt.plot(x_stretch_range, y_hat_stretch, label=label, c=color, alpha=0.7)

    plt.legend()
    plt.xlim(xticks.shape[0] * 4 / 7 - 13, xticks.shape[0])

    plt.show()

def graph4():
    logdirs = [x for x in os.listdir("experiments") if x.startswith("full-information_")]
    if len(logdirs) > 1:
        raise ValueError("More than 1 full-information logdir: {}".format(logdirs))

    predictions_path = os.path.join("experiments", logdirs[0], "experiments", "dcrnn", "P2_o12_h6_sb12", "predictions.npz")
    timestamps_path = os.path.join("experiments", logdirs[0], "inputs", "sensor_data", "P2_o12_h6_sb12_sensor_data", "test.npz")

    groundtruth, predictions = utils.load_predictions(predictions_path)
    groundtruth = groundtruth[..., 0]
    predictions = predictions[..., 0]

    timestamps_array = np.load(timestamps_path)["timestamps_y"]
    timestamps, groundtruth = extract_flat_data(timestamps_array, groundtruth)

    sensors = {508302: 0, 508306: 1}
    detector = 508306
    sensor = sensors[detector]

    graph_predictions(groundtruth, predictions, timestamps, timestamps_array, sensor=sensor,
                      title="Detector {} Full Information DCRNN Test Predictions".format(detector))

def graph5():
    logdirs = [x for x in os.listdir("experiments") if x.startswith("full-information_")]
    if len(logdirs) > 1:
        raise ValueError("More than 1 full-information logdir: {}".format(logdirs))

    dcrnn_predictions_path = os.path.join("experiments", logdirs[0], "experiments", "dcrnn", "P2_o12_h6_sb12", "predictions.npz")
    gru_predictions_path = os.path.join("experiments", logdirs[0], "experiments", "baselines", "rnn", "P2_o12_h6_sb12", "predictions.npz")
    arimax_predictions_path = os.path.join("experiments", logdirs[0], "experiments", "baselines", "arimax", "P2_o12_h6_sb12", "predictions.npz")
    timestamps_path = os.path.join("experiments", logdirs[0], "inputs", "sensor_data", "P2_o12_h6_sb12_sensor_data", "test.npz")

    groundtruth, dcrnn_predictions = utils.load_predictions(dcrnn_predictions_path)
    _, gru_predictions = utils.load_predictions(gru_predictions_path)
    _, arimax_predictions = utils.load_predictions(arimax_predictions_path)

    groundtruth = groundtruth[..., 0]
    dcrnn_predictions = dcrnn_predictions[..., 0]
    gru_predictions = gru_predictions[..., 0]

    timestamps_array = np.load(timestamps_path)["timestamps_y"]
    timestamps, groundtruth = extract_flat_data(timestamps_array, groundtruth)

    sensors = {508302: 0, 508306: 1}
    detector = 508306
    sensor = sensors[detector]

    graph_predictions_and_baselines(groundtruth, timestamps, timestamps_array,
                                    [dcrnn_predictions, gru_predictions, arimax_predictions],
                                    sensor=sensor,
                                    title="Detector {} Full Information Test Predictions".format(detector))


def main(args):
    graph = args.graph

    if graph == 1:
        graph_detector_data_time_series(508302)
    elif graph == 2:
        plot_fundamental_diagram_color_by_plan(508302)
        plot_fundamental_diagram_color_by_plan(508306)
    elif graph == 3:
        graph_detector_data_time_series(608107)
    elif graph == 4:
        graph4()
    elif graph == 5:
        graph5()
    else:
        raise ValueError("Invalid graph ID")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=int, help="id of graph to generate")
    args = parser.parse_args()

    main(args)
