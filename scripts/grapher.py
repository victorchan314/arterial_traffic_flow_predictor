import argparse
import os
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

from scripts.generate_training_data import get_detector_data


sns_colors = sns.color_palette("colorblind")
cycle = [sns_colors[i] for i in [2, 9, 6, 4, 8, 0, 3, 1, 7, 5]]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=cycle)
colors = cycle


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


def main(args):
    graph = args.graph

    if graph == 1:
        graph_detector_data_time_series(508302)
    elif graph == 2:
        plot_fundamental_diagram_color_by_plan(508302)
        plot_fundamental_diagram_color_by_plan(508306)
    elif graph == 3:
        graph_detector_data_time_series(608107)
    else:
        raise ValueError("Invalid graph ID")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=int, help="id of graph to generate")
    args = parser.parse_args()

    main(args)
