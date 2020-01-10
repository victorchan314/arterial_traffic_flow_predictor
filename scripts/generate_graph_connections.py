import argparse
import os
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

import pandas as pd

from lib import mysql_utils

INTERSECTIONS = [5083, 5082, 6081, 5091, 5072, 6082, 6083]
INTERSECTION_OR_BOOLEAN = " OR ".join(["IntersectionID = {}".format(intersection) for intersection in INTERSECTIONS])
MIN_RELEVANT_DATE_SUBQUERY = \
    "SELECT MAX(t.EndDate) FROM (\
        SELECT MIN(EndDate) AS EndDate FROM phase_time_2017\
            WHERE {}\
            GROUP BY IntersectionID\
    ) AS t"\
        .format(INTERSECTION_OR_BOOLEAN)
#PHASE_TIMINGS_QUERY = \
#    "SELECT IntersectionID, EndDate, EndTime, PhaseActualGreenTime FROM phase_time_2017 \
#    WHERE ({})\
#        AND (EndDate > ({})\
#        OR EndTime > (\
#            SELECT MAX(t2.EndTime) FROM (SELECT Min(EndTime) AS EndTime FROM phase_time_2017\
#            WHERE {}\
#            AND EndDate IN ({})\
#            GROUP BY IntersectionID) AS t2\
#        )\
#    );"\
#        .format(INTERSECTION_OR_BOOLEAN, MIN_RELEVANT_DATE_SUBQUERY, INTERSECTION_OR_BOOLEAN, MIN_RELEVANT_DATE_SUBQUERY)
DETECTOR_INVENTORY_QUERY = \
    "SELECT IF(SensorID < 10, CONCAT(IntersectionID, 0, SensorID), CONCAT(IntersectionID, SensorID)) AS Sensor,\
    IntersectionID, SensorID, Direction, Movement FROM detector_inventory\
    WHERE {}"\
        .format(INTERSECTION_OR_BOOLEAN)



def generate_graph_connections(detector_inventory, edges, phases, phase_plans, plan_name):
    edges.loc[:, "Distance"] = 0

    for i in range(edges.shape[0]):
        sensor_from = edges.iloc[i, 0]
        sensor_to = edges.iloc[i, 1]
        intersection_from, direction_from = detector_inventory.loc[str(sensor_from), ["IntersectionID", "Direction"]]
        intersection_to, direction_to = detector_inventory.loc[str(sensor_to), ["IntersectionID", "Direction"]]

        if intersection_from == intersection_to:
            edges.iloc[i, 2] = 1
            continue

        edge_phases = phases[(phases["From"] == direction_from) & (phases["To"] == direction_to)]["Phase"].values
        edge_plans = phase_plans[phase_plans["Intersection"] == intersection_from]
        edge_plans = edge_plans if not plan_name else edge_plans[edge_plans["PlanName"] == plan_name]
        total_hours = 0.0
        edge_green_time_fraction = 0.0

        for j in range(edge_plans.shape[0]):
            plan = edge_plans.iloc[j]
            plan_cycle = plan["Cycle"]
            green_times = plan["PhasePlannedGreenTime"].split(";")
            yr_times = plan["PhasePlannedYRTime"].split(";")
            phase_weight = plan["EndTime"] - plan["StartTime"]
            total_hours += phase_weight

            for k in edge_phases:
                phase_green_time = int(green_times[k-1])
                phase_yr_time = int(yr_times[k-1])

                edge_green_time_fraction += phase_weight * (phase_green_time + phase_yr_time) / plan_cycle

        edge_green_time_fraction = edge_green_time_fraction / total_hours if total_hours > 0 else 0
        edges.loc[edges.index[i], "Distance"] = edge_green_time_fraction

    return edges



def main(args):
    plan_name = args.plan_name
    detector_list = args.detector_list
    adjacency_matrix_path = args.adjacency_matrix_path

    #phase_timings = mysql_utils.execute_query(PHASE_TIMINGS_QUERY)
    detector_inventory = mysql_utils.execute_query(DETECTOR_INVENTORY_QUERY)

    # if phase_timings == None or detector_inventory == None:
    if detector_inventory == None:
        sys.exit()

    # phase_timings = pd.DataFrame(phase_timings, columns=["IntersectionID", "EndDate", "EndTime", "PhaseTimings"])
    detector_inventory = pd.DataFrame(detector_inventory,
                                      columns=["Sensor", "IntersectionID", "SensorID", "Direction", "Movement"])
    detector_inventory.set_index("Sensor", inplace=True)
    edges = pd.read_csv("data/inputs/model/edges.csv", header=None)
    phases = pd.read_csv("data/inputs/model/phases.csv")
    phase_plans = pd.read_csv("data/inputs/model/phase_plans.csv")

    # detector_list = detector_inventory.index.values
    # print(detector_list)
    # if args.detector_list:
    #     with open("data/inputs/model/sensors_advanced.txt", "w") as f:
    #         f.write(",".join(detector_list))
    #         f.close()

    relevant_edges = edges[edges[0].isin(detector_list) & edges[1].isin(detector_list)].copy()

    graph_connections = generate_graph_connections(detector_inventory, relevant_edges, phases, phase_plans, plan_name)

    if args.adjacency_matrix_path:
        graph_connections.to_csv(adjacency_matrix_path, header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_name", help="name of plan: E, P1, P2, or P3")
    parser.add_argument("--detector_list", "--dl", nargs="+", help="list of sensors to generate connections for")
    parser.add_argument("--adjacency_matrix_path", help="output file for adjacency matrix, if one is generated")
    args = parser.parse_args()

    main(args)
