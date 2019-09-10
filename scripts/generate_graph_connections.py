import argparse
import os
import sys

parent_dir = os.path.abspath("/Users/victorchan/Dropbox/UndergradResearch/Victor/Code")
sys.path.append(parent_dir)

#import numpy as np
import pandas as pd

from lib import mysql_utils

INTERSECTIONS = [5083, 5082, 6081, 5091, 5072]

#PHASE_TIMINGS_QUERY =\
#"SELECT IntersectionID, EndDate, EndTime, PhaseActualGreenTime FROM phase_time_2017 \
#WHERE (IntersectionID = 5083 OR IntersectionID = 6081 OR IntersectionID = 6082)\
#    AND (EndDate > (\
#        SELECT MAX(t.EndDate) FROM (SELECT MIN(EndDate) AS EndDate FROM phase_time_2017\
#        WHERE IntersectionID = 5083 OR IntersectionID = 6081 OR IntersectionID = 6082\
#        GROUP BY IntersectionID) AS t)\
#    OR EndTime > (\
#        SELECT MAX(t2.EndTime) FROM (SELECT Min(EndTime) AS EndTime FROM phase_time_2017\
#        WHERE IntersectionID = 5083 OR IntersectionID = 6081 OR IntersectionID = 6082\
#        AND EndDate IN\
#            (SELECT MAX(t.EndDate) FROM\
#            (SELECT MIN(EndDate) AS EndDate FROM phase_time_2017\
#            WHERE IntersectionID = 5083 OR IntersectionID = 6081 OR IntersectionID = 6082\
#            GROUP BY IntersectionID) AS t)\
#        GROUP BY IntersectionID) AS t2\
#    )\
#);"

DETECTOR_INVENTORY_QUERY =\
"SELECT IF(SensorID < 10, CONCAT(IntersectionID, 0, SensorID), CONCAT(IntersectionID, SensorID)) AS Sensor,\
IntersectionID, SensorID, Direction, Movement FROM detector_inventory \
WHERE Movement LIKE 'Advanced%' \
AND ({});"\
.format(" OR ".join(["IntersectionID = {}".format(intersection) for intersection in INTERSECTIONS]))



def generate_sensors_advanced_file(detector_inventory, path):
    with open(path, "w") as f:
        f.write(",".join(detector_inventory.index.values))
        f.close()

def generate_adjacency_matrix(detector_inventory, edges, phases, phase_plans, plan_name):
    edges["Distance"] = 0

    for i in range(edges.shape[0]):
        sensor_from = edges.iloc[i, 0]
        sensor_to = edges.iloc[i, 1]
        intersection_from, direction_from = detector_inventory.loc[str(sensor_from), ["IntersectionID", "Direction"]]
        intersection_to, direction_to = detector_inventory.loc[str(sensor_to), ["IntersectionID", "Direction"]]

        if intersection_from == intersection_to:
            edges.iloc[i, 2] = 1

        edge_phases = phases[(phases["From"] == direction_from) & (phases["To"] == direction_to)]["Phase"].values
        edge_plans = phase_plans[phase_plans["Intersection"] == intersection_from]
        edge_plans = edge_plans if not plan_name else edge_plans[edge_plans["PlanName"] == plan_name]
        total_hours = 0.0
        edge_greentime_fraction = 0.0

        for j in range(edge_plans.shape[0]):
            plan = edge_plans.iloc[j]
            plan_cycle = plan["Cycle"]
            green_times = plan["PhasePlannedGreenTime"].split(";")
            
            for k in edge_phases:
                phase_weight = plan["EndTime"] - plan["StartTime"]
                phase_greentime_fraction = int(green_times[k-1]) / plan_cycle
                total_hours += phase_weight

                edge_greentime_fraction += phase_weight * phase_greentime_fraction

        edge_greentime_fraction = edge_greentime_fraction / total_hours if total_hours > 0 else 0
        edges.loc[edges.index[i], "Distance"] = edge_greentime_fraction

    return edges



SENSORS_ADVANCED = False
ADJACENCY_MATRIX = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", help="intersection to focus on. Assumes all relevant data/model/ files have proper suffix.")
    parser.add_argument("--adjacency_matrix_path", help="output file for adjacency matrix, if one is generated")
    parser.add_argument("--plan_name", help="name of plan: E, P1, P2, or P3")
    args = parser.parse_args()

    intersection = "_{}".format(args.intersection) if args.intersection else ""
    adjacency_matrix_path = args.adjacency_matrix_path or "data/model/sensor_distances{}.csv".format(intersection)
    plan_name = args.plan_name

    #phase_timings = mysql_utils.execute_query(PHASE_TIMINGS_QUERY)
    detector_inventory = mysql_utils.execute_query(DETECTOR_INVENTORY_QUERY)
    
    #if phase_timings == None or detector_inventory == None:
    if detector_inventory == None:
        sys.exit()

    #phase_timings = pd.DataFrame(phase_timings, columns=["IntersectionID", "EndDate", "EndTime", "PhaseTimings"])
    detector_inventory = pd.DataFrame(detector_inventory, columns=["Sensor", "IntersectionID", "SensorID", "Direction", "Movement"])
    detector_inventory.set_index("Sensor", inplace=True)
    edges = pd.read_csv("data/model/edges{}.csv".format(intersection), header=None)
    phases = pd.read_csv("data/model/phases.csv")
    phase_plans = pd.read_csv("data/model/phase_plans{}.csv".format(intersection))

    if SENSORS_ADVANCED:
        generate_sensors_advanced_file(detector_inventory, "data/model/sensors_advanced{}.txt".format(intersection))
    
    edges = generate_adjacency_matrix(detector_inventory, edges, phases, phase_plans, plan_name)

    if ADJACENCY_MATRIX:
        edges.to_csv(adjacency_matrix_path, header=False, index=False)
