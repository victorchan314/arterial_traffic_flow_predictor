import os
import sys

parent_dir = os.path.abspath("/Users/victorchan/Dropbox/UndergradResearch/Victor/Code")
sys.path.append(parent_dir)

#import numpy as np
import pandas as pd

import mysql_utils

PHASE_TIMINGS_QUERY =\
"SELECT IntersectionID, EndDate, EndTime, PhaseActualGreenTime FROM phase_time_2017 \
WHERE (IntersectionID = 5083 OR IntersectionID = 6081 OR IntersectionID = 6082)\
    AND (EndDate > (\
        SELECT MAX(t.EndDate) FROM (SELECT MIN(EndDate) AS EndDate FROM phase_time_2017\
        WHERE IntersectionID = 5083 OR IntersectionID = 6081 OR IntersectionID = 6082\
        GROUP BY IntersectionID) AS t)\
    OR EndTime > (\
        SELECT MAX(t2.EndTime) FROM (SELECT Min(EndTime) AS EndTime FROM phase_time_2017\
        WHERE IntersectionID = 5083 OR IntersectionID = 6081 OR IntersectionID = 6082\
        AND EndDate IN\
            (SELECT MAX(t.EndDate) FROM\
            (SELECT MIN(EndDate) AS EndDate FROM phase_time_2017\
            WHERE IntersectionID = 5083 OR IntersectionID = 6081 OR IntersectionID = 6082\
            GROUP BY IntersectionID) AS t)\
        GROUP BY IntersectionID) AS t2\
    )\
);"

DETECTOR_INVENTORY_QUERY =\
"SELECT IF(SensorID < 10, CONCAT(IntersectionID, 0, SensorID), CONCAT(IntersectionID, SensorID)) AS Sensor,\
IntersectionID, SensorID, Direction, Movement FROM detector_inventory \
WHERE Movement = 'Advanced' \
AND (IntersectionID = 5083 OR IntersectionID = 6081 OR IntersectionID = 6082);"



generate_sensors_advanced = False

if __name__ == "__main__":
    phase_timings = mysql_utils.execute_query(PHASE_TIMINGS_QUERY)
    detector_inventory = mysql_utils.execute_query(DETECTOR_INVENTORY_QUERY)
    
    if phase_timings == None or detector_inventory == None:
        sys.exit()

    phase_timings = pd.DataFrame(phase_timings, columns=["IntersectionID", "EndDate", "EndTime", "PhaseTimings"])
    detector_inventory = pd.DataFrame(detector_inventory, columns=["Sensor", "IntersectionID", "SensorID", "Direction", "Movement"])
    detector_inventory.set_index("Sensor", inplace=True)
    edges = pd.read_csv("data/model/edges.csv", header=None)
    phases = pd.read_csv("data/model/phases.csv")

    if generate_sensors_advanced:
        with open("data/model/sensors_advanced.txt", "w") as f:
            f.write(",".join(detector_inventory.index.values))
            f.close()
    
    #print(phase_timings.head())
    print(detector_inventory)
