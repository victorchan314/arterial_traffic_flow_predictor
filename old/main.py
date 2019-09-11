import sys

import numpy as np
import datetime as dt

from config.config import config
from lib import visualization, mysql_utils as mysql

#DETECTOR_DATA_TABLE = "detector_data_processed_2017_1"
DETECTOR_DATA_TABLE = "detector_data_processed_2017"
DETECTOR_ID = "608219"
DETECTOR_DATA_QUERY = "SELECT DetectorID, Year, Month, Day, Time, Volume, Occupancy \
        FROM {} AS DD NATURAL JOIN detector_health AS DH \
        WHERE DetectorID = {} AND Health = 1 \
        ORDER BY Year, Month, Day, Time;"


def query_detector_data(cursor, table, detector_id, graph=False):
    query = DETECTOR_DATA_QUERY.format(table, detector_id)

    cursor = mysql.query(cursor, query)
    
    if cursor == None:
        return

    time = []
    volume = []
    occupancy = []
    #speed = []

    for row in cursor:
        d = dt.datetime(row[1], row[2], row[3], row[4] // 3600, (row[4] % 3600) // 60, row[4] % 60)
        time.append(d)

        volume.append(row[5])
        occupancy.append(row[6])
        #speed.append(row[7])

    time = np.array(time)
    volume = np.array(volume)
    occupancy = np.array(occupancy)
    occupancy_percentage = occupancy / 3600 * 100
    #speed = np.array(speed)

    if graph:
        visualization.plot_data_over_time(time, volume, title="Detector {} Volume 2017".format(detector_id), ylabel="Volume (vph)", figsize=(12, 5))
        visualization.plot_data_over_time(time, occupancy, title="Detector {} Occupancy 2017".format(detector_id), ylabel="Occupancy (s)", figsize=(12, 5))
        #visualization.plot_data_over_time(time, speed, title="Detector {} Speed 2017".format(detector_id), ylabel="Speed", figsize=(12, 5))
        visualization.plot_data_over_time(time, occupancy_percentage, title="Detector {} Occupancy 2017".format(detector_id), ylabel="Occupancy (%)", figsize=(12, 5))
        visualization.plot_fundamental_diagram(volume, occupancy_percentage, title="Detector {} Flow-Occupancy Diagram 2017".format(detector_id))

    return time, volume, occupancy



if __name__ == '__main__':
    cnx = mysql.connect_to_database(**config)

    if cnx == None:
        sys.exit()

    cursor = cnx.cursor()

    time, flow, occupancy = query_detector_data(cursor, DETECTOR_DATA_TABLE, DETECTOR_ID)

    cursor.close()
    cnx.close()
