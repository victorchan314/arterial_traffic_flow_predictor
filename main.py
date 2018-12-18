import sys

import numpy as np
import datetime as dt

from armax import armax
from config import config
import mysql_utils as mysql
import visualization


#DETECTOR_DATA_TABLE = "detector_data_processed_2017_1"
DETECTOR_DATA_TABLE = "detector_data_processed_2017"
DETECTOR_ID = "608219"
DETECTOR_DATA_QUERY = "SELECT * FROM {} WHERE DetectorID = {} ORDER BY Year, Month, Day, Time"


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
        visualization.plot_data_over_time(time, volume, title="Detector {} Volume January 2017".format(detector_id), ylabel="Volume (vph)", figsize=(12, 5))
        visualization.plot_data_over_time(time, occupancy, title="Detector {} Occupancy January 2017".format(detector_id), ylabel="Occupancy (s)", figsize=(12, 5))
        #visualization.plot_data_over_time(time, speed, title="Detector {} Speed January 2017".format(detector_id), ylabel="Speed", figsize=(12, 5))
        visualization.plot_data_over_time(time, occupancy_percentage, title="Detector {} Occupancy January 2017".format(detector_id), ylabel="Occupancy (%)", figsize=(12, 5))
        visualization.plot_fundamental_diagram(volume, occupancy_percentage, title="Detector {} Flow-Occupancy Diagram January 2017".format(detector_id))

    return time, volume, occupancy



if __name__ == '__main__':
    cnx = mysql.connect_to_database(**config)

    if cnx == None:
        sys.exit()

    cursor = cnx.cursor()

    time, flow, occupancy = query_detector_data(cursor, DETECTOR_DATA_TABLE, DETECTOR_ID)

    cursor.close()
    cnx.close()
