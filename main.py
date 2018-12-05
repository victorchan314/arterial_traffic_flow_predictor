import sys
import mysql.connector

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from config import config

DETECTOR_DATA_TABLE = "detector_data_processed_2017_1"
DETECTOR_ID = "608219"
DETECTOR_DATA_QUERY = "SELECT * FROM {} WHERE DetectorID = {}"

def connect_to_database(user, password, host, database):
    try:
        cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    except mysql.connector.Error as err:
        if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
            print("Error: Username or password denied")
        elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
            print("Error: Database does not exist")
        else:
            print(err)

        return None
    else:
        return cnx

def plot_data_over_time(title, time, y, xlabel="Date", ylabel=None, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    fig.autofmt_xdate()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(time, y)
    plt.show()

def plot_fundamental_diagram(flow, occupancy, detector_id):
    plt.title("Detector {} Flow-Occupancy Diagram January 2017".format(detector_id))
    plt.xlabel("Occupancy (%)")
    plt.ylabel("Flow (vph)")

    plt.scatter(occupancy, flow)
    plt.xlim(0, 100)
    plt.ylim(bottom=0)
    plt.show()

def query_detector_data(cursor, table, detector_id, graph=False):
    cursor.execute(DETECTOR_DATA_QUERY.format(table, detector_id))

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

    volume = np.array(volume)
    occupancy = np.array(occupancy)
    occupancy_percentage = occupancy / 3600 * 100
    #speed = np.array(speed)

    if graph:
        plot_data_over_time("Detector {} Volume January 2017".format(detector_id), time, volume, ylabel="Volume (vph)", figsize=(12, 5))
        plot_data_over_time("Detector {} Occupancy January 2017".format(detector_id), time, occupancy, ylabel="Occupancy (s)", figsize=(12, 5))
        #plot_data_over_time("Detector {} Speed January 2017".format(detector_id), time, speed, ylabel="Speed", figsize=(12, 5))
        plot_data_over_time("Detector {} Occupancy January 2017".format(detector_id), time, occupancy_percentage, ylabel="Occupancy (%)", figsize=(12, 5))
        plot_fundamental_diagram(volume, occupancy_percentage, detector_id)

    return time, volume, occupancy



if __name__ == '__main__':
    cnx = connect_to_database(**config)

    if cnx == None:
        sys.exit()

    cursor = cnx.cursor()

    time, volume, occupancy = query_detector_data(cursor, DETECTOR_DATA_TABLE, DETECTOR_ID, graph=True)

    cursor.close()
    cnx.close()
