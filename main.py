import sys
import mysql.connector

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from config import config

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

def plot_fundamental_diagram(flow, occupancy):
    plt.title("Detector 608219 Flow-Occupancy Diagram January 2017")
    plt.xlabel("Occupancy (%)")
    plt.ylabel("Flow (vph)")

    plt.scatter(occupancy, flow)
    plt.xlim(0, 100)
    plt.ylim(bottom=0)
    plt.show()

def query_detector_data(cursor, graph=False):
    query = ("SELECT * FROM {} WHERE DetectorID = {}".format("detector_data_processed_2017_1", "608219"))
    cursor.execute(query)

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
    #speed = np.array(speed)

    if graph:
        plot_data_over_time("Detector 608219 Volume January 2017", time, volume, ylabel="Volume (vph)", figsize=(12, 5))
        plot_data_over_time("Detector 608219 Occupancy January 2017", time, occupancy, ylabel="Occupancy (s)", figsize=(12, 5))
        #plot_data_over_time("Detector 608219 Speed January 2017", time, speed, ylabel="Speed", figsize=(12, 5))

    return time, volume, occupancy

def calculate_flow_occupancy(time, volume, occupancy, graph=False):
    occupancy_percentage = occupancy / 3600 * 100

    if graph:
        plot_data_over_time("Detector 608219 Occupancy January 2017", time, occupancy_percentage, ylabel="Occupancy (%)", figsize=(12, 5))
        plot_fundamental_diagram(volume, occupancy_percentage)



if __name__ == '__main__':
    cnx = connect_to_database(**config)

    if cnx == None:
        sys.exit()

    cursor = cnx.cursor()

    time, volume, occupancy = query_detector_data(cursor)
    calculate_flow_occupancy(time, volume, occupancy)

    cursor.close()
    cnx.close()
