import sys
import mysql.connector
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

def plot_data_over_time(title, x, y, xlabel="Date", ylabel=None, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    fig.autofmt_xdate()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(x, y)
    plt.show()

def query_detector_data(cursor, graph=False):
    query = ("SELECT * FROM {} WHERE DetectorID = {}".format("detector_data_processed_2017_1", "608219"))
    cursor.execute(query)

    time = []
    volume = []
    occupancy = []
    speed = []

    for row in cursor:
        d = dt.datetime(row[1], row[2], row[3], row[4] // 3600, (row[4] % 3600) // 60, row[4] % 60)
        time.append(d)

        volume.append(row[5])
        occupancy.append(row[6])
        speed.append(row[7])

    if graph:
        plot_detector_data_over_time(time, volume, occupancy, speed)

    return time, volume, occupancy

def plot_detector_data_over_time(time, volume, occupancy, speed):
    plot_data_over_time("Detector Volume January 2017", time, volume, ylabel="Volume", figsize=(12, 5))
    plot_data_over_time("Detector Occupancy January 2017", time, occupancy, ylabel="Occupancy", figsize=(12, 5))
    plot_data_over_time("Detector Speed January 2017", time, speed, ylabel="Speed", figsize=(12, 5))

def calculate_flow_occupancy(time, volume, occupancy, graph=False):
    plot_flow_occupancy_graphs()

def plot_flow_occupancy_graphs():
    pass

if __name__ == '__main__':
    cnx = connect_to_database(**config)

    if cnx == None:
        sys.exit()

    cursor = cnx.cursor()

    time, volume, occupancy = query_detector_data(cursor, graph=True)
    calculate_flow_occupancy(time, volume, occupancy, graph=True)

    cursor.close()
    cnx.close()
