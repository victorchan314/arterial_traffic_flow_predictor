import sys

import numpy as np
import pandas as pd
import datetime as dt

from armax import armax
from config import config
import mysql_utils as mysql
import visualization

DATA_FREQUENCY = dt.timedelta(minutes=5)

DETECTOR_DATA_TABLE = "detector_data_processed_2017"
INTERSECTION = [608217, 608219, 608103, 608102, 608106, 608114, 608108]
QUERY = "WITH intersection AS (SELECT Year, Month, Day, Time, Volume AS Flow, DetectorID FROM {} WHERE DetectorID = 608217 OR DetectorID = 608219 OR DetectorID = 608103 OR DetectorID = 608102 OR DetectorID = 608106 OR DetectorID = 608114 OR DetectorID = 608108) SELECT Year, Month, Day, Time, {} FROM intersection GROUP BY Year, Month, Day, Time ORDER BY Year, Month, Day, Time;".format(DETECTOR_DATA_TABLE, ", ".join(["SUM(IF (DetectorId = {}, Flow, NULL))".format(detector_id) for detector_id in INTERSECTION]))



def query_detector_data(cursor):
    query = QUERY

    cursor = mysql.query(cursor, query)
    
    if cursor == None:
        return
    
    lists = []

    for row in cursor:
        r = [dt.datetime(row[0], row[1], row[2], row[3] // 3600, (row[3] % 3600) // 60, row[3] % 60)]
        r.extend(row[4:])
        lists.append(r)
    
    df = pd.DataFrame(lists, columns=['Date'] + [str(detector) for detector in INTERSECTION]).set_index('Date')

    return df

if __name__ == "__main__":
    cnx = mysql.connect_to_database(**config)

    if cnx == None:
        sys.exit()

    cursor = cnx.cursor()

    intersection = query_detector_data(cursor)

    cursor.close()
    cnx.close()
