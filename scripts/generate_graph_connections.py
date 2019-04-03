import os
import sys

parent_dir = os.path.abspath("/Users/victorchan/Dropbox/UndergradResearch/Victor/Code")
sys.path.append(parent_dir)

import mysql_utils

QUERY =\
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

if __name__ == "__main__":
    phase_timings = mysql_utils.execute_query(QUERY)
    
    if phase_timings == None:
        sys.exit()

    print(len(phase_timings))
