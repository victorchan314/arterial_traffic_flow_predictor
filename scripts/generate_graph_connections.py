import os
import sys

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

import mysql_utils

results = mysql_utils.execute_query("SELECT IntersectionID, Direction, SensorID, Movement FROM detector_inventory;")
print(results)
