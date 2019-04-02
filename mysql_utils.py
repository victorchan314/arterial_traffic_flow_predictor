import mysql.connector
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

def query(cursor, query_string):
    try:
        cursor.execute(query_string)
        return cursor
    except mysql.connector.Error as err:
        cursor.close()
        print("Error: failed query: {}".format(query_string))
        cursor.close()
        return None

def execute_query(query_string):
    cnx = connect_to_database(**config)

    if cnx == None:
        sys.exit()

    cursor = cnx.cursor()
    cursor = query(cursor, query_string)
    
    if cursor == None:
        cnx.close()
        sys.exit()

    results = []

    for row in cursor:
        results.append([x for x in row])

    cursor.close()
    cnx.close()

    return results
