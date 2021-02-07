
import sqlite3
import pandas as pd
databasePath =  'database/marketpredix.db'

def queryExecute(queryString):
    conn = sqlite3.connect(databasePath)

    # definindo um cursor
    cursor = conn.cursor()
 
    cursor.execute(queryString)
    conn.commit()
 
    conn.close()

def queryData(queryString, dataframe = False):
    conn = sqlite3.connect(databasePath)
    retorno = ""
    if dataframe == False:
        # definindo um cursor
        cursor = conn.cursor()
        cursor.execute(queryString)
        retorno = cursor.fetchall()
        conn.close()
        return retorno
    else:
        retorno = pd.read_sql_query(queryString, conn)
        conn.close()
        return retorno

def queryExecute_Safe(queryString, valueString):
    conn = sqlite3.connect(databasePath)

    # definindo um cursor
    cursor = conn.cursor()
 
    retorno = cursor.execute(queryString,valueString)
    conn.commit()
    conn.close()

    return retorno        