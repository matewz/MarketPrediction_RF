import utils.dde as ddec
import time
import sqlite3
import datetime
import database.dbmanager as db

def arrangeData(Cotacao):
    return str(Cotacao).replace('\'','').replace('b','').replace(',','.')


def storeQuote(timestamp, ticker,Open,High,Low,Close):
    
    sqlConsulta = "SELECT * FROM " + ticker + " WHERE data = '" + str(timestamp) + "'"
    df_consulta = db.queryData(sqlConsulta)

    if len(df_consulta) > 0:
        str_SQL = "UPDATE " + ticker + " SET data = '" + str(timestamp) + "',open = '" + str(Open) + "',high = '" + str(High) + "',low = '" + str(Low) + "',close = '" + str(Close) + "'"
        str_UPDT = "  WHERE id = '" + str(df_consulta[0][0]) + "'"
        db.queryExecute(str_SQL + str_UPDT)
    else:
        str_SQL = 'INSERT INTO ' + ticker + ' (data,Open,High,Low,Close) VALUES '
        string_Insert = " ('" + str(timestamp) + "'," + "'" + str(Open) + "'," + "'" + str(High) + "'," + "'" + str(Low) + "'," + "'" + str(Close) + "')"
        try:
            db.queryExecute(str_SQL + string_Insert)
        except:
            time.sleep(2)
            db.queryExecute(str_SQL + string_Insert)



def updateRequest(symbols):
    try:
        print('Refresh')
        QUOTE_client = ''
        QUOTE_client = ddec.DDEClient('profitchart', 'cot')

        for i in symbols:
            QUOTE_client.advise(i)

    except ValueError:
        print("erro: " + ValueError)



def dataUpdate():
    symbols = ['VALE3','INDFUT']

    QUOTE_client = ddec.DDEClient('profitchart', 'cot')
    time.sleep(2)

    iteracao = 0

    updateRequest(symbols)
    time.sleep(2)
    string_Agora = datetime.datetime.today().strftime("%Y%m%d")

    for symb in symbols:
        ticker = symb 
        try:
            Open = float(arrangeData(QUOTE_client.request(symb + '.ABE')))
            High = float(arrangeData(QUOTE_client.request(symb + '.MAX')))
            Low = float(arrangeData(QUOTE_client.request(symb + '.MIN')))
            Close = float(arrangeData(QUOTE_client.request(symb + '.ULT')))

            storeQuote(string_Agora, ticker + "_ohlc_d1",Open,High,Low,Close)
            print(ticker, "Armazenado...", ticker)   
            #else:
            #    print(ticker, "Sem Alteracao de Volume...") 
        except ValueError:
            print("erro: " + ValueError)