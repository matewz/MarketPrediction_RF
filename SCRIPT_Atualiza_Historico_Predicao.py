import datetime
import database.dbmanager as db
import DataUtils_Market as d_util
import pandas as pd

tabelaDePlotagem = "Predicao_N_TOP50_20d"

def Processa_Historico_Predicao():
    tickers = ["INDFUT","VALE3"]
    for T in tickers:
        print ("Iniciando... : " + T)
        #df_Tabela_Predicao = db.queryData("select id, DATA from Predicao WHERE ticker='" + T + "' and DATA > 20190103 order by DATA ASC", True)
        df_Tabela_Predicao = db.queryData("select id, DATA from " + tabelaDePlotagem + " WHERE ticker='" + T + "' and real is null order by DATA ASC", True)

        df_Tabela_Cotacao = db.queryData("select * from " + T + "_ohlc_d1 order by DATA ASC", True)
        df_Tabela_Cotacao = df_Tabela_Cotacao.rename(columns = {"data": "Date", "close": "Close"})
        df_Tabela_Cotacao["Data"] = pd.to_datetime(df_Tabela_Cotacao["Date"], format='%Y%m%d').dt.normalize()
        df_Tabela_Cotacao.dropna(axis = 0)
        df_Tabela_Cotacao = df_Tabela_Cotacao.set_index("Data")
        df_Tabela_Cotacao.drop(["Date"], axis=1, inplace=True)

        df_Tabela_Cotacao = d_util.returnColumn(df_Tabela_Cotacao,1,False)

        df_Tabela_Predicao["data"] = pd.to_datetime(df_Tabela_Predicao["data"], format='%Y%m%d').dt.normalize()
        df_Tabela_Predicao = df_Tabela_Predicao.set_index("data")
        df_Tabela_Predicao['data_string'] = df_Tabela_Predicao.index
        df_Tabela_Cotacao = df_Tabela_Cotacao.dropna(axis = 0)

        for e in df_Tabela_Predicao.iterrows():
            data = e[1]["data_string"]
            id = e[1]["id"]
            if len(df_Tabela_Cotacao.loc[data:data]) > 0:
                pips = df_Tabela_Cotacao.loc[data:data]["Pips"][0]
                
                Variacao_Retorno = df_Tabela_Cotacao.loc[data:data]["Alvo1"][0]
                Real = ""
                if Variacao_Retorno > 0:
                    Real = "COMPRA"
                else:
                    Real = "VENDA"    
                    
                Variacao_Retorno = str(round(Variacao_Retorno*100,2))    
                SQL = "UPDATE " + tabelaDePlotagem + " SET pontos_real='" + str(pips) + "', real='" + str(Real) + "', variacao_real='" + str(Variacao_Retorno) + "' where id='" + str(id) + "'"
                db.queryExecute(SQL)
            
        print ("Concluido... : " + T)

if __name__ == "__main__":
    Processa_Historico_Predicao()