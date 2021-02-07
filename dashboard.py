from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import database.dbmanager as db
import numpy as np
import json
import ast
from datetime import datetime

tabelaDePlotagem = "Predicao"

st.set_page_config(page_title='MACHINE LEARNING para Mercado Financeiro')


"""
# Bem vindo ao nosso Dashboard
Vamos plotar os resultados de  `VALE3` e `INDFUT` de acordo com nossos modelos :sunglasses:\n
Não recomendo COMPRA ou VENDA.
Isto é apenas para estudos de MACHINE LEARNING aplicado ao mercado.
"""

class collectionData:
    def __init__(self, prediction_data):
        self.source_data = prediction_data["Item"]
        self.predict = prediction_data["Predicao"]
        self.Accuracy = prediction_data["Acuracia"]
        self.TrendDirection = prediction_data["coeffInclination"]
        self.CnfMatrix_LongPrecision = prediction_data["Matrix_confusao"]["1"]["precision"]
        self.CnfMatrix_ShortPrecision = prediction_data["Matrix_confusao"]["0"]["precision"]
        self.Return = prediction_data["Retorno"]
        
class predictionData:
    def __init__(self, ticker, data):
        prediction_data = db.queryData("SELECT * FROM " + tabelaDePlotagem  + " where ticker='" + ticker + "' and DATA='" + data + "' ORDER BY DATA ASC",True)
        if len(prediction_data) > 0:
            self.id = int(prediction_data["id"][0])
            self.date = str(prediction_data["data"][0])
            self.ticker = prediction_data["ticker"][0]
            self.overall_predict = prediction_data["predict"][0]
            predicted_collection_info = ast.literal_eval(prediction_data["datainfo"][0])
            self.collection = []
            
            for item in predicted_collection_info["modelSets"]:
                self.collection.append(collectionData(item))
            
            self.collection_short_signal = prediction_data["shortPercent"][0]
            self.collection_long_signal = prediction_data["longPercent"][0]
            self.real = prediction_data["real"][0]
            self.real_variation = prediction_data["variacao_real"][0]
            self.real_pips = prediction_data["pontos_real"][0]
        else:
            self.id = "error"

# st.header('Company overview')
# chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
# st.line_chart(chart_data)

# chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
# st.line_chart(chart_data)
# st.button("COMPRA")

optionTicker = st.selectbox(
    'Selecione o ticker',
    ('VALE3','INDFUT'))


data_max = db.queryData("SELECT max(data) as data FROM " + tabelaDePlotagem  + " where ticker='" + optionTicker + "' ORDER BY DATA ASC",True)
data_max = datetime.strptime(str(data_max["data"][0]),'%Y%m%d')
data_min = db.queryData("SELECT min(data) as data FROM " + tabelaDePlotagem  + " where ticker='" + optionTicker + "' ORDER BY DATA ASC",True)
data_min = datetime.strptime(str(data_min["data"][0]),'%Y%m%d')

overall_win_query= db.queryData("select data, (predicao==real) as validacao, pontos_real from " + tabelaDePlotagem  + " where ticker='" + optionTicker + "' ORDER BY DATA ASC",True)

overall_win_query["real_return"] = 0
overall_win_query['real_return'] = overall_win_query['real_return'].astype('float')

x = 0
for i in overall_win_query.iterrows():
    if i[1]["validacao"] == 1:
        if i[1]["pontos_real"] < 0:
            overall_win_query["real_return"][x] = i[1]["pontos_real"] * -1
        else:
            overall_win_query["real_return"][x] = i[1]["pontos_real"]
    if i[1]["validacao"] == 0:
        if i[1]["pontos_real"] > 0:
            overall_win_query["real_return"][x] = i[1]["pontos_real"] * -1
        else:
            overall_win_query["real_return"][x] = i[1]["pontos_real"]
    x = x + 1


total_days = len(overall_win_query)
days_loss = len(overall_win_query.loc[overall_win_query["validacao"] == 0])
days_win = len(overall_win_query.loc[overall_win_query["validacao"] == 1])

col1, col2, col3= st.beta_columns(3)

with col1:
    st.success("Gain: " + str(round((days_win / total_days) * 100,2)) + "%")
with col2:
    st.error("Loss: " + str(round((days_loss / total_days) * 100,2)) + "%")
with col3:
    st.info("Total: " + str(total_days))


st.header("Retorno geral dos modelSets")
overall_win_query["cum_return"] = overall_win_query["real_return"].cumsum()
overall_win_query['data'] = pd.to_datetime(overall_win_query['data'], format="%Y%m%d")
overall_win_query = overall_win_query.set_index("data")
overall_win_query.drop(["validacao", "pontos_real", "real_return"], axis=1, inplace=True)
overall_win_query.columns = ['retorno']

st.line_chart(overall_win_query)


start_time = st.slider(
     "Data da Previsão?",
     value=data_max,
     max_value=data_max,
     min_value=data_min,
     format="DD/MM/YYYY")

informacoes = predictionData(optionTicker,str(datetime.strftime(start_time, "%Y%m%d")))

if informacoes.id != "error":
    col1, col2= st.beta_columns(2)

    with col1:
        st.info("Previsão Geral: " + informacoes.overall_predict + "")
    with col2:
        if informacoes.overall_predict == informacoes.real:
            st.success("Real: " + str(informacoes.real))
        else:
            if informacoes.real == None:
                st.success("Aguardando Processamento")
            else:
                st.error("Real: " + str(informacoes.real))



    listModels = []
    selectedModel = 0
    for i in informacoes.collection:
        listModels.append(i.source_data)

    listModels = tuple(listModels)


    optionModel = st.selectbox(
        'Selecione o Modelo',
        listModels)

    selectedModel = listModels.index(optionModel)
    returnOfModel = informacoes.collection[selectedModel].Return

    dfReturn = pd.DataFrame.from_dict(returnOfModel, orient='index')
    dfReturn = dfReturn.reset_index()
    dfReturn.columns = ['data', 'retorno']
    dfReturn['data'] = dfReturn['data'].astype('datetime64[ns]')
    dfReturn['data'] = dfReturn['data'].dt.strftime('%d-%m-%Y')
    dfReturn['retorno'] = dfReturn['retorno'].astype('float')
    dfReturn['retorno'] = dfReturn["retorno"].cumsum()
    

    
    dfReturn = dfReturn.set_index("data")
    st.line_chart(dfReturn)
 
    st.header("Previsto pelo modelSet: `" + informacoes.collection[selectedModel].predict + "`")
    st.write("Acuracia: ", float(informacoes.collection[selectedModel].Accuracy), " - Inclinação da Tendencia:", float(informacoes.collection[selectedModel].TrendDirection))
    st.write("Assertividade -> Compras:", informacoes.collection[selectedModel].CnfMatrix_LongPrecision, "Vendas: ",informacoes.collection[selectedModel].CnfMatrix_ShortPrecision)
    st.header("Retorno em 15 Dias: `" + str(dfReturn['retorno'].tail(1)[0]) + "`")

    st.header('Visão geral de Resultados')
    st.table(dfReturn)
