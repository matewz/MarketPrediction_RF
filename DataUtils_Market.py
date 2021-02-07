import pandas as pd
import numpy as np
from math import sqrt

import warnings
warnings.filterwarnings('ignore')
from ta import add_all_ta_features
from ta.utils import dropna

def returnColumn(infoData, Periodos, drop = True):
    periodos = Periodos
    infoData["Retorno"] = infoData["Close"].pct_change(periodos)
    infoData["Alvo1"] = infoData["Retorno"].shift(-periodos)
    infoData["Pips"] = (infoData["Close"] - infoData["Close"].shift(periodos))
    infoData["Pips"] = infoData["Pips"].shift(-periodos)
    if drop == True:
        infoData =  infoData.dropna(axis = 0) 
        infoData =  infoData.dropna(axis = 1)
    return infoData

def returnColumnNextDay(infoData, Periodos, drop = True):
    periodos = Periodos
    infoData["Retorno"] = infoData["Close"].pct_change(periodos)
    infoData["Alvo1"] = infoData["Retorno"].shift(-periodos)
    infoData["Pips"] = (infoData["open"].shift(periodos) - infoData["Close"].shift(periodos))
    infoData["Pips"] = infoData["Pips"].shift(-periodos)
    if drop == True:
        infoData =  infoData.dropna(axis = 0) 
        infoData =  infoData.dropna(axis = 1)
    return infoData

def physicsIndicator(infoData, Deslocamento = 5, Cortes = 5, drop = True):
    p = Deslocamento
    corte = Cortes
    
    # Velocidade de p dias
    infoData["v"] = (infoData["Retorno"] - infoData["Retorno"].shift(p))/p

    # Aceleraçao de p dias
    infoData["a"] = (infoData["v"] - infoData["v"].shift(p))/p

    # Força
    # Calculando a massa
    infoData["m"] = infoData["Retorno"].rolling(p).sum()
    infoData["f"] = infoData["m"]*infoData["a"]

    # Energia cinética
    infoData["T"] = 0.5*infoData["m"]*infoData["v"]*infoData["v"]


    # Trabalho 
    # cateto_oposto
    cat_op = infoData["Retorno"].rolling(p).sum()-infoData["Retorno"].rolling(1).sum()
    cat_ad = p
    infoData["cat"] = cat_op/cat_ad
    #infoData =  infoData.dropna(axis = 0)
    infoData["w"] = infoData["f"]*np.cos(np.arctan(infoData["cat"]))

    # Energia potencial
    infoData["k"] = cat_op*infoData["m"]

    # Torque
    infoData["tau"] = infoData["f"]*np.sin(np.arctan(infoData["cat"]))

    # Momentum
    infoData["M"] = infoData["m"]*infoData["v"]

    # Gravidade
    infoData["g"] = infoData["m"]*infoData["Retorno"].rolling(p).sum()/(infoData["m"]/infoData["Retorno"].rolling(p).sum())*(infoData["m"]/infoData["Retorno"].rolling(p).sum())


    filteredData = infoData.copy() #.dropna(axis = 0) 

    filteredData["v"] = pd.qcut(filteredData["v"], corte, labels = False)
    filteredData["a"] = pd.qcut(filteredData["a"], corte, labels = False)
    filteredData["m"] = pd.qcut(filteredData["m"], corte, labels = False)
    filteredData["f"] = pd.qcut(filteredData["f"], corte, labels = False)
    filteredData["T"] = pd.qcut(filteredData["T"], corte, labels = False)
    filteredData["w"] = pd.qcut(filteredData["w"], corte, labels = False)
    filteredData["k"] = pd.qcut(filteredData["k"], corte, labels = False)
    filteredData["tau"] = pd.qcut(filteredData["tau"], corte, labels = False)
    filteredData["M"] = pd.qcut(filteredData["M"], corte, labels = False)
    filteredData["g"] = pd.qcut(filteredData["g"], corte, labels = False)
    
    if drop == True:
        filteredData = filteredData.dropna(axis = 0) 
        filteredData = filteredData.dropna(axis = 1)
        infoData =  infoData.dropna(axis = 0) 
        infoData =  infoData.dropna(axis = 1)

    return filteredData

def omIndicator(infoData, drop = True):
    
    infoData["std5"] = infoData["Retorno"].rolling(5).std()

    # Desvio Padrao de 10 dias
    infoData["std10"] = infoData["Retorno"].rolling(10).std()

    # Desvio Padrao de 15 dias
    infoData["std15"] = infoData["Retorno"].rolling(15).std()

    # Proporçao do corpo do candle em relacao ao range do dia
    infoData["prop"] = (infoData["Close"]-infoData["open"])/(infoData["high"]-infoData["low"])

    # Direçao do dia atual
    infoData["dir_D"] = np.where(infoData['Close'] > infoData['open'] , '1', '0')

    # Direçao D-1
    infoData["dir_D-1"] = infoData["dir_D"].shift(1)

    # Direçao D-2
    infoData["dir_D-2"] = infoData["dir_D"].shift(2)

    # Direçao D-3
    infoData["dir_D-3"] = infoData["dir_D"].shift(3)

    # Media Movel de 15 dias std5
    infoData["mm_std5"] = infoData["std5"].rolling(15).mean()

    # Media Movel de 15 dias std5
    infoData["mm_std10"] = infoData["std10"].rolling(15).mean()

    # Media Movel de 15 dias std5
    infoData["mm_std15"] = infoData["std15"].rolling(15).mean()

    # RSL std5
    infoData["RSL_std5"] = (infoData["std5"]/infoData["std5"].rolling(15).mean())-1

    # RSL std10
    infoData["RSL_std10"] = (infoData["std10"]/infoData["std10"].rolling(15).mean())-1

    # RSL std15
    infoData["RSL_std15"] = (infoData["std15"]/infoData["std15"].rolling(15).mean())-1

    infoData= infoData.drop(["std5","std10","std15","mm_std5","mm_std10","mm_std15"], axis = 1)

    infoData["RSL_std5"] = pd.qcut(infoData["RSL_std5"], 10, labels = False)
    infoData["RSL_std10"] = pd.qcut(infoData["RSL_std10"], 10, labels = False)
    infoData["RSL_std15"] = pd.qcut(infoData["RSL_std15"], 10, labels = False)
    
    if drop == True:
        infoData =  infoData.dropna(axis = 0) 
    
    return infoData

def techIndicator(df1):
        #OFtrader

    # Initialize Bollinger Bands Indicator
    from ta.volatility import BollingerBands
    indicator_bb = BollingerBands(close=df1["Close"], window=10, window_dev=2)

    # Add Bollinger Bands features
    df1['bb_bbh'] = indicator_bb.bollinger_hband()
    df1['bb_bbl'] = indicator_bb.bollinger_lband()

    # Initialize Bollinger Bands Indicator
    from ta.trend import PSARIndicator
    indicator_SAR = PSARIndicator(high=df1["high"], low=df1["low"], close=df1["Close"])

    # Add Bollinger Bands features
    df1['sar_high'] = indicator_SAR.psar_up()
    df1['sar_low'] = indicator_SAR.psar_down()

    from ta.trend import EMAIndicator
    indicator_EMA = EMAIndicator(close=df1["Close"], window=7)
    df1['Media7'] = indicator_EMA.ema_indicator()

    df1['sar_low'] = df1['sar_low'].fillna(0)
    df1['sar_high'] = df1['sar_high'].fillna(0)

    df1['Distancia_M7'] = df1['Close'] / df1['Media7']
    df1['Distancia_BBH'] = df1['Close'] / df1['bb_bbh']
    df1['Distancia_BBL'] = df1['Close'] / df1['bb_bbl']
    df1['Distancia_SAR'] = np.where(df1['sar_high'] > 0,  df1['Close'] / df1['sar_high'] , df1['sar_low'] / df1['Close'])
    df1['posicao_sar'] = np.where(df1['sar_high'] > 0 , '1', '0')

    corte = 3
    df1["Distancia_M7"] = pd.qcut(df1["Distancia_M7"], corte, labels = False)
    df1["Distancia_BBH"] = pd.qcut(df1["Distancia_BBH"], corte, labels = False)
    df1["Distancia_BBL"] = pd.qcut(df1["Distancia_BBL"], 15, labels = False) 
    df1["Distancia_SAR"] = pd.qcut(df1["Distancia_SAR"], 15, labels = False)
    #
    # Padrão Bom com M7: 3 - BBH: 3 - BBL: 15 - DSAR: 15


    df1 = df1.drop(["sar_high","sar_low","bb_bbh","bb_bbl","Media7"], axis = 1)
    #df1.tail(50)
    return df1
