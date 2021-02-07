import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from math import sqrt
import ast
import matplotlib.pyplot as plt
import gc
from sklearn.neural_network import MLPClassifier
import datetime as dt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import DataUtils_Market as d_util
from ta import add_all_ta_features
from ta.utils import dropna
import datetime
import json
import database.dbmanager as db

topAlgorithmToCollection = 15
daysOfTest = 10
predictTable = "Predicao"


def dataProcess():
    infoData_Train = {
        
        'INDFUT': {
            'Data': {
                'start_train': "2003-12-02",  
                'end_train': "2016-12-31"
            },
            'modelSets': [
                ['RSL_std15', 'v', 'a', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std15', 'v', 'a', 'tau', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std5', 'm', 'f', 'tau', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'a', 'T', 'Distancia_BBL', 'Distancia_SAR', 'Dia_Semana', 'Dia_Mes'],
                ['v', 'a', 'cat', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'k', 'tau', 'M', 'Distancia_BBL', 'Dia_Semana', 'Mes'],
                ['v', 'a', 'M', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std15', 'v', 'a', 'Distancia_SAR', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'a', 'cat', 'Distancia_BBL', 'Distancia_SAR', 'Dia_Semana', 'Dia_Mes'],
                ['RSL_std10', 'v', 'T', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'a', 'tau', 'Distancia_SAR', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'tau', 'g', 'Distancia_SAR', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std15', 'a', 'cat', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std5', 'm', 'w', 'tau', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'a', 'tau', 'Distancia_BBL', 'Distancia_SAR', 'Dia_Semana', 'Dia_Mes'],
                ['v', 'a', 'T', 'k', 'Distancia_BBL', 'Dia_Semana', 'Mes'],
                ['RSL_std10', 'v', 'M', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std15', 'v', 'tau', 'Distancia_SAR', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'a', 'T', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std5', 'a', 'M', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std10', 'RSL_std15', 'v', 'a', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'k', 'tau', 'Distancia_SAR', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'm', 'cat', 'M', 'Distancia_BBL', 'posicao_sar', 'Dia_Semana'],
                ['v', 'a', 'T', 'M', 'Distancia_BBL', 'posicao_sar', 'Dia_Semana'],
                ['RSL_std15', 'f', 'cat', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std15', 'cat', 'tau', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'm', 'T', 'cat', 'M', 'Distancia_BBL', 'Dia_Semana'],
                ['RSL_std15', 'm', 'cat', 'tau', 'M', 'Distancia_BBL', 'Dia_Semana'],
                ['v', 'a', 'k', 'tau', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes'],
                ['v', 'a', 'tau', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['v', 'm', 'T', 'cat', 'tau', 'M', 'Distancia_BBL'],
                ['v', 'cat', 'tau', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['RSL_std5', 'f', 'g', 'Distancia_BBL', 'Dia_Semana', 'Dia_Mes', 'Mes'],
            ]
        },
       
        'VALE3': {
            'Data': {
                'start_train': "2004-01-02",  
                'end_train': "2017-12-31"
            },
            'modelSets': [
                ['dir_D-2', 'dir_D-3', 'Dia_Mes', 'Mes'],
                ['RSL_std5', 'RSL_std15', 'posicao_sar', 'Dia_Mes'],
                ['dir_D-1', 'dir_D-3', 'RSL_std5', 'RSL_std15', 'Dia_Mes'],
                ['dir_D-1', 'dir_D-2', 'dir_D-3', 'RSL_std15', 'a'],
                ['dir_D-1', 'RSL_std5', 'RSL_std15', 'Dia_Mes'],
                ['dir_D-1', 'dir_D-2', 'RSL_std15', 'm', 'posicao_sar'],
                ['RSL_std10', 'M', 'Dia_Mes', 'Mes'],
                ['RSL_std15', 'm', 'Distancia_BBL', 'Dia_Mes'],
                ['tau', 'Distancia_BBL', 'Dia_Semana'],
                ['dir_D-2', 'RSL_std5', 'v', 'Distancia_BBL'],
                ['dir_D-2', 'RSL_std5', 'a', 'Distancia_BBL'],
                ['dir_D-2', 'v', 'k', 'Dia_Mes'],
                ['dir_D-1', 'dir_D-2', 'w', 'Dia_Mes'],
                ['dir_D-1', 'dir_D-2', 'f', 'Dia_Mes'],
                ['a', 'cat', 'Dia_Semana'],
                ['dir_D-2', 'tau', 'Distancia_BBH', 'Dia_Mes'],
                ['dir_D-1', 'dir_D-3', 'RSL_std15', 'Dia_Mes'],
                ['RSL_std10', 'k', 'g', 'Dia_Mes'],
                ['k', 'Dia_Mes'],
                ['RSL_std15', 'k', 'M', 'Dia_Mes'],
                ['a', 'tau', 'Distancia_BBL', 'Dia_Semana'],
                ['dir_D-1', 'dir_D-2', 'f', 'w', 'Dia_Mes'],
                ['tau', 'Dia_Semana', 'Dia_Mes', 'Mes'],
                ['dir_D-1', 'dir_D-2', 'T', 'tau', 'g'],
                ['a', 'Distancia_BBH', 'Distancia_BBL', 'Distancia_SAR'],
                ['k', 'Distancia_M7', 'Distancia_BBH', 'Dia_Mes'],
                ['dir_D-1', 'dir_D-2', 'k', 'Distancia_BBL', 'posicao_sar'],
                ['dir_D-2', 'tau', 'Dia_Mes'],
                ['dir_D-2', 'dir_D-3', 'posicao_sar', 'Dia_Mes'],
                ['RSL_std5', 'RSL_std10', 'w', 'Distancia_BBH'],
                ['RSL_std5', 'RSL_std10', 'f', 'Distancia_BBH'],
                ['dir_D-1', 'dir_D-2', 'RSL_std5', 'a', 'Distancia_BBL'],
                ['dir_D-1', 'dir_D-3', 'RSL_std15', 'm', 'Dia_Mes'],
                ['dir_D-1', 'k', 'Distancia_BBH', 'Dia_Mes'],
                ['dir_D-2', 'a', 'k', 'Distancia_BBL'],
                ['dir_D-1', 'RSL_std15', 'Distancia_M7', 'Distancia_BBL'],
            ]
        }
    }

    for T in infoData_Train:
        #T = "VALE3"
        print ("Processing: " + T)

        start_train = infoData_Train[T]["Data"]["start_train"]
        end_train = infoData_Train[T]["Data"]["end_train"]

        modelSets = infoData_Train[T]["modelSets"]

        df1 = db.queryData("select * from " + T + "_ohlc_d1", True)
        df1 = df1.rename(columns = {"data": "Date", "close": "Close"})

        df1["Data"] = pd.to_datetime(df1["Date"], format='%Y%m%d').dt.normalize()
        df1 = df1.sort_values(['Data'], ascending = True)

        df1.dropna(axis = 0)
        df1 = df1.set_index("Data")
        df1.drop(["Date"], axis=1, inplace=True)
        df1.drop(["id"], axis=1, inplace=True)

        dfOriginal = df1.copy()
        dfFiltered = df1.copy()

        dfFiltered = d_util.returnColumn(df1,1,False)
        dfFiltered = d_util.techIndicator(dfFiltered)
        dfFiltered = d_util.omIndicator(dfFiltered, False)
        dfFiltered = d_util.physicsIndicator(dfFiltered,5,5,False)

        dfFiltered["Dia_Semana"] = dfFiltered.index.strftime("%w")
        dfFiltered["Dia_Mes"] = dfFiltered.index.strftime("%d")
        dfFiltered["Mes"] = dfFiltered.index.strftime("%m")

        dfFiltered["Alvo_Bin"] = np.where(dfFiltered['Alvo1'] > 0 , 1, 0)

        df1_predict =  dfFiltered.tail(1).copy()

        dfFiltered = dfFiltered.dropna(axis = 0) 
        dfFiltered = dfFiltered.dropna(axis = 1)

        lastDate = datetime.datetime.strftime(df1_predict.index[0], "%d/%m/%Y")
        nextDate = datetime.datetime.strftime(df1_predict.index[0] + datetime.timedelta(days=1), "%d/%m/%Y")

        startTest = datetime.datetime.strftime(dfFiltered.tail(daysOfTest).index[0], "%Y-%m-%d")
        endTest = datetime.datetime.strftime(df1_predict.index[0] - datetime.timedelta(days=1), "%Y-%m-%d")

        d = ['algo', 'modelSet','currentAccuracy', 'returnResult', 'returnAccum', 'coeffInclination', 'confusionMatrix', 'resultPrediction']
        dfCurrentResult = pd.DataFrame(columns=d) 


        for itm in modelSets:
            df1_train1 = dfFiltered[start_train : end_train].copy()
            dfTesting = dfFiltered[startTest : endTest].copy()
                        
            x_train1 = df1_train1[itm]
            y_train1 = df1_train1['Alvo_Bin']

            x_test1 = dfTesting[itm]
            y_test1 = dfTesting['Alvo_Bin']


            rf1 = RandomForestClassifier(bootstrap = True,
                    criterion = 'gini', max_depth = 10, max_features = 'auto',
                    min_samples_leaf = 1, min_samples_split = 2,
                    n_estimators = 1500, n_jobs = 5, oob_score = True, random_state = 42)

            rf1.fit(x_train1, y_train1)

            lastDayPrediction = df1_predict[itm]
            y_pred_test1 = rf1.predict(x_test1)
            
            resultPrediction =  rf1.predict(lastDayPrediction)
            confusionMatrix = classification_report(y_test1, y_pred_test1, output_dict=True)
            currentAccuracy =  round(metrics.accuracy_score(y_test1, y_pred_test1)*100,3)

            stop = 100000
            dfTesting.loc[: , "Predicted"] = y_pred_test1
            dfTesting["Predicted"].astype(str)
            dfTesting.loc[: , "PipsReturn"] = np.where(dfTesting.loc[: , 'Predicted'] == 1 , dfTesting.loc[: , 'Pips'], '0')
            dfTesting.loc[: , "PipsReturn"] = np.where(dfTesting.loc[: , 'Predicted'] == 0 , -1*dfTesting.loc[: , 'Pips'], dfTesting.loc[: , "PipsReturn"])
            dfTesting.loc[: , "PipsReturn"] = dfTesting["PipsReturn"].astype(float)
            dfTesting.loc[: , "PipsReturn"] = np.where(dfTesting.loc[: , 'PipsReturn'] <= -stop , -stop, dfTesting.loc[: , "PipsReturn"])
            dfTesting.loc[: , "PipsReturn_Accum"] = dfTesting["PipsReturn"].cumsum()

            returnResult = str(dfTesting["PipsReturn_Accum"].tail(1)[0])

            dfReturn = dfTesting.copy()
            dfReturn = dfReturn.reset_index()

            returnAccum = { }
            for e in dfReturn.iterrows():
                data = str(datetime.datetime.strftime(e[1]["Data"], "%Y%m%d"))
                acc = str(round(e[1]["PipsReturn_Accum"],2))
                returnAccum[data] = acc

            print(str(itm) + " -- " + str(returnAccum))
            
            dfLinearRegression = dfTesting.copy()
            y3 = np.asarray(dfLinearRegression['PipsReturn_Accum']).reshape(-1,1)
            dfLinearRegression['Datetime'] = pd.to_datetime(dfLinearRegression.index.to_numpy())

            #X_test.columns = ["Date"]
            dfLinearRegression['Datetime'] = pd.to_datetime(dfLinearRegression['Datetime'])
            dfLinearRegression['Datetime'] = dfLinearRegression['Datetime'].map(dt.datetime.toordinal)

            x3 = np.asarray(dfLinearRegression['Datetime'])
            x3 = x3.reshape(-1,1)

            model = LinearRegression() #create linear regression object
            model.fit(x3, y3) #train model on train data
            model.score(x3, y3) #check score

            coeffInclination = round(float(model.coef_[0]),1)
            Intercept = round(float(model.intercept_[0]),1)

            d_frame = {'algo':'RANDOMFOREST','modelSet': str(list(itm)), 'currentAccuracy': currentAccuracy ,'returnResult': str(returnResult), 'returnAccum': str(returnAccum), 'coeffInclination': str(coeffInclination), 'confusionMatrix': str(confusionMatrix), 'resultPrediction': resultPrediction[0] }

            new_row = pd.Series(data=d_frame)
            dfCurrentResult = dfCurrentResult.append(new_row, ignore_index=True)


        dfCurrentResult['coeffInclination'] = dfCurrentResult['coeffInclination'].astype(float)
        dfCurrentResult = dfCurrentResult.sort_values(['coeffInclination'], ascending = False)
        dfTOP5Result = dfCurrentResult.head(topAlgorithmToCollection)
        dfResult = dfTOP5Result.copy()

        totalAnalyzed = dfResult["resultPrediction"].count()

        shortPercent = (dfResult[dfResult["resultPrediction"] == 0]["resultPrediction"].count()/totalAnalyzed)*100
        longPercent = (dfResult[dfResult["resultPrediction"] == 1]["resultPrediction"].count()/totalAnalyzed)*100

        position = ""
        percentual = 0

        if longPercent > shortPercent:
            position = "LONG"
            percentual = longPercent
        else:
            position = "SHORT"
            percentual = shortPercent


        infoData =  {
            'position': position,
            'percentual': str(round(percentual,2)),
            'modelSets':
            [
            ]   
        }
        
        for inteLine in dfResult.iterrows():
            if str(inteLine[1]["resultPrediction"]) == "1":
                Posicao = "LONG"
            else:
                Posicao = "SHORT"   

            x = { "Item": inteLine[1]["modelSet"],
                    "Position": Posicao,
                    "Accuracy": str(inteLine[1]["currentAccuracy"]),
                    "coeffInclination": str(inteLine[1]["coeffInclination"]),
                    "confusionMatrix": ast.literal_eval(inteLine[1]["confusionMatrix"]),
                    "Return": ast.literal_eval(inteLine[1]["returnAccum"])

            }

            infoData['modelSets'].append(x)

        infoData =  json.dumps(infoData)
        
 
        predictionDate = dt.datetime.strptime(lastDate,"%d/%m/%Y").strftime("%Y%m%d")
        ExisteData = "SELECT * FROM " + predictTable + " WHERE data = '" + str(predictionDate) + "' and ticker='" + T + "'"
        df_consulta = db.queryData(ExisteData)
        if len(df_consulta) > 0:
            ExisteData = "DELETE FROM " + predictTable + " WHERE data = '" + str(predictionDate) + "' and ticker='" + T + "'"
            db.queryExecute(ExisteData)

        stringAdicionaOuAltera = "INSERT INTO " + predictTable + " (data,ticker,predict,datainfo,shortPercent,longPercent) VALUES (?,?,?,?,?,?)"
        db.queryExecute_Safe(stringAdicionaOuAltera,[str(predictionDate),str(T),str(position),str(infoData),str(shortPercent),str(longPercent)])



if __name__ == "__main__":
    dataProcess()