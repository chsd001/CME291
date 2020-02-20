import pandas as pd
import numpy as np

def cross_center(data, colName, weightCol=None, newColName=None):
    if newColName==None:
        newColName=colName+'-c'

    #weightSum = data.ticker.drop_duplicates().count() if weightCol==None else data.groupby('datetime')[weightCol].transform('sum')
    data[newColName] = data[colName] - data.groupby('datetime')[colName].transform('mean')


def compute_return(data, priceName, colName, lag):
    data[colName] =  data.groupby('ticker')[priceName].transform(lambda x: 10000*np.sign(lag)*x.pct_change(periods=lag))

def cross_correlation(data, retName):
    return data.pivot(columns = 'ticker', values = retName).corr()

def lagged_correlation(data, retName, lag):
    t = data.pivot(columns = 'ticker', values = retName)
    tickers = t.columns
    for x in tickers:
        t[x+"-L"] = t[x].shift(lag)
    t = t.corr()
    t = t.drop(tickers, axis = 1)
    t.columns = tickers
    return t.loc[tickers]
