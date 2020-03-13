import pandas as pd
import numpy as np
import cvxpy as cvx
from statsmodels.regression.linear_model import WLS


def pred_stats(data, predName, retName, weightCol = None):
    if weightCol == None:
        data['tmp'] = 1
        weightCol = 'tmp'
    ret = data.groupby('datetime').apply(lambda x: (x[predName]*x[retName]*x[weightCol]).sum()/np.abs(x[predName]*x[weightCol]).sum())
    if weightCol == None:
        data.drop('tmp', axis=1, inplace=True)
    return pd.DataFrame({'return': ret.mean(), 'sharpe': ret.mean()/ret.std()}, index=[0])


def cross_center(data, colName, weightCol=None, newColName=None):
    if newColName==None:
        newColName=colName+'-c'

    #weightSum = data.ticker.drop_duplicates().count() if weightCol==None else data.groupby('datetime')[weightCol].transform('sum')
    data[newColName] = data[colName] - data.groupby(data.index.name)[colName].transform('mean')

def wmean(data, colName, weightCol, newColName=None):
    if newColName==None:
        newColName=colName+'-w'

    data[newColName] = data.eval(weightCol + "*" + colName).groupby('datetime').transform('sum') / data.groupby('datetime')[weightCol].transform('sum')

def zscore(data, colName, newColName=None):
    if newColName==None:
        newColName=colName+'-c'
    data[newColName] = (data[colName] - data.groupby('datetime')[colName].transform('mean'))/data.groupby('datetime')[colName].transform('std')

def compute_return(data, priceName, colName, lag):
    data[colName] =  data.groupby('ticker')[priceName].transform(lambda x: 10000*np.sign(lag)*x.pct_change(periods=lag))

def compute_volume(data, volumeName, colName, periods):
    #data[colName] =  data.groupby('ticker')[volumeName].rolling(periods).sum().reset_index(0,drop=True)
    data[colName] = data.groupby('ticker')[volumeName].transform(lambda x: x.rolling(periods).sum())

def compute_volatility(data, retName, colName, periods):
    data[colName] = data.groupby('ticker')[retName].transform(lambda x: x.rolling(periods).std())

def cross_correlation(data, retName, norm=True):
    return data.pivot(columns = 'ticker', values = retName).corr() if norm==True else data.pivot(columns = 'ticker', values = retName).cov()

def lagged_correlation(data, retName, lag):
    t = data.pivot(columns = 'ticker', values = retName)
    tickers = t.columns
    for x in tickers:
        t[x+"-L"] = t[x].shift(lag)
    t = t.corr()
    t = t.drop(tickers, axis = 1)
    t.columns = tickers
    return t.loc[tickers]

def run_regression(data, target, predictors, fpred, predict=None, weightCol = None):
    weights = 1 if weightCol == None else data[weightCol]
    model = WLS(data[target], data[predictors], weights = weights)
    results = model.fit()
    coeffs = pd.DataFrame(results.params).transpose()
    y = None if type(predict)==type(None) else predict[target] - results.predict(predict[fpred])
    return coeffs, y

def exposure_regression(data, target, predictors, resCol, fpred = None):
    fpred = predictors if type(fpred)==type(None) else fpred
    n_period = data.period.max()
    tickers = data.ticker.unique()
    coeffs = {}
    for ticker in tickers:
        coeffs[ticker] = []
        for i in range(0, n_period):
            coeffs_i,  y_i= run_regression(data[(data.period <= i) & (data.ticker==ticker)], target, predictors, fpred, data[(data.period == i+1) & (data.ticker==ticker)], None)
            coeffs[ticker].append(coeffs_i)
            data.loc[(data.period == i+1) & (data.ticker == ticker), resCol] = y_i
        coeffs[ticker] = pd.concat(coeffs[ticker]).reset_index(drop=True)
    return coeffs

def return_regression(data, target, predictors, weightCol=None):
    coeffs = []
    for d in data.index.unique():
        #print("Regressing on timestamp " + d.strftime("%Y-%m-%d %H:%M:%S"))
        coeffs_i, y_i = run_regression(data.loc[d], target, predictors, None, weightCol)
        coeffs.append(coeffs_i)
    coeffs = pd.concat(coeffs)
    coeffs.set_index(data.index.unique(), inplace=True)
    coeffs.columns = ['r60_'+c for c in coeffs.columns]
    return coeffs

def replicating_portfolio(data, factors, weightCol=None):
    n = len(factors)
    X = data[factors].values
    if n == 1:
        X = X.reshape((X.shape[0], 1))
    W = np.diag(data[weightCol]) if weightCol != None else np.eye(X.shape[0])
    F = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W)
    return pd.DataFrame(F, columns = data.ticker, index = pd.Series(factors,name='factor'))

def targetPos(prevx, p, c, l, sigma):
    x = cvx.Variable(prevx.size)
    constraints = []
    constraints.append(cvx.sum(cvx.abs(x)) <= 1)
    constraints.append(cvx.abs(x) <= 0.5)
    prob = cvx.Problem(cvx.Minimize(-p.T*x + c*cvx.sum(cvx.abs(x-prevx)) + l*cvx.quad_form(x, sigma)), constraints)
    prob.solve()
    return x.value

def markowitz(data, retName, predName, covMat, c, l):
    p = data.ticker.unique().size
    #results = pd.Series(index = data.index.unique(), name='return')
    results = pd.DataFrame(index = data.index.unique(), columns=['return', 'turnover'])
    curQty = np.zeros(p)
    for d in results.index:
        cross = data.loc[d].sort_values('ticker')
        x = targetPos(curQty, cross[predName].values, c, l, covMat[cross['period'].values[0] - 1])
        ret = x.dot(cross[retName].values)/np.abs(x).sum()
        results.loc[d, 'return'] = ret
        results.loc[d, 'turnover'] = np.abs(x-curQty).sum()
        curQty = x
    return results

def computeOU(data, colName):
    V = data.groupby('ticker')[colName].var()
    C = data.groupby('ticker')[colName].apply(lambda x:np.cov(x[1:], x.shift()[1:])[0,1])
    eps = np.log(V/C)
    eps = eps.mean().round(3)
    return eps

def evaluate_model(data, startDate, endDate, retName, predName):
    df = data.loc[startDate:endDate, [retName,predName]]
    df['day'] = df.index.round('1D') 
    df = df.groupby('day').apply(lambda x: x.corr().iat[0,1])
    return df
    #return pd.Series([df.mean(), df.std()], index = ["mean", "sdev"], name = predName)

def evaluate_strategies(returns, turnovers, fees = 0):
    returns = pd.DataFrame(returns)
    turnovers = pd.DataFrame(turnovers)
    returns = returns - fees*turnovers
    return pd.concat([0.01*24*returns.mean().rename('Daily return (%)'), np.sqrt(26*365)*(returns.mean()/returns.std()).rename('Annualized Sharpe'), (returns.sum()/turnovers.sum()).rename("Return per unit")], axis=1).round(2)
