import pandas as pd

from plotly.offline import plot, iplot
import plotly.graph_objs as go

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def pyplot_scatter(x, y, label, mode='markers+lines'):
    plt.figure(figsize=[15,7])
    for i in range(len(y)):
        if mode=='markers+lines':
            plt.plot(x, y[i], marker = 'o', label = label[i])
        else:
            plt.plot(x, y[i], label = label[i])
    if len(y)>1: plt.legend();
    plt.grid()
    plt.show()

def pyplot_df(data, mode='markers+lines'):
    df = pd.DataFrame(data)
    pyplot_scatter(df.index, df.T.values, df.columns, mode)

def plot_scatter(x, y, label, mode='markers+lines'):
    data_plot = []
    for i in range(len(y)):
        trace = go.Scattergl(
            x = x,
            y = y[i],
            mode = mode,
            name = str(label[i])
        )
        data_plot.append(trace)
    layout = go.Layout(
        xaxis = dict(
            showline = True,
            zeroline = True,
        ),
        yaxis = dict(
            showline = True,
            zeroline = True,
        )
    )
    fig = go.Figure(data=data_plot, layout=layout)
    iplot(fig)

def plot_df(data, mode='markers+lines'):
    df = pd.DataFrame(data)
    plot_scatter(df.index, df.T.values, df.columns, mode)

def plot_corr_heatmap(data, fixed=True):
    layout = go.Layout(
            autosize=True,
            xaxis=dict(side='bottom', automargin=True,tickangle=90),
            yaxis=dict( autorange='reversed', automargin=True),
            )

    if fixed:
        data_plot = [go.Heatmap(z=data.values.tolist(), x=data.columns, y=data.columns, colorscale='RdBu', zmin=-1, zmax=1)]
    else:
        data_plot = [go.Heatmap(z=data.values.tolist(), x=data.columns, y=data.columns, colorscale='RdBu', zmid=0)]
    fig = go.Figure(data=data_plot, layout=layout)
    iplot(fig)

def plot_df_heatmap(data, x, y):
    layout = go.Layout(xaxis=dict(side='top'), yaxis=dict( autorange='reversed'))
    data_plot = [go.Heatmap(z=data.values.tolist(), x=x, y=y, colorscale='RdBu')]
    fig = go.Figure(data=data_plot, layout=layout)
    iplot(fig)

def plot_pca(data, retName):
    df = data.pivot(columns = 'ticker', values = retName).dropna(axis=0, how='any')
    pca = PCA()
    pca.fit(df/df.std())
    strindex = [str(round(100*x, 3)) + " pct" for x in np.cumsum(pca.explained_variance_ratio_)]
    plotData =  pd.DataFrame(pca.components_.transpose(), columns = df.columns, index = strindex)
    plotData = plotData*np.sign(np.sum(pca.components_, axis=1))
    plot_df_heatmap(plotData, strindex, plotData.columns)
    return 

def plot_pred_corr(data, x, y):
    df = data.groupby('ticker')[[x,y]].apply(lambda x: x.corr().iat[0,1])
    return df
