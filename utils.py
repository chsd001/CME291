import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
from tqdm import tqdm

def load_data(path, start_date, end_date):
    data = []
    pbar = tqdm(pd.date_range(start_date, end_date))
    for date in pbar:
        filename = date.strftime("%Y%m%d") + ".csv"
        fullpath = os.path.join(path, filename)
        if os.path.isfile(fullpath):
            data.append(dd.read_csv(fullpath))
    if(len(data)==0):
        return pd.DataFrame()
    data = dd.concat(data)
    data = data.compute()
    return data

def index(data, timeCol='close_time', indexCol='datetime', unit='ms'):
    data.set_index(pd.to_datetime(data[timeCol], unit=unit).rename(indexCol), inplace=True)
    data.sort_values(by=indexCol, inplace=True)


