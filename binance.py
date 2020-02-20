import requests        
import json            
import pandas as pd    
import time

import datetime as dt 

dumpPath = '/home/charbel/Documents/Stanford/Project/data/'

## Get bars for a date and a set of tickers
def get_bars(date, tickers):
    startDates = [round(1000*pd.to_datetime(date).timestamp()), round(1000*(pd.to_datetime(date)+pd.to_timedelta('12H')).timestamp())] 
    root_url = 'https://api.binance.com/api/v1/klines'
    headers = ['open_time', 'open', 'high', 'low', 'close', 'volume','close_time', 'volume_quote', 'num_trades', 'buy_vol', 'buy_vol_quote', 'ignore', 'ticker']
    data = pd.DataFrame(columns = headers)
    for t in tickers:
        print("Date["+str(date)+"]||Ticker["+t+"]")
        #Split to avoid reaching request limit
        for d in startDates:
            url = root_url + '?symbol=' + t + '&interval=1m&startTime=' + str(d) + '&limit=720'
            raw = json.loads(requests.get(url).text)
            df = pd.DataFrame(raw)
            df.columns = headers[:-1]
            df.index = [dt.datetime.utcfromtimestamp((1+x)/1000.0) for x in df.close_time]
            df['ticker'] = t
            data = data.append(df)
    return data


## Dump the bar data for a set of dates, sleep to avoid timeout
def dump_bars(startDate, endDate, tickers):
        dr = pd.date_range(start=startDate, end=endDate, freq='1D')
        for d in dr:
                data = get_bars(d, tickers)
                data.to_csv(dumpPath+d.strftime(format="%Y%m%d")+'.csv', sep=',', index=False)
                time.sleep(2)

tickers = [
        'BTCUSDT',
        'ETHUSDT', 
        'LTCUSDT', 
        'BCHUSDT', 
        'BNBUSDT', 
        'XRPUSDT', 
        'HBARUSDT', 
        'LINKUSDT', 
        'EOSUSDT', 
        'ETCUSDT', 
        'XTZUSDT', 
        'ADAUSDT', 
        'TRXUSDT', 
        'XLMUSDT', 
        'VETUSDT', 
        'DASHUSDT', 
        'ONTUSDT', 
        'MATICUSDT', 
        'NEOUSDT', 
        'ZECUSDT', 
        'IOTAUSDT', 
        'ATOMUSDT', 
        'WRXUSDT', #
        'XMRUSDT', 
        'PAXUSDT', 
        'QTUMUSDT', 
        'BATUSDT', 
        'IOSTUSDT', 
        'ALGOUSDT', 
        'OGNUSDT', #
        'STXUSDT', 
        'VITEUSDT', #
        'IOTXUSDT', 
        'WAVESUSDT', 
        'BTTUSDT', 
        'RVNUSDT', 
        'LTOUSDT', 
        'HOTUSDT', 
        'FETUSDT', 
        'TOMOUSDT', 
        'KAVAUSDT', #
        'DOGEUSDT', 
        'ENJUSDT', 
        'BEAMUSDT', 
        'ZILUSDT', 
        'OMGUSDT', 
        'TROYUSDT', #
        'DUSKUSDT', #
        'LSKUSDT', 
        'ZRXUSDT' 
]

dump_bars('2019-03-12', '2019-03-12', tickers)
