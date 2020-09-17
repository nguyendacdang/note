import requests
import pandas as pd
import json
from datetime import datetime, timedelta,date
import matplotlib.pyplot as plt

def call_api(symbol,resolution,startDate,endDate):
    url_path = 'https://dchart-api.vndirect.com.vn/dchart/history?resolution={}&symbol={}&from={}&to={}'.format(resolution,symbol,startDate,endDate)
    response = requests.get(url= url_path)
    jdata = None
    if response is not None:
        jdata = response.json()
    
    data = None
    if jdata is not None:
        strindex = [str(date.fromtimestamp(d)) for d in jdata['t']]
        data = pd.DataFrame.from_dict(jdata)
        data['new_index'] = strindex
        data.drop(columns=['s','t'],axis=1,inplace=True)
        data.rename(columns={'o':'Open', 'c':'Close', 'h':'High','l':'Low','v':'Volume'}, inplace=True)
        
    return data

def get_close_data(symbol,start,end):
    resolution = 'D'
    startDate = str(int(datetime.fromisoformat(start).timestamp()))
    endDate = str(int(datetime.fromisoformat(end).timestamp()))
    dataPrime = call_api(symbol=symbol,resolution=resolution,startDate=startDate,endDate=endDate)
    return dataPrime['Close']


def get_daily_return(data):
    return 100*(data/data.shift(1) - 1)

def get_rolling_mean(data,window):
    return data.rolling(window=window, center=False).mean()

def get_rolling_std(data,window):
    return data.rolling(window=window, center=False).std()

def live_run():
    data = get_close_data('FPT','2020-06-01','2020-09-17')
    MA_data = get_rolling_mean(data,20)
    std_data = get_rolling_std(data,20)
    print(data.describe())

    fig, axes = plt.subplots(2, 2, figsize=(15,5), dpi=100, sharex=True)
    axes[0,0].plot(data,label='Close Price')
    MA_data.plot(ax=axes[0,0])


    axes[1,0].plot(get_daily_return(data))

    axes[1,1].plot(data.pct_change())

    # plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    live_run()