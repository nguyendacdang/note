import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta,date
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.stattools import jarque_bera as jb
from pandas.core.window.rolling import Rolling as pd_rolling
import statsmodels.api as api
from statsmodels import regression
import seaborn
from scipy.stats import norm
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

def forward_aic(Y, data):
    # This function will work with pandas dataframes and series
    
    # Initialize some variables
    explanatory = list(data.columns)
    selected = pd.Series(np.ones(data.shape[0]), name="Intercept")
    current_score, best_new_score = np.inf, np.inf
    
    # Loop while we haven't found a better model
    while current_score == best_new_score and len(explanatory) != 0:
        
        scores_with_elements = []
        count = 0
        
        # For each explanatory variable
        for element in explanatory:
            # Make a set of explanatory variables including our current best and the new one
            tmp = pd.concat([selected, data[element]], axis=1)
            # Test the set
            result = regression.linear_model.OLS(Y, tmp).fit()
            score = result.aic
            scores_with_elements.append((score, element, count))
            count += 1
        
        # Sort the scoring list
        scores_with_elements.sort(reverse = True)
        # Get the best new variable
        best_new_score, best_element, index = scores_with_elements.pop()
        if current_score > best_new_score:
            # If it's better than the best add it to the set
            explanatory.pop(index)
            selected = pd.concat([selected, data[best_element]],axis=1)
            current_score = best_new_score
    # Return the final model
    model = regression.linear_model.OLS(Y, selected).fit()
    return model

def live_run():
    vni = get_close_data('VNINDEX','2016-06-01','2020-09-22')
    fpt = get_close_data('FPT','2016-06-01','2020-09-22')
    hpg = get_close_data('VIC','2016-06-01','2020-09-22')
    # print(api.add_constant(vni))
    # MA_data = get_rolling_mean(data,20)
    # std_data = get_rolling_std(data,20)
    # print(data.describe())

    # returns = data.pct_change()

    # fig, axes = plt.subplots(2, 2, figsize=(15,5), dpi=100, sharex=True)
    # axes[0,0].plot(data,label='Close Price')
    # MA_data.plot(ax=axes[0,0])


    # axes[1,0].plot(get_daily_return(data))

    # axes[1,1].plot(data.pct_change())

    # plt.legend(loc='upper left')
    # print(jb(returns.dropna().values))
    # print(stats.skew(returns.dropna().values))
    # print(stats.kurtosis(returns.dropna().values))
    # plt.hist(returns, bins = 40, edgecolor= 'black', align='mid')

    # rolling_correlation = vni.rolling(60).corr(fpt)
    # plt.plot(rolling_correlation)
    # returns = fpt.pct_change().dropna()
    # plt.hist(returns, bins=40, edgecolor='black')
    # plt.show()
    fpt_returns = fpt.pct_change().dropna()
    vni_returns = vni.pct_change().dropna()

    # model = regression.linear_model.OLS(fpt_returns.values, vni_returns.values).fit()
    # print(model.summary())
    # plt.scatter(fpt_returns,vni_returns)
    # seaborn.regplot(fpt_returns.values, vni_returns.values)
    # plt.show()
   
    # mean = fpt_returns.values.mean()
    # std = fpt_returns.values.std()
    # x = np.linspace(-0.06,0.06,100)
    # pdf = norm.pdf(x, loc=mean, scale=std)
    # plt.hist(fpt_returns,bins=x,histtype='stepfilled', alpha=0.2,density=True)
    # plt.plot(x,pdf)
    # plt.show()
    # fig, ax = plt.subplots(1, 1)
    # mean, var, skew, kurt = norm.stats(moments='mvsk')
    # x = np.linspace(norm.ppf(0.01),
    #             norm.ppf(0.99), 100)
    # ax.plot(x, norm.pdf(x),'r-', lw=5, alpha=0.6, label='norm pdf')
    # plt.show()
    model = regression.linear_model.OLS(hpg,api.add_constant(np.column_stack((fpt,vni)))).fit()
    prediction = model.params[0] + model.params[1]*fpt + model.params[2]*vni
    prediction.name = "Prediction"
    print(model.summary())
    error = fpt.values - prediction.values
    
    # fg, ax = plt.subplots()
    # plt.hist(error, bins= 100)

    fpt.plot(color='r')
    # vni.plot()
    hpg.plot()
    prediction.plot(color='black')
    plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()

if __name__ == '__main__':
    live_run()