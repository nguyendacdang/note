from urllib import request
import pandas as pd
import json
from datetime import datetime, timedelta,date
import requests
import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm
from statsmodels import regression
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima_model import ARIMA


def loadJson(symbol,resolution,startDate,endDate):
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

# symbol = 'FPT'
# resolution = 'D'
# startDate = str(int(datetime.fromisoformat('2020-03-01').timestamp()))
# endDate = str(int(datetime.now().timestamp()))
# fpt = loadJson(symbol,resolution,startDate,endDate)
# vni = loadJson("VNINDEX",resolution,startDate,endDate)
# rFpt = fpt['Close']/fpt['Close'].shift(1) - 1
# rVni = vni['Close']/vni['Close'].shift(1) - 1
# ax = rFpt.plot(label='FPT' , color='y')
# bx = rVni.plot(label='VNINDEX' , color='b')
# fpt['Close'].plot()
# data = fpt['Close']
# train = data[:100]
# test = data[100:115]

# fig, axes = plt.subplots(3, 2, figsize=(10,5), dpi=100, sharex=True)

# result = adfuller(data)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])

# axes[0,0].plot(train, label='Original Series')
# axes[0,0].set_title('Original Series')
# plot_acf(data, ax=axes[0,1], title='Autocorrelation' , lags=50)

# axes[1,0].plot(train.dropna().diff(1), label='Usual Differencing 1')
# axes[1,0].set_title('Usual Differencing 1')
# plot_acf(train.diff().dropna(), ax=axes[1,1], title='Autocorrelation', lags=50)

# axes[2,0].plot(train.dropna().diff(1).diff(1), label='Usual Differencing 2')
# axes[2,0].set_title('Usual Differencing 2')
# plot_acf(train.diff(1).diff(1).dropna(), ax=axes[2,1], title='Autocorrelation', lags=50)

# plt.show()



# ## Adf Test
# print(ndiffs(train, test='adf'))

# # KPSS test
# print(ndiffs(train, test='kpss'))

# # PP test:
# print(ndiffs(train, test='pp'))


# model = ARIMA(train, order=(10,2,3))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())

# residuals = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1])
# model_fit.plot_predict(dynamic=False)
# fc , se , conf = model_fit.forecast(15, alpha=0.05)  # 95% conf

# fc_series = pd.Series(fc, index=test.index)
# lower_series = pd.Series(conf[:, 0], index=test.index)
# upper_series = pd.Series(conf[:, 1], index=test.index)

# plt.figure(figsize=(12,5), dpi=100)
# plt.plot(train, label='training')
# plt.plot(test, label='actual')
# plt.plot(fc_series, label='forecast')
# plt.fill_between(lower_series.index, lower_series, upper_series, 
#                  color='k', alpha=.15)
# plt.title('Forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()

# smodel = pm.auto_arima(data, start_p=1, start_q=1,
#                          test='adf',
#                          max_p=3, max_q=3, m=12,
#                          start_P=0, seasonal=True,
#                          d=None, D=1, trace=True,
#                          error_action='ignore',  
#                          suppress_warnings=True, 
#                          stepwise=True)

# smodel.summary()



def get_rolling_mean(values, window):
	"""Return rolling mean of given values, using specified window size."""
	return values.rolling(window=window, center=False).mean()

def get_bollinger_bands(rm, rstd):
	"""Return upper and lower Bollinger Bands."""
	upper_band = rm + 2*rstd
	lower_band = rm - 2*rstd
	return upper_band, lower_band

def get_rolling_std(values, window):
    return values.rolling(window=window, center=False).std()

def daily_return(values):
    return values/values.shift(1) - 1

def test_run():
    symbol = 'HPG'
    resolution = 'D'
    startDate = str(int(datetime.fromisoformat('2020-03-01').timestamp()))
    endDate = str(int(datetime.now().timestamp()))
    fpt = loadJson(symbol,resolution,startDate,endDate)
    data = fpt['Close']

    print(daily_return(data)[-1:]*100)
    rstd = get_rolling_std(data,window=20)
    rmean = get_rolling_mean(data, window=20)
    upper_band, lower_band = get_bollinger_bands(rmean, rstd)
    ax = data.plot(title="Bollinger Bands", label='FPT')
    rmean.plot(label='Rolling mean', ax = ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
	test_run()