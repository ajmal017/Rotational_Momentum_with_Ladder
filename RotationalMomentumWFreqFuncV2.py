# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:21:55 2019

@author: Daniel Weng & Gavin Guan
"""

import numpy as np
import pandas as pd
import math
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
import datetime
import matplotlib.pyplot as plt
import detrendPrice
import WhiteRealityCheckFor1
import os
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

def getDate(dt):
    if type(dt) != str:
        return dt
    try:
        datetime_object = datetime.datetime.strptime(dt, '%Y-%m-%d')
    except Exception:
        datetime_object = datetime.datetime.strptime(dt, '%m/%d/%Y')
        return datetime_object
    else:
        return datetime_object
       

def get_dfp(csv_name):
    yahoo_data = "yahoo_data"
    dfP = pd.read_csv(os.path.join(yahoo_data, csv_name), parse_dates=['Date'])
    dfAP = pd.read_csv(os.path.join(yahoo_data, csv_name.replace(".csv", "_AP.csv")), parse_dates=['Date'])
    dfP = dfP.sort_values(by='Date')
    dfAP = dfAP.sort_values(by='Date')
    dfP.set_index('Date', inplace=True)
    dfAP.set_index('Date', inplace=True)
    # for ticker in dfP.columns:
    #     if ticker not in ticker_list:
    #         del dfP[ticker]
    #         del dfAP[ticker]
    return dfP, dfAP


def single_type_return(csv):
    dfP, dfAP = get_dfp(csv)
    dfP.dropna(axis=1, inplace=True)
    dfAP.dropna(axis=1, inplace=True)
    # dfP, dfAP = get_dfp("BIL.SHY.IEI.IEF.TLH.TLT.TIP.csv")

    #logReturns = 1 means log returns will be used in the calculation of portfolio returns, 0 means pct_changes
    #momentum = 1 means A and B returns are ranked in increasing order (momentum), 0 in decreasing order (reversion to the mean)
    #volmomentum = 1 volatility ranked in increasing order (momentum), 0 in decreasing order (reversion to the mean)
    #calendar_month = 1 means trading respects the beginning and end of each calendar month, 0 it pays no attention to it
    #month is used if calendar_month is set to 1
    #month = 1 means every month, 2 means every other month, 3 means every third month
    #week is used if calendar_month is set to 0, no attention will be paid to beginning and end of months
    #week = 1 means every weak, 2 means every other week, 3 means every third week, 4 means every fourth week
    #if week is used trading always occurs on a Tuesday
    #the selection of the ETF is based on maximum weighted score of: A returns, B returns and volatility
    #Frequency="W" every weeek, "2W" for every 2 weeks, "3W" every 3 weeks etc
    #Frequency="W-TUE" every Tuesday, "2W-TUE" for every 2 Tuesdats, "3W-TUE" every 3 Tudsdays etc
    #Frequency= "BM" every month, "2BM" for every 2 months, "3BM" every 3 months etc; B relates to business days; 31 or previous business day if necessary
    #Frequency="SM" on the middle (15) and end (31) of the month, or previous business day if necessary
    #Delay = 1 if the trade occurs instantaneously with the signal, 2 if the trade occurs 1 day after the signal

    #regime 40 40 till 2015, then 120 200
    logReturns = 0
    momentum = 1
    volmomentum = 0 #do not change
    Aperiods = 20 #20 Default
    Bperiods = 66 #66 Default
    Speriods = 20
    Zperiods = 200
    CashFilter = 0
    MAperiods = 200 #for the cash filter
    Zboundary = -1.5 #alternative cash filter
    Frequency = "2W-FRI" #8W-FRI= 40 days, #40W-FRI = 200 days
    Delay = 1
    #Frequency = "2W-FRI"


    #dfA contains a short moving average of the daily percent changes, calculated for each ETF
    #dfB contains a long moving average of the daily percent changes, calculated for each ETF
    #dfS contains the annualized volatility, calculated for each ETF
    #dfMA contains 200 MA of price
    #dfDetrebd contains the detrended AP prices (for White's reality test)

    dfA = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    dfB = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    dfS = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    dfZ = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    dfMA = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    dfDetrend = dfP.drop(labels=None, axis=1, columns=dfP.columns)

    # calculating the three performance measures in accordance with their windows
    # first, calculate the percent change compared to Aperiods ago
    dfA = dfP.pct_change(periods=Aperiods-1, fill_method='pad', limit=None, freq=None) #is counting window from 0
    dfB = dfP.pct_change(periods=Bperiods-1, fill_method='pad', limit=None, freq=None) #is counting window from 0
    dfR = dfP.pct_change(periods=1, fill_method='pad', limit=None, freq=None) #is counting window from 0

    columns = dfP.shape[1]
    for column in range(columns):
        # Annualized Volatility
        dfS[dfP.columns[column]] = (dfR[dfP.columns[column]].rolling(window=Speriods).std())*math.sqrt(252)
        # standard deviation of price compared to 200 days MA
        dfZ[dfP.columns[column]] = (dfP[dfP.columns[column]]-dfP[dfP.columns[column]].rolling(window=Zperiods).mean())/dfP[dfP.columns[column]].rolling(window=Zperiods).std()
        # 200 days MA mean
        dfMA[dfP.columns[column]] = (dfP[dfP.columns[column]].rolling(window=MAperiods).mean())
        # detrended AP pirces, basically residual of the linear fit and the actual price after linear fit
        dfDetrend[dfAP.columns[column]] = detrendPrice.detrendPrice(dfAP[dfAP.columns[column]]).values

    # Ranking each ETF w.r.t. short moving average of returns
    dfA_ranks = dfP.copy(deep=True)
    # n columns of zero, where n is the number of stocks
    dfA_ranks[:] = 0

    # integer values
    columns = dfA_ranks.shape[1]
    rows = dfA_ranks.shape[0]

    # derive rank (based on short MA) from the short MA
    # put the derived values in dfA_ranks
    for row in range(rows):
        # short MA rows
        arr_row = dfA.iloc[row].values
        if momentum == 1:
            temp = arr_row.argsort() #sort momentum, best is ETF with largest return
        else:
            temp = (-arr_row).argsort()[:arr_row.size] #sort reversion, best is ETF with lowest return
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(1,len(arr_row)+1)
        for column in range(columns):
            dfA_ranks.iat[row, column] = ranks[column]

    dfB_ranks = dfP.copy(deep=True)
    dfB_ranks[:] = 0

    columns = dfB_ranks.shape[1]
    rows = dfB_ranks.shape[0]

    # this loop takes each row of the B dataframe, puts the row into an array,
    # within the array the contents are ranked,
    # then the ranks are placed into the B_ranks dataframe one by one
    for row in range(rows):
        arr_row = dfB.iloc[row].values
        if momentum == 1:
            temp = arr_row.argsort() #sort momentum
        else:
            temp = (-arr_row).argsort()[:arr_row.size] #sort reversion
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(1,len(arr_row)+1)
        for column in range(columns):
            dfB_ranks.iat[row, column] = ranks[column]

    dfS_ranks = dfP.copy(deep=True)
    dfS_ranks[:] = 0

    columns = dfS_ranks.shape[1]
    rows = dfS_ranks.shape[0]

    #this loop takes each row of the dfS dataframe, puts the row into an array,
    #within the array the contents are ranked,
    #then the ranks are placed into the dfS_ranks dataframe one by one
    for row in range(rows):
        arr_row = dfS.iloc[row].values
        if volmomentum == 1:
            temp = arr_row.argsort() #sort momentum, best is highest volatility
        else:
            temp = (-arr_row).argsort()[:arr_row.size] #sort reversion, best is lowest volatility
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(1,len(arr_row)+1)
        for column in range(columns):
            dfS_ranks.iat[row, column] = ranks[column]

    #Weights of the varous ranks ####################################################################################
    dfA_ranks = dfA_ranks.multiply(.3) #.3 default
    dfB_ranks = dfB_ranks.multiply(.4) #.4 default
    dfS_ranks = dfS_ranks.multiply(.3) #.3 default
    dfAll_ranks = dfA_ranks.add(dfB_ranks, fill_value=0)
    dfAll_ranks = dfAll_ranks.add(dfS_ranks, fill_value=0)

    # Gavin df select
    # dfSelect = pd.DataFrame(columns=["A", "B", "C", "D"])

    # first find out what dates are necessary
    date_list = []
    tiktok = True
    for date in dfAll_ranks.index:
        if date.dayofweek == 2:
            if tiktok:
                date_list.append(date)
                tiktok = False
            else:
                tiktok = True

    dfSelect = pd.DataFrame(index=date_list, columns=["A", "B", "C", "D"])
    tiktok_65_to_68 = 65   # chr("A") is 65
    prev_date = None
    for date in dfSelect.index:
        if prev_date:
            dfSelect.loc[date] = dfSelect.loc[prev_date]
        prev_date = date
        row = dfAll_ranks.loc[date]
        best_etf = row.idxmax()
        # print(row, best_etf)
        # print(best_etf)
        dfSelect.loc[date, chr(tiktok_65_to_68)] = best_etf
        if tiktok_65_to_68 == 68:   # chr("D") is 68
            tiktok_65_to_68 = 65
        else:
            tiktok_65_to_68 += 1
    return dfSelect


# P0 - Portfolio Initial Value
# N - Number of periods
# K - Number of “rungs” on the ladder
# X - Matrix with stock selections (KxN)
# D - Matrix with daily stock prices
# Returns dataframe Y of portfolio values from the end of periods 0 to N

def GetPortfolioValues(P0, N, K, X, D):
    Y = pd.DataFrame(np.zeros(N), index=X.index, columns=['Value'])
    values = Y['Value']
    values[0] = P0
    for i in range(0, N-1):
        for j in range(0, K):
            ticker = X.iloc[i,j]
            previous_day = X.iloc[i].name
            today = X.iloc[i+1].name
            previous_price = GetStockPrice(ticker, previous_day, D)
            today_price = GetStockPrice(ticker, today, D)
            gain = today_price / previous_price
            allocation = values[i] / K
            values[i+1] += gain * allocation
    return Y

# T - Ticker
# N - Date
# D - Daily Price Data
# Returns stock price p at end of Nth period for ticker T

def GetStockPrice(T, N, D):
    if type(T) is str:
        Y = D[T]
        p = Y[N]
        return p
    elif math.isnan(T):
        return 1

def vol(returns):
    # Return the annualized volatility
    return np.std(returns)*np.sqrt(252)

def sharpe_ratio(er, returns, rf):
    return (er - rf) / vol(returns)

# This calculates the momentum score and generates the ETF ladder. Uncomment to recalculate on a new set of ETFs
# result = single_type_return("My_ETF.csv")

# This saves the result of the previous function to a pickle file in order to save computation time
# result.to_pickle("result.pkl")

# This reads the saved pickle data file back so it can be used for the following calculations
df = pd.read_pickle("result.pkl")

# Gets the daily prices and adjusted prices
dfP, dfAP = get_dfp("My_ETF.csv")

# Removes NA or missing data
dfP.dropna(axis=1, inplace=True)

# Portfolio initial funding
portfolio_start = 10000

# Number of periods to calculate over. By default it is set to the total # of periods available in the dataframe
periods = len(df.index)

# Number of rungs of the ladder to consider in the calculation
rungs = 4

# Gets portfolio values over all periods
pdf = GetPortfolioValues(portfolio_start, periods, rungs, df, dfP)
values = pdf['Value']

start_val = portfolio_start
end_val = values.iloc[-1]
start_date = getDate(pdf.iloc[0].name)
end_date = getDate(pdf.iloc[-1].name)

days = (end_date - start_date).days
TotaAnnReturn = (end_val - start_val) / start_val / (days / 360)
TotaAnnReturn_trading = (end_val - start_val) / start_val / (days / 252)

CAGR_trading = round(((float(end_val) / float(start_val)) ** (1 / (days / 252.0))).real - 1,
                     4)  # when raised to an exponent I am getting a complex number, I need only the real part
CAGR = round(((float(end_val) / float(start_val)) ** (1 / (days / 350.0))).real - 1,
             4)  # when raised to an exponent I am getting a complex number, I need only the real part

sharpe = sharpe_ratio(TotaAnnReturn, values.pct_change(), 0)

txt = "TotaAnnReturn = %f" % (TotaAnnReturn * 100) + '\n'
txt += ("CAGR = %f" % (CAGR * 100)) + '\n'
txt += "Sharpe Ratio = %f" % (round(sharpe, 2)) + '\n'

fig = plt.figure()
plt.plot(values)
fig.suptitle('Portfolio Value Over Time', fontsize=20)
plt.xlabel('Time (Date)', fontsize=18)
plt.ylabel('Value (CAD $)', fontsize=16)
plt.figtext(0.5, 0.1, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig(r'Results\%s.png' %datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
plt.show()