# -*- coding: utf-8 -*-
"""
"""
import pandas as pd

pd.core.common.is_list_like = pd.api.types.is_list_like  # datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import datetime

import yfinance as yf
import os

yf.pdr_override()  # <== that's all it takes :-)

yahoo_data = "/home/guanyush/Pictures/APS1051/QUERCUS_5_MOMENTUM_LOOK_BACK_HOLDING_WHITES_REALITY_CHECK/yahoo_data"

# Example1
# download dataframe
# data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
# download Panel
# data2 = pdr.get_data_yahoo(["SPY", "IWM"], start="2017-01-01", end="2017-04-30")
# example2
# start = datetime.datetime(2017, 1, 1)
# symbol = 'SPY'
# data = pdr.get_data_yahoo(symbol, start=start, end=end)
# data.to_csv("C:\\Users\\Rosario\\Documents\\NeuralNetworksMachineLearning\\LSTMReturnPrediction\\data\\YahooSPY.csv")

# start_date=datetime.datetime(2003, 1, 1)
start_date = datetime.datetime(2010, 1, 4)
# end_date= datetime.datetime.now()
end_date = datetime.datetime(2017, 1, 1)

# stock_list = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
# stock_list = ["BIL", "SHY", "IEI", "IEF", "TLH", "TLT", "TIP"]
stock_list = ["SPY", "GLD", "XLF", "DIA", "XLK", "MDY", "SDY", "XLV", "XLY", "XLP", "XLE", "XLI", "XLU", "JNK", "BIL", "XLC", "SPYG", "SPSB", "SPIB", "SPDW", "XLB", "SPAB", "XBI", "FLRN", "CWB", "SHM", "SPTM", "XLRE", "TOTL", "SPYV", "SJNK", "TFI", "SPEM", "SPLG", "RWR", "SPTL", "RWO", "FEZ", "RWX", "SRLN", "KRE", "SLYV", "SLYG", "XOP", "SPYD", "KBE", "CWI", "ITE", "MDYG", "IPE", "SPMD", "XAR", "MDYV", "SPTS", "SPSM", "SLY", "GXC", "BWX", "GNR", "KIE", "GLDM", "PSK", "DWX", "HYMB", "EBND", "GWX", "NANR", "LGLV", "XHB", "XHE", "SPLB", "ONEV", "XME", "EWX", "GMF", "WIP", "EDIV", "ONEO", "ONEY", "QUS", "SPYX", "XNTK", "GII", "TIPX", "XSD", "QEFA", "BWZ", "QEMM", "SHE", "WDIV", "MBG", "GAL", "XSW", "XRT", "SMLV", "FEU", "XPH", "ACIM", "ULST", "IBND", "XES", "XTN", "STOT", "DWFI", "RLY", "INKM", "DGT", "XHS", "CBND", "XITK", "CJNK", "EFAX", "LOWC", "EMTL", "EEMX", "KOMP", "XTL", "MMTM", "KCE", "SYE", "GLDW", "SYG", "SYV", "XWEB", "QWLD", "ZCAN", "SPYB", "VLU", "SMEZ", "XLSR", "FITE", "FISR", "SIMS", "ZGBR", "ZHOK", "ZJPN", "HAIL", "ZDEU", "CNRG", "XTH", "ROKT"]

stock_str = ""
for i in range(len(stock_list)):
    stock_str = stock_str + stock_list[i] + "."

main_df = pd.DataFrame()

for stock in range(len(stock_list)):
    df = pdr.get_data_yahoo(stock_list[stock], start=start_date, end=end_date)
    df.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)
    df.rename(columns={'Adj Close': stock_list[stock]}, inplace=True)
    if main_df.empty:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.to_csv(os.path.join(yahoo_data, "My_ETF_AP.csv"))

main_df = pd.DataFrame()

for stock in range(len(stock_list)):
    df = pdr.get_data_yahoo(stock_list[stock], start=start_date, end=end_date)
    df.drop(['Adj Close', 'High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)
    df.rename(columns={'Close': stock_list[stock]}, inplace=True)
    if main_df.empty:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.to_csv(os.path.join(yahoo_data, "My_ETF.csv"))

