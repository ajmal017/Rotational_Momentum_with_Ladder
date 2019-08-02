import numpy as np

# P0 - Portfolio Initial Value
# N - Number of periods
# K - Number of “rungs” on the ladder
# X - Matrix with stock selections (KxN)
# Returns array Y of portfolio values from the end of periods 0 to N

def GetPortfolioValues(P0, N, K, X):
    Y = np.zeros(N)
    Y[0] = P0
    for i in range(1, N):
        for j in range(1, K):
            Y[i] += GetStockPrice(X[j,i], i) / GetStockPrice(X[j,i], i-1) * Y[i-1] / K
    return Y

# T - Ticker
# N - Period # or Date
# X - Hash table of [Ticker, Daily Prices of Ticker] pairs
# Returns stock price p at end of Nth period for ticker T

def GetStockPrice(T, N, X):
    Y = X[T]
    p = Y[N]
    return p