import bs4 as bs
import requests
import tiingoconnect_mod
import pandas as pd
import datetime as dt
import os
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web

def DJIA_tickers():
	'''

	Parses Slickcharts Webpage to obtain the tickers for each current listed stock in the DJIA.

	'''

	# resp = requests.get('https://money.cnn.com/data/dow30/')
	# soup = bs.BeautifulSoup(resp.text, "lxml")
	# table = soup.find('table', {'class': 'wsod_dataTable wsod_dataTableBig'})
	tickers = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'SHY']
	#tickers = ['SHY', 'BIL', 'IEI', 'IEF', 'TLH', 'TLT', 'TIP']
	#for row in table.findAll('tr')[1:]:
		#ticker = row.findAll('td')[0].text
		#ticker = ticker.split('\xa0', 1)[0]
		#tickers.append(ticker)
    
	return tickers

# DJIA_tickers()


def API_data():
	'''

	Uses Tiingo API to access all the historical data for the current stocks listed in the DJIA.

	'''
	#DJIA = "DIA" #not '.DJI', doesn't work
	tickers = DJIA_tickers()
	#tickers.append(DJIA)

	if not os.path.exists('our_ETFs'):
		os.makedirs('our_ETFs')

	end = dt.datetime.now()
	start = end - dt.timedelta(days=365*5)
	
	for ticker in tickers:
		print(ticker)
		if not os.path.exists('our_ETFs/{}.csv'.format(ticker)):
			df = tiingoconnect_mod.DataReader(ticker, start, end)
			df.to_csv('our_ETFs/{}.csv'.format(ticker))
		else:
			print('Information already acquired for {}'.format(ticker))


# API_data()

def compile_data():
	'''

	Creates a dataframe with the compiled adjusted closing for all the stocks in the DJIA.

	'''

	tickers = DJIA_tickers()

	API_data()

	main_df = pd.DataFrame()

	for count,ticker in enumerate(tickers):
		df = pd.read_csv('our_ETFs/{}.csv'.format(ticker))
		df.set_index('Date', inplace= True)

		df.rename(columns = {'Adj Close': ticker}, inplace=True)

		df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor'], 1, inplace=True)

		if main_df.empty:
			main_df = df
		else:
			main_df = main_df.join(df, how='outer')

		if count % 10 == 0:
			print(count)

	print(main_df.head())
	main_df.to_csv('our_ETFs_adjcloses.csv')

	for count,ticker in enumerate(tickers):
		df = pd.read_csv('our_ETFs/{}.csv'.format(ticker))
		df.set_index('Date', inplace=True)

		df.rename(columns={'Close': ticker}, inplace=True)

		df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor'], 1, inplace=True)

		if main_df.empty:
			main_df = df
		else:
			main_df = main_df.join(df, how='outer', rsuffix='_')

		if count % 10 == 0:
			print(count)

	print(main_df.head())
	main_df.to_csv('our_ETFs_closes.csv')

compile_data()




