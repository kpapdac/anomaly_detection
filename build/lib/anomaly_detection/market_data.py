import pandas as pd
from datetime import datetime
import yfinance as yf
import pandas_datareader.data as web

sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_constituents = pd.read_html(sp_url, header=0)[0]

start = '2014'
end = datetime(2017, 5, 24)


my_data = yf.download('TSLA', start='2021-12-17', end='2022-12-17', progress=False)