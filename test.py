import yfinance as yf

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'QQQ', 'SPY']
start_date = '2020-01-01'
end_date = '2020-01-10'

data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

# 打印前几行
print(data.head())