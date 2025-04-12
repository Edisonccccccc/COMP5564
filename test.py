import yfinance as yf

ticker = yf.Ticker("META")
hist = ticker.history(start="2017-01-01", end="2017-01-10")
print(hist.head())