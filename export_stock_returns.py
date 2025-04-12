"""
导出 MAGN7 股票每日收益率数据

✅ 输入：
- 股票代码列表：MAGN7（AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM, QQQ, SPY）
- 时间范围（可调整）

📤 输出：
- data/stock_returns.csv：包含 date + 各股票每日收益率的表格
"""

import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'QQQ', 'SPY']
start_date = '2017-01-20'
end_date = '2021-01-20'

print("📥 正在下载股票收盘价数据...")
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

returns_df = pd.DataFrame()
returns_df['date'] = data.index

for ticker in tickers:
    if isinstance(data[ticker], pd.DataFrame):
        returns_df[ticker] = data[ticker]['Close'].pct_change()

returns_df.dropna(how='all', subset=tickers, inplace=True)

# 保存到 CSV
returns_df.to_csv("data/stock_returns.csv", index=False)
print("✅ 已保存至 data/stock_returns.csv")
