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
start_date = '2015-01-01'
end_date = '2021-01-08'

print("📥 正在下载股票收盘价数据...")
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

returns_df = pd.DataFrame(index=data.index)

for ticker in tickers:
    if isinstance(data[ticker], pd.DataFrame):
        close_series = data[ticker]['Close']
        returns_df[ticker] = close_series.pct_change()

# 删除第一行（因 pct_change 后为 NaN）
returns_df = returns_df.iloc[1:]

# 重设索引并添加日期列
returns_df.reset_index(inplace=True)
returns_df.rename(columns={'Date': 'date'}, inplace=True)

# 保存到 CSV
returns_df.to_csv("data/stock_returns.csv", index=False)
print("✅ 已保存至 data/stock_returns.csv")
