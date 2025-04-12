"""
导出 MAGN7 股票每日收盘价数据

✅ 输入：
- 股票代码列表：MAGN7（AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM, QQQ, SPY）
- 时间范围（可调整）

📤 输出：
- data/stock_closes.csv：包含 date + 各股票每日收盘价的表格
"""

import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'QQQ', 'SPY']
start_date = '1962-01-02'
end_date = '2024-06-21'

print("📥 正在下载股票收盘价数据...")
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

# 提取收盘价
closes_df = pd.DataFrame(index=data.index)
for ticker in tickers:
    if isinstance(data[ticker], pd.DataFrame):
        closes_df[ticker] = data[ticker]['Close']

# 重设索引并保存
closes_df = closes_df.reset_index()
closes_df.rename(columns={'Date': 'date'}, inplace=True)
closes_df.to_csv("data/stock_closes.csv", index=False)

print("✅ 已保存至 data/stock_closes.csv")