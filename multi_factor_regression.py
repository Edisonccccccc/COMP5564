"""
多因子回归分析：股票收益率 ~ 国债收益率变动 + Trump 情绪得分

✅ 输入：
- data/stock_returns.csv：包含 MAGN7 股票的每日收益率，字段包括 date, AAPL, MSFT 等
- data/us_treasury_yields_daily.csv：美债收益率原始数据（包含 US10Y 等字段，每日绝对水平）
- data/trump_sentiment_daily.csv：每日 Trump 情绪得分（sentiment_score）

📤 输出：
- 控制台输出每支股票的回归系数、R²、MSE

"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 加载数据
stock_df = pd.read_csv("data/stock_returns.csv", parse_dates=["date"]).set_index("date")
yield_df = pd.read_csv("data/us_treasury_yields_daily.csv", parse_dates=["date"]).set_index("date")
sentiment_df = pd.read_csv("data/trump_sentiment_daily.csv", parse_dates=["date"]).set_index("date")

# 合并所有数据
merged_df = stock_df.join([yield_df["US10Y"], sentiment_df["sentiment_score"]], how="inner").dropna()

# 拟合每支股票的回归模型
tickers = [col for col in stock_df.columns if col != "date"]

print("📊 多因子回归分析结果：\n")

for ticker in tickers:
    X = merged_df[["US10Y", "sentiment_score"]]
    y = merged_df[ticker]

    # 标准化 X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拟合模型
    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    coefs = dict(zip(["US10Y", "sentiment_score"], model.coef_))

    print(f"📌 {ticker}")
    print(f"  R²: {r2:.4f}, MSE: {mse:.6f}")
    print(f"  系数: {coefs}")
    print()
