import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. 定义美股权重股
MAGN7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'QQQ','SPY']


# 2. 获取真实股票数据
def get_stock_data(tickers, start_date, end_date):
    stock_data = pd.DataFrame()
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            df[f'{ticker}'] = df['Close'].pct_change()
            if stock_data.empty:
                stock_data = df[[ticker]]
            else:
                stock_data = stock_data.join(df[[ticker]], how='outer')
            print(f"Successfully retrieved data for {ticker}")
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {str(e)}")
    stock_data = stock_data.dropna()
    return stock_data


# 3. 生成模拟的特朗普推文情感得分并进行3天滚动平均处理
def generate_simulated_sentiment(dates):
    np.random.seed(42)
    # 生成初始随机情感得分
    initial_sentiment = np.random.uniform(-1, 1, len(dates))

    # 创建DataFrame用于处理
    sentiment_df = pd.DataFrame({
        'date': dates,
        'initial_sentiment': initial_sentiment
    })
    sentiment_df.set_index('date', inplace=True)

    # 向前滚动3天取平均值
    # 用rolling计算前3天的均值（包括当天），min_periods=1确保即使不足3天也有值
    sentiment_df['tweet_sentiment'] = (sentiment_df['initial_sentiment']
                                       .rolling(window=3, min_periods=1)
                                       .mean()
                                       .shift(-2))  # 向前滚动2天加上当天共3天

    # 填充空值为0
    sentiment_df['tweet_sentiment'] = sentiment_df['tweet_sentiment'].fillna(0)

    return sentiment_df['tweet_sentiment'].values


# 4. 主分析函数
def analyze_sentiment_impact(start_date, end_date):
    stock_df = get_stock_data(MAGN7_TICKERS, start_date, end_date)
    dates = stock_df.index
    tweet_sentiment = generate_simulated_sentiment(dates)
    tweet_df = pd.DataFrame({
        'date': dates,
        'tweet_sentiment': tweet_sentiment
    })
    merged_df = pd.merge(
        tweet_df,
        stock_df,
        left_on='date',
        right_index=True,
        how='inner'
    )
    results = {}
    scaler = StandardScaler()
    for ticker in MAGN7_TICKERS:
        X = scaler.fit_transform(merged_df[['tweet_sentiment']])
        y = merged_df[ticker]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        results[ticker] = {
            'coefficient': model.params[1],
            'p_value': model.pvalues[1],
            'r_squared': model.rsquared
        }
    return pd.DataFrame(results).T, merged_df


# 5. 绘图函数
def plot_analysis_results(merged_df):
    # 为每只股票创建单独的时间序列图
    for ticker in MAGN7_TICKERS:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 左Y轴 - 股票收益率
        ax1.plot(merged_df['date'], merged_df[ticker],
                 label=f'{ticker} Returns', color='tab:blue', alpha=0.7)
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'{ticker} Daily Returns', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)

        # 右Y轴 - 情感得分
        ax2 = ax1.twinx()
        ax2.plot(merged_df['date'], merged_df['tweet_sentiment'],
                 label='Simulated Tweet Sentiment', color='red', linewidth=1)
        ax2.set_ylabel('Sentiment Score (-1 to 1)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(f'Simulated Trump Tweet Sentiment vs {ticker} Returns')
        plt.tight_layout()

    # 相关性热图
    plt.figure(figsize=(10, 8))
    correlation_matrix = merged_df[MAGN7_TICKERS + ['tweet_sentiment']].corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(correlation_matrix)),
               correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix)),
               correlation_matrix.columns)
    plt.title('Correlation Heatmap: Tweet Sentiment vs Stock Returns')

    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                     ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()


# 6. 主程序
if __name__ == "__main__":
    # 设置分析时间范围
    start_date = "2020-01-01"
    end_date = "2020-12-31"

    # 运行分析
    results, merged_df = analyze_sentiment_impact(start_date, end_date)

    # 输出结果
    print("模拟的特朗普推文情感对纳斯达克7巨头真实收益率的影响分析结果：")
    print("\n回归系数（影响方向和强度）：")
    print(results['coefficient'])
    print("\nP值（统计显著性）：")
    print(results['p_value'])
    print("\nR平方（解释力度）：")
    print(results['r_squared'])

    # 调用绘图函数
    plot_analysis_results(merged_df)