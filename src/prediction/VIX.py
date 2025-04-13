import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. 定义美股权重股
MAGN7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'QQQ', 'SPY']

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

    # 统一日期格式为年-月-日
    stock_data.index = stock_data.index.strftime('%Y-%m-%d')
    stock_data.index = pd.to_datetime(stock_data.index)

    stock_data = stock_data.dropna()
    return stock_data

# 3. 读取 VIX 数据并计算日变化率
def load_vix_data(file_path, start_date, end_date):
    # 读取CSV文件
    vix_df = pd.read_csv(file_path)

    # 将日期列转换为datetime格式，并统一为年-月-日
    vix_df['date'] = pd.to_datetime(vix_df['date']).dt.strftime('%Y-%m-%d')
    vix_df['date'] = pd.to_datetime(vix_df['date'])

    # 筛选指定时间范围
    vix_df = vix_df[(vix_df['date'] >= start_date) & (vix_df['date'] <= end_date)]

    # 设置日期为索引
    vix_df.set_index('date', inplace=True)

    # 使用收盘价作为 VIX 指数值，并计算日变化率
    vix_df['VIX'] = vix_df['close']
    vix_df['VIX_pct_change'] = vix_df['VIX'].pct_change()

    # 只保留变化率列
    vix_df = vix_df[['VIX_pct_change']]

    # 重命名列
    vix_df.columns = ['VIX']

    # 填充空值为0
    vix_df = vix_df.fillna(0)

    return vix_df

# 4. 主分析函数
def analyze_vix_impact(start_date, end_date, vix_file_path):
    # 获取股票数据
    stock_df = get_stock_data(MAGN7_TICKERS, start_date, end_date)

    # 加载 VIX 数据
    vix_df = load_vix_data(vix_file_path, start_date, end_date)

    # 合并股票和 VIX 数据
    merged_df = stock_df.join(vix_df, how='inner')

    # 回归分析（以 VIX 变化率为自变量）
    results = {}
    scaler = StandardScaler()
    for ticker in MAGN7_TICKERS:
        X = scaler.fit_transform(merged_df[['VIX']])
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
        ax1.plot(merged_df.index, merged_df[ticker],
                 label=f'{ticker} Returns', color='tab:blue', alpha=0.7)
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'{ticker} Daily Returns', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)

        # 右Y轴 - VIX 变化率
        ax2 = ax1.twinx()
        ax2.plot(merged_df.index, merged_df['VIX'],
                 label='VIX Change', color='red', linewidth=1)
        ax2.set_ylabel('VIX Change', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(f'VIX Change vs {ticker} Returns')
        plt.tight_layout()

    # 相关性热图
    plt.figure(figsize=(12, 10))
    correlation_matrix = merged_df[MAGN7_TICKERS + ['VIX']].corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(correlation_matrix)),
               correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix)),
               correlation_matrix.columns)
    plt.title('Correlation Heatmap: VIX Change vs Stock Returns')

    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                     ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()

# 6. 主程序
if __name__ == "__main__":
    # 设置分析时间范围
    start_date = "2017-01-20"
    end_date = "2021-01-20"

    # VIX 数据CSV文件路径（请替换为实际路径）
    vix_file_path = "D:\PolyU\Machine Learning\project\code\data\\vix_daily.csv"

    # 运行分析
    results, merged_df = analyze_vix_impact(start_date, end_date, vix_file_path)

    # 输出结果
    print("VIX 变化对美股权重股真实收益率的影响分析结果：")
    print("\n回归系数（影响方向和强度）：")
    print(results['coefficient'])
    print("\nP值（统计显著性）：")
    print(results['p_value'])
    print("\nR平方（解释力度）：")
    print(results['r_squared'])

    # 调用绘图函数
    plot_analysis_results(merged_df)