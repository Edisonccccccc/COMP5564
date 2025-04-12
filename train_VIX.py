import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import matplotlib.pyplot as plt

# 1. 设置时间范围和股票代码
start_date = '2017-01-20'
end_date = '2021-01-20'
tickers = ['QQQ', 'SPY']  # 你可以在这里添加更多股票代码，例如 ['QQQ', 'SPY', 'AAPL', 'GOOGL']

# 2. 读取恐慌指数
treasury_data_path = r'D:\PolyU\Machine Learning\project\code\data\vix_daily.csv'
treasury_data = pd.read_csv(treasury_data_path)

# 假设CSV文件中的日期列名为 'date'，10年期国债收益率列名为 'US10Y'
treasury_data['date'] = pd.to_datetime(treasury_data['date'])
treasury_data.set_index('date', inplace=True)

# 筛选指定时间范围内的国债收益率数据
treasury_data = treasury_data[start_date:end_date]

# 3. 获取所有股票的数据
stock_data = yf.download(tickers, start=start_date, end=end_date)[['Close', 'Volume']]

# 重命名列名以便合并
close_columns = [f"{ticker}_Close" for ticker in tickers]
volume_columns = [f"{ticker}_Volume" for ticker in tickers]
stock_data.columns = close_columns + volume_columns

# 4. 合并国债收益率和股票数据
data = stock_data.join(treasury_data[['close']], how='inner')

# 5. 数据预处理
# 填充缺失值（如果有）
data.fillna(method='ffill', inplace=True)


# 6. 定义函数：为每个股票训练和预测
def train_and_predict_for_ticker(ticker, data, time_step=60):
    # 提取特征和目标变量
    close_col = f"{ticker}_Close"
    volume_col = f"{ticker}_Volume"
    features = [close_col, volume_col, 'close']  # 特征：该股票的收盘价、交易量和恐慌指数
    target = close_col  # 目标：该股票的收盘价

    # 标准化数据
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    # 创建训练数据集
    def create_dataset(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, 0])  # 预测收盘价（第0列）
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, time_step)

    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), verbose=0)

    # 预测
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # 反标准化预测结果
    train_predict_full = np.zeros((train_predict.shape[0], len(features)))
    test_predict_full = np.zeros((test_predict.shape[0], len(features)))
    train_predict_full[:, 0] = train_predict[:, 0]
    test_predict_full[:, 0] = test_predict[:, 0]

    train_predict = scaler.inverse_transform(train_predict_full)[:, 0]
    test_predict = scaler.inverse_transform(test_predict_full)[:, 0]

    # 反标准化真实值
    y_train_full = np.zeros((y_train.shape[0], len(features)))
    y_test_full = np.zeros((y_test.shape[0], len(features)))
    y_train_full[:, 0] = y_train
    y_test_full[:, 0] = y_test

    y_train_inv = scaler.inverse_transform(y_train_full)[:, 0]
    y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]

    # 计算MSE和R²
    train_mse = mean_squared_error(y_train_inv, train_predict)
    train_r2 = r2_score(y_train_inv, train_predict)
    test_mse = mean_squared_error(y_test_inv, test_predict)
    test_r2 = r2_score(y_test_inv, test_predict)

    # 绘制折线图
    plt.figure(figsize=(14, 5))
    plt.plot(data.index[time_step:train_size + time_step], y_train_inv, label='Train Actual')
    plt.plot(data.index[time_step:train_size + time_step], train_predict, label='Train Predict')
    plt.plot(data.index[train_size + time_step:], y_test_inv, label='Test Actual')
    plt.plot(data.index[train_size + time_step:], test_predict, label='Test Predict')
    plt.title(f'{ticker} Price Prediction using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return train_mse, train_r2, test_mse, test_r2


# 7. 逐个股票进行训练和预测
for ticker in tickers:
    print(f"\n正在处理股票: {ticker}")
    train_mse, train_r2, test_mse, test_r2 = train_and_predict_for_ticker(ticker, data)

    # 输出MSE和R²
    print(f"{ticker} 训练集 MSE: {train_mse:.4f}")
    print(f"{ticker} 训练集 R²: {train_r2:.4f}")
    print(f"{ticker} 测试集 MSE: {test_mse:.4f}")
    print(f"{ticker} 测试集 R²: {test_r2:.4f}")