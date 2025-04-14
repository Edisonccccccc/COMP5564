
# COMP5564_MLAF_Project: F(Trump) = Market

## 🕒 数据时间范围说明

- **分析时间段**：2015-01-01 至 2021-01-08
  - ✅ 2015-01-01：保留特朗普当选前一年的历史言论数据
  - ✅ 2016-11-08：特朗普当选总统
  - ✅ 2017-01-20：特朗普正式就任总统
  - ✅ 2021-01-08：特朗普被永久封禁 Twitter，数据集终止
  - ✅ 2024-11-05：特朗普赢得大选（先搁置）

## 📁 数据文件说明（data/）

| 文件名                      | 描述                                                         | 时间范围                 |
|---------------------------|--------------------------------------------------------------|--------------------------|
| `realdonaldtrump.csv`     | Trump 官方账号推文原始数据（含文本、日期、转发、点赞等字段）       | 2009-05-04 ~ 2020-06-17 |
| `trumptweets.csv`         | 补充 Trump 推文数据，结构与上者相同，用于数据合并提升覆盖度         | 2009-05-04 ~ 2020-01-20 |
| `trump_sentiment_daily.csv` | 从上述推文中提取的每日平均情绪得分（VADER 分析）                     | 2009-05-04 ~ 2020-06-17 |
| `stock_closes.csv`        | MAGN7 及指数（AAPL, MSFT, NVDA 等）的每日收盘价（用于 LSTM 等模型）     | 2015-01-02 ~ 2021-01-07 |
| `stock_returns.csv`       | MAGN7 股票每日收益率（基于收盘价 pct_change 后计算）                | 2015-01-05 ~ 2021-01-07 |
| `us_treasury_yields_daily.csv` | 美国国债（10Y 等）收益率原始数据（日频，来源于外部 CSV）              | 1962-01-02 ~ 2024-06-21 |
| `sentiment_classified_tweets_2017_2021_batch.csv` | 经人工标注的 Trump 推文情绪分类数据（pos/neg/neu 三类）         | 2017-01-20 ~ 2021-01-08 |
| `trump_sentiment_labeled_no_classification.csv`   | Trump 推文每日情绪得分（未分类标签，可能用于回归）             | 2017-01-20 ~ 2021-01-08 |

## 项目进展总结

### ✅ 已完成
1. **数据准备**
   - 获取 MAGN7（AAPL, MSFT, NVDA 等）股票日收益率（使用 yfinance，时间范围为 2015-01-01 至 2021-01-08）
   - 导入美国国债收益率 CSV 文件
   - 构造 Trump 情绪因子：
     - 前期使用模拟情绪（范围 [-1, 1]，3日滚动平均）
     - 当前已替换为基于真实推文文本的每日情绪得分（使用 VADER）
     - 输出 Trump 推文情绪得分的日期对照表 CSV（`output/trump_sentiment_daily.csv`）

2. **建模分析**
   - 单因子回归：股票收益率 ~ 国债收益率变动率（基线模型）
   - 单因子回归：股票收益率 ~ 模拟 Trump 情绪（原型测试）

---

### ❗待完成
1. 多因子建模：股票收益率 ~ 国债 + Trump 情绪
2. 考察情绪滞后效应（.shift(n)）
3. 增加显著性检验与模型稳健性分析
4. 输出最终图表与可视化报告

## 📦 各代码模块结构说明

### `src/preprocessing/` - 数据预处理模块

#### 1. `export_stock_closes.py`
- **输入**：
  - 股票代码列表（如 MAGN7）
  - 时间区间（如 2015-01-01 ~ 2021-01-08）
- **处理**：
  - 使用 yfinance 抓取每日收盘价数据
- **输出**：
  - `data/raw/stock_closes.csv`

#### 2. `export_stock_returns.py`
- **输入**：
  - `stock_closes.csv`
- **处理**：
  - 对收盘价做 `pct_change()` 并对齐处理
- **输出**：
  - `data/raw/stock_returns.csv`

---

### `src/regression/` - 探因/解释性建模模块

#### 1. `analyze_trump_sentiment.py`
- **输入**：
  - `kaggle_realdonaldtrump.csv`, `kaggle_trumptweets.csv`
- **处理**：
  - 合并、去重、按日期聚合情绪得分（VADER）
- **输出**：
  - `data/processed/trump_sentiment_daily.csv`

#### 3. `us_treasury_yields_daily.py`
- **输入**：
  - 股票每日收益率
  - 美债收益率（us_treasury_yields_daily.csv）
- **处理**：
  - 美债收益率变动率计算
  - OLS 回归（收益率 ~ 美债变动率）
- **输出**：
  - 回归结果表格
  - 对应图表与热图

#### 4. `trump_tweet.py`
- **输入**：
  - MAGN7 股票历史收益率
  - Trump 每日情绪得分
- **处理**：
  - 对每支股票做 OLS 回归（收益率 ~ 情绪得分）
- **输出**：
  - 回归系数、p 值、R² 表格
  - 情绪-股价双轴图与热图

---

### `src/prediction/` - 时间序列预测模块

#### 1. `train_VIX.py`
- **输入**：
  - `vix_daily.csv`：VIX 恐慌指数（字段含日期和 close）
  - `yfinance` 拉取的股票历史收盘价与成交量（如 QQQ, SPY）
- **处理**：
  - 合并 VIX 与股票数据，标准化后构造滑动窗口序列
  - 使用 2 層 LSTM 进行收盘价预测
  - 训练集与测试集分开计算 MSE 与 R²
- **输出**：
  - 训练与测试集的预测曲线图
  - 控制台输出每只股票的 MSE 与 R²

#### 2. `train_contrast.py`
- **输入**：
  - `yfinance` 拉取的 QQQ 与 SPY 的历史收盘价与成交量（Close, Volume）
- **处理**：
  - 为每支股票构建特征（自身的 Close + Volume），标准化后构造滑动窗口序列
  - 构建双层 LSTM 模型进行收盘价预测
  - 对每支股票分别训练并输出训练/测试集的预测曲线图
- **输出**：
  - 每支股票的预测可视化图
  - 控制台输出训练/测试集的 MSE 与 R²

#### 3. `train_treasury_yields.csv.py`
- **输入**：
  - 收盘价、国债收益率
- **处理**：
  - 使用 LSTM 进行时间序列预测
- **输出**：
  - 预测 vs 实际图，误差指标等

#### 4. `VIX.py`
- **输入**：
  - `vix_daily.csv`：VIX 恐慌指数每日变化率
  - `yfinance` 拉取的 MAGN7 权重股收盘价（按日计算收益率）
- **处理**：
  - 计算 VIX 日变化率并标准化
  - 构建回归模型：各股票收益率 ~ VIX 日变化率
  - 输出回归系数、P 值和 R²
  - 绘制每支股票收益率与 VIX 的双轴图 + 相关性热图
- **输出**：
  - 控制台输出每支股票的回归结果表
  - 所有股票回归图与 VIX 联动可视化

#### 2. `multi_factor_regression.py`
- **输入**：
  - `stock_returns.csv`：MAGN7 股票的每日收益率（字段包括 AAPL, MSFT 等）
  - `us_treasury_yields_daily.csv`：美债收益率（每日 US10Y）
  - `trump_sentiment_daily.csv`：每日 Trump 情绪得分
- **处理**：
  - 合并数据后标准化 US10Y 与情绪得分
  - 对每支股票构建线性回归模型（收益率 ~ 国债 + 情绪）
- **输出**：
  - 控制台打印每支股票的回归系数、R² 和 MSE