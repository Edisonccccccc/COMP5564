import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 初始化情绪分析器
analyzer = SentimentIntensityAnalyzer()

# 读取两个 CSV 文件
df1 = pd.read_csv('data/realdonaldtrump.csv')
df2 = pd.read_csv('data/trumptweets.csv')

# 合并数据并保留必要字段
df = pd.concat([df1, df2], ignore_index=True)
df['date'] = pd.to_datetime(df['date']).dt.date  # 转换为日期（去除时分秒）

# 进行情绪打分
def compute_sentiment(text):
    if pd.isna(text):
        return 0.0
    score = analyzer.polarity_scores(str(text))
    return score['compound']  # 综合情绪值（-1 到 1）

df['sentiment_score'] = df['content'].apply(compute_sentiment)

# 聚合为每日平均情绪
daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()

# 输出为 CSV
daily_sentiment.to_csv('output/trump_sentiment_daily.csv', index=False)

print("✅ 情绪得分已生成并保存到 output/trump_sentiment_daily.csv")
