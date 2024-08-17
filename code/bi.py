import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud
import jieba
from collections import Counter
import re
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import squarify

# Paths
conversation_file = '../data/conversation/chatbot.json'
output_folder = '../BI'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Font path for supporting Chinese characters
chinese_font_path = "/Users/azure/Library/Fonts/Songti.ttc"  # Update this to Songti SC font path

# 1. 数据加载和预处理
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

data = load_data(conversation_file)

# 常见停用词列表
dedicated_stopwords = set([
    '的', '了', '在', '是', '和', '有', '也', '就', '不', '与', '为', '上', '对', '与', '而', '之', '中', '及', '这', '或', 
    '但', '很', '到', '都', '并', '和', '等', '等', '去', '要', '所', '我们', '你们', '他们', '她们', '它们', '这个', '那个',
    '因此', '已经', '如果', '没有', '可以', '一个', '两个', '这些', '那些', '因为', '所以', '而且', '并且', '从', '通过', '由于', '请问', '吗', '数据',
    '关于', '比如', '然后', '同时', '就是', '而', '并', '可能', '当然', '仍然', '之间', '以上', '以下', '之后', '以前', '虽然', '我', 'K11', '商场'
])

# 停用词与标点符号过滤
def clean_text(text, stopwords):
    text = re.sub(r'[^\w\s]', '', text)
    words = ' '.join(jieba.cut(text))
    return ' '.join([word for word in words.split() if word not in stopwords])

# 2. 基本统计分析
def basic_stats(df):
    total_conversations = len(df)
    avg_question_length = df['question'].str.len().mean()
    avg_answer_length = df['answer'].str.len().mean()
    
    print(f"总对话数: {total_conversations}")
    print(f"平均问题长度: {avg_question_length:.2f}")
    print(f"平均回答长度: {avg_answer_length:.2f}")
    
    df['hour'] = df['timestamp'].dt.hour
    hour_counts = df['hour'].value_counts().sort_index()
    
    bar_fig = go.Figure(data=[
        go.Bar(x=hour_counts.index, y=hour_counts.values, marker_color='rgb(8, 48, 107)')
    ])
    bar_fig.update_layout(
        title='对话时间分布',
        xaxis_title='小时',
        yaxis_title='对话数量',
        font=dict(family="Songti SC", size=14),
        template='plotly_white'
    )
    bar_fig.write_image(f"{output_folder}/time_distribution.png")

    return df

df = basic_stats(data)

# 3. 文本分析
def text_analysis(df, stopwords):
    df['question_words'] = df['question'].apply(lambda x: clean_text(x, stopwords))
    df['answer_words'] = df['answer'].apply(lambda x: clean_text(x, stopwords))
    
    all_words = ' '.join(df['question_words'])
    
    # Generate WordCloud with darker shades of blue
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          font_path=chinese_font_path,
                          relative_scaling=0.17,
                        #   colormap='Blues'  # This colormap will generate darker shades of blue
                          ).generate(all_words)
    
    wordcloud_fig = go.Figure(go.Image(z=wordcloud.to_array()))
    wordcloud_fig.update_layout(
        title="对话内容词云", 
        xaxis={'visible': False}, 
        yaxis={'visible': False},
        font=dict(family="Songti SC", size=14)
    )
    wordcloud_fig.write_image(f"{output_folder}/wordcloud.png")

    word_freq = Counter(all_words.split())
    top_words = word_freq.most_common(20)
    words, counts = zip(*top_words)
    
    bar_fig = go.Figure(data=[
        go.Bar(x=words, y=counts, marker_color='rgb(8, 48, 107)')
    ])
    bar_fig.update_layout(
        title='Top 20 最常见词',
        xaxis_title='词语',
        yaxis_title='频率',
        font=dict(family="Songti SC", size=14),
        template='plotly_white'
    )
    bar_fig.write_image(f"{output_folder}/top_words.png")

text_analysis(df, dedicated_stopwords)

# 4. 情感分析
def analyze_sentiment(df):
    df['sentiment_score'] = df['answer'].apply(lambda x: SnowNLP(x).sentiments)
    df['sentiment_category'] = pd.cut(df['sentiment_score'], 
                                      bins=[-0.01, 0.3, 0.7, 1.01], 
                                      labels=['Negative', 'Neutral', 'Positive'])
    
    hist_fig = go.Figure(data=[
        go.Histogram(x=df['sentiment_score'], nbinsx=30, marker_color='rgb(8, 48, 107)')
    ])
    hist_fig.update_layout(
        title='情感分数分布',
        xaxis_title='情感分数',
        yaxis_title='频率',
        font=dict(family="Songti SC", size=14),
        template='plotly_white'
    )
    hist_fig.write_image(f"{output_folder}/sentiment_distribution.png")
    
    pie_fig = go.Figure(data=[
        go.Pie(labels=df['sentiment_category'].value_counts().index, 
               values=df['sentiment_category'].value_counts().values,
               marker=dict(colors=['rgb(8, 48, 107)', 'rgb(33, 113, 181)', 'rgb(39, 130, 190)'])
        )
    ])
    pie_fig.update_layout(
        title='情感类别分布',
        font=dict(family="Songti SC", size=14)
    )
    pie_fig.write_image(f"{output_folder}/sentiment_category_pie.png")
    
    print("情感分数统计:")
    print(df['sentiment_score'].describe())
    print("\n情感类别统计:")
    print(df['sentiment_category'].value_counts(normalize=True))

analyze_sentiment(df)

# 5. 主题分析
def topic_analysis(df):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf = vectorizer.fit_transform(df['question_words'] + ' ' + df['answer_words'])
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf)
        
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")

topic_analysis(df)

# 6. 位置分析
def location_analysis(df):
    location_counts = df['location'].apply(lambda x: x['name']).value_counts()

    # Load the font properties
    font_properties = fm.FontProperties(fname=chinese_font_path)

    # Create the treemap with darker blues
    fig, ax = plt.subplots(1, figsize=(12, 8))
    squarify.plot(sizes=location_counts.values, label=location_counts.index, 
                  color=['#08306b', '#2171b5', '#4292c6', '#6baed6'], alpha=0.7, ax=ax)
    plt.axis('off')

    # Apply the font to all text elements
    for text in ax.texts:
        text.set_fontproperties(font_properties)
        text.set_fontsize(10)
        text.set_color('white')

    # Title with font support
    plt.title('对话位置分布', fontproperties=font_properties, size=14)

    plt.savefig(f"{output_folder}/location_distribution.png")
    plt.close()

location_analysis(df)

# 7. FAQ识别
def identify_faqs(df):
    question_freq = df['question'].value_counts()
    top_questions = question_freq.head(10)
    print("Top 10 常见问题:")
    for question, count in top_questions.items():
        print(f"{question}: {count}")

identify_faqs(df)
