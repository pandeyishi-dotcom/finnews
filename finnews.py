# app.py
import streamlit as st
import pandas as pd
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------

@st.cache_data
def fetch_news(fund_name, days=30):
    """Fetch news for a given fund and time range using GNews."""
    google_news = GNews(language='en', period=f'{days}d')
    results = google_news.get_news(fund_name)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df[['title', 'published date', 'description', 'url']]
        df['fund'] = fund_name
        df['published date'] = pd.to_datetime(df['published date'])
    return df


def analyze_sentiment(df):
    """Add sentiment scores and labels to dataframe."""
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['title'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    df['sentiment_label'] = df['sentiment'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    return df


def make_wordcloud(text, title):
    """Generate and display a word cloud."""
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


def plot_sentiment_trend(df):
    """Plot average sentiment over time for each fund."""
    df['date'] = df['published date'].dt.date
    trend = df.groupby(['fund', 'date'])['sentiment'].mean().reset_index()
    fig = px.line(trend, x='date', y='sentiment', color='fund',
                  title='📈 Sentiment Trend Over Time', markers=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_sentiment_comparison(df):
    """Bar chart showing positive/negative ratio per fund."""
    comp = df.groupby(['fund', 'sentiment_label']).size().reset_index(name='count')
    fig = px.bar(comp, x='fund', y='count', color='sentiment_label',
                 title='📊 Sentiment Comparison', barmode='group')
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------
# Streamlit Layout
# -----------------------------------------------------
st.set_page_config(page_title="Bluechip Fund Sentiment Analyzer Pro", layout="wide")
st.title("💹 Bluechip Fund Sentiment Analyzer Pro")
st.caption("AI-driven dashboard for sentiment & trend insights on Indian Bluechip Funds")

# Sidebar
funds = [
    "HDFC Bluechip Fund", "SBI Bluechip Fund", "ICICI Prudential Bluechip Fund",
    "Axis Bluechip Fund", "Kotak Bluechip Fund", "Mirae Asset Large Cap Fund",
    "Canara Robeco Bluechip Equity Fund"
]
selected_funds = st.sidebar.multiselect("Select Bluechip Funds", funds, default=["HDFC Bluechip Fund"])
period = st.sidebar.selectbox("Select Time Period", ["7 Days", "30 Days", "90 Days", "180 Days"])
days = int(period.split()[0])

# Fetch and process data
data_frames = []
for f in selected_funds:
    df = fetch_news(f, days)
    if not df.empty:
        df = analyze_sentiment(df)
        data_frames.append(df)

if not data_frames:
    st.error("No news articles found for the selected funds and time period.")
    st.stop()

news_data = pd.concat(data_frames)

# -----------------------------------------------------
# Tabs for visualization
# -----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🗞️ News Feed", "📈 Trends", "📊 Comparison", "☁️ Word Cloud"])

with tab1:
    st.header("🗞️ Latest News")
    for _, row in news_data.iterrows():
        st.markdown(f"### [{row['title']}]({row['url']})")
        st.write(row['description'])
        st.caption(f"📅 {row['published date'].date()} | Fund: {row['fund']} | Sentiment: **{row['sentiment_label']}**")
        st.divider()

with tab2:
    plot_sentiment_trend(news_data)

with tab3:
    plot_sentiment_comparison(news_data)

with tab4:
    text_all = " ".join(news_data['title'].dropna().tolist())
    make_wordcloud(text_all, "Top Keywords Across Funds")

st.success("✅ Dashboard loaded successfully!")
st.caption("Built with ❤️ Streamlit + GNews + VADER + Plotly")
