# finnews.py
import datetime as dt
import math
from functools import lru_cache

import streamlit as st
import pandas as pd
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bluechip Fund News Intelligence", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# CONFIG / CONSTANTS
# -------------------------
analyzer = SentimentIntensityAnalyzer()
google_news = GNews(language='en', country='IN', max_results=100)

BLUECHIP_FUNDS = [
    "HDFC Bluechip Fund",
    "SBI Bluechip Fund",
    "ICICI Prudential Bluechip Fund",
    "Aditya Birla SL Focused Equity Fund",
    "Nippon India Large Cap Fund",
    "Kotak Bluechip Fund",
    "Axis Bluechip Fund",
    "UTI Mastershare Fund"
]

TIME_OPTIONS = {
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180
}

# -------------------------
# HELPERS
# -------------------------
def days_ago_to_iso(days):
    dt_from = dt.datetime.utcnow() - dt.timedelta(days=days)
    return dt_from.strftime("%Y-%m-%d")

@lru_cache(maxsize=128)
def fetch_news_gnews(query: str, days: int):
    """Fetch recent news using GNews (no API key required)."""
    google_news.period = f"{days}d"
    results = google_news.get_news(query)
    return results or []

def analyze_sentiment(text):
    s = analyzer.polarity_scores(text)
    return s['compound']

def make_df_from_articles(articles, fund_name):
    rows = []
    for a in articles:
        title = a.get("title", "")
        desc = a.get("description", "")
        url = a.get("url", "")
        source = a.get("publisher", {}).get("title", "")
        pub_date = a.get("published date") or a.get("published_date")
        try:
            pub_dt = dt.datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
        except Exception:
            pub_dt = None
        days_old = (dt.datetime.utcnow() - pub_dt).days if pub_dt else None
        content = f"{title}. {desc}".strip()
        sentiment = analyze_sentiment(content)
        rows.append({
            "fund": fund_name,
            "title": title,
            "description": desc,
            "url": url,
            "source": source,
            "published_dt": pub_dt,
            "days_old": days_old,
            "sentiment": sentiment
        })
    return pd.DataFrame(rows)

def compute_buzz_score(df):
    if df.empty:
        return 0.0
    count = len(df)
    mean_sent = df['sentiment'].mean()
    recency_weights = df['days_old'].apply(lambda d: math.exp(-d/30) if pd.notnull(d) else 0.0)
    mean_recency_weight = recency_weights.mean() if not recency_weights.empty else 0.0
    return float(mean_sent * count * mean_recency_weight)

def generate_wordcloud(text, max_words=100):
    wc = WordCloud(width=800, height=400, background_color="white", max_words=max_words)
    wc.generate(text)
    return wc

def plot_sentiment_trend(df):
    if df.empty:
        return None
    df2 = df.copy()
    df2['date'] = df2['published_dt'].dt.date
    daily = df2.groupby('date').agg({'sentiment': 'mean', 'title': 'count'}).reset_index()
    fig = px.line(daily, x='date', y='sentiment', markers=True, title='Daily Average Sentiment')
    fig.update_layout(xaxis_title="Date", yaxis_title="Sentiment Score")
    return fig

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("Filters & Controls")
selected_time_label = st.sidebar.radio("Timeframe", list(TIME_OPTIONS.keys()), index=1)
days = TIME_OPTIONS[selected_time_label]
fund_mode = st.sidebar.radio("Mode", ["Single Fund", "Compare Funds"], index=0)

if fund_mode == "Single Fund":
    selected_fund = st.sidebar.selectbox("Select Fund", BLUECHIP_FUNDS)
    compare_funds = []
else:
    compare_funds = st.sidebar.multiselect("Funds to compare", BLUECHIP_FUNDS, default=BLUECHIP_FUNDS[:2])
    selected_fund = None

# -------------------------
# MAIN
# -------------------------
st.title("Bluechip Fund News Intelligence Dashboard")
st.write("📈 Tracks recent bluechip mutual fund news using Google News and performs sentiment & buzz analysis — no API key needed.")

funds_to_fetch = [selected_fund] if fund_mode == "Single Fund" else compare_funds
if not funds_to_fetch:
    st.warning("Select at least one fund.")
    st.stop()

all_dfs = []
with st.spinner("Fetching news from Google..."):
    for fund in funds_to_fetch:
        articles = fetch_news_gnews(fund, days)
        df = make_df_from_articles(articles, fund)
        all_dfs.append(df)

if not any(len(df) for df in all_dfs):
    st.warning("No news found for selected timeframe.")
    st.stop()

full_df = pd.concat(all_dfs, ignore_index=True)
full_df['published_dt'] = pd.to_datetime(full_df['published_dt'], errors='coerce')

# Summary metrics
st.subheader("Overview")
cols = st.columns(len(funds_to_fetch))
for i, fund in enumerate(funds_to_fetch):
    df_f = full_df[full_df['fund'] == fund]
    count = len(df_f)
    avg_sent = df_f['sentiment'].mean()
    buzz = compute_buzz_score(df_f)
    with cols[i]:
        st.metric(label=fund, value=f"{count} articles", delta=f"Avg Sent {avg_sent:.2f}")
        st.caption(f"Buzz Score: {buzz:.2f}")

# Left/Right layout
left, right = st.columns((1, 2))

with left:
    st.markdown("### News Feed")
    keyword = st.text_input("Search headlines", value="")
    filtered = full_df.copy()
    if keyword:
        filtered = filtered[filtered['title'].str.contains(keyword, case=False, na=False)]
    for _, row in filtered.iterrows():
        st.markdown(f"**[{row['title']}]({row['url']})**")
        st.caption(f"{row['source']} — {row['published_dt']} — Sent: {row['sentiment']:.2f}")
        st.write(row['description'])
        st.write("---")

with right:
    st.markdown("### Visuals")
    # Sentiment trend
    if fund_mode == "Single Fund":
        fig_sent = plot_sentiment_trend(full_df)
        if fig_sent:
            st.plotly_chart(fig_sent, use_container_width=True)
    else:
        df_copy = full_df.copy()
        df_copy['date'] = df_copy['published_dt'].dt.date
        df_avg = df_copy.groupby(['fund', 'date']).agg({'sentiment': 'mean'}).reset_index()
        fig = px.line(df_avg, x='date', y='sentiment', color='fund', title='Sentiment by Fund')
        st.plotly_chart(fig, use_container_width=True)

    # Word cloud
    st.markdown("#### Word Cloud")
    combined_text = " ".join(full_df['title'].dropna().tolist())
    if combined_text.strip():
        wc = generate_wordcloud(combined_text)
        fig = plt.figure(figsize=(10, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)

# Buzz leaderboard
st.subheader("Buzz Leaderboard")
buzz_rows = []
for fund in funds_to_fetch:
    df_f = full_df[full_df['fund'] == fund]
    buzz_rows.append({
        "fund": fund,
        "articles": len(df_f),
        "avg_sentiment": df_f['sentiment'].mean(),
        "buzz_score": compute_buzz_score(df_f)
    })
buzz_df = pd.DataFrame(buzz_rows).sort_values('buzz_score', ascending=False)
st.dataframe(buzz_df)

st.caption("✅ Uses Google News (no API key). ✅ Built with Streamlit + Sentiment Analysis + WordCloud.")
