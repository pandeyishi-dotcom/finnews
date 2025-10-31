# app.py
import os
import math
import datetime as dt
from functools import lru_cache

import streamlit as st
import pandas as pd
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Bluechip Fund News Intelligence", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# CONFIG / CONSTANTS
# -------------------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    st.sidebar.error("Please set the NEWSAPI_KEY environment variable (get one from https://newsapi.org/).")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
analyzer = SentimentIntensityAnalyzer()

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
def fetch_news(query: str, from_date: str, page_size=100):
    """Fetch articles from NewsAPI. Returns list of article dicts."""
    if not newsapi:
        return []
    all_articles = []
    page = 1
    while True:
        res = newsapi.get_everything(q=query,
                                     from_param=from_date,
                                     language='en',
                                     sort_by='publishedAt',
                                     page_size=page_size,
                                     page=page)
        articles = res.get("articles", [])
        if not articles:
            break
        all_articles.extend(articles)
        if len(articles) < page_size:
            break
        page += 1
        if page > 5:  # safety limit
            break
    return all_articles

def analyze_sentiment(text):
    s = analyzer.polarity_scores(text)
    return s['compound']

def make_df_from_articles(articles, fund_name):
    rows = []
    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        content = (title + ". " + desc).strip()
        published = a.get("publishedAt")
        source = a.get("source", {}).get("name", "")
        url = a.get("url", "")
        sentiment = analyze_sentiment(content) if content else 0.0
        # compute days since published for recency weight
        try:
            pub_dt = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
        except Exception:
            pub_dt = None
        days_old = (dt.datetime.utcnow() - pub_dt).days if pub_dt else None
        rows.append({
            "fund": fund_name,
            "title": title,
            "description": desc,
            "content": content,
            "publishedAt": published,
            "published_dt": pub_dt,
            "days_old": days_old,
            "source": source,
            "url": url,
            "sentiment": sentiment
        })
    return pd.DataFrame(rows)

def compute_buzz_score(df):
    # Buzz = mean_sentiment * count * mean_recency_weight
    if df.empty:
        return 0.0
    count = len(df)
    mean_sent = df['sentiment'].mean()
    # recency weight: average of exp(-days/30) to weight recent articles more
    recency_weights = df['days_old'].apply(lambda d: math.exp(-d/30) if pd.notnull(d) else 0.0)
    mean_recency_weight = recency_weights.mean() if not recency_weights.empty else 0.0
    buzz = mean_sent * count * mean_recency_weight
    return float(buzz)

def generate_wordcloud(text, max_words=100):
    wc = WordCloud(width=800, height=400, background_color="white", max_words=max_words)
    wc.generate(text)
    return wc

def plot_sentiment_trend(df, freq='D'):
    if df.empty:
        return None
    df2 = df.copy()
    df2['date'] = df2['published_dt'].dt.date
    daily = df2.groupby('date').agg({'sentiment': 'mean', 'title': 'count'}).reset_index()
    fig = px.line(daily, x='date', y='sentiment', markers=True, title='Daily Average Sentiment')
    fig.update_layout(xaxis_title="Date", yaxis_title="Avg Sentiment (VADER compound)")
    return fig

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("Filters & Controls")
selected_time_label = st.sidebar.radio("Timeframe", list(TIME_OPTIONS.keys()), index=1)
days = TIME_OPTIONS[selected_time_label]
from_date = days_ago_to_iso(days)

fund_mode = st.sidebar.radio("Mode", ["Single Fund", "Compare Funds"], index=0)

if fund_mode == "Single Fund":
    selected_fund = st.sidebar.selectbox("Select Fund", BLUECHIP_FUNDS)
    compare_funds = []
else:
    compare_funds = st.sidebar.multiselect("Choose funds to compare (2+)", BLUECHIP_FUNDS, default=BLUECHIP_FUNDS[:2])
    selected_fund = None

max_articles = st.sidebar.slider("Max articles per fund (NewsAPI pages)", 20, 200, 100, step=20)

st.sidebar.markdown("---")
st.sidebar.markdown("Optional: Upload NAV CSV to correlate (columns: fund, date (YYYY-MM-DD), nav)")
nav_file = st.sidebar.file_uploader("NAV CSV", type=["csv"], accept_multiple_files=False)

st.sidebar.markdown("---")
st.sidebar.markdown("Usage notes:")
st.sidebar.markdown("- You need a NewsAPI key in env var `NEWSAPI_KEY`.")
st.sidebar.markdown("- News coverage depends on NewsAPI results and may miss some local-language articles.")

# -------------------------
# MAIN
# -------------------------
st.title("Bluechip Fund News Intelligence Dashboard")
st.write("A Streamlit dashboard that fetches recent news for bluechip funds, performs sentiment analysis, creates buzz scores and visualizations.")

# Determine funds to fetch
if fund_mode == "Single Fund":
    funds_to_fetch = [selected_fund]
else:
    funds_to_fetch = compare_funds

if not funds_to_fetch:
    st.warning("Choose at least one fund.")
    st.stop()

# fetch articles
all_dfs = []
with st.spinner("Fetching articles from NewsAPI..."):
    for fund in funds_to_fetch:
        # Query: fund name OR 'fund' + name; keep it simple but effective
        query = f'"{fund}" OR "{fund.split()[0]} fund" OR "{fund.split()[0]} Bluechip"'
        articles = fetch_news(query=query, from_date=from_date, page_size=max_articles)
        df = make_df_from_articles(articles, fund)
        all_dfs.append(df)

if not any(len(df) for df in all_dfs):
    st.warning("No articles found for selected fund(s) in this timeframe.")
    st.stop()

full_df = pd.concat(all_dfs, ignore_index=True)
# ensure published_dt is datetime
full_df['published_dt'] = pd.to_datetime(full_df['published_dt'], errors='coerce')

# Top summary cards
st.subheader("Overview")
cols = st.columns(len(funds_to_fetch))
for i, fund in enumerate(funds_to_fetch):
    df_f = full_df[full_df['fund'] == fund]
    count = len(df_f)
    avg_sent = df_f['sentiment'].mean() if count else 0.0
    buzz = compute_buzz_score(df_f)
    with cols[i]:
        st.metric(label=f"{fund}", value=f"{count} articles", delta=f"Avg Sent {avg_sent:.2f}")
        st.caption(f"Buzz score: {buzz:.2f}")

# Main layout: left = controls + list, right = charts
left, right = st.columns((1,2))

with left:
    st.markdown("### News Feed")
    # search/filter box for headlines
    keyword = st.text_input("Filter headlines by keyword (title/description)", value="")
    filtered = full_df.copy()
    if keyword:
        filtered = filtered[filtered['content'].str.contains(keyword, case=False, na=False)]
    sort_by = st.selectbox("Sort by", ["published_dt (newest)", "sentiment (highest)", "sentiment (lowest)"])
    if sort_by.startswith("published"):
        filtered = filtered.sort_values("published_dt", ascending=False)
    elif "highest" in sort_by:
        filtered = filtered.sort_values("sentiment", ascending=False)
    else:
        filtered = filtered.sort_values("sentiment", ascending=True)

    page_size = 10
    for idx, row in filtered.head(200).iterrows():  # limit displayed
        st.markdown(f"**[{row['title']}]({row['url']})**")
        st.write(f"*{row['source']} — {row['published_dt']} — Sent: {row['sentiment']:.2f}*")
        st.write(row['description'] or "")
        st.write("---")

    st.download_button("Download filtered CSV", data=filtered.to_csv(index=False).encode('utf-8'),
                       file_name=f"bluechip_news_{selected_time_label.replace(' ','')}.csv", mime="text/csv")

with right:
    st.markdown("### Visualizations")
    # Article counts by fund
    counts = full_df.groupby('fund').size().reset_index(name='count')
    fig_counts = px.bar(counts, x='fund', y='count', title='Article count by fund', text='count')
    st.plotly_chart(fig_counts, use_container_width=True)

    # Sentiment trend (single fund shows fund trend; compare shows aggregated)
    if fund_mode == "Single Fund":
        df_f = full_df[full_df['fund'] == selected_fund]
        fig_sent = plot_sentiment_trend(df_f)
        if fig_sent:
            st.plotly_chart(fig_sent, use_container_width=True)
    else:
        # show per-fund average sentiment
        df_copy = full_df.copy()
        df_copy['date'] = df_copy['published_dt'].dt.date
        df_avg = df_copy.groupby(['fund', 'date']).agg({'sentiment':'mean'}).reset_index()
        fig = px.line(df_avg, x='date', y='sentiment', color='fund', title='Sentiment trend by fund')
        st.plotly_chart(fig, use_container_width=True)

    # Wordcloud for selected funds combined
    st.markdown("#### Word Cloud (titles + descriptions)")
    combined_text = " ".join(full_df['content'].dropna().astype(str).tolist())
    if combined_text.strip():
        wc = generate_wordcloud(combined_text)
        fig = plt.figure(figsize=(10,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)
    else:
        st.info("No text to generate word cloud.")

    # Timeline (article spikes)
    st.markdown("#### Article timeline (spikes)")
    timeline_df = full_df.copy()
    timeline_df['date'] = timeline_df['published_dt'].dt.date
    timeline_counts = timeline_df.groupby(['date', 'fund']).size().reset_index(name='n')
    fig_tl = px.bar(timeline_counts, x='date', y='n', color='fund', title='Articles by date')
    st.plotly_chart(fig_tl, use_container_width=True)

# NAV correlation (optional)
if nav_file:
    try:
        nav_df = pd.read_csv(nav_file)
        nav_df['date'] = pd.to_datetime(nav_df['date']).dt.date
        st.success("NAV CSV loaded.")
        # simple merge and plot for first selected fund
        # user must have matching fund name
        target_fund = funds_to_fetch[0]
        nav_f = nav_df[nav_df['fund'] == target_fund]
        article_daily = full_df[full_df['fund'] == target_fund].groupby(full_df['published_dt'].dt.date).size().reset_index(name='articles')
        merged = pd.merge(nav_f, article_daily, left_on='date', right_on='published_dt', how='left')
        merged = merged.sort_values('date')
        if not merged.empty:
            fig_nav = px.line(merged, x='date', y=['nav', 'articles'], title=f'NAV vs Article Count for {target_fund}')
            st.plotly_chart(fig_nav, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read NAV CSV: {e}")

# Buzz leaderboard
st.subheader("Buzz leaderboard")
buzz_rows = []
for fund in funds_to_fetch:
    df_f = full_df[full_df['fund'] == fund]
    buzz_rows.append({
        "fund": fund,
        "articles": len(df_f),
        "avg_sentiment": df_f['sentiment'].mean() if len(df_f) else 0.0,
        "buzz_score": compute_buzz_score(df_f)
    })
buzz_df = pd.DataFrame(buzz_rows).sort_values('buzz_score', ascending=False)
st.dataframe(buzz_df)

# Quick auto-summary (lightweight)
st.subheader("Auto News Digest (short)")
digest = []
for fund in funds_to_fetch:
    df_f = full_df[full_df['fund'] == fund].sort_values('published_dt', ascending=False).head(6)
    if df_f.empty:
        digest.append(f"No recent articles for {fund}")
        continue
    positive = (df_f['sentiment'] > 0.05).sum()
    negative = (df_f['sentiment'] < -0.05).sum()
    neutral = len(df_f) - positive - negative
    top_headlines = "\n".join([f"- {r['title']} ({r['source']})" for _, r in df_f.iterrows()])
    summary = f"**{fund}** — {len(df_f)} recent articles (pos:{positive}, neg:{negative}, neutral:{neutral}). Top headlines:\n{top_headlines}"
    digest.append(summary)

st.markdown("\n\n".join(digest))

st.markdown("---")
st.caption("Built with ❤️ for financial news monitoring. Tips: increase timeframe or page size to fetch more articles; add your own fund aliases to improve coverage.")
