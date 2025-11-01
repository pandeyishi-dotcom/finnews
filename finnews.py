# app.py
"""
Bluechip Fund Sentiment Analyzer Pro — Modular Skeleton
- Fetchers: NewsAPI (newsapi-python) or GNews (gnews)
- Sentiment: VADER (vaderSentiment)
- Visuals: Plotly (preferred) with Streamlit fallbacks
- Wordcloud and a lightweight summary function included
- Designed to be extended: add transformers, NAV fetch, alerts.
"""

import os
from datetime import datetime, timedelta
from functools import lru_cache

import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter

# Optional libs; we'll import them later with try/except to allow graceful degradation
try:
    from newsapi import NewsApiClient
except Exception:
    NewsApiClient = None

try:
    from gnews import GNews
except Exception:
    GNews = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except Exception:
    WordCloud = None
    plt = None

# -------------------------
# Config & constants
# -------------------------
st.set_page_config(page_title="Bluechip Fund Sentiment Analyzer Pro", layout="wide")
st.title("Bluechip Fund Sentiment Analyzer Pro — Modular Skeleton")
st.markdown("Modular app: fetch → clean → sentiment → visualize → summarize. Toggle components in the sidebar.")

DEFAULT_FUNDS = [
    "HDFC Bluechip Fund",
    "SBI Bluechip Fund",
    "ICICI Prudential Bluechip Fund",
    "Axis Bluechip Fund",
    "Kotak Bluechip Fund",
    "Mirae Asset Large Cap Fund",
    "Canara Robeco Bluechip Equity Fund"
]

TIME_PERIODS = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "180d": 180
}

# -------------------------
# Utility helpers
# -------------------------
def safe_lower(text):
    return text.lower() if isinstance(text, str) else ""

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)  # drop urls
    text = re.sub(r"[^\w\s\.\-]", "", text)  # keep words punctuation limited
    return text.strip()

def date_n_days_ago(n):
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")

# -------------------------
# FETCHERS
# -------------------------
@lru_cache(maxsize=64)
def fetch_news_via_newsapi(api_key, query, from_date, page_size=50, max_pages=2):
    """Fetch using NewsAPI. Returns list of articles (dict)."""
    if not NewsApiClient:
        st.error("newsapi-python package not installed.")
        return []
    client = NewsApiClient(api_key=api_key)
    articles = []
    page = 1
    while page <= max_pages:
        try:
            res = client.get_everything(q=query,
                                        from_param=from_date,
                                        language='en',
                                        sort_by='publishedAt',
                                        page_size=page_size,
                                        page=page)
        except Exception as e:
            st.error(f"NewsAPI error: {e}")
            break
        arts = res.get("articles", []) or []
        if not arts:
            break
        articles.extend(arts)
        if len(arts) < page_size:
            break
        page += 1
    return articles

@lru_cache(maxsize=64)
def fetch_news_via_gnews(query, days=7, max_results=30):
    """Fetch using GNews (scraping-based). Returns list of dicts similar to NewsAPI's minimal fields."""
    if not GNews:
        st.error("gnews package not installed.")
        return []
    gn = GNews(language='en', country='IN', max_results=max_results)
    gn.period = f"{days}d"
    try:
        results = gn.get_news(query)
    except Exception as e:
        st.error(f"GNews error: {e}")
        results = []
    # Normalize results to match the fields we use later
    normalized = []
    for r in results:
        normalized.append({
            "title": r.get("title"),
            "description": r.get("description"),
            "url": r.get("url"),
            "publishedAt": r.get("published date") or r.get("published_date") or None,
            "source": {"name": r.get("publisher", {}).get("title", "") if isinstance(r.get("publisher"), dict) else r.get("publisher")}
        })
    return normalized

# -------------------------
# CLEAN & DEDUP
# -------------------------
def make_article_df(raw_articles, fund_name):
    """Normalize a list of article dicts to a DataFrame with specific columns"""
    rows = []
    for a in raw_articles:
        title = clean_text(a.get("title") or "")
        desc = clean_text(a.get("description") or "")
        url = a.get("url") or ""
        # support both 'publishedAt' or 'published date' formats
        pub = a.get("publishedAt") or a.get("published date") or a.get("published_date") or None
        # try safe parse; keep as string if parse fails
        try:
            pub_dt = pd.to_datetime(pub)
        except Exception:
            pub_dt = None
        source = a.get("source", {}).get("name") if isinstance(a.get("source"), dict) else a.get("source") or ""
        rows.append({
            "fund": fund_name,
            "title": title,
            "description": desc,
            "content": (title + ". " + desc).strip(),
            "url": url,
            "publishedAt": pub_dt,
            "source": source
        })
    df = pd.DataFrame(rows)
    # dedupe by title + url
    if not df.empty:
        df["dupe_key"] = df["title"].str.slice(0, 200).fillna("") + "|" + df["url"].fillna("")
        df = df.drop_duplicates(subset="dupe_key")
        df = df.drop(columns=["dupe_key"])
    return df

# -------------------------
# SENTIMENT
# -------------------------
def init_sentiment_analyzer():
    if SentimentIntensityAnalyzer:
        return SentimentIntensityAnalyzer()
    st.warning("VADER not installed. Sentiment values will be zero.")
    class Dummy:
        def polarity_scores(self, t): return {"compound": 0.0}
    return Dummy()

def add_sentiment(df, analyzer):
    if df.empty:
        df["sentiment"] = []
        df["sentiment_label"] = []
        return df
    df["sentiment"] = df["content"].fillna("").apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment_label"] = df["sentiment"].apply(lambda s: "Positive" if s > 0.05 else ("Negative" if s < -0.05 else "Neutral"))
    return df

# -------------------------
# AGGREGATION & METRICS
# -------------------------
def aggregate_sentiment(df, freq="D"):
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = d["publishedAt"].dt.date
    agg = d.groupby(["fund", "date"]).agg(
        articles=("title", "count"),
        avg_sentiment=("sentiment", "mean")
    ).reset_index()
    return agg

def compute_buzz_score(df):
    if df.empty:
        return 0.0
    count = len(df)
    mean_sent = df["sentiment"].mean()
    days = (pd.Timestamp.utcnow() - df["publishedAt"].fillna(pd.Timestamp.utcnow())).dt.days.clip(lower=0)
    recency_weights = np.exp(-days/30)
    mean_recency = recency_weights.mean()
    return float(mean_sent * count * mean_recency)

# -------------------------
# SIMPLE SUMMARY (lightweight)
# -------------------------
def simple_headline_summary(df, top_n=3):
    """Return a short bullet summary from most common keywords and top headlines."""
    if df.empty:
        return "No articles to summarize."
    titles = df["title"].dropna().tolist()
    # top headlines
    top_headlines = titles[:top_n]
    # extract most common words excluding stopwords
    stop = set(["the","and","in","to","of","for","on","with","a","is","at","by","from"])
    words = []
    for t in titles:
        for w in re.findall(r"\w{3,}", t.lower()):
            if w not in stop:
                words.append(w)
    common = Counter(words).most_common(6)
    common_str = ", ".join([w for w,_ in common])
    summary = f"Top themes: {common_str}. Top headlines:\n" + "\n".join([f"- {h}" for h in top_headlines])
    return summary

# -------------------------
# VISUALIZATIONS
# -------------------------
def plot_sentiment_trend(agg_df, funds_selected):
    if agg_df.empty:
        st.info("No data to plot.")
        return
    if px:
        fig = px.line(agg_df, x="date", y="avg_sentiment", color="fund", markers=True,
                      title="Sentiment trend (avg sentiment per day)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # fallback: pivot and st.line_chart
        pivot = agg_df.pivot(index="date", columns="fund", values="avg_sentiment").fillna(0)
        st.line_chart(pivot)

def plot_heatmap_latest(df):
    if df.empty:
        st.info("No data for heatmap.")
        return
    # compute average sentiment per fund
    avg = df.groupby("fund").agg(avg_sentiment=("sentiment", "mean")).reset_index()
    if avg.empty:
        st.info("Heatmap: no averages.")
        return
    if px:
        fig = px.imshow(avg[["avg_sentiment"]].T, x=avg["fund"], y=["avg_sentiment"], color_continuous_scale="RdYlGn",
                        aspect="auto", title="Average sentiment per fund")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(avg.set_index("fund")["avg_sentiment"])

def show_wordcloud(df, fund_name=None):
    if WordCloud is None or plt is None:
        st.info("WordCloud / matplotlib not installed.")
        return
    texts = df["content"].dropna().tolist()
    if fund_name:
        texts = df[df["fund"] == fund_name]["content"].dropna().tolist()
    text = " ".join(texts)
    if not text.strip():
        st.info("No text for wordcloud.")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10,4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# -------------------------
# STREAMLIT UI
# -------------------------
def sidebar_controls():
    st.sidebar.header("Controls")
    funds = st.sidebar.multiselect("Select funds (multi)", DEFAULT_FUNDS, default=DEFAULT_FUNDS[:3])
    if not funds:
        funds = DEFAULT_FUNDS[:1]
    days_label = st.sidebar.selectbox("Time window", list(TIME_PERIODS.keys()), index=1)
    days = TIME_PERIODS[days_label]
    source_choice = st.sidebar.radio("News source", ["NewsAPI (requires key)", "GNews (no key)"], index=1)
    api_key = None
    if source_choice.startswith("NewsAPI"):
        api_key = st.sidebar.text_input("NewsAPI key", value=os.getenv("NEWSAPI_KEY",""), placeholder="paste your NewsAPI key here")
    options = {
        "funds": funds,
        "days": days,
        "source": source_choice,
        "api_key": api_key
    }
    return options

def main():
    opts = sidebar_controls()
    funds = opts["funds"]
    days = opts["days"]
    source = opts["source"]
    api_key = opts["api_key"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("Extensions:\n- Add transformer summaries\n- Add NAV fetch (AMFI)\n- Add alerting (email/telegram)")

    st.header("Fetch & Analyze")
    st.write(f"Fetching {len(funds)} funds over last {days} days using {source}.")

    raw_all = []
    for fund in funds:
        query = f'"{fund}" OR "{fund.split()[0]} fund"'
        st.write(f"Fetching: {fund} ...")
        if source.startswith("NewsAPI"):
            if not api_key:
                st.warning("NewsAPI selected but no key provided. Skipping NewsAPI fetch.")
                articles = []
            else:
                articles = fetch_news_via_newsapi(api_key=api_key, query=query, from_date=date_n_days_ago(days))
        else:
            articles = fetch_news_via_gnews(query=query, days=days)
        df_f = make_article_df(articles, fund)
        raw_all.append(df_f)

    if len(raw_all) == 0:
        st.error("No fetchers configured or no data fetched.")
        return

    all_df = pd.concat(raw_all, ignore_index=True) if any(not d.empty for d in raw_all) else pd.DataFrame()
    if all_df.empty:
        st.warning("No articles found for chosen funds/timeframe.")
        return

    # sentiment
    analyzer = init_sentiment_analyzer()
    all_df = add_sentiment(all_df, analyzer)

    # Show table and metrics
    st.subheader("Raw articles sample")
    st.dataframe(all_df[["fund","publishedAt","title","source","sentiment","sentiment_label"]].sort_values("publishedAt", ascending=False).head(200))

    # Aggregation
    agg = aggregate_sentiment(all_df)

    # Visuals
    st.subheader("Sentiment Trend")
    plot_sentiment_trend(agg, funds)

    st.subheader("Fund heatmap (avg sentiment)")
    plot_heatmap_latest(all_df)

    st.subheader("Buzz leaderboard")
    buzz_list = []
    for f in funds:
        df_f = all_df[all_df["fund"] == f]
        buzz_list.append({
            "fund": f,
            "articles": len(df_f),
            "avg_sentiment": df_f["sentiment"].mean() if not df_f.empty else 0.0,
            "buzz": compute_buzz_score(df_f)
        })
    buzz_df = pd.DataFrame(buzz_list).sort_values("buzz", ascending=False)
    st.table(buzz_df)

    st.subheader("Word Clouds (per fund)")
    cols = st.columns(min(3, len(funds)))
    for i, f in enumerate(funds):
        with cols[i % 3]:
            st.markdown(f"**{f}**")
            show_wordcloud(all_df, fund_name=f)

    # Summaries
    st.subheader("Auto Summaries (lightweight)")
    for f in funds:
        st.markdown(f"**{f}**")
        df_f = all_df[all_df["fund"] == f].sort_values("publishedAt", ascending=False)
        st.write(simple_headline_summary(df_f, top_n=3))

    st.success("Analysis complete. Extend the modules to add model summaries, NAV correlation, and alerts.")

if __name__ == "__main__":
    main()

