# app.py
"""
Bluechip Fund Sentiment Analyzer Pro — Fully working prototype
- Uses NewsAPI if NEWSAPI_KEY provided, else falls back to GNews
- VADER sentiment
- Optional transformer summarization (if transformers installed)
- Plotly visuals, wordclouds, Slack alerting (optional)
"""

import os
import re
import time
import json
from datetime import datetime, timedelta
from functools import lru_cache
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import requests

# optional libs (graceful degradation)
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

# transformers summarizer (optional; heavy)
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# ------------------------
# Config
# ------------------------
st.set_page_config(page_title="Bluechip Fund Sentiment Analyzer Pro", layout="wide")
st.title("Bluechip Fund Sentiment Analyzer Pro")
st.markdown("Multi-source news → sentiment → visualization. Use NewsAPI (key) or GNews (no key).")

DEFAULT_FUNDS = [
    "HDFC Bluechip Fund",
    "SBI Bluechip Fund",
    "ICICI Prudential Bluechip Fund",
    "Axis Bluechip Fund",
    "Kotak Bluechip Fund",
    "Mirae Asset Large Cap Fund",
    "Canara Robeco Bluechip Equity Fund"
]

TIME_OPTIONS = {
    "7 Days": 7,
    "30 Days": 30,
    "90 Days": 90,
    "180 Days": 180
}

# ------------------------
# Helpers
# ------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"http\S+", "", text)
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    return t

def format_date(dt):
    if pd.isna(dt):
        return ""
    if isinstance(dt, str):
        return dt
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt)

# ------------------------
# News fetchers
# ------------------------
@lru_cache(maxsize=128)
def fetch_news_newsapi(api_key, query, from_date_iso, page_size=50, max_pages=2):
    if not NewsApiClient:
        return []
    client = NewsApiClient(api_key=api_key)
    all_articles = []
    page = 1
    while page <= max_pages:
        try:
            res = client.get_everything(q=query,
                                        from_param=from_date_iso,
                                        to=datetime.utcnow().strftime("%Y-%m-%d"),
                                        language="en",
                                        sort_by="publishedAt",
                                        page_size=page_size,
                                        page=page)
        except Exception as e:
            st.error(f"NewsAPI fetch error: {e}")
            break
        arts = res.get("articles", []) or []
        if not arts:
            break
        all_articles.extend(arts)
        if len(arts) < page_size:
            break
        page += 1
    return all_articles

@lru_cache(maxsize=128)
def fetch_news_gnews(query, days=7, max_results=40):
    if not GNews:
        return []
    gn = GNews(language="en", country="IN", max_results=max_results)
    gn.period = f"{days}d"
    try:
        res = gn.get_news(query)
    except Exception as e:
        st.error(f"GNews fetch error: {e}")
        return []
    normalized = []
    for r in res:
        normalized.append({
            "title": r.get("title"),
            "description": r.get("description"),
            "url": r.get("url"),
            "publishedAt": r.get("published date") or r.get("published_date") or None,
            "source": {"name": (r.get("publisher") or {}).get("title") if isinstance(r.get("publisher"), dict) else r.get("publisher")}
        })
    return normalized

# ------------------------
# Normalizer / dedupe
# ------------------------
def normalize_articles(raw_articles, fund_name):
    rows = []
    for a in raw_articles:
        title = clean_text(a.get("title") or "")
        desc = clean_text(a.get("description") or "")
        url = a.get("url") or ""
        published = a.get("publishedAt") or a.get("published date") or a.get("published_date") or None
        try:
            published_dt = pd.to_datetime(published)
        except Exception:
            published_dt = None
        source = a.get("source", {}).get("name") if isinstance(a.get("source"), dict) else a.get("source") or ""
        rows.append({
            "fund": fund_name,
            "title": title,
            "description": desc,
            "content": (title + ". " + desc).strip(),
            "url": url,
            "publishedAt": published_dt,
            "source": source
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["dupe_key"] = df["title"].str.slice(0,200).fillna("") + "|" + df["url"].fillna("")
    df = df.drop_duplicates(subset="dupe_key").drop(columns=["dupe_key"])
    return df

# ------------------------
# Sentiment
# ------------------------
def get_sentiment_analyzer():
    if SentimentIntensityAnalyzer:
        return SentimentIntensityAnalyzer()
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

# ------------------------
# Aggregation & buzz
# ------------------------
def aggregate_by_day(df):
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = d["publishedAt"].dt.date
    agg = d.groupby(["fund", "date"]).agg(articles=("title", "count"), avg_sentiment=("sentiment", "mean")).reset_index()
    return agg

def compute_buzz(df):
    if df.empty:
        return 0.0
    count = len(df)
    mean_sent = df["sentiment"].mean()
    days = (pd.Timestamp.utcnow() - df["publishedAt"].fillna(pd.Timestamp.utcnow())).dt.days.clip(lower=0)
    recency_weights = np.exp(-days/30)
    mean_recency = recency_weights.mean() if len(recency_weights)>0 else 0.0
    return float(mean_sent * count * mean_recency)

# ------------------------
# Summarizer (optional transformers)
# ------------------------
SUMMARIZER = None
if HAS_TRANSFORMERS:
    try:
        # small & general summarizer — transformer download may take time
        SUMMARIZER = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        SUMMARIZER = None

def summarize_articles_transformers(texts, max_chars=600):
    if not SUMMARIZER:
        return None
    joined = " ".join([t for t in texts if t])
    if not joined.strip():
        return ""
    # transformers summarizers usually accept up to a token limit; truncate conservatively
    joined = joined[:8000]
    try:
        out = SUMMARIZER(joined, max_length=130, min_length=30, do_sample=False)
        return out[0]["summary_text"]
    except Exception:
        return None

def simple_headline_summary(df, top_n=3):
    if df.empty:
        return "No articles to summarize."
    titles = df["title"].dropna().tolist()
    top_headlines = titles[:top_n]
    stop = set(["the","and","in","to","of","for","on","with","a","is","at","by","from","fund","et","al"])
    words = []
    for t in titles:
        for w in re.findall(r"\w{3,}", t.lower()):
            if w not in stop:
                words.append(w)
    common = Counter(words).most_common(6)
    themes = ", ".join([w for w,_ in common]) if common else ""
    summary = f"Top themes: {themes}. Top headlines:\n" + "\n".join([f"- {h}" for h in top_headlines])
    return summary

# ------------------------
# Visuals
# ------------------------
def plot_sentiment_trend(agg_df):
    if agg_df.empty:
        st.info("No sentiment trend data.")
        return
    if px:
        fig = px.line(agg_df, x="date", y="avg_sentiment", color="fund", markers=True,
                      title="Average sentiment over time (by fund)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        pivot = agg_df.pivot(index="date", columns="fund", values="avg_sentiment").fillna(0)
        st.line_chart(pivot)

def plot_article_counts(agg_df):
    if agg_df.empty:
        return
    dfc = agg_df.groupby("fund").agg(total_articles=("articles","sum")).reset_index()
    st.bar_chart(dfc.set_index("fund")["total_articles"])

def show_wordcloud(df, fund_name=None):
    if WordCloud is None or plt is None:
        st.info("WordCloud not available (missing dependencies).")
        return
    texts = df["content"].dropna().tolist()
    if fund_name:
        texts = df[df["fund"]==fund_name]["content"].dropna().tolist()
    text = " ".join(texts)
    if not text.strip():
        st.info("No text for word cloud.")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10,4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# ------------------------
# Alerts (Slack webhook optional)
# ------------------------
def send_slack_alert(webhook_url, message):
    if not webhook_url:
        return False
    try:
        payload = {"text": message}
        r = requests.post(webhook_url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ------------------------
# Streamlit UI
# ------------------------
def sidebar():
    st.sidebar.header("Controls")
    funds = st.sidebar.multiselect("Select funds (multi)", DEFAULT_FUNDS, default=DEFAULT_FUNDS[:3])
    days_label = st.sidebar.selectbox("Time window", list(TIME_OPTIONS.keys()), index=1)
    days = TIME_OPTIONS[days_label]
    source = st.sidebar.radio("Source", ["Auto: NewsAPI if key, else GNews", "Force NewsAPI", "Force GNews"], index=0)
    api_key = st.sidebar.text_input("NewsAPI Key (optional)", value=os.getenv("NEWSAPI_KEY",""))
    slack_webhook = st.sidebar.text_input("Slack webhook URL (optional)", value=os.getenv("SLACK_WEBHOOK",""))
    st.sidebar.markdown("---")
    st.sidebar.write("Transformer summarizer:", "✅ available" if SUMMARIZER else "❌ not available")
    st.sidebar.markdown("Upload NAV CSV to correlate (columns: fund, date (YYYY-MM-DD), nav)")
    nav_file = st.sidebar.file_uploader("NAV CSV", type=["csv"])
    return {
        "funds": funds,
        "days": days,
        "source": source,
        "api_key": api_key.strip(),
        "slack_webhook": slack_webhook.strip(),
        "nav_file": nav_file
    }

def main():
    opts = sidebar()
    funds = opts["funds"]
    days = opts["days"]
    source_choice = opts["source"]
    api_key = opts["api_key"]
    slack_webhook = opts["slack_webhook"]
    nav_file = opts["nav_file"]

    if not funds:
        st.warning("Select at least one fund.")
        st.stop()

    st.write(f"Fetching news for {len(funds)} funds over last {days} days.")

    raw_frames = []
    for fund in funds:
        q = f'"{fund}" OR "{fund.split()[0]} fund"'
        st.info(f"Fetching: {fund}")
        use_newsapi = False
        if source_choice.startswith("Force NewsAPI"):
            use_newsapi = True
        elif source_choice.startswith("Force GNews"):
            use_newsapi = False
        else:
            use_newsapi = bool(api_key and NewsApiClient)
        articles = []
        if use_newsapi:
            if not api_key:
                st.warning("NewsAPI key missing — skipping NewsAPI fetch.")
            else:
                from_iso = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
                articles = fetch_news_newsapi(api_key, q, from_iso, page_size=50, max_pages=2)
        else:
            if not GNews:
                st.warning("GNews package not installed — cannot use GNews.")
                articles = []
            else:
                articles = fetch_news_gnews(q, days=days, max_results=60)
        df_f = normalize_articles(articles, fund)
        raw_frames.append(df_f)
        time.sleep(0.2)  # polite pause

    if not raw_frames:
        st.error("No fetchers executed.")
        st.stop()

    combined = pd.concat(raw_frames, ignore_index=True) if any(len(x)>0 for x in raw_frames) else pd.DataFrame()
    if combined.empty:
        st.warning("No articles found for selected funds/timeframe.")
        st.stop()

    analyzer = get_sentiment_analyzer()
    combined = add_sentiment(combined, analyzer)

    # show top table
    st.subheader("Articles (sample)")
    show_cols = ["fund","publishedAt","source","title","sentiment","sentiment_label","url"]
    st.dataframe(combined[show_cols].sort_values("publishedAt", ascending=False).head(300))

    agg = aggregate_by_day(combined)

    st.subheader("Sentiment trend")
    plot_sentiment_trend(agg)

    st.subheader("Article counts (total)")
    plot_article_counts(agg)

    st.subheader("Buzz leaderboard")
    buzz = []
    for f in funds:
        df_f = combined[combined["fund"]==f]
        buzz.append({
            "fund": f,
            "articles": len(df_f),
            "avg_sentiment": float(df_f["sentiment"].mean()) if len(df_f)>0 else 0.0,
            "buzz": compute_buzz(df_f)
        })
    buzz_df = pd.DataFrame(buzz).sort_values("buzz", ascending=False)
    st.table(buzz_df)

    # Word clouds
    st.subheader("Word clouds")
    cols = st.columns(min(3, len(funds)))
    for i, f in enumerate(funds):
        with cols[i % 3]:
            st.markdown(f"**{f}**")
            show_wordcloud(combined, fund_name=f)

    # Summaries
    st.subheader("Automated summaries")
    for f in funds:
        st.markdown(f"**{f}**")
        df_f = combined[combined["fund"]==f].sort_values("publishedAt", ascending=False)
        texts = df_f["title"].fillna("").tolist()
        summary = None
        if SUMMARIZER and len(texts)>0:
            summary = summarize_articles_transformers(texts, max_chars=400)
        if not summary:
            summary = simple_headline_summary(df_f, top_n=3)
        st.write(summary)

    # Optional: NAV correlation if uploaded
    if nav_file:
        try:
            nav_df = pd.read_csv(nav_file)
            nav_df["date"] = pd.to_datetime(nav_df["date"]).dt.date
            st.subheader("NAV correlation (upload provided)")
            # pick first fund for example correlation
            target_fund = funds[0]
            nav_f = nav_df[nav_df["fund"]==target_fund]
            if nav_f.empty:
                st.info(f"No NAV rows for fund {target_fund} in uploaded file.")
            else:
                articles_daily = combined[combined["fund"]==target_fund].groupby(combined["publishedAt"].dt.date).size().reset_index(name="articles")
                merged = pd.merge(nav_f, articles_daily, left_on="date", right_on="publishedAt", how="left").fillna(0)
                if not merged.empty:
                    st.line_chart(merged.set_index("date")[["nav","articles"]])
        except Exception as e:
            st.error(f"Failed to process NAV CSV: {e}")

    # Alerts: if buzz spike for top fund beyond threshold, send Slack notification
    try:
        top = buzz_df.iloc[0]
        if top["articles"] >= 10 and abs(top["avg_sentiment"]) > 0.3:
            # a heuristic threshold for noisy spike
            if slack_webhook := opts.get("slack_webhook", ""):
                msg = f"Alert: {top['fund']} has {top['articles']} articles and avg sentiment {top['avg_sentiment']:.2f}. Buzz: {top['buzz']:.2f}"
                ok = send_slack_alert(slack_webhook, msg)
                if ok:
                    st.success("Slack alert sent for top buzz fund.")
    except Exception:
        pass

    st.success("Analysis complete.")

if __name__ == "__main__":
    main()
