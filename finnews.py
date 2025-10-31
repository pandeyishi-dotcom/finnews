# finnews.py
import streamlit as st
import pandas as pd
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(page_title="Bluechip Fund News Dashboard", layout="wide")
st.title("📈 Bluechip Fund News Intelligence Dashboard")
st.markdown("Stay updated with the latest news and sentiment around India’s top Bluechip Mutual Funds.")

# --------------------------------
# Fund list
# --------------------------------
funds = [
    "HDFC Bluechip Fund",
    "SBI Bluechip Fund",
    "ICICI Prudential Bluechip Fund",
    "Axis Bluechip Fund",
    "Kotak Bluechip Fund",
    "Mirae Asset Large Cap Fund",
    "Canara Robeco Bluechip Equity Fund"
]

# --------------------------------
# Sidebar
# --------------------------------
selected_fund = st.sidebar.selectbox("Select a Bluechip Fund", funds)
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["Last 1 Week", "Last 1 Month", "Last 3 Months", "Last 6 Months"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("🔍 *Data Source:* Google News via GNews API")

# --------------------------------
# News fetching
# --------------------------------
st.subheader(f"📰 Recent News for {selected_fund}")

google_news = GNews(language='en', country='IN', max_results=30)
news = google_news.get_news(selected_fund)

if not news:
    st.warning("No recent news found for this fund. Try another or adjust the time range.")
else:
    df = pd.DataFrame(news)
    df = df[["title", "description", "published date", "url"]]
    df.rename(columns={"published date": "date"}, inplace=True)

    # --------------------------------
    # Sentiment Analysis
    # --------------------------------
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment_label"] = df["sentiment"].apply(
        lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral")
    )

    # --------------------------------
    # Display News
    # --------------------------------
    for _, row in df.iterrows():
        st.markdown(f"### [{row['title']}]({row['url']})")
        st.write(row["description"])
        st.caption(f"Published: {row['date']} | Sentiment: **{row['sentiment_label']}**")
        st.markdown("---")

    # --------------------------------
    # Sentiment Summary
    # --------------------------------
    st.subheader("📊 Sentiment Summary")
    sentiment_summary = df["sentiment_label"].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_summary)

    # --------------------------------
    # Word Cloud
    # --------------------------------
    st.subheader("☁️ Word Cloud of Topics")
    text = " ".join(df["title"].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, GNews, and VADER Sentiment.")
