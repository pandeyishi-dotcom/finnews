import streamlit as st
from gnews import GNews

st.title("GNews Test Successful")
st.success("The 'gnews' library was imported correctly!")

# Optional: Display a headline to confirm functionality
google_news = GNews()
st.write("Fetching one headline to confirm functionality...")
try:
    data = google_news.get_news_by_topic('TECHNOLOGY')
    if data:
        st.info(f"First headline: {data[0]['title']}")
    else:
        st.warning("Could not fetch any news data.")
except Exception as e:
    st.error(f"Error fetching data: {e}")
