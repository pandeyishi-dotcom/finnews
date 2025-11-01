# Bluechip Fund Sentiment Analyzer Pro

## Quick start (local)
1. Clone repo
2. Create venv & install:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

3. (Optional) Export NewsAPI key:
   export NEWSAPI_KEY="your_key_here"  # Linux/Mac
   set NEWSAPI_KEY="your_key_here"     # Windows (PowerShell)

4. Run:
   streamlit run app.py

## Deploy to Streamlit Cloud
1. Push repo to GitHub.
2. In Streamlit Cloud app settings, add `NEWSAPI_KEY` and `SLACK_WEBHOOK` as secrets if needed.
3. Deploy; note that installing transformers/torch can make build time long—omit if you don't need summaries.

## Notes
- If no NewsAPI key provided, the app will attempt to use GNews (no key).
- Transformer summarization is optional; fallback to a simple headline summary is provided.
- Upload a NAV CSV with columns: fund,date,nav to visualize NAV vs article volume.
