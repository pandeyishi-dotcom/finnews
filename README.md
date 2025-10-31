# Bluechip Fund News Intelligence (Streamlit)

## Setup (local)
1. Clone:
   git clone <your-repo>
   cd <your-repo>

2. Create virtualenv & install:
   python -m venv venv
   source venv/bin/activate  # or venv\\Scripts\\activate on Windows
   pip install -r requirements.txt

3. Export your NEWSAPI key:
   export NEWSAPI_KEY="your_key_here"   # Linux / Mac
   set NEWSAPI_KEY="your_key_here"      # Windows (PowerShell)

4. Run:
   streamlit run app.py

## Deploy to Streamlit Community Cloud
1. Push repo to GitHub.
2. Go to https://share.streamlit.io, connect GitHub, select your repo & branch.
3. Add `NEWSAPI_KEY` under "Secrets" in your Streamlit app settings.

## Notes & Next steps
- NewsAPI has rate limits and coverage; consider combining other sources (GNews, Google CSE) for broader local coverage.
- You can swap VADER for a transformer-based sentiment model for better nuance (but heavier).
- Add scheduling/alerts using GitHub Actions or a small backend if you want push notifications.
