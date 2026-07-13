---
title: Stock Market Analysis & Prediction
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.58.0"
app_file: streamlit_app.py
pinned: false
license: mit
short_description: Technical analysis, ML/DL forecasting, sentiment, and buy/sell signals for stocks.
---

# Stock Market Analysis & Prediction System

**🔴 Live Demo:** [stock-market-analysis-prediction-1.onrender.com](https://stock-market-analysis-prediction-1.onrender.com)

> Deployed on Render's free tier using `requirements-cloud.txt`, so the
> Deep Learning Forecast page (LSTM/Prophet) isn't available on this
> deployment - everything else is fully live. Free tier also spins down
> after 15 minutes of inactivity, so the first load after a while may take
> 30-60 seconds to wake up.

A CLI + Streamlit toolkit for downloading stock data, analyzing it, and
generating multi-factor buy/sell recommendations.

## Features

- **Data Pipeline** - download (Yahoo Finance), clean, and feature-engineer
  OHLCV data for any ticker (RSI, MACD, Bollinger Bands, ATR, moving
  averages, volatility, and more).
- **Visualization** - 19+ chart types: candlesticks, correlation heatmaps,
  Monte Carlo simulations, relative strength, and more.
- **Machine Learning Prediction** - Linear Regression, Decision Tree,
  Random Forest, and XGBoost, auto-compared by RMSE.
- **Deep Learning Forecasting** - LSTM (Keras/TensorFlow) and Prophet
  models for multi-day price forecasts.
- **Portfolio Analysis** - annualized return/volatility, Sharpe ratio,
  max drawdown, correlation matrix, and a Monte Carlo efficient frontier
  (max-Sharpe / min-volatility portfolios).
- **News Sentiment Analysis** - scrapes free RSS feeds (Google News +
  Yahoo Finance, no API key needed) and scores headlines with a
  finance-tuned VADER model.
- **Multi-Stock Comparison** - normalized growth charts and a
  return/risk/Sharpe comparison table across any set of tickers.
- **Buy/Sell Recommendation Engine** - blends ML prediction (40%),
  technical indicators (35%), and news sentiment (25%) into one
  transparent, weighted score and call (STRONG BUY -> STRONG SELL).
- **Streamlit Web Dashboard** - a real-time interactive dashboard with a
  live scrolling ticker tape, candlestick/intraday charts, and a page for
  every feature above.

## Project Structure

```
main.py                   CLI entry point
streamlit_app.py          Streamlit web dashboard entry point
src/
    download_data.py      Yahoo Finance downloader
    preprocessing.py      Data cleaning
    feature_engineering.py Technical indicator generation
    visualization.py      Matplotlib chart menu (CLI)
    prediction.py         ML regression models + prediction menu
    model_utils.py        Shared ML training/eval/save utilities
    deep_learning.py       LSTM + Prophet forecasting
    portfolio.py           Portfolio risk/return analysis
    sentiment.py           News sentiment analysis
    comparison.py          Multi-stock comparison
    recommendation.py      Buy/Sell recommendation engine
data/raw/                 Downloaded OHLCV CSVs
data/processed/           Cleaned + feature-engineered CSVs
models/                   Saved ML models (.pkl), LSTM (.keras), Prophet
graphs/                   Saved chart images
reports/                  Saved text/CSV reports
```

## Setup

```bash
pip install -r requirements.txt
```

> Note: `tensorflow-cpu` and `prophet` are the heaviest installs. If you
> only need the classic ML models + dashboard basics, you can comment
> those two out and skip the Deep Learning Forecast page/menu.

## Usage

### CLI

```bash
python main.py
```

Walk through the numbered menu: download data (1) -> preprocess (2) ->
feature engineer (3), then explore Visualization, Prediction, Deep
Learning, Portfolio, Sentiment, Comparison, and Recommendation from
there. Each new module expects data to already exist in
`data/processed/`, i.e. steps 1-3 need to run first for any ticker
you want to analyze.

### Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Use the sidebar to type a ticker and click **Download & Process** to
pull it in (this runs the same download -> clean -> feature-engineer
pipeline as the CLI), then navigate between pages using the sidebar
radio buttons. The dashboard reads from the same `data/processed/`
folder as the CLI, so anything downloaded in one is available in the
other.

## Notes on the Recommendation Engine

The Buy/Sell score is a weighted blend:

```
final_score = 0.40 * ml_signal + 0.35 * technical_signal + 0.25 * sentiment_signal
```

- **ML signal**: the % change predicted by your best-performing trained
  regression model (train it first via the Prediction menu/page),
  clipped to [-1, 1].
- **Technical signal**: average of RSI, MACD-vs-signal, Bollinger Band
  position, and price-vs-moving-average trend, each mapped to [-1, 1].
- **Sentiment signal**: average VADER compound score across recent
  headlines, already in roughly [-1, 1].

If no ML model has been trained yet, its weight is redistributed
proportionally across the technical and sentiment signals so the score
stays meaningful. This is a research/education tool, not financial
advice - always do your own due diligence before trading.

## Deploying (Making It Live)

### Option A: Streamlit Community Cloud (free, easiest)

1. Push this project to a **public** GitHub repo (private repos need a paid plan).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app" -> select your repo/branch -> set the main file to
   `streamlit_app.py` -> Deploy.
4. You get a public URL like `yourname-stockapp.streamlit.app`.

**Resource limits matter here.** The free tier gives ~1GB RAM. `tensorflow-cpu`
and `prophet` are heavy enough that Deep Learning Forecasting may fail on that
tier. Two ways to handle it:

- Deploy as-is with `requirements.txt` and accept that the Deep Learning page
  may be flaky/unavailable there (the app is built to degrade gracefully -
  it'll show a warning and disable that page's button rather than crash).
- Deploy with `requirements-cloud.txt` instead (rename it to `requirements.txt`
  in your deployed branch, or point Streamlit Cloud's advanced settings at
  it) to skip `tensorflow-cpu`/`prophet` entirely and keep everything else
  fully working: Technical Analysis, Portfolio Analysis, News Sentiment,
  Comparison, classic ML Prediction, and Recommendation.

A `.streamlit/config.toml` is already included so the deployed app keeps the
same dark navy/emerald/coral theme.

### Option B: More headroom - Hugging Face Spaces

Hugging Face Spaces supports Streamlit directly, and its free CPU tier
(2 vCPU / 16GB RAM) has far more headroom than Streamlit Community Cloud's
~1GB, so it's the better free option if you want Deep Learning Forecasting
(LSTM/Prophet) to actually work reliably.

1. Create a free account at huggingface.co/join.
2. Go to huggingface.co/new-space -> pick a name -> SDK = **Streamlit** ->
   Hardware = **CPU basic (free)** -> Create Space.
3. Clone the Space's repo and copy this project into it:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   # copy in main.py, streamlit_app.py, src/, requirements.txt, .streamlit/
   ```
4. Push:
   ```bash
   git add .
   git commit -m "Deploy stock app"
   git push
   ```
   Use a Hugging Face **access token** (huggingface.co/settings/tokens, write
   scope) as the password when prompted, not your account password.
5. Watch the **Logs** tab while it builds - Prophet's Stan compile step takes
   a few minutes, that's normal. Once green, it's live at
   `https://huggingface.co/spaces/<username>/<space-name>`.

This `README.md` already has the YAML metadata block Hugging Face needs at
the top (title, sdk, `app_file: streamlit_app.py`, etc.) so it should work
as-is once pushed.

### Option C: Full control - your own server/container

For a VM (AWS/GCP/DigitalOcean/etc.) or Docker:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 80 --server.address 0.0.0.0
```

Put it behind a reverse proxy (nginx/Caddy) with HTTPS if it's public-facing.

### Notes for any option

- No API keys or secrets are required anywhere in this app (news sentiment
  uses free RSS feeds, not a paid API), so there's nothing to configure in
  `.streamlit/secrets.toml`.
- `data/`, `models/`, `reports/`, and `graphs/` are regenerated on demand and
  are gitignored - most free hosts have an ephemeral filesystem anyway, so
  don't rely on them persisting between restarts. If you need persistence
  (e.g. trained models surviving a redeploy), add external storage (S3, a
  database, etc.) - that's not included here.

### Option D: Render (already deployed here)

This project is live at
[stock-market-analysis-prediction-1.onrender.com](https://stock-market-analysis-prediction-1.onrender.com).

Unlike Hugging Face Spaces, Render deploys from a **GitHub repository** -
there's no drag-and-drop file upload option. Its free web service tier is
also only 512MB RAM, which is too tight for `tensorflow-cpu` + `prophet`,
so this deployment uses the lightweight `requirements-cloud.txt` (Deep
Learning Forecast page isn't available there).

To redeploy this yourself:

1. Push this project to a GitHub repo (public or private).
2. Go to render.com, sign in with GitHub.
3. Click **New > Web Service**, pick the repo, and set:
   - Build Command: `pip install -r requirements-cloud.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
   - Instance Type: Free
4. Wait for the build to finish, then open the URL Render gives you.

(A `render.yaml` blueprint is also included if you'd rather use **New >
Blueprint** instead of filling those fields in by hand.)

Note: free web services on Render spin down after 15 minutes of
inactivity, with a 30-60 second cold start on the next visit - same
tradeoff as the other free options above.

## Developer

Deepak
