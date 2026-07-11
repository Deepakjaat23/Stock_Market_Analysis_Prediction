# Stock Market Analysis & Prediction System

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

## Developer

Deepak
