"""
sentiment.py
------------
News Sentiment Analysis Module

Fetches recent headlines for a company/ticker from free, no-key-required
sources (Google News RSS and Yahoo Finance news feed) and scores them with
VADER sentiment analysis (tuned here with a small finance-specific lexicon
boost) to produce an overall Bullish / Bearish / Neutral signal.
"""

import os
import time
import feedparser
import pandas as pd
from urllib.parse import quote_plus
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

REPORTS_FOLDER = "reports"

# --------------------------------------------------------------------
# A handful of finance-specific words VADER doesn't score well by default.
# This nudges the lexicon without needing a heavy model like FinBERT.
# --------------------------------------------------------------------
FINANCE_LEXICON = {
    "beat": 2.0, "beats": 2.0, "misses": -2.0, "miss": -2.0,
    "surge": 2.5, "surges": 2.5, "soar": 2.8, "soars": 2.8,
    "plunge": -2.8, "plunges": -2.8, "crash": -3.0, "crashes": -3.0,
    "rally": 2.2, "rallies": 2.2, "slump": -2.2, "slumps": -2.2,
    "upgrade": 2.0, "upgraded": 2.0, "downgrade": -2.0, "downgraded": -2.0,
    "bullish": 2.5, "bearish": -2.5, "outperform": 2.0, "underperform": -2.0,
    "cut": -1.5, "cuts": -1.5, "layoffs": -2.5, "lawsuit": -2.0,
    "profit": 1.8, "profits": 1.8, "loss": -1.8, "losses": -1.8,
    "record high": 2.5, "record low": -2.5, "bankruptcy": -3.5,
    "buyback": 1.5, "dividend": 1.0, "recall": -2.0, "probe": -1.8,
    "investigation": -1.8, "fraud": -3.0, "growth": 1.5, "decline": -1.5,
}

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(FINANCE_LEXICON)


# --------------------------------------------------------------------
# Fetch headlines
# --------------------------------------------------------------------

def fetch_google_news(query, limit=20):
    """
    Scrape recent headlines from Google News RSS. No API key required.
    """

    url = f"https://news.google.com/rss/search?q={quote_plus(query)}+stock&hl=en-US&gl=US&ceid=US:en"

    headlines = []

    try:

        feed = feedparser.parse(url)

        for entry in feed.entries[:limit]:

            headlines.append({
                "title": entry.get("title", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", ""),
                "source": "Google News"
            })

    except Exception as e:

        print(f"Error fetching Google News: {e}")

    return headlines


def fetch_yahoo_news(ticker, limit=20):
    """
    Fetch recent news for a ticker via Yahoo Finance's RSS feed.
    No API key required.
    """

    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US"

    headlines = []

    try:

        feed = feedparser.parse(url)

        for entry in feed.entries[:limit]:

            headlines.append({
                "title": entry.get("title", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", ""),
                "source": "Yahoo Finance"
            })

    except Exception as e:

        print(f"Error fetching Yahoo Finance news: {e}")

    return headlines


def fetch_all_news(ticker, company_name=None, limit=15):
    """
    Combine Google News + Yahoo Finance headlines for a ticker.
    Falls back gracefully if one source returns nothing (e.g. offline,
    rate limited, or an unusual ticker symbol).
    """

    query = company_name if company_name else ticker

    headlines = fetch_google_news(query, limit=limit)

    headlines += fetch_yahoo_news(ticker, limit=limit)

    # De-duplicate by headline text
    seen = set()
    unique_headlines = []

    for h in headlines:

        key = h["title"].strip().lower()

        if key and key not in seen:

            seen.add(key)

            unique_headlines.append(h)

    return unique_headlines


# --------------------------------------------------------------------
# Sentiment Scoring
# --------------------------------------------------------------------

def score_headline(text):
    """
    Returns VADER compound score in range [-1, 1] for a single headline.
    """

    return _analyzer.polarity_scores(text)["compound"]


def classify_score(score):

    if score >= 0.15:
        return "Bullish"

    elif score <= -0.15:
        return "Bearish"

    else:
        return "Neutral"


def analyze_headlines(headlines):
    """
    Scores every headline and returns a DataFrame + summary stats.
    """

    if not headlines:

        return pd.DataFrame(), {
            "average_score": 0.0,
            "label": "No Data",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "total": 0
        }

    rows = []

    for h in headlines:

        score = score_headline(h["title"])

        rows.append({
            "Headline": h["title"],
            "Source": h["source"],
            "Published": h["published"],
            "Link": h["link"],
            "Sentiment_Score": round(score, 4),
            "Sentiment": classify_score(score)
        })

    df = pd.DataFrame(rows)

    avg_score = df["Sentiment_Score"].mean()

    summary = {
        "average_score": round(avg_score, 4),
        "label": classify_score(avg_score),
        "positive_count": int((df["Sentiment"] == "Bullish").sum()),
        "negative_count": int((df["Sentiment"] == "Bearish").sum()),
        "neutral_count": int((df["Sentiment"] == "Neutral").sum()),
        "total": len(df)
    }

    return df, summary


def get_stock_sentiment(ticker, company_name=None, limit=15):
    """
    Main entry point: fetch news + analyze it for a given ticker.

    Returns
    -------
    (headlines_df, summary_dict)
    """

    print(f"\nFetching news for {ticker}...")

    headlines = fetch_all_news(ticker, company_name=company_name, limit=limit)

    print(f"Found {len(headlines)} unique headlines.")

    df, summary = analyze_headlines(headlines)

    return df, summary


def save_sentiment_report(ticker, df, summary):

    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    csv_path = os.path.join(REPORTS_FOLDER, f"{ticker}_sentiment.csv")

    if not df.empty:

        df.to_csv(csv_path, index=False)

    txt_path = os.path.join(REPORTS_FOLDER, f"{ticker}_sentiment_summary.txt")

    lines = [
        "=" * 50,
        f"NEWS SENTIMENT REPORT - {ticker}",
        "=" * 50,
        f"Overall Sentiment : {summary['label']}",
        f"Average Score     : {summary['average_score']}",
        f"Bullish Headlines : {summary['positive_count']}",
        f"Bearish Headlines : {summary['negative_count']}",
        f"Neutral Headlines : {summary['neutral_count']}",
        f"Total Headlines   : {summary['total']}",
        "=" * 50
    ]

    text = "\n".join(lines)

    with open(txt_path, "w") as f:
        f.write(text)

    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {txt_path}")


# --------------------------------------------------------------------
# CLI Menu
# --------------------------------------------------------------------

def sentiment_menu():

    while True:

        print("\n" + "=" * 55)
        print("        NEWS SENTIMENT ANALYSIS MENU")
        print("=" * 55)

        print("1. Analyze Sentiment for a Ticker")
        print("0. Back")

        choice = input("\nEnter your choice : ")

        if choice == "1":

            ticker = input("\nEnter Stock Symbol (e.g. AAPL): ").upper().strip()

            if not ticker:

                print("Please enter a valid symbol.")

                continue

            df, summary = get_stock_sentiment(ticker)

            print("\n" + "-" * 40)
            print(f"Overall Sentiment : {summary['label']}")
            print(f"Average Score     : {summary['average_score']}")
            print(f"Bullish / Bearish / Neutral : "
                  f"{summary['positive_count']} / "
                  f"{summary['negative_count']} / "
                  f"{summary['neutral_count']}")
            print("-" * 40)

            if not df.empty:

                for _, row in df.head(10).iterrows():

                    print(f"[{row['Sentiment']:>7}] {row['Headline']}")

                save_sentiment_report(ticker, df, summary)

            else:

                print("No headlines found. Try again later or check the symbol.")

        elif choice == "0":

            break

        else:

            print("Invalid Choice")


if __name__ == "__main__":

    sentiment_menu()
