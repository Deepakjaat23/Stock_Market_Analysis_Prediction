"""
streamlit_app.py
----------------
Real-Time Stock Market Analysis & Prediction Dashboard

Run with:
    streamlit run streamlit_app.py

Ties together every module in src/ (download, preprocessing, feature
engineering, technical visualization, ML prediction, deep learning,
portfolio analysis, sentiment analysis, comparison, and the buy/sell
recommendation engine) into a single interactive app.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

from src.download_data import download_stock_data, save_stock_data
from src.preprocessing import clean_stock_data
from src.feature_engineering import create_features
from src import portfolio as portfolio_module
from src import comparison as comparison_module
from src import sentiment as sentiment_module
from src import recommendation as recommendation_module
from src.model_utils import FEATURE_COLUMNS

warnings.filterwarnings("ignore")

RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"

os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# ======================================================================
# PAGE CONFIG + THEME
# ======================================================================

st.set_page_config(
    page_title="Stock Market Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = {
    "bg": "#0E1117",
    "panel": "#161B27",
    "border": "#262D3D",
    "text": "#E7EAF1",
    "muted": "#8892A6",
    "buy": "#2FBF71",
    "sell": "#E5484D",
    "hold": "#F2B84B",
    "accent": "#3E8EDE",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {PALETTE['text']};
}}

.stApp {{
    background-color: {PALETTE['bg']};
}}

section[data-testid="stSidebar"] {{
    background-color: {PALETTE['panel']};
    border-right: 1px solid {PALETTE['border']};
}}

.ticker-wrap {{
    width: 100%;
    overflow: hidden;
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 10px 0;
    margin-bottom: 18px;
}}

.ticker-move {{
    display: inline-block;
    white-space: nowrap;
    animation: ticker-scroll 30s linear infinite;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px;
}}

.ticker-wrap:hover .ticker-move {{
    animation-play-state: paused;
}}

@keyframes ticker-scroll {{
    0%   {{ transform: translateX(0%); }}
    100% {{ transform: translateX(-50%); }}
}}

.ticker-item {{ padding: 0 28px; }}
.up {{ color: {PALETTE['buy']}; }}
.down {{ color: {PALETTE['sell']}; }}

.metric-card {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 16px 18px;
}}

.metric-label {{
    color: {PALETTE['muted']};
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.metric-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
}}

.big-call {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 34px;
    font-weight: 700;
    text-align: center;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid {PALETTE['border']};
}}

h1, h2, h3 {{
    font-family: 'Inter', sans-serif;
    font-weight: 700;
}}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# HELPERS
# ======================================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_quote(ticker):
    """Lightweight, frequently-refreshed quote for the ticker tape / dashboard header."""

    try:

        t = yf.Ticker(ticker)

        hist = t.history(period="2d", interval="1d")

        if hist.empty:

            return None

        last_close = hist["Close"].iloc[-1]

        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else last_close

        change = last_close - prev_close

        pct = (change / prev_close) * 100 if prev_close else 0

        return {
            "ticker": ticker,
            "price": round(float(last_close), 2),
            "change": round(float(change), 2),
            "pct": round(float(pct), 2),
        }

    except Exception:

        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker, period="5d", interval="15m"):

    try:

        return yf.Ticker(ticker).history(period=period, interval=interval)

    except Exception:

        return pd.DataFrame()


def list_available_companies():

    if not os.path.exists(PROCESSED_FOLDER):

        return []

    return sorted(f[:-4] for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".csv"))


@st.cache_data(show_spinner=False)
def load_processed(company):

    path = os.path.join(PROCESSED_FOLDER, f"{company}.csv")

    return pd.read_csv(path, parse_dates=["Date"])


def fetch_and_prepare_ticker(ticker, period="5y"):
    """
    Full pipeline for a brand-new ticker typed into the sidebar:
    download -> clean -> feature engineer -> save to data/processed.
    """

    df = download_stock_data(ticker, period=period)

    if df is None or df.empty:

        return None

    save_stock_data(df, ticker)

    df = clean_stock_data(df)

    df = create_features(df)

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    df.to_csv(os.path.join(PROCESSED_FOLDER, f"{ticker}.csv"), index=False)

    return df


def plotly_candlestick(df, title):

    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"] if "Date" in df.columns else df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=PALETTE["buy"],
        decreasing_line_color=PALETTE["sell"],
    )])

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False
    )

    return fig


# ======================================================================
# SIDEBAR
# ======================================================================

st.sidebar.markdown("## 📈 Control Panel")

watchlist_input = st.sidebar.text_input(
    "Watchlist (comma separated)",
    value="AAPL, MSFT, TSLA",
    help="Symbols shown in the live ticker tape at the top of the dashboard."
)

watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

st.sidebar.markdown("---")

st.sidebar.markdown("### Add / Refresh a Company")

new_ticker = st.sidebar.text_input("Ticker to fetch (e.g. AAPL, RELIANCE.NS)", value="")

fetch_period = st.sidebar.selectbox("History period", ["1y", "2y", "5y", "10y", "max"], index=2)

if st.sidebar.button("Download & Process"):

    if new_ticker.strip():

        with st.spinner(f"Downloading and processing {new_ticker.upper()}..."):

            result = fetch_and_prepare_ticker(new_ticker.upper().strip(), period=fetch_period)

        if result is not None:

            st.sidebar.success(f"{new_ticker.upper()} ready.")

            st.cache_data.clear()

        else:

            st.sidebar.error("No data found for that symbol.")

    else:

        st.sidebar.warning("Enter a ticker symbol first.")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Live Dashboard",
        "📊 Technical Analysis",
        "🧠 Deep Learning Forecast",
        "💼 Portfolio Analysis",
        "📰 News Sentiment",
        "⚖️ Compare Stocks",
        "✅ Buy/Sell Recommendation",
    ]
)

companies = list_available_companies()


# ======================================================================
# TICKER TAPE HEADER
# ======================================================================

def render_ticker_tape(symbols):

    quotes = [fetch_live_quote(s) for s in symbols]

    quotes = [q for q in quotes if q is not None]

    if not quotes:

        return

    items = []

    for q in quotes * 2:  # duplicate for seamless scroll

        cls = "up" if q["change"] >= 0 else "down"

        arrow = "▲" if q["change"] >= 0 else "▼"

        items.append(
            f'<span class="ticker-item {cls}">{q["ticker"]} '
            f'{q["price"]:.2f} {arrow} {q["change"]:+.2f} ({q["pct"]:+.2f}%)</span>'
        )

    html = f'<div class="ticker-wrap"><div class="ticker-move">{"".join(items)}</div></div>'

    st.markdown(html, unsafe_allow_html=True)


st.title("Stock Market Analysis & Prediction")

render_ticker_tape(watchlist if watchlist else ["AAPL"])


# ======================================================================
# PAGE: LIVE DASHBOARD
# ======================================================================

if page == "🏠 Live Dashboard":

    if not companies:

        st.info("No processed companies yet. Use **Add / Refresh a Company** in the sidebar to get started.")

    else:

        selected = st.selectbox("Select a company", companies)

        col_refresh, _ = st.columns([1, 5])

        if col_refresh.button("🔄 Refresh Live Price"):

            st.cache_data.clear()

        quote = fetch_live_quote(selected)

        df = load_processed(selected)

        c1, c2, c3, c4 = st.columns(4)

        if quote:

            change_color = PALETTE["buy"] if quote["change"] >= 0 else PALETTE["sell"]

            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Last Price</div>
                    <div class="metric-value">${quote['price']:.2f}</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Change</div>
                    <div class="metric-value" style="color:{change_color}">
                        {quote['change']:+.2f} ({quote['pct']:+.2f}%)
                    </div>
                </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">RSI (Latest)</div>
                <div class="metric-value">{df['RSI'].iloc[-1]:.1f}</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Volatility (20d)</div>
                <div class="metric-value">{df['Volatility'].iloc[-1]*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Candlestick (Historical)", "Intraday (Live-ish)"])

        with tab1:

            fig = plotly_candlestick(df.tail(150), f"{selected} - Last 150 Trading Days")

            st.plotly_chart(fig, width='stretch')

        with tab2:

            intraday = fetch_intraday(selected)

            if not intraday.empty:

                intraday = intraday.reset_index()

                date_col = "Datetime" if "Datetime" in intraday.columns else intraday.columns[0]

                fig2 = go.Figure(go.Scatter(
                    x=intraday[date_col], y=intraday["Close"],
                    mode="lines", line=dict(color=PALETTE["accent"], width=2)
                ))

                fig2.update_layout(
                    title=f"{selected} - Intraday (15m interval)",
                    template="plotly_dark",
                    paper_bgcolor=PALETTE["panel"],
                    plot_bgcolor=PALETTE["panel"],
                    height=400
                )

                st.plotly_chart(fig2, width='stretch')

            else:

                st.warning("Intraday data unavailable for this symbol right now.")


# ======================================================================
# PAGE: TECHNICAL ANALYSIS
# ======================================================================

elif page == "📊 Technical Analysis":

    if not companies:

        st.info("No processed companies yet. Add one from the sidebar first.")

    else:

        selected = st.selectbox("Select a company", companies)

        df = load_processed(selected)

        indicator = st.radio(
            "Indicator", ["Moving Averages", "Bollinger Bands", "RSI", "MACD", "Volume"],
            horizontal=True
        )

        fig = go.Figure()

        if indicator == "Moving Averages":

            for col, color in [("Close", PALETTE["text"]), ("MA20", PALETTE["accent"]),
                                ("MA50", PALETTE["hold"]), ("MA100", PALETTE["sell"])]:

                fig.add_trace(go.Scatter(x=df["Date"], y=df[col], name=col, line=dict(color=color)))

        elif indicator == "Bollinger Bands":

            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="Upper Band",
                                      line=dict(color=PALETTE["sell"], dash="dash")))

            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close",
                                      line=dict(color=PALETTE["text"])))

            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="Lower Band",
                                      line=dict(color=PALETTE["buy"], dash="dash"),
                                      fill="tonexty", fillcolor="rgba(62,142,222,0.08)"))

        elif indicator == "RSI":

            fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color=PALETTE["accent"])))

            fig.add_hline(y=70, line_dash="dash", line_color=PALETTE["sell"], annotation_text="Overbought")

            fig.add_hline(y=30, line_dash="dash", line_color=PALETTE["buy"], annotation_text="Oversold")

        elif indicator == "MACD":

            fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color=PALETTE["accent"])))

            fig.add_trace(go.Scatter(x=df["Date"], y=df["Signal_Line"], name="Signal Line",
                                      line=dict(color=PALETTE["hold"])))

        elif indicator == "Volume":

            fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color=PALETTE["accent"]))

        fig.update_layout(
            title=f"{selected} - {indicator}",
            template="plotly_dark",
            paper_bgcolor=PALETTE["panel"],
            plot_bgcolor=PALETTE["panel"],
            height=500
        )

        st.plotly_chart(fig, width='stretch')


# ======================================================================
# PAGE: DEEP LEARNING FORECAST
# ======================================================================

elif page == "🧠 Deep Learning Forecast":

    if not companies:

        st.info("No processed companies yet. Add one from the sidebar first.")

    else:

        selected = st.selectbox("Select a company", companies)

        model_choice = st.radio("Model", ["LSTM", "Prophet"], horizontal=True)

        days = st.slider("Forecast horizon (days)", 5, 60, 30)

        if model_choice == "LSTM":

            epochs = st.slider("Training epochs", 5, 50, 15)

        run = st.button("Train & Forecast")

        if run:

            from src import deep_learning as dl

            df = load_processed(selected)

            if model_choice == "LSTM":

                with st.spinner("Training LSTM... this can take a minute."):

                    model, scaler, close_prices = dl.train_lstm_model(selected, epochs=epochs)

                if model is None:

                    st.error("Not enough historical data to train an LSTM for this company.")

                else:

                    predictions = dl.predict_lstm_future(model, scaler, close_prices, days=days)

                    future_dates = pd.bdate_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=days)

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(x=df["Date"].tail(150), y=df["Close"].tail(150),
                                              name="Historical", line=dict(color=PALETTE["text"])))

                    fig.add_trace(go.Scatter(x=future_dates, y=predictions,
                                              name="LSTM Forecast", line=dict(color=PALETTE["accent"], dash="dash")))

                    fig.update_layout(title=f"{selected} - LSTM Forecast", template="plotly_dark",
                                       paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"], height=500)

                    st.plotly_chart(fig, width='stretch')

                    st.dataframe(pd.DataFrame({"Date": future_dates, "Predicted_Close": np.round(predictions, 2)}))

            else:

                with st.spinner("Training Prophet model..."):

                    model = dl.train_prophet_model(selected)

                    forecast = dl.predict_prophet_future(model, days=days)

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=df["Date"].tail(150), y=df["Close"].tail(150),
                                          name="Historical", line=dict(color=PALETTE["text"])))

                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                                          name="Prophet Forecast", line=dict(color=PALETTE["buy"], dash="dash")))

                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"],
                                          line=dict(width=0), showlegend=False))

                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                                          line=dict(width=0), fill="tonexty",
                                          fillcolor="rgba(47,191,113,0.1)", name="Confidence Interval"))

                fig.update_layout(title=f"{selected} - Prophet Forecast", template="plotly_dark",
                                   paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"], height=500)

                st.plotly_chart(fig, width='stretch')

                st.dataframe(forecast.round(2))


# ======================================================================
# PAGE: PORTFOLIO ANALYSIS
# ======================================================================

elif page == "💼 Portfolio Analysis":

    if len(companies) < 1:

        st.info("Add at least one company from the sidebar first.")

    else:

        selected = st.multiselect("Select holdings", companies, default=companies[:min(3, len(companies))])

        if len(selected) >= 1:

            weight_mode = st.radio("Weights", ["Equal Weight", "Custom Weights"], horizontal=True)

            weights = None

            if weight_mode == "Custom Weights":

                weights = []

                cols = st.columns(len(selected))

                for i, c in enumerate(selected):

                    weights.append(cols[i].number_input(f"{c} weight", min_value=0.0, max_value=1.0,
                                                          value=round(1/len(selected), 2), step=0.05))

            if st.button("Analyze Portfolio"):

                result = portfolio_module.portfolio_summary(selected, weights)

                if result:

                    summary, prices, returns, port_returns, cumulative = result

                    stats_table = portfolio_module.individual_asset_stats(prices, returns, summary["weights"])

                    # ---- Sharpe rating badge ----
                    sharpe = summary["sharpe_ratio"]

                    if sharpe >= 1.5:
                        rating, rating_color = "EXCELLENT", PALETTE["buy"]
                    elif sharpe >= 1.0:
                        rating, rating_color = "GOOD", PALETTE["accent"]
                    elif sharpe >= 0.5:
                        rating, rating_color = "FAIR", PALETTE["hold"]
                    else:
                        rating, rating_color = "NEEDS IMPROVEMENT", PALETTE["sell"]

                    st.markdown(f"""
                        <div class="big-call" style="color:{rating_color}; font-size:24px;">
                            PORTFOLIO HEALTH: {rating}
                        </div>
                        <p style="text-align:center; color:{PALETTE['muted']}; margin-top:6px;">
                            Based on a Sharpe ratio of {sharpe}
                        </p>
                        <br>
                    """, unsafe_allow_html=True)

                    # ---- Styled metric cards ----
                    ret_color = PALETTE["buy"] if summary["annual_return"] >= 0 else PALETTE["sell"]

                    m1, m2, m3, m4 = st.columns(4)

                    for col, label, value, color in [
                        (m1, "Annual Return", f"{summary['annual_return']:+.2f}%", ret_color),
                        (m2, "Annual Volatility", f"{summary['annual_volatility']:.2f}%", PALETTE["hold"]),
                        (m3, "Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}", rating_color),
                        (m4, "Max Drawdown", f"{summary['max_drawdown']:.2f}%", PALETTE["sell"]),
                    ]:
                        col.markdown(f"""<div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value" style="color:{color}">{value}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ---- Allocation donut + Risk/Return bubble map ----
                    col_pie, col_bubble = st.columns([1, 1.3])

                    with col_pie:

                        st.subheader("Allocation")

                        pie_colors = px.colors.qualitative.Set2

                        fig_pie = go.Figure(go.Pie(
                            labels=stats_table["Ticker"],
                            values=stats_table["Weight_%"],
                            hole=0.55,
                            marker=dict(colors=pie_colors),
                            textinfo="label+percent"
                        ))

                        fig_pie.update_layout(
                            template="plotly_dark", paper_bgcolor=PALETTE["panel"],
                            height=380, showlegend=False,
                            margin=dict(l=10, r=10, t=10, b=10),
                            annotations=[dict(text=f"{len(selected)} Holdings", showarrow=False, font=dict(size=14))]
                        )

                        st.plotly_chart(fig_pie, width='stretch')

                    with col_bubble:

                        st.subheader("Risk vs. Return by Holding")

                        fig_bubble = px.scatter(
                            stats_table, x="Annual_Volatility_%", y="Annual_Return_%",
                            size="Weight_%", color="Sharpe_Ratio", text="Ticker",
                            color_continuous_scale="Tealrose", size_max=55,
                            template="plotly_dark"
                        )

                        fig_bubble.update_traces(textposition="top center")

                        fig_bubble.update_layout(
                            paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"], height=380,
                            xaxis_title="Annual Volatility (%)", yaxis_title="Annual Return (%)",
                            margin=dict(l=10, r=10, t=10, b=10)
                        )

                        st.plotly_chart(fig_bubble, width='stretch')

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ---- Detail tabs ----
                    tab_perf, tab_holdings, tab_corr, tab_frontier = st.tabs(
                        ["📈 Performance", "🧾 Holdings Breakdown", "🔗 Correlation", "🎯 Efficient Frontier"]
                    )

                    with tab_perf:

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=cumulative.index, y=(cumulative - 1) * 100,
                            name="Portfolio", line=dict(color=PALETTE["accent"], width=3),
                            fill="tozeroy", fillcolor="rgba(62,142,222,0.12)"
                        ))

                        for ticker in prices.columns:

                            asset_cum = (1 + returns[ticker]).cumprod()

                            fig.add_trace(go.Scatter(
                                x=asset_cum.index, y=(asset_cum - 1) * 100,
                                name=ticker, line=dict(width=1.2, dash="dot"), opacity=0.7
                            ))

                        fig.update_layout(title="Cumulative Return: Portfolio vs. Individual Holdings (%)",
                                           template="plotly_dark", paper_bgcolor=PALETTE["panel"],
                                           plot_bgcolor=PALETTE["panel"], height=420)

                        st.plotly_chart(fig, width='stretch')

                        col_dd, col_vol = st.columns(2)

                        with col_dd:

                            dd = portfolio_module.drawdown_series(cumulative) * 100

                            fig_dd = go.Figure(go.Scatter(
                                x=dd.index, y=dd, line=dict(color=PALETTE["sell"], width=1.5),
                                fill="tozeroy", fillcolor="rgba(229,72,77,0.2)"
                            ))

                            fig_dd.update_layout(title="Underwater Curve (Drawdown %)", template="plotly_dark",
                                                  paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
                                                  height=320)

                            st.plotly_chart(fig_dd, width='stretch')

                        with col_vol:

                            roll_vol = portfolio_module.rolling_volatility(port_returns) * 100

                            fig_vol = go.Figure(go.Scatter(
                                x=roll_vol.index, y=roll_vol, line=dict(color=PALETTE["hold"], width=1.5)
                            ))

                            fig_vol.update_layout(title="Rolling 30-Day Annualized Volatility (%)",
                                                   template="plotly_dark", paper_bgcolor=PALETTE["panel"],
                                                   plot_bgcolor=PALETTE["panel"], height=320)

                            st.plotly_chart(fig_vol, width='stretch')

                    with tab_holdings:

                        st.dataframe(
                            stats_table.style.format({
                                "Weight_%": "{:.1f}%", "Total_Return_%": "{:+.2f}%",
                                "Annual_Return_%": "{:+.2f}%", "Annual_Volatility_%": "{:.2f}%",
                                "Sharpe_Ratio": "{:.2f}", "Contribution_%": "{:+.2f}%"
                            }),
                            width='stretch'
                        )

                        fig_contrib = px.bar(
                            stats_table.sort_values("Contribution_%"), x="Contribution_%", y="Ticker",
                            orientation="h", color="Contribution_%", color_continuous_scale="RdYlGn",
                            template="plotly_dark"
                        )

                        fig_contrib.update_layout(title="Contribution to Portfolio Annual Return (%)",
                                                   paper_bgcolor=PALETTE["panel"], height=350)

                        st.plotly_chart(fig_contrib, width='stretch')

                    with tab_corr:

                        if len(selected) >= 2:

                            corr = returns.corr()

                            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                                  zmin=-1, zmax=1, template="plotly_dark")

                            fig_corr.update_layout(paper_bgcolor=PALETTE["panel"], height=420)

                            st.plotly_chart(fig_corr, width='stretch')

                            st.caption("Lower (bluer) correlations between holdings mean more diversification benefit.")

                        else:

                            st.info("Select at least 2 holdings to see correlation.")

                    with tab_frontier:

                        if len(selected) >= 2:

                            with st.spinner("Simulating random portfolios..."):

                                results_df, max_sharpe, min_vol = portfolio_module.simulate_random_portfolios(
                                    returns, num_portfolios=2000
                                )

                            fig_ef = px.scatter(results_df, x="Volatility", y="Return", color="Sharpe",
                                                 color_continuous_scale="Viridis", template="plotly_dark",
                                                 opacity=0.6)

                            fig_ef.add_trace(go.Scatter(
                                x=[summary["annual_volatility"] / 100], y=[summary["annual_return"] / 100],
                                mode="markers", marker=dict(size=20, color=PALETTE["text"], symbol="diamond",
                                                             line=dict(width=2, color=PALETTE["accent"])),
                                name="Your Portfolio"
                            ))

                            fig_ef.add_trace(go.Scatter(x=[max_sharpe["Volatility"]], y=[max_sharpe["Return"]],
                                                         mode="markers", marker=dict(size=18, color=PALETTE["sell"], symbol="star"),
                                                         name="Max Sharpe"))

                            fig_ef.add_trace(go.Scatter(x=[min_vol["Volatility"]], y=[min_vol["Return"]],
                                                         mode="markers", marker=dict(size=18, color=PALETTE["accent"], symbol="star"),
                                                         name="Min Volatility"))

                            fig_ef.update_layout(paper_bgcolor=PALETTE["panel"], height=480,
                                                  xaxis_title="Annualized Volatility", yaxis_title="Annualized Return")

                            st.plotly_chart(fig_ef, width='stretch')

                            st.caption(
                                "Max Sharpe weights: "
                                + ", ".join(f"{t}: {max_sharpe[t]:.0%}" for t in selected if t in max_sharpe)
                            )

                        else:

                            st.info("Select at least 2 holdings to simulate an efficient frontier.")


# ======================================================================
# PAGE: NEWS SENTIMENT
# ======================================================================

elif page == "📰 News Sentiment":

    ticker = st.text_input("Ticker for news sentiment", value=companies[0] if companies else "AAPL")

    if st.button("Fetch & Analyze News"):

        with st.spinner("Fetching headlines..."):

            df, summary = sentiment_module.get_stock_sentiment(ticker.upper().strip())

        color = PALETTE["buy"] if summary["label"] == "Bullish" else (
            PALETTE["sell"] if summary["label"] == "Bearish" else PALETTE["hold"])

        st.markdown(f"""<div class="big-call" style="color:{color}">{summary['label']}</div>""",
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Average Score", summary["average_score"])
        c2.metric("Bullish", summary["positive_count"])
        c3.metric("Bearish", summary["negative_count"])
        c4.metric("Neutral", summary["neutral_count"])

        if not df.empty:

            fig = px.bar(df, x="Sentiment_Score", y="Headline", orientation="h",
                          color="Sentiment",
                          color_discrete_map={"Bullish": PALETTE["buy"], "Bearish": PALETTE["sell"],
                                               "Neutral": PALETTE["hold"]},
                          template="plotly_dark")

            fig.update_layout(paper_bgcolor=PALETTE["panel"], height=max(400, len(df) * 30),
                               yaxis=dict(autorange="reversed"))

            st.plotly_chart(fig, width='stretch')

            st.dataframe(df[["Headline", "Source", "Sentiment", "Sentiment_Score"]], width='stretch')

        else:

            st.warning("No headlines found. This sandbox may not have live internet access to news "
                       "sources — this will work normally once run on your machine.")


# ======================================================================
# PAGE: COMPARE STOCKS
# ======================================================================

elif page == "⚖️ Compare Stocks":

    if len(companies) < 2:

        st.info("Add at least 2 companies from the sidebar to compare.")

    else:

        selected = st.multiselect("Companies to compare", companies, default=companies)

        if len(selected) >= 2:

            datasets = comparison_module.load_companies(selected)

            fig = go.Figure()

            for company, df in datasets.items():

                normalized = df["Close"] / df["Close"].iloc[0] * 100

                fig.add_trace(go.Scatter(x=df["Date"], y=normalized, name=company))

            fig.update_layout(title="Normalized Growth (Base = 100)", template="plotly_dark",
                               paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"], height=450)

            st.plotly_chart(fig, width='stretch')

            table = comparison_module.build_comparison_table(datasets)

            st.dataframe(table, width='stretch')

            fig_bar = px.bar(table, x="Company", y="Total_Return_%", color="Total_Return_%",
                              color_continuous_scale="RdYlGn", template="plotly_dark")

            fig_bar.update_layout(paper_bgcolor=PALETTE["panel"], height=400)

            st.plotly_chart(fig_bar, width='stretch')


# ======================================================================
# PAGE: BUY/SELL RECOMMENDATION
# ======================================================================

elif page == "✅ Buy/Sell Recommendation":

    if not companies:

        st.info("Add at least one company from the sidebar first.")

    else:

        selected = st.selectbox("Select a company", companies)

        best_model_name = st.selectbox(
            "Trained regression model to use for the ML signal (optional)",
            ["None"] + ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
        )

        if st.button("Generate Recommendation"):

            with st.spinner("Blending ML, technical, and sentiment signals..."):

                result = recommendation_module.generate_recommendation(
                    selected,
                    best_model_name=None if best_model_name == "None" else best_model_name
                )

            color = PALETTE["buy"] if "BUY" in result["recommendation"] else (
                PALETTE["sell"] if "SELL" in result["recommendation"] else PALETTE["hold"])

            st.markdown(
                f"""<div class="big-call" style="color:{color}">{result['recommendation']}</div>
                <p style="text-align:center; color:{PALETTE['muted']}">Blended score: {result['final_score']}</p>""",
                unsafe_allow_html=True
            )

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["final_score"],
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [-1, 1]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [-1, -0.15], "color": "rgba(229,72,77,0.25)"},
                        {"range": [-0.15, 0.15], "color": "rgba(242,184,75,0.25)"},
                        {"range": [0.15, 1], "color": "rgba(47,191,113,0.25)"},
                    ],
                }
            ))

            gauge.update_layout(paper_bgcolor=PALETTE["panel"], height=300,
                                 margin=dict(l=20, r=20, t=20, b=20))

            st.plotly_chart(gauge, width='stretch')

            c1, c2, c3 = st.columns(3)

            with c1:

                st.subheader("ML Signal")

                st.write(result["ml_signal"]["details"] or "Not available - select a trained model above.")

            with c2:

                st.subheader("Technical Signal")

                st.write(result["technical_signal"]["details"])

            with c3:

                st.subheader("Sentiment Signal")

                st.write(result["sentiment_signal"]["details"])
