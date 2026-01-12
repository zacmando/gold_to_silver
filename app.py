from datetime import datetime
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Gold & Silver", layout="wide")

TIMEFRAMES = [
    "1 Day",
    "3 Days",
    "1 Week",
    "1 Month",
    "1 Year",
    "3 Years",
    "5 Years",
    "10 Years",
    "Year to Date",
    "All Time",
]

# How far back to fetch from Yahoo for each timeframe
TIMEFRAME_CONFIG = {
    "1 Day": {"period": "5d", "interval": "5m"},
    "3 Days": {"period": "10d", "interval": "15m"},
    "1 Week": {"period": "1mo", "interval": "30m"},
    "1 Month": {"period": "3mo", "interval": "1h"},
    "1 Year": {"period": "2y", "interval": "1d"},
    "3 Years": {"period": "5y", "interval": "1d"},
    "5 Years": {"period": "10y", "interval": "1d"},
    "10 Years": {"period": "max", "interval": "1d"},
    "Year to Date": {"period": "ytd", "interval": "1d"},
    "All Time": {"period": "max", "interval": "1d"},
}

FX_MAP = {
    "USD": {"ticker": None, "mode": "direct"},
    "EUR": {"ticker": "EURUSD=X", "mode": "divide"},  # USD per EUR -> USD price / rate
    "GBP": {"ticker": "GBPUSD=X", "mode": "divide"},
    "CAD": {"ticker": "USDCAD=X", "mode": "multiply"},  # CAD per USD
    "JPY": {"ticker": "USDJPY=X", "mode": "multiply"},  # JPY per USD
}

PLOTLY_TEMPLATE = "plotly_dark"
PAPER_BG = "#0e1117"
PLOT_BG = "#0e1117"
FONT_COLOR = "#e6e6e6"
GRID_X = "rgba(255,255,255,0.25)"
GRID_Y = "rgba(255,255,255,0.2)"

PLOTLY_CFG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": False,
    "doubleClick": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
        "toImage",
    ],
    "modeBarButtonsToAdd": ["toggleFullscreen"],
}


def timeframe_start(option: str, end: pd.Timestamp) -> pd.Timestamp | None:
    if option == "All Time":
        return None
    if option == "Year to Date":
        return pd.Timestamp(datetime(end.year, 1, 1))
    deltas = {
        "1 Day": pd.Timedelta(days=1),
        "3 Days": pd.Timedelta(days=3),
        "1 Week": pd.Timedelta(days=7),
        "1 Month": pd.DateOffset(months=1),
        "1 Year": pd.DateOffset(years=1),
        "3 Years": pd.DateOffset(years=3),
        "5 Years": pd.DateOffset(years=5),
        "10 Years": pd.DateOffset(years=10),
    }
    delta = deltas.get(option)
    return end - delta if delta is not None else None


@st.cache_data(ttl=10 * 60)
def fetch_price_series(ticker: str, timeframe: str) -> pd.Series:
    cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["1 Year"])
    try:
        df = yf.download(
            ticker,
            period=cfg["period"],
            interval=cfg["interval"],
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.Series(dtype=float, name=ticker)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if col not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    s = df[col].dropna().copy()
    # If yfinance returns a DataFrame (MultiIndex columns), reduce to the first column
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return pd.Series(dtype=float, name=ticker)
        s = s.iloc[:, 0]
    s.name = ticker
    # Trim to timeframe window
    start = timeframe_start(timeframe, s.index.max())
    if start:
        s = s.loc[start:]
    return s


def convert_currency(series: pd.Series, currency: str, timeframe: str) -> pd.Series:
    # If a DataFrame is passed, reduce to first column series
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return pd.Series(dtype=float)
        series = series.iloc[:, 0]

    if currency == "USD":
        return series.rename(series.name.replace("USD", currency))
    cfg = FX_MAP.get(currency)
    if not cfg or not cfg["ticker"]:
        return series
    fx = fetch_price_series(cfg["ticker"], timeframe)
    if fx.empty:
        return pd.Series(dtype=float, name=series.name)
    fx = fx.reindex(series.index).ffill().bfill()
    if cfg["mode"] == "divide":
        converted = series / fx
    else:
        converted = series * fx
    label = series.name.replace("USD", currency) if "USD" in series.name else f"{series.name} ({currency})"
    converted.name = label
    return converted


def filter_timeframe(series: pd.Series, option: str) -> pd.Series:
    if series.empty or option == "All Time":
        return series
    end = series.index.max()
    start = timeframe_start(option, end)
    return series.loc[start:] if start else series


def metric_info(series: pd.Series, decimals: int = 2, suffix: str = "") -> Tuple[str, str | None]:
    if series.empty:
        return "n/a", None
    current = series.iloc[-1]
    start_val = series.iloc[0]
    value = f"{current:,.{decimals}f}{suffix}"
    if len(series) >= 2 and start_val != 0:
        delta = (current - start_val) / start_val * 100
        delta_str = f"{delta:+.2f}%"
    else:
        delta_str = None
    return value, delta_str


st.title("Gold & Silver")
st.caption("Data source: Yahoo Finance via yfinance (intervals per timeframe).")

with st.sidebar:
    st.header("Settings")
    timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=5)

    st.subheader("Currencies")
    gold_ccy = st.selectbox("Gold chart currency", list(FX_MAP.keys()), index=0)
    silver_ccy = st.selectbox("Silver chart currency", list(FX_MAP.keys()), index=0)

    ratio_choice = st.radio("Ratio", ["Gold / Silver", "Silver / Gold"], index=0)

    st.subheader("Stock in ounces")
    stock_symbol = st.text_input("Stock ticker", value="AAPL").strip().upper()
    stock_ccy = st.selectbox("Stock chart currency (fiat)", list(FX_MAP.keys()), index=0)
    stock_denom = st.radio("Stock denominator", ["Gold (oz)", "Silver (oz)"], index=0)

# Metals from yfinance futures (priced in USD/oz)
gold_usd = fetch_price_series("GC=F", timeframe)
silver_usd = fetch_price_series("SI=F", timeframe)
missing = []
if gold_usd.empty:
    missing.append("gold")
if silver_usd.empty:
    missing.append("silver")
if missing:
    st.error("No data returned from Yahoo Finance for " + ", ".join(missing))
    st.stop()

# Convert currencies
gold_price = convert_currency(gold_usd, gold_ccy, timeframe)
silver_price = convert_currency(silver_usd, silver_ccy, timeframe)

# Filter timeframe explicitly (post-conversion)
gold_price = filter_timeframe(gold_price, timeframe)
silver_price = filter_timeframe(silver_price, timeframe)
ratio_gold = filter_timeframe(gold_usd, timeframe)
ratio_silver = filter_timeframe(silver_usd, timeframe)

# Ratio
ratio_df = pd.concat([ratio_gold, ratio_silver], axis=1).dropna()
if ratio_choice.startswith("Gold"):
    ratio_series = (ratio_df.iloc[:, 0] / ratio_df.iloc[:, 1]).rename("Gold / Silver")
    ratio_decimals = 2
else:
    ratio_series = (ratio_df.iloc[:, 1] / ratio_df.iloc[:, 0]).rename("Silver / Gold")
    ratio_decimals = 4

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Prices")
    price_tabs = st.tabs(["Gold", "Silver"])

    with price_tabs[0]:
        if gold_price.empty:
            st.warning("No gold data in this timeframe.")
        else:
            val, delta = metric_info(gold_price, decimals=2, suffix=f" {gold_ccy}/oz")
            st.metric("Gold price", val, delta)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=gold_price.index,
                    y=gold_price.values,
                    mode="lines",
                    name=f"Gold ({gold_ccy})",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Gold: %{y:.2f} {gold_ccy}/oz<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=f"Price ({gold_ccy}/oz)",
                hovermode="x unified",
                template=PLOTLY_TEMPLATE,
                title=f"Gold Price ({timeframe})",
                paper_bgcolor=PAPER_BG,
                plot_bgcolor=PLOT_BG,
                font_color=FONT_COLOR,
                dragmode="pan",
            )
            fig.update_xaxes(
                showgrid=True,
                gridcolor=GRID_X,
                griddash="dot",
                fixedrange=True,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=1,
            )
            fig.update_yaxes(showgrid=True, gridcolor=GRID_Y, griddash="dot", fixedrange=True)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    with price_tabs[1]:
        if silver_price.empty:
            st.warning("No silver data in this timeframe.")
        else:
            val, delta = metric_info(silver_price, decimals=2, suffix=f" {silver_ccy}/oz")
            st.metric("Silver price", val, delta)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=silver_price.index,
                    y=silver_price.values,
                    mode="lines",
                    name=f"Silver ({silver_ccy})",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Silver: %{y:.2f} {silver_ccy}/oz<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=f"Price ({silver_ccy}/oz)",
                hovermode="x unified",
                template=PLOTLY_TEMPLATE,
                title=f"Silver Price ({timeframe})",
                paper_bgcolor=PAPER_BG,
                plot_bgcolor=PLOT_BG,
                font_color=FONT_COLOR,
                dragmode="pan",
            )
            fig.update_xaxes(
                showgrid=True,
                gridcolor=GRID_X,
                griddash="dot",
                fixedrange=True,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=1,
            )
            fig.update_yaxes(showgrid=True, gridcolor=GRID_Y, griddash="dot", fixedrange=True)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

with col2:
    st.subheader("Ratio")
    if ratio_series.empty:
        st.warning("No ratio data in this timeframe.")
    else:
        val, delta = metric_info(ratio_series, decimals=ratio_decimals, suffix=" oz")
        st.metric(f"{ratio_series.name}", val, delta)
        fig_ratio = go.Figure()
        fig_ratio.add_trace(
            go.Scatter(
                x=ratio_series.index,
                y=ratio_series.values,
                mode="lines",
                name=ratio_series.name,
                hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>Ratio: %{{y:.{ratio_decimals}f}} oz<extra></extra>",
            )
        )
        fig_ratio.update_layout(
            xaxis_title="Date",
            yaxis_title="Ratio (oz)",
            hovermode="x unified",
            template=PLOTLY_TEMPLATE,
            title=f"{ratio_series.name} ({timeframe})",
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG,
            font_color=FONT_COLOR,
            dragmode="pan",
        )
        fig_ratio.update_xaxes(
            showgrid=True,
            gridcolor=GRID_X,
            griddash="dot",
            fixedrange=True,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
        )
        fig_ratio.update_yaxes(showgrid=True, gridcolor=GRID_Y, griddash="dot", fixedrange=True)
        st.plotly_chart(fig_ratio, use_container_width=True, config=PLOTLY_CFG)

# --------------------- Stock in ounces ---------------------
st.divider()
st.header("Stock in ounces")

if stock_symbol:
    stock_usd = fetch_price_series(stock_symbol, timeframe)
    if stock_usd.empty:
        st.warning(f"No data returned for {stock_symbol} from Yahoo Finance.")
    else:
        if isinstance(stock_usd, pd.DataFrame):
            if stock_usd.shape[1] == 0:
                st.warning(f"No data returned for {stock_symbol} from Yahoo Finance.")
                st.stop()
            stock_usd = stock_usd.iloc[:, 0]
        stock_usd = stock_usd.rename(f"{stock_symbol} (USD)")
        stock_price = convert_currency(stock_usd, stock_ccy, timeframe)
        stock_price = filter_timeframe(stock_price, timeframe)

        stock_gold = pd.concat([stock_usd, gold_usd], axis=1).dropna()
        stock_silver = pd.concat([stock_usd, silver_usd], axis=1).dropna()
        stock_gold_oz = (stock_gold.iloc[:, 0] / stock_gold.iloc[:, 1]).rename(
            f"{stock_symbol} in gold (oz)"
        )
        stock_silver_oz = (stock_silver.iloc[:, 0] / stock_silver.iloc[:, 1]).rename(
            f"{stock_symbol} in silver (oz)"
        )
        stock_gold_oz = filter_timeframe(stock_gold_oz, timeframe)
        stock_silver_oz = filter_timeframe(stock_silver_oz, timeframe)

        col_a, col_b, col_c = st.columns(3)
        price_val, price_delta = metric_info(stock_price, decimals=2, suffix=f" {stock_ccy}")
        col_a.metric(f"{stock_symbol} price", price_val, price_delta)

        gold_val, gold_delta = metric_info(stock_gold_oz, decimals=4, suffix=" oz")
        col_b.metric(f"{stock_symbol} in gold", gold_val, gold_delta)

        silver_val, silver_delta = metric_info(stock_silver_oz, decimals=4, suffix=" oz")
        col_c.metric(f"{stock_symbol} in silver", silver_val, silver_delta)

        st.subheader(f"{stock_symbol} chart")
        denom_series = stock_gold_oz if stock_denom.startswith("Gold") else stock_silver_oz
        denom_name = "gold" if stock_denom.startswith("Gold") else "silver"
        if denom_series.empty:
            st.warning(f"No {denom_name}-denominated data in this timeframe.")
        else:
            fig_stock = go.Figure()
            fig_stock.add_trace(
                go.Scatter(
                    x=denom_series.index,
                    y=denom_series.values,
                    mode="lines",
                    name=f"{stock_symbol} in {denom_name}",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Value: %{y:.4f} oz<extra></extra>",
                )
            )
            fig_stock.update_layout(
                xaxis_title="Date",
                yaxis_title=f"Ounces of {denom_name}",
                hovermode="x unified",
                template=PLOTLY_TEMPLATE,
                title=f"{stock_symbol} in {denom_name} ({timeframe})",
                paper_bgcolor=PAPER_BG,
                plot_bgcolor=PLOT_BG,
                font_color=FONT_COLOR,
                dragmode="pan",
            )
            fig_stock.update_xaxes(
                showgrid=True,
                gridcolor=GRID_X,
                griddash="dot",
                fixedrange=True,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=1,
            )
            fig_stock.update_yaxes(showgrid=True, gridcolor=GRID_Y, griddash="dot", fixedrange=True)
            st.plotly_chart(fig_stock, use_container_width=True, config=PLOTLY_CFG)
else:
    st.info("Enter a stock ticker to view it in ounces.")
