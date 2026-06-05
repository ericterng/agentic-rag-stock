import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from langchain_core.tools import tool
from src.config import CHARTS_DIR


def get_stock_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    return df


def get_fundamental_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "name": info.get("longName", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A"),
        "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        "current_price": info.get("currentPrice", "N/A"),
    }


def plot_stock_chart(ticker: str, df: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(df.index, df["Close"], linewidth=2, color="#2196F3", label="Close Price")
    ax1.fill_between(df.index, df["Close"], alpha=0.1, color="#2196F3")
    ax1.set_title(f"{ticker} Stock Price", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Price", fontsize=12)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(True, alpha=0.3)

    ax2.bar(df.index, df["Volume"], color="#90CAF9", alpha=0.7)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = CHARTS_DIR / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return str(filename)


@tool
def tool_get_stock_history(ticker: str, period: str = "3mo") -> str:
    """Get historical stock price data for a given ticker symbol.

    Args:
        ticker: Stock ticker (e.g. '2330.TW' for TSMC, '0050.TW' for Taiwan 50 ETF)
        period: Time period — '1mo', '3mo', '6mo', '1y'
    """
    try:
        df = get_stock_history(ticker, period)
        return (
            f"Stock: {ticker} | Period: {period}\n"
            f"Latest Close: {df['Close'].iloc[-1]:.2f}\n"
            f"Period High:  {df['High'].max():.2f}\n"
            f"Period Low:   {df['Low'].min():.2f}\n"
            f"Avg Volume:   {df['Volume'].mean():.0f}\n"
            f"Price Change: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%\n"
            f"Trading Days: {len(df)}"
        )
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {e}"


@tool
def tool_get_fundamental_data(ticker: str) -> str:
    """Get fundamental financial data (P/E, dividend yield, market cap) for a stock.

    Args:
        ticker: Stock ticker (e.g. '2330.TW')
    """
    try:
        d = get_fundamental_data(ticker)
        return (
            f"Fundamentals for {d['name']} ({ticker}):\n"
            f"  Current Price:  {d['current_price']}\n"
            f"  Market Cap:     {d['market_cap']}\n"
            f"  P/E Ratio:      {d['pe_ratio']}\n"
            f"  Dividend Yield: {d['dividend_yield']}\n"
            f"  52-Week High:   {d['52w_high']}\n"
            f"  52-Week Low:    {d['52w_low']}"
        )
    except Exception as e:
        return f"Error fetching fundamental data for {ticker}: {e}"


@tool
def tool_plot_stock_chart(ticker: str, period: str = "3mo") -> str:
    """Generate and save a stock price + volume chart as a PNG file.

    Args:
        ticker: Stock ticker (e.g. '2330.TW')
        period: Time period — '1mo', '3mo', '6mo', '1y'
    """
    try:
        df = get_stock_history(ticker, period)
        path = plot_stock_chart(ticker, df)
        return f"Chart saved to: {path}"
    except Exception as e:
        return f"Error generating chart for {ticker}: {e}"
