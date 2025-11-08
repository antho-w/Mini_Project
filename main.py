import yfinance as yf
import pandas as pd
import datetime as dt
import sys, os

def get_price_data(symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV price data for a given stock symbol using yfinance.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        period: Data period (e.g. "1d", "5d", "1mo", "1y").
        interval: Data interval (e.g. "1m", "1h", "1d").

    Returns:
        pandas.DataFrame with datetime index and columns: Open, High, Low, Close, Volume, etc.

    Raises:
        ValueError: if no data is returned for the given symbol.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, actions=False)
    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return df

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\antho\OneDrive\Documents\Anthony\Uni\7-3\MATH5816\Mini_Project"    

    TICKER = "AAPL"
    try:
        df = get_price_data(TICKER, period="5d", interval="1d")
        
        print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}: {df.shape[0]} row(s) fetched for symbol {TICKER}")
        df.to_csv(os.path.join(DATA_DIR, f"{TICKER}_stock_data.csv"), index = True)
    except Exception as e:
        print("Error:", e)

