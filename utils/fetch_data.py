import yfinance as yf
import pandas as pd
import requests
from pathlib import Path
from utils.cleaners import clean_stock_data
from utils.config import stock_tickers, crypto_ids


def fetch_stock_data(tickers, start="2020-01-01", end="2025-06-30"):
    data_dir = Path("data/stocks")
    data_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = yf.download(ticker, start=start, end=end)
        if not df.empty:
            df_cleaned = clean_stock_data(df, ticker)
            df_cleaned.to_csv(data_dir / f"{ticker}.csv", index=False)
        else:
            print(f"[Warning] No data found for {ticker}")


def get_crypto_ohlcv(coin_id, name=None, days=365):
    """Fetch OHLCV crypto data using CoinGecko"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    res = requests.get(url)

    if res.status_code != 200 or "error" in res.text.lower():
        raise Exception(f"[ERROR] Failed to fetch {coin_id}: {res.text}")
    
    data = res.json()

    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
    volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])

    df = prices.merge(volumes, on='timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    df = df.resample('1D').agg({'close': 'ohlc', 'volume': 'mean'})
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.dropna(inplace=True)
    df = df.sort_index()
    df.reset_index(inplace=True)

    coin_name = name or coin_id.upper()
    output_path = Path(f"data/crypto/{coin_name}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df['Ticker'] = coin_name
    df.to_csv(output_path, index=False)
    print(f"[SAVED] {coin_name} crypto data to {output_path}")


def fetch_crypto_data(crypto_ids, days=365):
    for name, coin_id in crypto_ids.items():
        try:
            get_crypto_ohlcv(coin_id=coin_id, name=name, days=days)
        except Exception as e:
            print(f"[ERROR] Failed to fetch {name} ({coin_id}): {e}")


def ensure_stock_data(tickers):
    for ticker in tickers:
        path = Path(f"data/stocks/{ticker}.csv")
        if not path.exists():
            print(f"[DATA] {ticker} not found. Downloading stock data...")
            fetch_stock_data([ticker])


def ensure_crypto_data(crypto_ids_subset):
    for name in crypto_ids_subset:
        path = Path(f"data/crypto/{name}.csv")
        if not path.exists():
            print(f"[DATA] {name} not found. Downloading crypto data...")
            fetch_crypto_data({name: crypto_ids_subset[name]})
