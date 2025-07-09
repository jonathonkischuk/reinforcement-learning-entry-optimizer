import pandas as pd

def clean_stock_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()

    # Flatten multi-index columns (in case of 'Adj Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    rename_map = {
        "Adj Close": "Close",
        "Close": "Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Volume": "Volume"
    }

    # Rename only relevant columns
    df = df.rename(columns=rename_map)

    keep_cols = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df = df[[col for col in keep_cols if col in df.columns]]
    df["Ticker"] = ticker

    # Drop rows where Date is null (e.g., accidental extra headers)
    df = df[df["Date"].notnull()]
    df = df.dropna()
    return df

