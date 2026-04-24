"""
expert_identification/load_data.py
------------------------------------
Loads and preprocesses all dataset files:
  - psudo_combine_all.csv  (expert signals, all stocks)
  - psudo_sp500.csv        (S&P 500 subset)
  - psudo_stocktable.csv   (StockNet subset)
  - NASDAQ100.csv          (sector labels)
  - industry_sp500.csv     (S&P 500 sector labels)
  - stocktable_new.csv     (additional sector labels)

Columns in pseudo files:
  stock        : ticker symbol
  stock_time   : date of the expert post
  gt_sentiment : ACTUAL price direction  (Bullish = rose, Bearish = fell)
  pseudo_gt    : EXPERT's predicted sentiment
"""

import os
import pandas as pd
import numpy as np

# ── Default data directory ─────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_expert_signals(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load and merge all three pseudo-signal CSV files."""
    files = {
        "combine": "psudo_combine_all.csv",
        "sp500":   "psudo_sp500.csv",
        "stock":   "psudo_stocktable.csv",
    }
    frames = []
    for source, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [WARN] {fname} not found, skipping.")
            continue
        df = pd.read_csv(path)
        df["source"] = source
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No expert signal CSV files found in data/")

    combined = pd.concat(frames, ignore_index=True)
    combined = _clean_signals(combined)
    return combined


def _clean_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, normalise sentiment labels, compute is_correct."""
    df = df.copy()
    df["stock_time"]    = pd.to_datetime(df["stock_time"], errors="coerce")
    df["date"]          = pd.to_datetime(df["stock_time"].dt.date)
    df["gt_sentiment"]  = df["gt_sentiment"].str.strip().str.capitalize()
    df["pseudo_gt"]     = df["pseudo_gt"].str.strip().str.capitalize()
    df                  = df.dropna(subset=["stock_time", "gt_sentiment", "pseudo_gt"])

    # is_correct: expert prediction matched actual direction
    df["is_correct"] = (df["gt_sentiment"] == df["pseudo_gt"]).astype(int)

    # Remove exact duplicates
    df = df.drop_duplicates(subset=["stock", "date", "pseudo_gt"])
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    return df


def load_sector_map(data_dir: str = DATA_DIR) -> dict:
    """Build a ticker → sector dictionary from all three sector files."""
    sector_map = {}

    # NASDAQ100
    path = os.path.join(data_dir, "NASDAQ100.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            sector_map[str(row["Ticker"]).strip()] = str(row["GICS Sector"]).strip()

    # industry_sp500
    path = os.path.join(data_dir, "industry_sp500.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["Symbol"]        = df["Symbol"].astype(str).str.strip()
        df["INDUSTRY_GICS"] = df["INDUSTRY_GICS"].astype(str).str.strip()
        for _, row in df.iterrows():
            sector_map[row["Symbol"]] = row["INDUSTRY_GICS"]

    # stocktable_new
    path = os.path.join(data_dir, "stocktable_new.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            sector_map[str(row["Symbol"]).strip()] = str(row["Sector"]).strip()

    return sector_map


def get_available_stocks(data_dir: str = DATA_DIR) -> list:
    """Return sorted list of all unique stock tickers in the dataset."""
    df = load_expert_signals(data_dir)
    return sorted(df["stock"].unique().tolist())


def get_stock_signals(ticker: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Return all expert signal rows for a given ticker."""
    df = load_expert_signals(data_dir)
    result = df[df["stock"] == ticker.upper()].copy()
    if result.empty:
        raise ValueError(f"No signals found for ticker '{ticker}'")
    return result


if __name__ == "__main__":
    print("Loading expert signals …")
    df = load_expert_signals()
    print(f"  Total rows   : {len(df):,}")
    print(f"  Unique stocks: {df['stock'].nunique():,}")
    print(f"  Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Accuracy     : {df['is_correct'].mean()*100:.2f}%")
    print()
    sm = load_sector_map()
    print(f"  Sector map   : {len(sm)} tickers with known sectors")
