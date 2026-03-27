"""
STEP 1: Load and Preprocess Data
=================================
Paper Section III (Problem Formulation) + Section IV (Expert User Identification)

What this step does:
  - Load StockTwits posts CSV
  - Load stock price CSV
  - Parse dates, clean columns
  - Compute next-day return ratio for every stock-day pair
  - Save merged dataset for use in Step 2

Expected input files (from the DualGAT GitHub repo):
  - data/stocktwits_posts.csv  (social media posts)
  - data/stock_prices.csv      (OHLCV price data)

Expected columns in stocktwits_posts.csv:
  user_id, stock_symbol, created_at, sentiment   (bullish / bearish)

Expected columns in stock_prices.csv:
  symbol, date, open, high, low, close, volume
"""

import pandas as pd
import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────────────────────
# Change these to match where YOUR downloaded files actually live
POSTS_CSV   = "data/stocktwits_posts.csv"
PRICES_CSV  = "data/stock_prices.csv"
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1-A  LOAD STOCKTWITS POSTS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1-A: Loading StockTwits posts …")
print("=" * 60)

posts = pd.read_csv(POSTS_CSV)

# Rename columns to standard names if needed
# Adjust the dict below to match your actual column names
posts = posts.rename(columns={
    "user_id":      "user_id",
    "stock_symbol": "symbol",       # ticker, e.g. "AAPL"
    "created_at":   "post_time",    # datetime string
    "sentiment":    "sentiment",    # "bullish" or "bearish"
})

# Keep only the columns we need
posts = posts[["user_id", "symbol", "post_time", "sentiment"]].copy()

# Parse the timestamp
posts["post_time"] = pd.to_datetime(posts["post_time"], utc=True, errors="coerce")
posts = posts.dropna(subset=["post_time"])          # drop rows with bad timestamps

# Extract calendar date (no time) – used later to join with price data
posts["post_date"] = posts["post_time"].dt.date
posts["post_date"] = pd.to_datetime(posts["post_date"])

# Normalise sentiment to lowercase
posts["sentiment"] = posts["sentiment"].str.lower().str.strip()

# Keep only known sentiment labels
posts = posts[posts["sentiment"].isin(["bullish", "bearish"])]

print(f"  Loaded {len(posts):,} posts  |  {posts['user_id'].nunique():,} unique users"
      f"  |  {posts['symbol'].nunique():,} unique stocks")
print(f"  Date range: {posts['post_date'].min()} → {posts['post_date'].max()}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 1-B  LOAD STOCK PRICES  &  COMPUTE NEXT-DAY RETURN RATIO
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1-B: Loading stock prices and computing return ratios …")
print("=" * 60)

prices = pd.read_csv(PRICES_CSV)

prices = prices.rename(columns={
    "symbol": "symbol",
    "date":   "price_date",
    "close":  "close",
})

prices["price_date"] = pd.to_datetime(prices["price_date"])
prices = prices.sort_values(["symbol", "price_date"]).reset_index(drop=True)

# ── Return ratio formula from paper (Equation 1) ──────────────────────────────
#   r_{u, t+1} = (close_{t+1} − close_t) / close_t
#
# We add the NEXT day's close as a new column and compute the ratio.
prices["close_next"] = prices.groupby("symbol")["close"].shift(-1)
prices["return_ratio"] = (prices["close_next"] - prices["close"]) / prices["close"]

# "rise" = positive return, "fall" = negative or zero return
prices["actual_direction"] = np.where(prices["return_ratio"] > 0, "rise", "fall")

# Drop the last row of each stock (no next-day price available)
prices = prices.dropna(subset=["return_ratio"])

print(f"  Loaded {len(prices):,} price rows  |  {prices['symbol'].nunique():,} stocks")
print(f"  Date range: {prices['price_date'].min()} → {prices['price_date'].max()}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 1-C  ATTACH NEXT-DAY OUTCOME TO EACH POST
# ══════════════════════════════════════════════════════════════════════════════
# Each post is made on day t.
# The paper evaluates whether the post's sentiment matches the stock's
# direction on day t+1.
# So we join each post (post_date = t) with the price row where price_date = t
# to get the actual_direction for t+1.

print("=" * 60)
print("STEP 1-C: Joining posts with price outcomes …")
print("=" * 60)

price_lookup = prices[["symbol", "price_date", "return_ratio",
                        "actual_direction", "close"]].copy()

merged = posts.merge(
    price_lookup,
    left_on=["symbol", "post_date"],
    right_on=["symbol", "price_date"],
    how="inner"
)

# Was the post's sentiment correct?
#   • bullish post → correct if stock rose next day
#   • bearish post → correct if stock fell next day
merged["is_correct"] = (
    ((merged["sentiment"] == "bullish") & (merged["actual_direction"] == "rise")) |
    ((merged["sentiment"] == "bearish") & (merged["actual_direction"] == "fall"))
)

print(f"  Merged dataset: {len(merged):,} post-price pairs")
print(f"  Overall naive accuracy: "
      f"{merged['is_correct'].mean() * 100:.2f}%  "
      f"(paper reports ~47.6% for naive aggregation)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 1-D  SAVE
# ══════════════════════════════════════════════════════════════════════════════
merged_path = os.path.join(OUTPUT_DIR, "posts_with_outcomes.parquet")
merged.to_parquet(merged_path, index=False)
print(f"  ✅ Saved merged dataset → {merged_path}")

prices_path = os.path.join(OUTPUT_DIR, "prices_clean.parquet")
prices.to_parquet(prices_path, index=False)
print(f"  ✅ Saved clean prices   → {prices_path}")

print()
print("STEP 1 complete. Run step2_bot_filter.py next.")
