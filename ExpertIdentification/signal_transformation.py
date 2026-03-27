"""
STEP 5: Signal Transformation — Binary → Continuous Return Ratio
=================================================================
Paper Section V-A  (Trend Signals Transformation)

The expert's prediction is binary: bullish (rise) or bearish (fall).
But the downstream model (DualGAT) needs a CONTINUOUS return ratio as input.

Transformation rule (from paper):
  • For stocks predicted to RISE:
      expert_signal = average of POSITIVE daily returns over the past 30 days
  • For stocks predicted to FALL:
      expert_signal = average of NEGATIVE daily returns over the past 30 days
  • For stocks with NO expert prediction on day d:
      expert_signal = 0   (paper's baseline)

This gives each expert signal a magnitude, not just a direction.

Output: a full (symbol × trade_date) matrix of expert signal values,
        with 0 where no expert was available.
"""

import pandas as pd
import numpy as np
import os

SIGNALS_PATH = "output/expert_signals_raw.parquet"
PRICES_PATH  = "output/prices_clean.parquet"
OUTPUT_DIR   = "output"

WINDOW = 30    # 30-day historical return window (as stated in paper)


# ══════════════════════════════════════════════════════════════════════════════
# 5-A  LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5: Signal Transformation (Binary → Return Ratio)")
print("=" * 60)

signals = pd.read_parquet(SIGNALS_PATH)
prices  = pd.read_parquet(PRICES_PATH)

# Make sure prices are sorted
prices = prices.sort_values(["symbol", "price_date"]).reset_index(drop=True)
print(f"  Expert signals : {len(signals):,}")
print(f"  Price rows     : {len(prices):,}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 5-B  PRE-COMPUTE 30-DAY AVG POSITIVE / NEGATIVE RETURNS PER STOCK-DAY
# ══════════════════════════════════════════════════════════════════════════════
# For each (symbol, date), we look back 30 trading days and compute:
#   avg_up_return   = mean(return_ratio) where return_ratio > 0
#   avg_down_return = mean(return_ratio) where return_ratio < 0

print("  Pre-computing 30-day rolling up/down averages …")

def rolling_avg_returns(group, window=WINDOW):
    group = group.sort_values("price_date").copy()
    r = group["return_ratio"].values
    avg_up   = []
    avg_down = []

    for i in range(len(r)):
        start = max(0, i - window)
        window_returns = r[start:i]    # past 'window' days (not including today)
        ups   = window_returns[window_returns > 0]
        downs = window_returns[window_returns < 0]
        avg_up.append(  ups.mean()   if len(ups)   > 0 else 0.0)
        avg_down.append(downs.mean() if len(downs) > 0 else 0.0)

    group["avg_up_return"]   = avg_up
    group["avg_down_return"] = avg_down
    return group

prices_with_avgs = (
    prices
    .groupby("symbol", group_keys=False)
    .apply(rolling_avg_returns)
)

# Build a fast lookup
up_lookup   = dict(zip(
    zip(prices_with_avgs["symbol"], prices_with_avgs["price_date"]),
    prices_with_avgs["avg_up_return"]
))
down_lookup = dict(zip(
    zip(prices_with_avgs["symbol"], prices_with_avgs["price_date"]),
    prices_with_avgs["avg_down_return"]
))

print("  Done.")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 5-C  RESOLVE MULTIPLE EXPERTS PER STOCK-DAY → ONE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
# When multiple experts post about the same stock on the same day,
# randomly sample ONE (as the paper does).

signals_deduped = (
    signals
    .groupby(["symbol", "trade_date"], group_keys=False)
    .apply(lambda g: g.sample(n=1, random_state=42))
    .reset_index(drop=True)
)
print(f"  Unique stock-day expert signals: {len(signals_deduped):,}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 5-D  COMPUTE EXPERT SIGNAL VALUE
# ══════════════════════════════════════════════════════════════════════════════
expert_signal_values = []

for row in signals_deduped.itertuples(index=False):
    symbol     = row.symbol
    trade_date = row.trade_date
    signal     = row.signal    # "bullish" (predict rise) or "bearish" (predict fall)

    if signal == "bullish":
        # Use the 30-day average of UP days as the expected return magnitude
        val = up_lookup.get((symbol, trade_date), 0.0)
    else:
        # Use the 30-day average of DOWN days (this is a negative number)
        val = down_lookup.get((symbol, trade_date), 0.0)

    expert_signal_values.append({
        "symbol"        : symbol,
        "trade_date"    : trade_date,
        "signal_dir"    : signal,
        "expert_signal" : val,     # continuous return ratio
        "has_expert"    : 1,
    })

expert_signal_df = pd.DataFrame(expert_signal_values)
print(f"  Sample of transformed signals:")
print(expert_signal_df.head(10).to_string(index=False))
print()


# ══════════════════════════════════════════════════════════════════════════════
# 5-E  BUILD FULL SIGNAL MATRIX  (all stock-days, fill missing with 0)
# ══════════════════════════════════════════════════════════════════════════════
# Get all stock-day combinations that appear in prices
all_stock_days = prices[["symbol", "price_date"]].rename(
    columns={"price_date": "trade_date"}
)

full_matrix = all_stock_days.merge(
    expert_signal_df[["symbol", "trade_date", "expert_signal", "has_expert"]],
    on=["symbol", "trade_date"],
    how="left"
)
full_matrix["expert_signal"] = full_matrix["expert_signal"].fillna(0.0)
full_matrix["has_expert"]    = full_matrix["has_expert"].fillna(0).astype(int)

total    = len(full_matrix)
covered  = full_matrix["has_expert"].sum()
coverage = covered / total * 100

print(f"  Full signal matrix: {total:,} stock-day pairs")
print(f"  With expert signal: {covered:,}  ({coverage:.2f}%)")
print(f"  Without (zero)    : {total-covered:,}  ({100-coverage:.2f}%)")
print(f"  → Paper reports ~4% coverage before graph propagation")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 5-F  SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_signals = os.path.join(OUTPUT_DIR, "expert_signals_continuous.parquet")
expert_signal_df.to_parquet(out_signals, index=False)

out_matrix = os.path.join(OUTPUT_DIR, "full_signal_matrix.parquet")
full_matrix.to_parquet(out_matrix, index=False)

print(f"  ✅ Saved continuous expert signals → {out_signals}")
print(f"  ✅ Saved full signal matrix (all stock-days) → {out_matrix}")
print()
print("STEP 5 complete.")
print()
print("══════════════════════════════════════════════════════════════")
print("  EXPERT IDENTIFICATION PIPELINE COMPLETE!")
print("  Next stage in the paper: MS-LSTM pre-training + DualGAT")
print("══════════════════════════════════════════════════════════════")
