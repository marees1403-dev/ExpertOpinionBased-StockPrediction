"""
expert_identification/expert_signal_generator.py
--------------------------------------------------
Orchestrates the full expert identification pipeline and exposes
a clean generate_signals() function used by downstream modules.

Pipeline:
  1. Load data
  2. Bot filter
  3. Compute per-stock accuracy & classify expert/inverse/noisy
  4. Attach signal type and direction-weighted signal strength
  5. Return final signal DataFrame
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from load_data       import load_expert_signals, load_sector_map, DATA_DIR
from bot_filter      import filter_bots
from compute_accuracy import (compute_per_stock_accuracy,
                               compute_horizon_accuracy,
                               get_yearly_accuracy)


def generate_signals(data_dir: str = DATA_DIR,
                     save_path: str = None) -> pd.DataFrame:
    """
    Full expert identification pipeline.

    Returns a DataFrame with one row per (stock, date) signal containing:
        stock, date, pseudo_gt, gt_sentiment, is_correct,
        signal_type, signal_strength, expert_signal
    """
    print("=" * 60)
    print("Expert Identification Pipeline")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/4] Loading data …")
    df = load_expert_signals(data_dir)
    print(f"      {len(df):,} rows | {df['stock'].nunique():,} stocks")

    # Step 2: Bot filter
    print("\n[2/4] Applying bot/spammer filter …")
    df = filter_bots(df)

    # Step 3: Compute accuracy & classify
    print("\n[3/4] Computing accuracy & classifying experts …")
    acc_df = compute_per_stock_accuracy(df)
    type_counts = acc_df["signal_type"].value_counts()
    for t, c in type_counts.items():
        print(f"      {t:20s}: {c:4d} stocks")

    # Merge signal_type back onto the signal DataFrame
    df = df.merge(
        acc_df[["stock", "signal_type", "accuracy", "A_recent", "A_long"]],
        on="stock", how="left"
    )
    df["signal_type"] = df["signal_type"].fillna("noisy")

    # Step 4: Signal strength (30-day rolling)
    print("\n[4/4] Computing rolling signal strength …")
    df = _add_signal_strength(df)

    overall_acc = df["is_correct"].mean() * 100
    print(f"\n  Overall expert accuracy : {overall_acc:.2f}%")
    print(f"  (Paper reports ~72.8% at T+1)")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n  Saved → {save_path}")

    return df


def _add_signal_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a continuous expert_signal in [-1, +1] for each row using a
    30-day rolling window of past correctness.

      signal_strength = (2 * n_correct - n_total) / n_total  in past 30 days
      expert_signal   = direction * |signal_strength|
         where direction = +1 if pseudo_gt=Bullish, -1 if Bearish
         (inverted for inverse_experts)
    """
    WINDOW_DAYS = 30
    records = []

    for stock, group in df.groupby("stock"):
        group = group.sort_values("date").reset_index(drop=True)
        dates   = group["date"].values
        correct = group["is_correct"].values
        sents   = group["pseudo_gt"].values
        stypes  = group["signal_type"].values

        for i in range(len(group)):
            cutoff     = pd.Timestamp(dates[i]) - pd.Timedelta(days=WINDOW_DAYS)
            past_mask  = [j < i and pd.Timestamp(dates[j]) >= cutoff
                          for j in range(len(group))]
            past_correct = [correct[j] for j in range(len(group)) if past_mask[j]]

            if len(past_correct) > 0:
                n_c = sum(past_correct)
                n_t = len(past_correct)
                strength = (2 * n_c - n_t) / n_t
            else:
                strength = 0.0

            direction  = 1.0 if sents[i] == "Bullish" else -1.0
            stype      = stypes[i]

            # Inverse experts: flip the direction
            if stype == "inverse_expert":
                direction *= -1.0

            expert_signal = direction * abs(strength)

            records.append({
                "stock"          : stock,
                "date"           : dates[i],
                "pseudo_gt"      : sents[i],
                "gt_sentiment"   : group["gt_sentiment"].values[i],
                "is_correct"     : correct[i],
                "signal_type"    : stype,
                "signal_strength": round(strength, 4),
                "expert_signal"  : round(expert_signal, 4),
                "accuracy"       : group["accuracy"].values[i]
                                   if "accuracy" in group.columns else None,
            })

    return pd.DataFrame(records)


def get_signal_for_stock(ticker: str,
                          data_dir: str = DATA_DIR) -> dict:
    """
    Return the latest expert signal for a single stock ticker.
    Used by the backend API.
    """
    df = generate_signals(data_dir)
    stock_df = df[df["stock"] == ticker.upper()]

    if stock_df.empty:
        return {"error": f"No signals found for {ticker}"}

    latest = stock_df.sort_values("date").iloc[-1]
    return {
        "stock"          : ticker.upper(),
        "latest_date"    : str(latest["date"].date()),
        "pseudo_gt"      : latest["pseudo_gt"],
        "gt_sentiment"   : latest["gt_sentiment"],
        "is_correct"     : bool(latest["is_correct"]),
        "signal_type"    : latest["signal_type"],
        "signal_strength": float(latest["signal_strength"]),
        "expert_signal"  : float(latest["expert_signal"]),
        "n_signals"      : len(stock_df),
        "accuracy"       : round(stock_df["is_correct"].mean() * 100, 2),
    }


if __name__ == "__main__":
    signals = generate_signals()
    print("\nSample output:")
    print(signals[["stock","date","pseudo_gt","signal_type",
                    "signal_strength","expert_signal"]].head(10).to_string(index=False))
