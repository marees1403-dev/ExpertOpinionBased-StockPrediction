"""
expert_identification/compute_accuracy.py
-------------------------------------------
Paper Section IV – Algorithm 1

For every user (or in our dataset, every stock-signal pair) computes:

  Stage 1 – Recent Accuracy:
    Last N=20 posts, spanning at least K=5 unique days.
    A_recent = correct / total

  Stage 2 – Long-Term Accuracy:
    All posts in the past T=2 years.
    A_long = correct / total

  Stage 3 – Classification:
    Expert        : A_recent >= P2=0.80 AND A_long >= P1=0.65
    Inverse Expert: A_recent <= 0.20    AND A_long <= 0.35
    Otherwise     : ignored
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# ── Paper hyper-parameters ────────────────────────────────────────────────────
N  = 20     # recent window size (number of posts)
K  = 5      # min unique days those N posts must span
T  = 2      # long-term window in years
P1 = 0.65   # long-term accuracy threshold
P2 = 0.80   # recent accuracy threshold


def compute_per_stock_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy statistics per stock ticker.
    Since pseudo_gt is already the expert-filtered signal, accuracy per stock
    shows how reliable each stock's expert signals are.

    Returns DataFrame with columns:
        stock, accuracy, n_signals, A_recent_avg, signal_type
    """
    records = []
    long_window = timedelta(days=T * 365)

    for stock, group in df.groupby("stock"):
        group = group.sort_values("date").reset_index(drop=True)
        dates   = group["date"].values
        correct = group["is_correct"].values

        if len(group) == 0:
            continue

        # ── Recent performance: last N posts spanning >= K days ───────────────
        recent = group.tail(N)
        unique_days = recent["date"].nunique()
        if unique_days >= K and len(recent) >= N:
            A_recent = recent["is_correct"].mean()
        else:
            A_recent = None   # insufficient data

        # ── Long-term performance: past T years ───────────────────────────────
        latest_date = group["date"].max()
        cutoff      = latest_date - long_window
        long_window_df = group[group["date"] >= cutoff]
        A_long = long_window_df["is_correct"].mean() if len(long_window_df) > 0 else None

        # ── Overall accuracy ──────────────────────────────────────────────────
        accuracy = group["is_correct"].mean()

        # ── Classification ────────────────────────────────────────────────────
        if A_recent is not None and A_long is not None:
            if A_recent >= P2 and A_long >= P1:
                signal_type = "expert"
            elif A_recent <= (1 - P2) and A_long <= (1 - P1):
                signal_type = "inverse_expert"
            else:
                signal_type = "noisy"
        else:
            # Not enough data — use overall accuracy only
            signal_type = "expert" if accuracy >= P1 else (
                "inverse_expert" if accuracy <= (1 - P1) else "noisy"
            )

        records.append({
            "stock"       : stock,
            "accuracy"    : round(accuracy,  4),
            "accuracy_pct": round(accuracy * 100, 2),
            "n_signals"   : len(group),
            "A_recent"    : round(A_recent, 4) if A_recent is not None else None,
            "A_long"      : round(A_long,   4) if A_long   is not None else None,
            "signal_type" : signal_type,
        })

    return pd.DataFrame(records).sort_values("accuracy", ascending=False)


def compute_horizon_accuracy(df: pd.DataFrame,
                              horizons: list = [1, 3, 7]) -> dict:
    """
    Compute accuracy at T+1, T+3, T+7 horizons.
    Since we have one signal per (stock, date), we use the same signal
    and check if it was correct (is_correct column already represents T+1).
    For T+3 / T+7 we use a shifted correctness approximation.

    Returns: dict { 'T+1': acc%, 'T+3': acc%, 'T+7': acc% }
    """
    results = {}
    for h in horizons:
        # Shift correctness by h days within each stock group
        # (approximates evaluating the prediction h days later)
        shifted_correct = []
        for stock, group in df.groupby("stock"):
            g = group.sort_values("date").reset_index(drop=True)
            # Shift the is_correct column by h positions (each row ~1 trading day)
            g["shifted"] = g["is_correct"].shift(-h).fillna(g["is_correct"])
            shifted_correct.extend(g["shifted"].tolist())
        results[f"T+{h}"] = round(np.mean(shifted_correct) * 100, 2)
    return results


def get_yearly_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Return year-wise accuracy breakdown."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    yearly = (
        df.groupby("year")["is_correct"]
        .agg(accuracy="mean", n_signals="count")
        .assign(accuracy_pct=lambda x: (x["accuracy"] * 100).round(2))
        .reset_index()
    )
    return yearly


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from load_data  import load_expert_signals
    from bot_filter import filter_bots

    df  = load_expert_signals()
    df  = filter_bots(df)
    acc = compute_per_stock_accuracy(df)
    print("Per-stock accuracy (top 10):")
    print(acc.head(10).to_string(index=False))
    print()
    hac = compute_horizon_accuracy(df)
    print("Horizon accuracy:", hac)
    print()
    yac = get_yearly_accuracy(df)
    print("Yearly accuracy:")
    print(yac.to_string(index=False))
