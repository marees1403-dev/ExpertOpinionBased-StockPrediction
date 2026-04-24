"""
expert_identification/bot_filter.py
--------------------------------------
Paper Section IV, Point 1:
  "Filtering spammers: for each user-stock pair, we only retain
   the post closest to market closing time on any given day."

Bots post hundreds of times per day at fixed intervals.
Keeping only ONE post per (stock, date) eliminates their distorting
effect on accuracy calculations.
"""

import pandas as pd


def filter_bots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the latest post per (stock, date) pair.

    Args:
        df: DataFrame with columns [stock, date, stock_time, ...]

    Returns:
        Deduplicated DataFrame — one row per (stock, date).
    """
    before = len(df)

    # Sort so the latest post on each day comes first
    df_sorted = df.sort_values("stock_time", ascending=False)

    # Keep the first (= latest) post per stock-date
    df_clean = (
        df_sorted
        .drop_duplicates(subset=["stock", "date"], keep="first")
        .sort_values(["stock", "date"])
        .reset_index(drop=True)
    )

    after   = len(df_clean)
    removed = before - after
    print(f"  Bot filter: {before:,} → {after:,} rows  "
          f"({removed:,} duplicates removed)")
    return df_clean


def get_duplicate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame showing stocks/dates with most duplicates."""
    counts = (
        df.groupby(["stock", "date"])
        .size()
        .reset_index(name="n_posts")
    )
    return counts[counts["n_posts"] > 1].sort_values("n_posts", ascending=False)


if __name__ == "__main__":
    from load_data import load_expert_signals
    raw = load_expert_signals()
    clean = filter_bots(raw)
    dups  = get_duplicate_stats(raw)
    print(f"  Duplicate (stock, date) groups: {len(dups)}")
    if not dups.empty:
        print(dups.head())
