"""
expert_opinion_propagation/signal_transformation.py
-----------------------------------------------------
Paper Section V-A: Trend Signals Transformation

Converts binary expert signals (Bullish/Bearish) into continuous
return ratios using a 30-day rolling average:

  • Bullish signal → avg positive returns over past 30 days
  • Bearish signal → avg negative returns over past 30 days
  • No signal      → 0.0

Also builds the full (stock × date) signal matrix, filling zeros
for stock-days without any expert signal (~96% of all pairs).
"""

import pandas as pd
import numpy as np

WINDOW = 30   # 30-day return window (paper Section V-A)


def transform_signals(signal_df: pd.DataFrame,
                       price_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Transform binary expert signals to continuous return ratios.

    Args:
        signal_df : output of expert_signal_generator.generate_signals()
                    must have: stock, date, pseudo_gt, expert_signal
        price_df  : optional OHLCV DataFrame with columns:
                    stock, date, close, return_ratio
                    If None, uses expert_signal as the magnitude proxy.

    Returns:
        DataFrame with columns:
            stock, date, signal_dir, continuous_signal, has_expert
    """
    if price_df is not None:
        return _transform_with_prices(signal_df, price_df)
    else:
        return _transform_without_prices(signal_df)


def _transform_with_prices(signal_df: pd.DataFrame,
                             price_df: pd.DataFrame) -> pd.DataFrame:
    """Use actual 30-day average returns as signal magnitude."""
    price_df = price_df.copy().sort_values(["stock", "date"])

    # Pre-compute rolling avg positive and negative returns per stock-day
    records = []
    for stock, group in price_df.groupby("stock"):
        group = group.sort_values("date").reset_index(drop=True)
        returns = group["return_ratio"].values
        dates   = group["date"].values

        for i in range(len(group)):
            start = max(0, i - WINDOW)
            window_r = returns[start:i]
            avg_up   = window_r[window_r > 0].mean() if (window_r > 0).any() else 0.0
            avg_down = window_r[window_r < 0].mean() if (window_r < 0).any() else 0.0
            records.append({
                "stock"   : stock,
                "date"    : dates[i],
                "avg_up"  : avg_up,
                "avg_down": avg_down,
            })

    avg_df = pd.DataFrame(records)

    # Merge with signals
    merged = signal_df.merge(avg_df, on=["stock", "date"], how="left")
    merged["avg_up"]   = merged["avg_up"].fillna(0.0)
    merged["avg_down"] = merged["avg_down"].fillna(0.0)

    def pick_signal(row):
        if row["pseudo_gt"] == "Bullish":
            return row["avg_up"]
        else:
            return row["avg_down"]

    merged["continuous_signal"] = merged.apply(pick_signal, axis=1)
    merged["has_expert"]        = 1

    return merged[["stock", "date", "pseudo_gt", "continuous_signal",
                   "has_expert", "signal_type", "expert_signal"]]


def _transform_without_prices(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: use expert_signal (already in [-1,+1]) directly as the
    continuous signal. This is equivalent to using normalised return proxy.
    """
    df = signal_df.copy()
    df["continuous_signal"] = df["expert_signal"]
    df["has_expert"]        = 1
    return df[["stock", "date", "pseudo_gt", "continuous_signal",
               "has_expert", "signal_type", "expert_signal"]]


def build_full_signal_matrix(signal_df: pd.DataFrame,
                              all_stocks: list,
                              all_dates: list) -> pd.DataFrame:
    """
    Build the complete (stock × date) matrix.
    Fills 0 where no expert signal exists.

    Args:
        signal_df  : transformed signal DataFrame
        all_stocks : complete list of stock tickers
        all_dates  : complete list of trading dates

    Returns:
        DataFrame with columns: stock, date, continuous_signal, has_expert
    """
    # Create full grid
    grid = pd.MultiIndex.from_product(
        [all_stocks, all_dates], names=["stock", "date"]
    ).to_frame(index=False)

    sig = signal_df[["stock", "date", "continuous_signal", "has_expert"]].copy()

    full = grid.merge(sig, on=["stock", "date"], how="left")
    full["continuous_signal"] = full["continuous_signal"].fillna(0.0)
    full["has_expert"]        = full["has_expert"].fillna(0).astype(int)

    coverage = full["has_expert"].mean() * 100
    print(f"  Signal matrix: {len(full):,} stock-day pairs | "
          f"coverage: {coverage:.2f}%")
    return full


def compute_return_ratio(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute next-day return ratio from price data.
    r_{t+1} = (close_{t+1} - close_t) / close_t   (Paper Eq. 1)
    """
    price_df = price_df.copy().sort_values(["stock", "date"])
    price_df["close_next"]   = price_df.groupby("stock")["close"].shift(-1)
    price_df["return_ratio"] = (
        (price_df["close_next"] - price_df["close"]) / price_df["close"]
    )
    price_df["direction"] = np.where(price_df["return_ratio"] > 0, "Bullish", "Bearish")
    return price_df.dropna(subset=["return_ratio"])


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from expert_identification.expert_signal_generator import generate_signals
    sigs = generate_signals()
    transformed = transform_signals(sigs)
    print(f"Transformed {len(transformed):,} signals")
    print(transformed.head(10).to_string(index=False))
