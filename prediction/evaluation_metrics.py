"""
prediction/evaluation_metrics.py
-----------------------------------
Computes all quantitative metrics from the paper + standard finance metrics.

Metrics:
  IC        - Information Coefficient (Pearson corr of predictions vs actuals)
  ICIR      - IC / std(IC)  — risk-adjusted IC
  RIC       - Rank IC (Spearman correlation)
  ACC       - Directional accuracy
  AR        - Annualised Return (long top-10%, short bottom-10%)
  SR        - Sharpe Ratio
  Volatility- Std dev of daily returns
  Max DD    - Maximum Drawdown
  Cumulative- Cumulative returns over time
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Information Coefficient = Pearson correlation."""
    if len(y_pred) < 2:
        return 0.0
    corr, _ = stats.pearsonr(y_pred, y_true)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_ric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Rank IC = Spearman rank correlation."""
    if len(y_pred) < 2:
        return 0.0
    corr, _ = stats.spearmanr(y_pred, y_true)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_icir(ic_series: np.ndarray) -> float:
    """ICIR = mean(IC) / std(IC) — signal quality adjusted for volatility."""
    if len(ic_series) < 2 or np.std(ic_series) == 0:
        return 0.0
    return float(np.mean(ic_series) / np.std(ic_series))


def compute_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Directional accuracy: fraction of correct up/down predictions."""
    pred_dir = np.sign(y_pred)
    true_dir = np.sign(y_true)
    return float((pred_dir == true_dir).mean())


def compute_sharpe_ratio(returns: np.ndarray,
                          risk_free: float = 0.0,
                          annualise: bool  = True) -> float:
    """Sharpe ratio = (mean_return - rf) / std_return."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    sr = (np.mean(returns) - risk_free) / np.std(returns)
    if annualise:
        sr *= np.sqrt(252)
    return float(sr)


def compute_annualised_return(returns: np.ndarray,
                               trading_cost_bps: float = 4) -> float:
    """
    Annualised return of a long-short strategy.
    Paper: long top-10%, short bottom-10%, 4bps trading cost.
    """
    tc   = trading_cost_bps / 10_000
    net  = returns - tc
    cum  = np.prod(1 + net) - 1
    years = len(returns) / 252
    if years == 0:
        return 0.0
    ar = (1 + cum) ** (1 / years) - 1
    return float(ar * 100)


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum drawdown from peak to trough."""
    if len(cumulative_returns) == 0:
        return 0.0
    peak  = np.maximum.accumulate(cumulative_returns)
    dd    = (cumulative_returns - peak) / (peak + 1e-8)
    return float(dd.min())


def compute_cumulative_returns(returns: np.ndarray) -> np.ndarray:
    """Compound cumulative returns."""
    return np.cumprod(1 + returns) - 1


def compute_volatility(returns: np.ndarray, annualise: bool = True) -> float:
    """Annualised volatility of daily returns."""
    if len(returns) < 2:
        return 0.0
    vol = float(np.std(returns))
    if annualise:
        vol *= np.sqrt(252)
    return vol


def compute_horizon_accuracy(signal_df: pd.DataFrame,
                              horizons: list = [1, 3, 7]) -> dict:
    """Accuracy at T+1, T+3, T+7 horizons."""
    results = {}
    for h in horizons:
        shifted = []
        for stock, group in signal_df.groupby("stock"):
            g = group.sort_values("date").reset_index(drop=True)
            shifted.extend(
                g["is_correct"].shift(-h).fillna(g["is_correct"]).tolist()
            )
        results[f"T+{h}"] = round(float(np.mean(shifted)) * 100, 2)
    return results


def full_metrics_for_stock(ticker: str,
                             signal_df: pd.DataFrame,
                             predicted_return: float = None) -> dict:
    """
    Compute all metrics for a given stock ticker.

    Args:
        ticker          : stock symbol
        signal_df       : full signal DataFrame (all stocks)
        predicted_return: optional DualGAT prediction

    Returns:
        dict of all metrics
    """
    stock_df = signal_df[signal_df["stock"] == ticker].sort_values("date")

    if stock_df.empty:
        return {"error": f"No data for {ticker}"}

    # Use expert_signal as proxy for returns (when no raw price data)
    returns  = stock_df["expert_signal"].values.astype(float)
    correct  = stock_df["is_correct"].values.astype(float)

    # Simulate predicted vs actual
    y_pred   = returns
    y_true   = np.where(correct == 1, np.abs(returns), -np.abs(returns))

    cum_ret  = compute_cumulative_returns(returns)
    sharpe   = compute_sharpe_ratio(returns)
    ann_ret  = compute_annualised_return(returns)
    vol      = compute_volatility(returns)
    max_dd   = compute_max_drawdown(cum_ret + 1)
    acc      = round(float(stock_df["is_correct"].mean() * 100), 2)
    ic       = compute_ic(y_pred, y_true)
    ric      = compute_ric(y_pred, y_true)

    # IC series (rolling 20-day)
    ic_series = []
    for i in range(20, len(y_pred)):
        ic_series.append(compute_ic(y_pred[i-20:i], y_true[i-20:i]))
    icir = compute_icir(np.array(ic_series)) if ic_series else 0.0

    # Horizon accuracy
    horizon_acc = compute_horizon_accuracy(
        signal_df[signal_df["stock"] == ticker]
    )

    # Cumulative return history for chart
    cum_hist = [round(float(c * 100), 4) for c in cum_ret]
    date_hist = [str(d)[:10] for d in stock_df["date"].values]

    return {
        "ticker"            : ticker,
        "accuracy_pct"      : acc,
        "horizon_accuracy"  : horizon_acc,
        "ic"                : round(float(ic),   4),
        "ric"               : round(float(ric),  4),
        "icir"              : round(float(icir), 4),
        "sharpe_ratio"      : round(float(sharpe), 4),
        "annualised_return" : round(float(ann_ret), 4),
        "volatility"        : round(float(vol),   4),
        "max_drawdown"      : round(float(max_dd), 4),
        "predicted_return"  : round(float(predicted_return or 0.0), 4),
        "n_signals"         : int(len(stock_df)),
        "cumulative_returns": {
            "dates"  : date_hist,
            "values" : cum_hist,
        },
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from expert_identification.expert_signal_generator import generate_signals
    df = generate_signals()
    m  = full_metrics_for_stock("AAPL", df, predicted_return=0.012)
    import json
    print(json.dumps(m, indent=2))
