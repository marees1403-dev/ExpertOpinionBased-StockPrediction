"""
backend/api.py
----------------
FastAPI backend exposing all endpoints for the frontend.

Endpoints:
  GET /stocks              — list all available tickers
  GET /predict?stock=AAPL  — full prediction for a ticker
  GET /metrics?stock=AAPL  — evaluation metrics for a ticker
  GET /system-usage        — CPU / memory / inference time
  GET /expert-signals?stock=AAPL — raw expert signals history
  GET /health              — health check

Run with:
  uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import psutil

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from prediction.predictor          import get_predictor
from prediction.evaluation_metrics import full_metrics_for_stock

app = FastAPI(
    title     = "Stock Prediction API — DualGAT",
    version   = "1.0.0",
    description = "Expert Opinion-based Stock Prediction using Dual Graph Attention Network",
)

# Allow frontend (React on port 3000) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── /health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "DualGAT v1.0"}


# ── /stocks ────────────────────────────────────────────────────────────────────
@app.get("/stocks")
def get_stocks():
    """Return list of all tickers available in the dataset."""
    predictor = get_predictor()
    return {
        "stocks": predictor.stocks,
        "count" : len(predictor.stocks),
    }


# ── /predict ───────────────────────────────────────────────────────────────────
@app.get("/predict")
def predict(stock: str = Query(..., description="Ticker symbol, e.g. AAPL")):
    """
    Run full prediction pipeline for a given stock.

    Returns:
      predicted_return, trend, confidence, expert_signal,
      signal_type, history, accuracy, etc.
    """
    t0        = time.time()
    predictor = get_predictor()

    result = predictor.predict(stock.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    result["inference_time_ms"] = round((time.time() - t0) * 1000, 2)
    return result


# ── /metrics ───────────────────────────────────────────────────────────────────
@app.get("/metrics")
def metrics(stock: str = Query(..., description="Ticker symbol, e.g. AAPL")):
    """
    Return evaluation metrics for a stock.

    Returns:
      IC, RIC, ICIR, Sharpe Ratio, Annualised Return,
      Volatility, Max Drawdown, Accuracy (T+1/T+3/T+7),
      Cumulative Returns history
    """
    predictor = get_predictor()
    ticker    = stock.upper()

    if ticker not in predictor.stocks:
        raise HTTPException(status_code=404,
                            detail=f"Ticker '{ticker}' not found.")

    pred_return = predictor.gat_preds.get(ticker, 0.0)
    result      = full_metrics_for_stock(ticker, predictor.signal_df,
                                          predicted_return=pred_return)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


# ── /expert-signals ────────────────────────────────────────────────────────────
@app.get("/expert-signals")
def expert_signals(stock: str = Query(..., description="Ticker symbol")):
    """Return full expert signal history for a stock."""
    predictor = get_predictor()
    ticker    = stock.upper()

    df = predictor.signal_df
    stock_df = df[df["stock"] == ticker].sort_values("date")

    if stock_df.empty:
        raise HTTPException(status_code=404,
                            detail=f"No signals for '{ticker}'.")

    records = stock_df[[
        "date", "pseudo_gt", "gt_sentiment", "is_correct",
        "expert_signal", "signal_strength", "signal_type"
    ]].copy()
    records["date"] = records["date"].astype(str)

    return {
        "ticker"  : ticker,
        "n"       : len(records),
        "accuracy": round(float(stock_df["is_correct"].mean() * 100), 2),
        "signals" : records.to_dict(orient="records"),
    }


# ── /system-usage ──────────────────────────────────────────────────────────────
@app.get("/system-usage")
def system_usage():
    """Return current system resource utilisation."""
    cpu_pct  = psutil.cpu_percent(interval=0.5)
    mem      = psutil.virtual_memory()
    disk     = psutil.disk_usage("/")

    # Measure model inference time
    t0        = time.time()
    predictor = get_predictor()
    # Quick forward pass on 3 random stocks
    sample = predictor.stocks[:3]
    for s in sample:
        predictor.predict(s)
    inf_time = round((time.time() - t0) * 1000 / max(len(sample), 1), 2)

    return {
        "cpu_percent"       : cpu_pct,
        "memory_used_mb"    : round(mem.used  / 1024**2, 1),
        "memory_total_mb"   : round(mem.total / 1024**2, 1),
        "memory_percent"    : mem.percent,
        "disk_used_gb"      : round(disk.used  / 1024**3, 2),
        "disk_total_gb"     : round(disk.total / 1024**3, 2),
        "disk_percent"      : disk.percent,
        "avg_inference_ms"  : inf_time,
        "loaded_stocks"     : len(predictor.stocks),
        "model_status"      : "loaded" if predictor._loaded else "loading",
    }


# ── /top-stocks ────────────────────────────────────────────────────────────────
@app.get("/top-stocks")
def top_stocks(n: int = Query(10, description="Number of top stocks to return")):
    """Return top-N stocks by predicted return (long candidates)."""
    predictor = get_predictor()
    preds = predictor.get_all_predictions()
    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    return {
        "top_long" : [{"stock": s, "predicted_return": round(p, 4)}
                       for s, p in sorted_preds[:n]],
        "top_short": [{"stock": s, "predicted_return": round(p, 4)}
                       for s, p in sorted_preds[-n:]],
    }
