"""
prediction/predictor.py
-------------------------
Main predictor: loads trained models and generates final predictions.

predict(ticker) returns:
  - predicted_return   : float (continuous return ratio)
  - trend              : "Bullish" | "Bearish"
  - confidence         : float (0-1)
  - expert_signal      : float
  - signal_type        : str
  - latest_date        : str
"""

import os
import sys
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from expert_identification.expert_signal_generator import generate_signals
from expert_identification.load_data               import load_sector_map, DATA_DIR
from expert_opinion_propagation.signal_transformation import transform_signals
from expert_opinion_propagation.graph_builder       import (build_industry_graph,
                                                             build_correlation_graph,
                                                             normalize_adjacency)
from expert_opinion_propagation.ms_lstm             import MSLSTM, infer_ms_lstm
from expert_opinion_propagation.dual_gat            import (DualGAT, infer_dual_gat,
                                                             build_node_features)

MODELS_DIR = os.path.join(ROOT, "models")


class StockPredictor:
    """
    End-to-end stock return predictor.

    Usage:
        predictor = StockPredictor()
        result    = predictor.predict("AAPL")
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir   = data_dir
        self.signal_df  = None
        self.sector_map = None
        self.ms_lstm    = None
        self.dual_gat   = None
        self.stocks     = None
        self.lstm_preds = None
        self.gat_preds  = None
        self._loaded    = False

    def load(self):
        """Load data, run expert identification, build graphs, load/train models."""
        print("[Predictor] Initialising …")

        # 1. Expert signals
        print("[Predictor] Loading expert signals …")
        self.signal_df  = generate_signals(self.data_dir)
        self.sector_map = load_sector_map(self.data_dir)
        self.stocks     = sorted(self.signal_df["stock"].unique().tolist())

        # 2. Signal transformation
        transformed = transform_signals(self.signal_df)

        # 3. Graphs
        print("[Predictor] Building graphs …")
        self.ind_graph  = build_industry_graph(self.stocks, self.sector_map)
        self.corr_graph = build_correlation_graph(self.stocks, transformed)

        # Normalise adjacency matrices
        n = len(self.stocks)
        stock_idx = {s: i for i, s in enumerate(self.stocks)}

        # Align graph nodes with self.stocks
        adj_ind  = _align_adj(self.ind_graph,  self.stocks, n)
        adj_cor  = _align_adj(self.corr_graph, self.stocks, n)
        adj_ind  = normalize_adjacency(adj_ind)
        adj_cor  = normalize_adjacency(adj_cor)

        self._adj_ind = adj_ind
        self._adj_cor = adj_cor

        # 4. MS-LSTM
        lstm_path = os.path.join(MODELS_DIR, "ms_lstm.pt")
        if os.path.exists(lstm_path):
            print("[Predictor] Loading MS-LSTM weights …")
            ckpt = torch.load(lstm_path, map_location="cpu")
            self.ms_lstm = MSLSTM(
                input_dim=ckpt["n_features"],
                seq_len=ckpt["seq_len"],
            )
            self.ms_lstm.load_state_dict(ckpt["model_state"])
            self.lstm_preds = infer_ms_lstm(self.ms_lstm, self.signal_df,
                                             seq_len=ckpt["seq_len"])
        else:
            print("[Predictor] MS-LSTM weights not found — using signal proxy.")
            self.lstm_preds = {
                s: float(self.signal_df[self.signal_df["stock"]==s]
                          ["expert_signal"].iloc[-1])
                for s in self.stocks
                if len(self.signal_df[self.signal_df["stock"]==s]) > 0
            }

        # 5. DualGAT
        gat_path = os.path.join(MODELS_DIR, "dual_gat.pt")
        if os.path.exists(gat_path):
            print("[Predictor] Loading DualGAT weights …")
            ckpt = torch.load(gat_path, map_location="cpu")
            self.dual_gat = DualGAT(
                in_dim=ckpt["n_features"], hidden_dim=32, out_dim=1
            )
            self.dual_gat.load_state_dict(ckpt["model_state"])
            self.gat_preds = infer_dual_gat(
                self.dual_gat, self.signal_df,
                self.lstm_preds, adj_ind, adj_cor,
                ckpt["stocks"]
            )
        else:
            print("[Predictor] DualGAT weights not found — using LSTM preds.")
            self.gat_preds = self.lstm_preds

        self._loaded = True
        print(f"[Predictor] Ready. {len(self.stocks)} stocks loaded.")

    def predict(self, ticker: str) -> dict:
        """
        Predict return ratio for a given ticker.

        Returns dict with all prediction fields.
        """
        if not self._loaded:
            self.load()

        ticker = ticker.upper()
        if ticker not in self.stocks:
            return {
                "error": f"Ticker '{ticker}' not found in dataset.",
                "available_stocks": self.stocks[:20],
            }

        # Get latest expert signal for this stock
        stock_sigs = self.signal_df[self.signal_df["stock"] == ticker]
        if stock_sigs.empty:
            return {"error": f"No signals for {ticker}"}

        latest = stock_sigs.sort_values("date").iloc[-1]

        # Final return prediction from DualGAT
        predicted_return = self.gat_preds.get(ticker,
                            self.lstm_preds.get(ticker, 0.0))

        # Trend
        trend      = "Bullish" if predicted_return > 0 else "Bearish"
        confidence = min(abs(predicted_return) * 5, 0.99)   # scale to [0,1]

        # Historical signals for this stock
        history = stock_sigs.sort_values("date").tail(20)

        return {
            "ticker"          : ticker,
            "predicted_return": round(float(predicted_return), 4),
            "trend"           : trend,
            "confidence"      : round(float(confidence), 3),
            "expert_signal"   : round(float(latest["expert_signal"]), 4),
            "signal_type"     : str(latest["signal_type"]),
            "latest_date"     : str(latest["date"])[:10],
            "pseudo_gt"       : str(latest["pseudo_gt"]),
            "accuracy"        : round(float(stock_sigs["is_correct"].mean()*100), 2),
            "n_signals"       : len(stock_sigs),
            "history"         : history[["date","pseudo_gt","gt_sentiment",
                                          "is_correct","expert_signal"]]
                                .assign(date=lambda x: x["date"].astype(str))
                                .to_dict(orient="records"),
        }

    def get_all_predictions(self) -> dict:
        """Return predictions for ALL stocks."""
        if not self._loaded:
            self.load()
        return {s: self.gat_preds.get(s, 0.0) for s in self.stocks}


def _align_adj(graph_dict: dict,
                all_stocks: list,
                n: int) -> np.ndarray:
    """Re-index adjacency matrix to match all_stocks order."""
    adj_full = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(adj_full, 1.0)

    graph_idx = graph_dict.get("idx", {})
    graph_adj = graph_dict.get("adj", np.array([]))

    if graph_adj.size == 0:
        return adj_full

    full_idx = {s: i for i, s in enumerate(all_stocks)}

    for gn, gi in graph_idx.items():
        if gn not in full_idx:
            continue
        fi = full_idx[gn]
        for gm, gj in graph_idx.items():
            if gm not in full_idx:
                continue
            fj = full_idx[gm]
            if gi < graph_adj.shape[0] and gj < graph_adj.shape[1]:
                adj_full[fi, fj] = graph_adj[gi, gj]

    return adj_full


# Singleton predictor (loaded once, reused across API calls)
_predictor_instance = None

def get_predictor() -> StockPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = StockPredictor()
        _predictor_instance.load()
    return _predictor_instance


if __name__ == "__main__":
    p   = StockPredictor()
    p.load()
    res = p.predict("AAPL")
    import json
    print(json.dumps(res, indent=2, default=str))
