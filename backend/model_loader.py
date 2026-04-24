"""
backend/model_loader.py
-------------------------
Utility functions to train and save models,
and to check which models are already saved.
"""

import os
import sys
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LSTM_PATH = os.path.join(MODELS_DIR, "ms_lstm.pt")
GAT_PATH  = os.path.join(MODELS_DIR, "dual_gat.pt")


def models_exist() -> bool:
    return os.path.exists(LSTM_PATH) and os.path.exists(GAT_PATH)


def train_and_save_all(data_dir: str = None, epochs: int = 30):
    """
    Run full training pipeline and save models.
    Call this once before starting the API server.
    """
    from expert_identification.expert_signal_generator import generate_signals
    from expert_identification.load_data               import load_sector_map, DATA_DIR
    from expert_opinion_propagation.signal_transformation import transform_signals
    from expert_opinion_propagation.graph_builder       import (build_industry_graph,
                                                                 build_correlation_graph,
                                                                 normalize_adjacency)
    from expert_opinion_propagation.ms_lstm             import train_ms_lstm, infer_ms_lstm
    from expert_opinion_propagation.dual_gat            import train_dual_gat

    if data_dir is None:
        data_dir = DATA_DIR

    print("=" * 60)
    print("Training Pipeline")
    print("=" * 60)

    # 1. Expert signals
    print("\n[1/5] Generating expert signals …")
    signal_df   = generate_signals(data_dir)
    sector_map  = load_sector_map(data_dir)
    stocks      = sorted(signal_df["stock"].unique().tolist())

    # 2. Signal transformation
    print("\n[2/5] Transforming signals …")
    transformed = transform_signals(signal_df)

    # 3. Train MS-LSTM
    print("\n[3/5] Training MS-LSTM …")
    ms_lstm, lstm_stocks = train_ms_lstm(
        signal_df, epochs=epochs, save_path=LSTM_PATH
    )
    lstm_preds = infer_ms_lstm(ms_lstm, signal_df)

    # 4. Build graphs
    print("\n[4/5] Building graphs …")
    ind_graph  = build_industry_graph(stocks, sector_map)
    corr_graph = build_correlation_graph(stocks, transformed)
    n          = len(stocks)

    from prediction.predictor import _align_adj
    adj_ind  = normalize_adjacency(_align_adj(ind_graph,  stocks, n))
    adj_cor  = normalize_adjacency(_align_adj(corr_graph, stocks, n))

    # 5. Train DualGAT
    print("\n[5/5] Training DualGAT …")
    train_dual_gat(
        signal_df, lstm_preds,
        adj_ind, adj_cor,
        stocks,
        epochs=epochs,
        save_path=GAT_PATH,
    )

    print("\n✅ Training complete. Models saved:")
    print(f"   {LSTM_PATH}")
    print(f"   {GAT_PATH}")


def get_model_info() -> dict:
    return {
        "ms_lstm_exists"  : os.path.exists(LSTM_PATH),
        "dual_gat_exists" : os.path.exists(GAT_PATH),
        "ms_lstm_path"    : LSTM_PATH,
        "dual_gat_path"   : GAT_PATH,
    }


if __name__ == "__main__":
    train_and_save_all(epochs=20)
