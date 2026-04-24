"""
expert_opinion_propagation/graph_builder.py
---------------------------------------------
Paper Section V-C: Graph Construction

Builds two complementary graphs used by DualGAT:

1. INDUSTRY GRAPH
   - Nodes  : stock tickers
   - Edges  : stocks in the same GICS sector are connected
   - Static  (updated quarterly in practice; we use current sector data)

2. CORRELATION GRAPH
   - Nodes  : stock tickers
   - Edges  : stocks with price-return correlation > threshold over past 30 days
   - Thresholds (paper):
       θ1 = 0.77  (general)
       θ2 = 0.67  (stocks that already have expert signals — lower bar)
   - Dynamic (recalculated each trading day)

Output: adjacency matrices / edge-lists for use in DualGAT.
"""

import pandas as pd
import numpy as np
from itertools import combinations

# ── Paper thresholds ──────────────────────────────────────────────────────────
THETA1 = 0.77   # general correlation threshold
THETA2 = 0.67   # threshold for stocks that have expert signals


def build_industry_graph(stock_list: list,
                          sector_map: dict) -> dict:
    """
    Build the static industry graph.

    Args:
        stock_list : list of ticker strings
        sector_map : dict mapping ticker → sector name

    Returns:
        dict with keys:
          'nodes'    : list of tickers
          'edges'    : list of (ticker_a, ticker_b) tuples
          'adj'      : np.ndarray adjacency matrix (N×N)
          'sector_of': dict mapping ticker → sector
    """
    nodes       = [s for s in stock_list if s in sector_map]
    sector_of   = {s: sector_map[s] for s in nodes}

    # Group by sector
    from collections import defaultdict
    sector_groups = defaultdict(list)
    for s, sec in sector_of.items():
        sector_groups[sec].append(s)

    edges = []
    for sec, members in sector_groups.items():
        for a, b in combinations(members, 2):
            edges.append((a, b))

    # Build adjacency matrix
    n   = len(nodes)
    idx = {s: i for i, s in enumerate(nodes)}
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in edges:
        if a in idx and b in idx:
            adj[idx[a], idx[b]] = 1.0
            adj[idx[b], idx[a]] = 1.0
    np.fill_diagonal(adj, 1.0)   # self-loops

    print(f"  Industry graph: {len(nodes)} nodes | {len(edges)} edges")
    return {
        "nodes"    : nodes,
        "edges"    : edges,
        "adj"      : adj,
        "idx"      : idx,
        "sector_of": sector_of,
    }


def build_correlation_graph(stock_list: list,
                              signal_df: pd.DataFrame,
                              window: int = 30) -> dict:
    """
    Build dynamic correlation graph using expert signal values
    as the proxy for price-return correlation.

    Args:
        stock_list : list of tickers
        signal_df  : DataFrame with columns [stock, date, expert_signal]
        window     : rolling window in days (default 30)

    Returns:
        Same structure as build_industry_graph output.
    """
    # Pivot: rows=dates, cols=stocks, values=expert_signal
    pivot = (
        signal_df[["stock", "date", "expert_signal"]]
        .pivot_table(index="date", columns="stock",
                     values="expert_signal", aggfunc="last")
        .sort_index()
    )

    # Stocks that appear in expert signals (have signals — use θ2)
    stocks_with_signals = set(signal_df["stock"].unique())

    # Use the last 'window' dates for correlation
    recent = pivot.tail(window)

    # Compute pairwise correlation
    corr_matrix = recent.corr()

    nodes = [s for s in stock_list if s in corr_matrix.columns]
    idx   = {s: i for i, s in enumerate(nodes)}
    n     = len(nodes)
    adj   = np.zeros((n, n), dtype=np.float32)
    edges = []

    for a, b in combinations(nodes, 2):
        if a not in corr_matrix.columns or b not in corr_matrix.columns:
            continue
        corr_val = corr_matrix.loc[a, b]
        if np.isnan(corr_val):
            continue

        # Paper: lower threshold if either stock has expert signals
        has_expert = (a in stocks_with_signals or b in stocks_with_signals)
        threshold  = THETA2 if has_expert else THETA1

        if corr_val >= threshold:
            edges.append((a, b))
            adj[idx[a], idx[b]] = corr_val
            adj[idx[b], idx[a]] = corr_val

    np.fill_diagonal(adj, 1.0)
    print(f"  Correlation graph: {len(nodes)} nodes | {len(edges)} edges "
          f"(θ1={THETA1}, θ2={THETA2})")
    return {
        "nodes": nodes,
        "edges": edges,
        "adj"  : adj,
        "idx"  : idx,
    }


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Symmetric normalisation: D^{-1/2} A D^{-1/2}
    Used in GCN-style message passing.
    """
    deg  = adj.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D    = np.diag(dinv)
    return D @ adj @ D


def get_neighbors(adj: np.ndarray, node_idx: int, top_k: int = 10) -> list:
    """Return indices of top-k neighbors for a given node."""
    row  = adj[node_idx].copy()
    row[node_idx] = 0   # exclude self
    return np.argsort(row)[::-1][:top_k].tolist()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from expert_identification.load_data             import load_sector_map
    from expert_identification.expert_signal_generator import generate_signals

    sm   = load_sector_map()
    sigs = generate_signals()
    stocks = sigs["stock"].unique().tolist()

    ind_g  = build_industry_graph(stocks, sm)
    corr_g = build_correlation_graph(stocks, sigs)

    print("\nSample industry edges:")
    for e in ind_g["edges"][:5]:
        print(f"  {e[0]} ↔ {e[1]}  "
              f"[sector: {ind_g['sector_of'].get(e[0], '?')}]")
