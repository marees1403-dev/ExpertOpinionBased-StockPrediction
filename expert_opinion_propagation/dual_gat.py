"""
expert_opinion_propagation/dual_gat.py
-----------------------------------------
Paper Section V-C: Dual Graph Attention Network (DualGAT)

Architecture (Fig. 3 from paper):
  Input per node v: [ms_lstm_output, expert_signal, has_expert]

  Two-hop message passing:
    Hop 1: GAT on Industry graph + GAT on Correlation graph
           → Attentive feature fusion (weighted sum)
    Hop 2: GAT on fused features
           → Second attentive fusion
    Output: MLP → scalar return prediction per stock

Key design choices from paper:
  - Separate GAT layers for each graph (pink blocks in Fig 3)
  - Learnable attention to combine the two graph outputs (blue in Fig 3)
  - Two hops gives ~89% node coverage after propagation
  - Attention coefficients adapt per-node (some stocks rely more on
    industry graph, others on correlation graph)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GATLayer(nn.Module):
    """
    Single Graph Attention layer (Paper Eq. 8-9).

    Computes:
      α_{vu} = softmax over neighbors of LeakyReLU(a^T [Wh_v || Wh_u])
      h'_v   = σ( Σ_{u∈N(v)} α_{vu} W h_u )
    """

    def __init__(self, in_dim: int, out_dim: int,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.n_heads   = n_heads
        self.head_dim  = out_dim // n_heads

        self.W   = nn.Linear(in_dim, out_dim, bias=False)
        # Attention vector a ∈ R^{2*out_dim}
        self.a   = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h   : (N, in_dim)   node features
            adj : (N, N)        adjacency matrix (0/1 or weighted)
        Returns:
            h'  : (N, out_dim)
        """
        N   = h.size(0)
        Wh  = self.W(h)                               # (N, out_dim)
        Wh  = Wh.view(N, self.n_heads, self.head_dim) # (N, H, D)

        # Compute attention coefficients
        # For each head, e_{ij} = a^T [Wh_i || Wh_j]
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # (N, N, H, D)
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1, -1)  # (N, N, H, D)
        e    = torch.cat([Wh_i, Wh_j], dim=-1)         # (N, N, H, 2D)
        e    = (e * self.a).sum(-1)                     # (N, N, H)
        e    = self.leaky_relu(e)

        # Mask non-edges
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e    = e.masked_fill(mask, float("-inf"))

        # Softmax over neighbors
        alpha = F.softmax(e, dim=1)                    # (N, N, H)
        alpha = self.dropout(alpha)

        # Aggregate
        Wh_j_exp = Wh_j.permute(0, 1, 2, 3)           # (N, N, H, D)
        out      = (alpha.unsqueeze(-1) * Wh_j_exp).sum(1)  # (N, H, D)
        out      = out.view(N, self.out_dim)            # (N, out_dim)
        return F.elu(out)


class DualGraphAttentionFusion(nn.Module):
    """
    Attentive fusion of features from two graphs (Paper Eq. 10-12).
    Learns per-node weights β_ind and β_cor via softmax.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Learnable query vectors q_ind and q_cor ∈ R^{hidden_dim}
        self.q_ind = nn.Parameter(torch.empty(hidden_dim))
        self.q_cor = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.q_ind)
        nn.init.normal_(self.q_cor)

    def forward(self,
                H_ind: torch.Tensor,
                H_cor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_ind : (N, hidden_dim)  features from industry graph
            H_cor : (N, hidden_dim)  features from correlation graph
        Returns:
            H_fused : (N, hidden_dim)
        """
        # Scalar attention scores per node (Paper Eq. 10)
        alpha_ind = (H_ind * self.q_ind).sum(dim=-1)   # (N,)
        alpha_cor = (H_cor * self.q_cor).sum(dim=-1)   # (N,)

        # Normalise (Paper Eq. 11)
        betas = F.softmax(
            torch.stack([alpha_ind, alpha_cor], dim=-1), dim=-1
        )  # (N, 2)
        beta_ind = betas[:, 0].unsqueeze(-1)   # (N, 1)
        beta_cor = betas[:, 1].unsqueeze(-1)   # (N, 1)

        # Fused features (Paper Eq. 12)
        H_fused = beta_ind * H_ind + beta_cor * H_cor  # (N, hidden_dim)
        return H_fused


class DualGAT(nn.Module):
    """
    Full Dual Graph Attention Network (Paper Section V-C).

    Two-hop architecture:
      Hop 1: GATLayer(ind) + GATLayer(cor) → DualFusion → H1
      Hop 2: GATLayer(ind) + GATLayer(cor) → DualFusion → H2
      Output: MLP(H2) → scalar per node

    Args:
        in_dim     : input feature dimension per node
        hidden_dim : intermediate representation size
        out_dim    : output dimension (1 for return ratio)
        n_heads    : attention heads in each GAT layer
    """

    def __init__(self,
                 in_dim: int     = 3,
                 hidden_dim: int = 32,
                 out_dim: int    = 1,
                 n_heads: int    = 4):
        super().__init__()

        # Hop 1 — transforms input (in_dim) → hidden_dim
        self.gat1_ind = GATLayer(in_dim, hidden_dim, n_heads)
        self.gat1_cor = GATLayer(in_dim, hidden_dim, n_heads)
        self.fuse1    = DualGraphAttentionFusion(hidden_dim)

        # Hop 2 — transforms hidden_dim → hidden_dim
        self.gat2_ind = GATLayer(hidden_dim, hidden_dim, n_heads)
        self.gat2_cor = GATLayer(hidden_dim, hidden_dim, n_heads)
        self.fuse2    = DualGraphAttentionFusion(hidden_dim)

        # Final MLP (Paper Eq. 13)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self,
                x: torch.Tensor,
                adj_ind: torch.Tensor,
                adj_cor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (N, in_dim)   node feature matrix
            adj_ind : (N, N)        industry adjacency
            adj_cor : (N, N)        correlation adjacency
        Returns:
            y_hat   : (N, out_dim)  predicted return ratios
        """
        # ── Hop 1 ────────────────────────────────────────────────────────────
        H1_ind = self.gat1_ind(x, adj_ind)       # (N, hidden_dim)
        H1_cor = self.gat1_cor(x, adj_cor)       # (N, hidden_dim)
        H1     = self.fuse1(H1_ind, H1_cor)      # (N, hidden_dim)

        # ── Hop 2 ────────────────────────────────────────────────────────────
        H2_ind = self.gat2_ind(H1, adj_ind)      # (N, hidden_dim)
        H2_cor = self.gat2_cor(H1, adj_cor)      # (N, hidden_dim)
        H2     = self.fuse2(H2_ind, H2_cor)      # (N, hidden_dim)

        # ── Prediction ───────────────────────────────────────────────────────
        y_hat  = self.mlp(H2)                    # (N, out_dim)
        return y_hat


def build_node_features(stocks: list,
                         lstm_preds: dict,
                         signal_df) -> torch.Tensor:
    """
    Build per-node input feature matrix.

    Features per node (paper Section V-C):
      [lstm_output, has_expert, expert_signal_value]

    Args:
        stocks     : list of tickers (graph node order)
        lstm_preds : { ticker: lstm_predicted_return }
        signal_df  : expert signals DataFrame

    Returns:
        x : torch.Tensor (N, 3)
    """
    import pandas as pd

    # Latest expert signal per stock
    latest_signals = (
        signal_df.sort_values("date")
        .groupby("stock")
        .last()
        .reset_index()
        [["stock", "expert_signal"]]
        .set_index("stock")
    )

    rows = []
    for s in stocks:
        lstm_val    = lstm_preds.get(s, 0.0)
        exp_sig     = float(latest_signals.loc[s, "expert_signal"]) \
                      if s in latest_signals.index else 0.0
        has_expert  = 1.0 if s in latest_signals.index else 0.0
        rows.append([lstm_val, has_expert, exp_sig])

    return torch.tensor(rows, dtype=torch.float32)


def train_dual_gat(signal_df,
                    lstm_preds: dict,
                    adj_ind: np.ndarray,
                    adj_cor: np.ndarray,
                    stocks: list,
                    epochs: int    = 30,
                    lr: float      = 1e-3,
                    save_path: str = None) -> "DualGAT":
    """Train DualGAT and return the trained model."""
    from expert_opinion_propagation.ms_lstm import ICLoss

    adj_ind_t = torch.tensor(adj_ind, dtype=torch.float32)
    adj_cor_t = torch.tensor(adj_cor, dtype=torch.float32)
    x         = build_node_features(stocks, lstm_preds, signal_df)

    # Labels: use expert_signal as proxy for true return
    import pandas as pd
    latest = (signal_df.sort_values("date")
               .groupby("stock").last().reset_index())
    y_dict = dict(zip(latest["stock"], latest["expert_signal"]))
    y      = torch.tensor([y_dict.get(s, 0.0) for s in stocks],
                           dtype=torch.float32)

    n_features = x.shape[-1]
    model      = DualGAT(in_dim=n_features, hidden_dim=32, out_dim=1)
    criterion  = ICLoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training DualGAT: {len(stocks)} nodes | {epochs} epochs")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(x, adj_ind_t, adj_cor_t).squeeze(-1)
        loss   = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            ic = -loss.item()
            print(f"  Epoch {epoch:3d}/{epochs}  |  IC = {ic:.4f}")

    if save_path:
        torch.save({
            "model_state": model.state_dict(),
            "n_features" : n_features,
            "stocks"     : stocks,
        }, save_path)
        print(f"  DualGAT saved → {save_path}")

    return model


def infer_dual_gat(model: DualGAT,
                    signal_df,
                    lstm_preds: dict,
                    adj_ind: np.ndarray,
                    adj_cor: np.ndarray,
                    stocks: list) -> dict:
    """Run inference and return predicted return per stock."""
    adj_ind_t = torch.tensor(adj_ind, dtype=torch.float32)
    adj_cor_t = torch.tensor(adj_cor, dtype=torch.float32)
    x         = build_node_features(stocks, lstm_preds, signal_df)

    model.eval()
    with torch.no_grad():
        preds = model(x, adj_ind_t, adj_cor_t).squeeze(-1).numpy()

    return {s: float(p) for s, p in zip(stocks, preds)}
