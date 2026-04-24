"""
expert_opinion_propagation/ms_lstm.py
----------------------------------------
Paper Section V-B: Temporal Pre-Training Model (MS-LSTM)

Multi-Scale LSTM captures both short-term and long-term temporal
dependencies in stock price sequences by processing the input at
multiple sampling scales.

Architecture:
  Input  : X ∈ R^{N × L × d}   (N stocks, L timesteps, d features)
  Scales : [1, 2, 4]            (sample every 1st, 2nd, 4th timestep)
  Each scale has its own independent LSTM
  Output : ĥ = LayerNorm(mean of last hidden states across scales)
           ŷ = MLP(ĥ)   ∈ R^N   (predicted return ratios)

Loss:
  IC loss = −Pearson correlation(ŷ, y_true)
  Maximising IC → model ranks stocks correctly by return
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd   # needed for type hints in build_feature_tensor


class SingleScaleLSTM(nn.Module):
    """One LSTM branch for a given sampling scale."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → output: (batch, hidden_dim)"""
        out, _ = self.lstm(x)
        return out[:, -1, :]   # last hidden state


class MSLSTM(nn.Module):
    """
    Multi-Scale LSTM pre-training model.

    Args:
        input_dim  : number of input features (e.g. 5 for OHLCV)
        hidden_dim : LSTM hidden size
        output_dim : 1 (return ratio regression)
        scales     : list of sampling intervals (paper uses [1, 2, 4])
        seq_len    : input sequence length (rolling window L)
    """

    def __init__(self,
                 input_dim: int  = 5,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 scales: list    = None,
                 seq_len: int    = 20):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.scales    = scales
        self.seq_len   = seq_len
        self.hidden_dim = hidden_dim

        # One LSTM per scale
        self.lstm_branches = nn.ModuleList([
            SingleScaleLSTM(input_dim, hidden_dim)
            for _ in scales
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def _subsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Sample the sequence at every 'scale' timesteps.
        Paper Eq. 2: Extract(X, si) = [X_0, X_si, X_2si, …]
        """
        return x[:, ::scale, :]   # (batch, L//scale, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch_size, seq_len, input_dim)
        Returns:
            ŷ : (batch_size, output_dim)
        """
        hidden_states = []
        for scale, lstm in zip(self.scales, self.lstm_branches):
            x_sub = self._subsample(x, scale)   # (B, L//s, d)
            h     = lstm(x_sub)                  # (B, hidden_dim)
            hidden_states.append(h)

        # Average across scales (Paper Eq. 4)
        h_mid = torch.stack(hidden_states, dim=0).mean(dim=0)  # (B, hidden_dim)
        h_mid = self.layer_norm(h_mid)

        # MLP prediction (Paper Eq. 5)
        y_hat = self.mlp(h_mid)  # (B, 1)
        return y_hat


class ICLoss(nn.Module):
    """
    Information Coefficient Loss.
    IC = Pearson correlation between predicted and actual returns.
    L_IC = -IC  (maximise correlation ≡ minimise negative IC)
    Paper Section V-B, Eq. 6-7.
    """

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        # Flatten to 1-D
        p = y_pred.view(-1)
        t = y_true.view(-1)

        p_mean = p - p.mean()
        t_mean = t - t.mean()

        numerator   = (p_mean * t_mean).sum()
        denominator = (p_mean.pow(2).sum() * t_mean.pow(2).sum()).sqrt() + 1e-8
        ic          = numerator / denominator
        return -ic   # minimise negative IC


def build_feature_tensor(signal_df: pd.DataFrame,
                          seq_len: int = 20) -> tuple:
    """
    Build input tensors from expert signal DataFrame.
    Uses expert_signal, signal_strength as proxy features when
    raw OHLCV data is not available.

    Returns:
        (X_tensor, y_tensor, stock_list)
        X : (N, seq_len, n_features)
        y : (N,)
    """
    import pandas as pd

    stocks = sorted(signal_df["stock"].unique())
    X_list, y_list, valid_stocks = [], [], []

    for stock in stocks:
        g = signal_df[signal_df["stock"] == stock].sort_values("date")
        if len(g) < seq_len + 1:
            continue

        # Feature matrix: [expert_signal, signal_strength, is_correct]
        feats = g[["expert_signal", "signal_strength", "is_correct"]].values
        feats = feats.astype(np.float32)

        # Use last seq_len rows as input window
        X = feats[-seq_len - 1: -1]     # (seq_len, n_features)
        y = feats[-1, 0]                 # predict next expert_signal as proxy

        X_list.append(X)
        y_list.append(y)
        valid_stocks.append(stock)

    if not X_list:
        raise ValueError("Not enough data to build feature tensors.")

    X_tensor = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.float32)
    return X_tensor, y_tensor, valid_stocks


def train_ms_lstm(signal_df,
                  epochs: int    = 30,
                  lr: float      = 1e-3,
                  seq_len: int   = 20,
                  save_path: str = None) -> MSLSTM:
    """
    Train the MS-LSTM model on expert signal features.

    Args:
        signal_df : output of generate_signals()
        epochs    : training epochs
        lr        : learning rate
        seq_len   : input sequence length
        save_path : if provided, save model weights here

    Returns:
        Trained MSLSTM model
    """
    import pandas as pd

    print("Building feature tensors …")
    X, y, stocks = build_feature_tensor(signal_df, seq_len)
    n_features   = X.shape[-1]

    model     = MSLSTM(input_dim=n_features, hidden_dim=64,
                        seq_len=seq_len, scales=[1, 2, 4])
    criterion = ICLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"Training MS-LSTM: {X.shape[0]} stocks | "
          f"{epochs} epochs | features={n_features}")

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(X).squeeze(-1)
        loss   = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            ic_val = -loss.item()
            print(f"  Epoch {epoch:3d}/{epochs}  |  IC = {ic_val:.4f}")

    if save_path:
        torch.save({"model_state": model.state_dict(),
                    "n_features" : n_features,
                    "seq_len"    : seq_len,
                    "stocks"     : stocks}, save_path)
        print(f"  Model saved → {save_path}")

    return model, stocks


def infer_ms_lstm(model: MSLSTM,
                  signal_df,
                  seq_len: int = 20) -> dict:
    """
    Run inference: return predicted return ratio per stock.

    Returns:
        dict { stock: predicted_return_ratio }
    """
    X, y, stocks = build_feature_tensor(signal_df, seq_len)
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze(-1).numpy()
    return {s: float(p) for s, p in zip(stocks, preds)}





if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from expert_identification.expert_signal_generator import generate_signals

    sigs  = generate_signals()
    model, stks = train_ms_lstm(sigs, epochs=20,
                                  save_path="../models/ms_lstm.pt")
    preds = infer_ms_lstm(model, sigs)
    print("\nSample predictions:")
    for s, p in list(preds.items())[:5]:
        print(f"  {s}: {p:.4f}")
