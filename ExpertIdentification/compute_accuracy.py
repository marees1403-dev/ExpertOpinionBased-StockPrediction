"""
STEP 3: Compute Per-User Accuracy
====================================
Paper Section IV – Expert User Identification, Algorithm 1

This is the core of the expert tracing system.
For EVERY trading day d and EVERY user who posted on that day, we compute:

  Stage 1 – Recent Performance:
    • Look at user's LAST N=20 posts (before day d)
    • These 20 posts must span at least K=5 different trading days
      (filters out "shotgun" posters who post 20 times in one day)
    • Compute recent accuracy  A_recent = correct / total

  Stage 2 – Long-Term Performance:
    • Look at ALL user posts over the past T=2 years (before day d)
    • Compute long-term accuracy  A_long = correct / total

  Stage 3 – Classify:
    • Expert        : A_recent >= P2 (80%)  AND  A_long >= P1 (65%)
    • Inverse Expert: A_recent <= 1-P2 (20%) AND  A_long <= 1-P1 (35%)
    • Otherwise     : ignore (not used as a signal)

Output: one row per (user_id, symbol, post_date) that qualifies as
        an expert or inverse expert signal.
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

# ── Hyper-parameters (exactly as in the paper) ────────────────────────────────
N  = 20          # recent post window size
K  = 5           # minimum unique days those N posts must span
T  = 2           # long-term window in years
P1 = 0.65        # long-term accuracy threshold for expert
P2 = 0.80        # recent accuracy threshold  for expert
# Inverse thresholds are (1-P2)=0.20 and (1-P1)=0.35 automatically

INPUT_PATH = "output/posts_deduped.parquet"
OUTPUT_DIR = "output"


# ══════════════════════════════════════════════════════════════════════════════
# 3-A  LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3: Compute Per-User Accuracy (Algorithm 1 from paper)")
print("=" * 60)

df = pd.read_parquet(INPUT_PATH)

# Sort chronologically — critical for the "last N posts" window
df = df.sort_values(["user_id", "post_date"]).reset_index(drop=True)

print(f"  Total post-day pairs: {len(df):,}")
print(f"  Unique users        : {df['user_id'].nunique():,}")
print(f"  Parameters: N={N}, K={K}, T={T}yr, P1={P1}, P2={P2}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 3-B  BUILD A HISTORY TABLE FOR EVERY USER
# ══════════════════════════════════════════════════════════════════════════════
# For speed we pre-group all posts by user, then iterate over trading days.
# For each trading day d:
#   - We only look at users who POSTED on day d
#   - We evaluate their history BEFORE day d

user_history = {}    # user_id → list of (post_date, is_correct) sorted by date

for row in df.itertuples(index=False):
    uid  = row.user_id
    date = row.post_date
    correct = row.is_correct
    if uid not in user_history:
        user_history[uid] = []
    user_history[uid].append((date, correct))

print(f"  Built history table for {len(user_history):,} users")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 3-C  EVALUATE EACH USER ON EACH DAY THEY POSTED  (Algorithm 1)
# ══════════════════════════════════════════════════════════════════════════════
results = []
long_window = timedelta(days=T * 365)

# Unique trading days that have at least one post
trading_days = sorted(df["post_date"].unique())

print(f"  Processing {len(trading_days):,} trading days …")
print()

for d in trading_days:
    # All users who posted on day d
    day_posts = df[df["post_date"] == d]

    for row in day_posts.itertuples(index=False):
        uid    = row.user_id
        symbol = row.symbol
        sent   = row.sentiment

        history = user_history.get(uid, [])

        # ── Filter history to posts BEFORE day d ─────────────────────────────
        past_posts = [(dt, ok) for dt, ok in history if dt < d]

        if len(past_posts) == 0:
            continue   # no history at all

        # ── STAGE 1: Recent Performance ──────────────────────────────────────
        # Take the LAST N posts (most recent ones before day d)
        recent_posts = past_posts[-N:]

        # Must span at least K unique days
        unique_days_recent = len(set(dt for dt, _ in recent_posts))
        if unique_days_recent < K:
            continue   # "shotgun" poster — skip

        n_correct_recent = sum(ok for _, ok in recent_posts)
        n_total_recent   = len(recent_posts)
        A_recent = n_correct_recent / n_total_recent

        # ── STAGE 2: Long-Term Performance ───────────────────────────────────
        # All posts within the past T years before day d
        long_start   = d - long_window
        long_posts   = [(dt, ok) for dt, ok in past_posts if dt >= long_start]

        if len(long_posts) == 0:
            continue   # no data in 2-year window

        n_correct_long = sum(ok for _, ok in long_posts)
        n_total_long   = len(long_posts)
        A_long = n_correct_long / n_total_long

        # ── STAGE 3: Classify ─────────────────────────────────────────────────
        if A_recent >= P2 and A_long >= P1:
            user_type = "expert"
        elif A_recent <= (1 - P2) and A_long <= (1 - P1):
            user_type = "inverse_expert"
        else:
            continue   # not an expert or inverse expert

        results.append({
            "trade_date"      : d,
            "user_id"         : uid,
            "symbol"          : symbol,
            "sentiment"       : sent,
            "user_type"       : user_type,
            "A_recent"        : round(A_recent, 4),
            "A_long"          : round(A_long,   4),
            "n_recent_posts"  : n_total_recent,
            "n_long_posts"    : n_total_long,
            # The TRADING SIGNAL:
            # expert      → follow sentiment  (bullish=long, bearish=short)
            # inv. expert → flip  sentiment   (bullish=short, bearish=long)
            "signal": sent if user_type == "expert"
                      else ("bullish" if sent == "bearish" else "bearish"),
        })

expert_df = pd.DataFrame(results)
print(f"  Found {len(expert_df):,} expert / inverse-expert signals")

if len(expert_df) > 0:
    print(f"  Breakdown:")
    print(expert_df["user_type"].value_counts().to_string())
    print()

    # Coverage: what % of stock-day combinations have a signal?
    total_stock_days = len(df[["symbol", "post_date"]].drop_duplicates())
    signal_stock_days = len(
        expert_df[["symbol", "trade_date"]].drop_duplicates()
    )
    coverage = signal_stock_days / total_stock_days * 100
    print(f"  Signal coverage: {coverage:.2f}%  "
          f"(paper reports ~4% — expect a similar number)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 3-D  SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(OUTPUT_DIR, "expert_signals_raw.parquet")
expert_df.to_parquet(out_path, index=False)
print(f"  ✅ Saved expert signals → {out_path}")
print()
print("STEP 3 complete. Run step4_evaluate_accuracy.py next.")
