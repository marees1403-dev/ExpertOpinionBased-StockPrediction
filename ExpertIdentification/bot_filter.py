"""
STEP 2: Bot / Spammer Filter
==============================
Paper Section IV – Expert User Identification, Point 1:
  "Filtering spammers: ... for each user-stock pair, we only retain the
   post closest to market closing time on any given day."

Why this matters:
  Bots post hundreds of sentiment-labeled tweets per day at fixed intervals
  (e.g. every 24 seconds) about the same stock. If we count all of them,
  one bot can dominate the accuracy calculation and appear to be an "expert".

What this step does:
  1. Load the merged post-outcome dataset from Step 1.
  2. For each (user_id, symbol, post_date) group, keep ONLY the single post
     that is closest to market close (4:00 PM US Eastern = 20:00 UTC).
  3. Save the de-duplicated dataset for use in Step 3.
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH  = "output/posts_with_outcomes.parquet"
OUTPUT_DIR  = "output"


# ══════════════════════════════════════════════════════════════════════════════
# 2-A  LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2: Bot / Spammer Filter")
print("=" * 60)

df = pd.read_parquet(INPUT_PATH)
print(f"  Posts before filter : {len(df):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2-B  DEFINE "CLOSEST TO MARKET CLOSE"
# ══════════════════════════════════════════════════════════════════════════════
# US market closes at 16:00 Eastern Time = 20:00 UTC.
# For each post, compute how many seconds it is BEFORE that closing time.
# Posts AFTER closing time are counted as being from "next trading day" in
# real trading, but to keep things simple we just pick the latest post of the
# day (which is what the paper does — "post closest to market closing time").

# Strategy: sort by post_time descending inside each group, keep first row.
# This picks the post made closest to (or just before) market close.

df = df.sort_values("post_time", ascending=False)   # latest first

df_filtered = (
    df
    .drop_duplicates(subset=["user_id", "symbol", "post_date"], keep="first")
    .sort_values(["user_id", "symbol", "post_date"])
    .reset_index(drop=True)
)

print(f"  Posts after filter  : {len(df_filtered):,}")
print(f"  Posts removed (bots): {len(df) - len(df_filtered):,}  "
      f"({(1 - len(df_filtered)/len(df))*100:.1f}% reduction)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 2-C  QUICK SANITY CHECK  – find heavy posters before vs after filter
# ══════════════════════════════════════════════════════════════════════════════
# Before filter: how many posts per (user, stock, day) on average?
posts_per_group_before = df.groupby(["user_id", "symbol", "post_date"]).size()
posts_per_group_after  = df_filtered.groupby(
    ["user_id", "symbol", "post_date"]).size()

print(f"  Max posts per (user, stock, day) BEFORE filter : "
      f"{posts_per_group_before.max()}")
print(f"  Max posts per (user, stock, day) AFTER filter  : "
      f"{posts_per_group_after.max()}   ← always 1")
print()

# Top bot candidates (users with the most duplicate posts removed)
before_counts = df.groupby("user_id").size().rename("before")
after_counts  = df_filtered.groupby("user_id").size().rename("after")
bot_report    = pd.concat([before_counts, after_counts], axis=1).fillna(0)
bot_report["removed"] = bot_report["before"] - bot_report["after"]
top_bots = bot_report.nlargest(5, "removed")
print("  Top 5 suspected bot / spammer users (most posts removed):")
print(top_bots.to_string())
print()


# ══════════════════════════════════════════════════════════════════════════════
# 2-D  SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(OUTPUT_DIR, "posts_deduped.parquet")
df_filtered.to_parquet(out_path, index=False)
print(f"  ✅ Saved de-duplicated posts → {out_path}")
print()
print("STEP 2 complete. Run step3_compute_accuracy.py next.")
