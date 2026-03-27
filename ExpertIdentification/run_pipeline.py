"""
Expert Identification Pipeline — Master Runner
================================================
Runs all 5 steps in sequence.
Edit the DATA PATHS section to point to your actual downloaded files.

Usage:
    python run_pipeline.py
"""

import os
import sys
import time

# ── DATA PATHS — EDIT THESE ────────────────────────────────────────────────────
# After downloading from https://github.com/wanyunzh/DualGAT
# set these paths to your actual CSV files:

POSTS_CSV  = "data/stocktwits_posts.csv"    # StockTwits sentiment posts
PRICES_CSV = "data/stock_prices.csv"        # OHLCV daily stock prices

# ──────────────────────────────────────────────────────────────────────────────

# Patch paths into step 1 at runtime
os.environ["POSTS_CSV"]  = POSTS_CSV
os.environ["PRICES_CSV"] = PRICES_CSV

steps = [
    ("step1_load_data.py",         "Load & Preprocess Data"),
    ("step2_bot_filter.py",        "Bot / Spammer Filter"),
    ("step3_compute_accuracy.py",  "Compute Per-User Accuracy (Algorithm 1)"),
    ("step4_evaluate_accuracy.py", "Evaluate Expert Accuracy (Fig 1)"),
    ("step5_signal_transformation.py", "Signal Transformation (Binary → Return Ratio)"),
]

print()
print("╔══════════════════════════════════════════════════════════════╗")
print("║         EXPERT IDENTIFICATION PIPELINE                      ║")
print("║   Paper: Unleashing Expert Opinion From Social Media        ║")
print("║          for Stock Prediction  (Zhou et al., 2025)          ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()

total_start = time.time()

for i, (script, description) in enumerate(steps, 1):
    print(f"{'─'*62}")
    print(f"  Running Step {i}/5: {description}")
    print(f"{'─'*62}")
    t0 = time.time()
    exec(open(script).read())
    elapsed = time.time() - t0
    print(f"  ⏱  Step {i} finished in {elapsed:.1f}s")
    print()

total_elapsed = time.time() - total_start
print(f"{'═'*62}")
print(f"  ✅ All steps complete in {total_elapsed:.1f}s")
print(f"  Output files are in the  output/  directory:")
print()
print(f"    posts_with_outcomes.parquet     — posts joined with price labels")
print(f"    posts_deduped.parquet           — after bot/spammer filter")
print(f"    expert_signals_raw.parquet      — raw expert + inverse signals")
print(f"    accuracy_summary.csv            — T+1/T+3/T+7 accuracy table")
print(f"    expert_signals_continuous.parquet — transformed return ratios")
print(f"    full_signal_matrix.parquet      — all stock-days (0 if no expert)")
print(f"{'═'*62}")
