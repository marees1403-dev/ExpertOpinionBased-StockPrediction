# Expert Identification Pipeline
### Based on: "Unleashing Expert Opinion From Social Media for Stock Prediction"
### Zhou et al., IEEE TKDE 2025 | GitHub: github.com/wanyunzh/DualGAT

---

## What This Implements

The **Expert Identification** component of the DualGAT paper (Section IV +
Section V-A, Algorithm 1). This is the first and most novel part of the paper
— it identifies "true experts" and "inverse experts" from StockTwits posts who
can serve as reliable trading signals.

---

## File Structure

```
expert_identification/
│
├── run_pipeline.py               ← Run this to execute all steps
│
├── step1_load_data.py            ← Load CSVs, compute next-day return ratio
├── step2_bot_filter.py           ← Remove bot/spammer duplicates
├── step3_compute_accuracy.py     ← Algorithm 1: classify experts
├── step4_evaluate_accuracy.py    ← Evaluate T+1/T+3/T+7 accuracy (Fig 1)
├── step5_signal_transformation.py ← Convert binary signal → return ratio
│
├── data/                         ← Put your downloaded CSV files here
│   ├── stocktwits_posts.csv
│   └── stock_prices.csv
│
└── output/                       ← Generated automatically
    ├── posts_with_outcomes.parquet
    ├── posts_deduped.parquet
    ├── expert_signals_raw.parquet
    ├── accuracy_summary.csv
    ├── expert_signals_continuous.parquet
    └── full_signal_matrix.parquet
```

---

## Setup

```bash
pip install pandas numpy
```

---

## How to Run

1. Download the dataset from https://github.com/wanyunzh/DualGAT
2. Put `stocktwits_posts.csv` and `stock_prices.csv` inside the `data/` folder
3. Edit the column name mappings at the top of `step1_load_data.py` to match
   your actual CSV column names
4. Run:

```bash
python run_pipeline.py
```

Or run each step individually:

```bash
python step1_load_data.py
python step2_bot_filter.py
python step3_compute_accuracy.py
python step4_evaluate_accuracy.py
python step5_signal_transformation.py
```

---

## Step-by-Step Explanation

### Step 1 — Load Data
- Reads StockTwits posts: `user_id`, `symbol`, `post_time`, `sentiment`
- Reads stock prices: OHLCV daily data
- Computes **next-day return ratio**: `r = (close_{t+1} - close_t) / close_t`
- Labels each day's price movement as "rise" or "fall"
- Joins each post with the actual price outcome on the next day

### Step 2 — Bot / Spammer Filter
> *Paper quote:* "for each user-stock pair, we only retain the post closest
> to market closing time on any given day"

- Keeps only **1 post per (user, stock, day)** — the latest one
- This removes bots that post hundreds of times per day
- Typically removes 30–60% of raw posts

### Step 3 — Compute Per-User Accuracy (Algorithm 1)
For every user who posts on day `d`:

**Stage 1 — Recent Performance:**
- Take their last **N=20 posts** before day `d`
- Those 20 posts must span at least **K=5 unique days** (filters shotgun posters)
- Compute: `A_recent = correct_predictions / total_predictions`

**Stage 2 — Long-Term Performance:**
- Take all their posts over the past **T=2 years** before day `d`
- Compute: `A_long = correct_predictions / total_predictions`

**Stage 3 — Classify:**
| User Type | A_recent | A_long |
|-----------|----------|--------|
| Expert | ≥ 80% | ≥ 65% |
| Inverse Expert | ≤ 20% | ≤ 35% |
| Neither | — | — |

For **experts**: follow their sentiment (bullish → go long, bearish → go short)
For **inverse experts**: do the opposite (bullish → go short, bearish → go long)

### Step 4 — Evaluate Accuracy
- Measures expert accuracy at T+1, T+3, T+7 horizons
- Computes naive baseline (~47.6%) to reproduce Fig 1 from the paper
- Our experts should achieve ~72.8% at T+1 (paper's result)
- When multiple experts post the same stock-day, randomly sample ONE

### Step 5 — Signal Transformation
> *Paper Section V-A:* "for stocks predicted to rise, we compute the average
> return ratio from days within the past 30 days when the stock showed an
> upward trend"

- Bullish signal → use 30-day average of positive returns as signal value
- Bearish signal → use 30-day average of negative returns as signal value
- No expert → signal = 0
- Produces a **continuous return ratio** instead of binary ±1
- Expert signals cover **~4%** of all stock-day pairs

---

## Key Paper Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| N | 20 | Recent posts window |
| K | 5 | Min unique days in recent window |
| T | 2 years | Long-term evaluation window |
| P1 | 65% | Long-term accuracy threshold |
| P2 | 80% | Recent accuracy threshold |
| Window | 30 days | Return ratio computation window |
| Coverage | ~4% | Stock-days with expert signals |

---

## What Comes After (Not Implemented Here)

After expert signals are extracted, the paper's pipeline continues with:

1. **MS-LSTM Pre-training** — Multi-scale LSTM trained on OHLCV+fundamentals
   to generate baseline return ratio estimates for every stock-day

2. **DualGAT** — Dual Graph Attention Network that:
   - Constructs an **Industry Graph** (stocks in same GICS sector connected)
   - Constructs a **Correlation Graph** (stocks with 30-day correlation > 0.77)
   - Propagates the sparse 4% expert signals across connected stocks
   - Increases signal coverage from 4% to ~89% after 2-hop propagation

3. **Prediction & Strategy** — Long top 10% of predicted returns, short bottom 10%

---

## Expected Results

After running this pipeline you should see:

```
Expert accuracy at T+1: ~72.8%    (paper: 72.8%)
Expert accuracy at T+3: ~68.0%
Expert accuracy at T+7: ~65.0%
Naive accuracy  at T+1: ~47.6%    (paper: 47.6%)
Signal coverage        : ~4.0%    (paper: ~4%)
```
