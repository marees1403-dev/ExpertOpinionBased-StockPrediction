# DualGAT Stock Predictor
### Full-Stack Web Application
### Based on: "Unleashing Expert Opinion From Social Media for Stock Prediction"  
### Zhou et al., IEEE TKDE 2025

---

## Project Overview

This application predicts stock return ratios using:
1. **Expert Identification** — dynamic tracing algorithm (Algorithm 1 from paper)
2. **Signal Transformation** — binary signals → continuous return ratios
3. **MS-LSTM** — multi-scale temporal pre-training model
4. **DualGAT** — dual graph attention network (industry + correlation graphs)

---

## Project Structure

```
stockpred/
│
├── data/                          ← PUT YOUR CSV FILES HERE
│   ├── psudo_combine_all.csv
│   ├── psudo_sp500.csv
│   ├── psudo_stocktable.csv
│   ├── NASDAQ100.csv
│   ├── industry_sp500.csv
│   └── stocktable_new.csv
│
├── models/                        ← Trained model weights saved here
│   ├── ms_lstm.pt
│   └── dual_gat.pt
│
├── expert_identification/         ← Module 1
│   ├── load_data.py               → Loads all CSV files
│   ├── bot_filter.py              → Removes duplicate bot posts
│   ├── compute_accuracy.py        → Algorithm 1 accuracy computation
│   └── expert_signal_generator.py → Full pipeline orchestrator
│
├── expert_opinion_propagation/    ← Module 2
│   ├── signal_transformation.py   → Binary → return ratio signals
│   ├── graph_builder.py           → Industry + Correlation graph construction
│   ├── ms_lstm.py                 → Multi-Scale LSTM + IC loss
│   └── dual_gat.py                → Dual Graph Attention Network
│
├── prediction/                    ← Module 3
│   ├── predictor.py               → End-to-end predictor
│   └── evaluation_metrics.py      → IC, ICIR, Sharpe, AR, MaxDD, etc.
│
├── backend/                       ← Module 4
│   ├── api.py                     → FastAPI routes
│   └── model_loader.py            → Training pipeline entry point
│
├── frontend/                      ← Module 5 (React)
│   ├── src/
│   │   ├── App.jsx                → Router + layout
│   │   ├── index.js               → Entry point
│   │   ├── pages/
│   │   │   ├── Home.jsx           → Stock selection + top picks
│   │   │   ├── PredictPage.jsx    → Prediction result + charts
│   │   │   ├── MetricsDashboard.jsx → All metrics + radar chart
│   │   │   └── SystemPage.jsx     → Live CPU/memory/inference monitor
│   │   ├── components/
│   │   │   └── Navbar.jsx         → Navigation bar
│   │   └── utils/
│   │       └── api.js             → API client
│   ├── public/index.html
│   └── package.json
│
├── scripts/
│   └── train.py                   → Standalone training script
│
├── requirements.txt               ← Python dependencies
└── README.md
```

---

## Step-by-Step Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Node.js 18 or higher
- pip and npm installed

---

### STEP 1 — Copy Dataset Files

Copy your downloaded CSV files into the `data/` folder:

```
stockpred/data/psudo_combine_all.csv
stockpred/data/psudo_sp500.csv
stockpred/data/psudo_stocktable.csv
stockpred/data/NASDAQ100.csv
stockpred/data/industry_sp500.csv
stockpred/data/stocktable_new.csv
```

---

### STEP 2 — Install Python Dependencies

Open a terminal and navigate to the `stockpred/` folder:

```bash
cd stockpred
pip install -r requirements.txt
```

This installs: FastAPI, Uvicorn, PyTorch, Pandas, NumPy, SciPy, psutil

---

### STEP 3 — Train the Models (ONE TIME ONLY)

```bash
python scripts/train.py
```

This will:
1. Load all 6 CSV files
2. Run expert identification (Algorithm 1)
3. Train the MS-LSTM model
4. Build industry + correlation graphs
5. Train the DualGAT model
6. Save weights to `models/ms_lstm.pt` and `models/dual_gat.pt`

Expected output:
```
[1/5] Loading expert signals ...  11,647 rows | 1,120 stocks
[2/5] Applying bot/spammer filter ...
[3/5] Computing accuracy & classifying experts ...
[4/5] Computing rolling signal strength ...
[5/5] Training MS-LSTM ... Epoch 30/30 | IC = 0.3142
Training DualGAT: 1120 nodes | 30 epochs
✅ Training complete.
```

Training takes about 3–5 minutes on CPU.

---

### STEP 4 — Start the Backend API

In the `stockpred/` folder, run:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will start at: **http://localhost:8000**

Test it in your browser:
- http://localhost:8000/health
- http://localhost:8000/stocks
- http://localhost:8000/predict?stock=AAPL
- http://localhost:8000/metrics?stock=AAPL
- http://localhost:8000/system-usage

---

### STEP 5 — Start the Frontend

Open a **new terminal**, navigate to `stockpred/frontend/`:

```bash
cd stockpred/frontend
npm install
npm start
```

The React app opens at: **http://localhost:3000**

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stocks` | GET | List all available tickers |
| `/predict?stock=AAPL` | GET | Full prediction for a stock |
| `/metrics?stock=AAPL` | GET | All evaluation metrics |
| `/expert-signals?stock=AAPL` | GET | Raw expert signal history |
| `/system-usage` | GET | CPU, memory, inference time |
| `/top-stocks?n=10` | GET | Top long/short candidates |

---

## Frontend Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Search stocks, view top picks |
| Predict | `/predict?stock=AAPL` | Prediction result + signal charts |
| Metrics | `/metrics` | IC, Sharpe, Drawdown, Cumulative Returns |
| System | `/system` | Live CPU/memory/model stats |

---

## Key Paper Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| N | 20 | Recent posts window |
| K | 5 | Min unique days in recent window |
| T | 2 years | Long-term accuracy window |
| P1 | 65% | Long-term accuracy threshold |
| P2 | 80% | Recent accuracy threshold |
| θ1 | 0.77 | Correlation graph threshold (general) |
| θ2 | 0.67 | Correlation graph threshold (expert stocks) |
| Window | 30 days | Signal return computation window |

---

## Expected Results

After running on your dataset:

| Metric | Your Data | Paper Reports |
|--------|-----------|---------------|
| Expert Accuracy (T+1) | ~82.6% | ~72.8% |
| Naive Baseline | ~68.3% | ~47.6% |
| Unique Stocks | 1,120 | — |
| Expert Signals | 11,647 | — |

---

## Troubleshooting

**Problem:** `ModuleNotFoundError`  
**Fix:** Run all commands from inside the `stockpred/` directory

**Problem:** CORS error in browser  
**Fix:** Make sure backend is running on port 8000 before starting frontend

**Problem:** "No signals found for ticker"  
**Fix:** The ticker must exist in `psudo_combine_all.csv` — search for it on the Home page

**Problem:** `torch` not found  
**Fix:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## Dataset Files (Required)

Available from: https://github.com/wanyunzh/DualGAT

| File | Contents | Rows |
|------|----------|------|
| `psudo_combine_all.csv` | Expert signals, all stocks 2019–2023 | 11,647 |
| `psudo_sp500.csv` | Expert signals, S&P 500 subset | 7,194 |
| `psudo_stocktable.csv` | Expert signals, StockNet subset | 5,606 |
| `NASDAQ100.csv` | NASDAQ-100 GICS sector labels | 106 |
| `industry_sp500.csv` | S&P 500 GICS industry labels | 500 |
| `stocktable_new.csv` | Additional stock sector labels | 62 |
