"""
scripts/train.py
-----------------
Run this ONCE before starting the API server.
Trains MS-LSTM and DualGAT and saves weights to models/

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 50
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.model_loader import train_and_save_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data directory (default: data/)")
    args = parser.parse_args()
    train_and_save_all(data_dir=args.data_dir, epochs=args.epochs)
