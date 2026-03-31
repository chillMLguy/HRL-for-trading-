"""
Pre-train PricePatternCNN on a binary direction-prediction task.

The CNN learns to predict whether the 20-bar forward return is positive,
using a vol-normalized window of recent log returns as input. The 3-dim
latent bottleneck is then used as observation features for the RL agents.

Data leakage prevention:
  Full data:    |======== TRAIN (80%) ========|=== EVAL (20%) ===|
  CNN training: |=== CNN_TRAIN (70%) ===|= CNN_VAL (30%) =|
  RL agents see the full TRAIN split with frozen CNN features.

Usage:
  python pretrain_cnn.py --ticker SPY --start 2024-04-01 --end 2025-10-31
  python train_agents.py --ticker SPY --start 2024-04-01 --end 2025-10-31
"""
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from env.cnn_model import PricePatternCNN


BARS_PER_YEAR = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552}


def download_data(ticker, start, end, interval="1h"):
    print(f"  Downloading {ticker} {start} → {end} (interval={interval})...")
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"  {len(prices)} bars loaded.")
    return prices


def build_dataset(returns, vol_20, w_cnn, w_fwd, ann_factor):
    """
    Build (X, y) dataset for CNN pre-training.
    """
    n = len(returns)

    X_list, y_list = [], []
    for t in range(w_cnn, n - w_fwd):
        window = returns[t - w_cnn: t].copy()
        v = vol_20[t]
        if v > 1e-6:
            window = window / (v / ann_factor)

        fwd_ret = returns[t: t + w_fwd].sum()
        label = 1.0 if fwd_ret > 0 else 0.0

        X_list.append(window)
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def compute_vol_20(returns, w20):
    n = len(returns)
    ANN = np.sqrt(1638)  
    vol = np.zeros(n + 1, dtype=np.float32)
    for t in range(n + 1):
        tc = min(t, n)
        w = returns[max(0, tc - w20): tc]
        vol[t] = np.std(w) * ANN if len(w) > 1 else 0.0
    return vol


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train CNN for price pattern features")
    parser.add_argument("--ticker",     default="SPY")
    parser.add_argument("--start",      default="2024-04-01")
    parser.add_argument("--end",        default="2025-10-31")
    parser.add_argument("--interval",   default="1h",
                        choices=list(BARS_PER_YEAR.keys()))
    parser.add_argument("--outdir",     default=".")
    parser.add_argument("--epochs",     default=100, type=int)
    parser.add_argument("--patience",   default=10, type=int)
    parser.add_argument("--latent_dim", default=3, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr",         default=1e-3, type=float)
    parser.add_argument("--seed",       default=42, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bars_per_year = BARS_PER_YEAR[args.interval]
    scale = max(1, bars_per_year // 252)
    w_cnn = 60 * scale
    w_fwd = 20 * scale   
    w20   = 20 * scale   

    print(f"\n=== CNN Pre-Training ===")
    print(f"  Ticker      : {args.ticker}")
    print(f"  Interval    : {args.interval} ({bars_per_year} bars/year)")
    print(f"  Period      : {args.start} → {args.end}")
    print(f"  CNN window  : {w_cnn} bars ({60} trading days)")
    print(f"  Forward tgt : {w_fwd} bars ({20} trading days)")
    print(f"  Latent dim  : {args.latent_dim}")


    prices = download_data(args.ticker, args.start, args.end, args.interval)
    returns = np.diff(np.log(prices.values)).astype(np.float32)


    train_end = int(len(returns) * 0.8)
    train_returns = returns[:train_end]


    ANN = np.float32(np.sqrt(bars_per_year))


    vol_20 = np.zeros(len(train_returns) + 1, dtype=np.float32)
    for t in range(len(train_returns) + 1):
        tc = min(t, len(train_returns))
        w = train_returns[max(0, tc - w20): tc]
        vol_20[t] = np.std(w) * ANN if len(w) > 1 else 0.0


    X, y = build_dataset(train_returns, vol_20, w_cnn, w_fwd, ANN)
    print(f"  Dataset     : {len(X)} samples from training split")
    print(f"  Class balance: {y.mean():.1%} positive")

    if len(X) < 100:
        print("  [ERROR] Not enough data for CNN training. Need more bars.")
        return


    cnn_split = int(len(X) * 0.7)
    X_train, X_val = X[:cnn_split], X[cnn_split:]
    y_train, y_val = y[:cnn_split], y[cnn_split:]

    print(f"  CNN train   : {len(X_train)} samples")
    print(f"  CNN val     : {len(X_val)} samples")


    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)


    model = PricePatternCNN(input_length=w_cnn, latent_dim=args.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    print(f"\n  Training CNN ({sum(p.numel() for p in model.parameters()):,} params)...")


    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_correct += ((pred > 0.5).float() == yb).sum().item()
            train_total += len(xb)


        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
                val_correct += ((pred > 0.5).float() == yb).sum().item()
                val_total += len(xb)

        train_loss /= train_total
        val_loss   /= val_total
        train_acc = train_correct / train_total
        val_acc   = val_correct / val_total

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(patience={args.patience})")
                break


    if best_state is not None:
        model.load_state_dict(best_state)


    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for xb, yb in val_dl:
            pred = model(xb)
            val_correct += ((pred > 0.5).float() == yb).sum().item()
            val_total += len(xb)
    print(f"\n  Best val accuracy: {val_correct / val_total:.3f}")

    save_dir = os.path.join(args.outdir, "models", "cnn_features")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "cnn_model.pt")
    torch.save(model.state_dict(), model_path)

    meta = {
        "ticker":       args.ticker,
        "interval":     args.interval,
        "start":        args.start,
        "end":          args.end,
        "input_length": w_cnn,
        "latent_dim":   args.latent_dim,
        "bars_per_year": bars_per_year,
        "val_accuracy": round(val_correct / val_total, 4),
    }
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved → {model_path}")
    print(f"  Saved → {meta_path}")
    print(f"\n✓ CNN pre-training complete.")


if __name__ == "__main__":
    main()
