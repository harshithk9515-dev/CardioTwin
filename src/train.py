"""Training entry point for CardioTwinNet.

- 80/20 patient-level split (no temporal leakage).
- BCELoss + Adam(lr=1e-3), class-weighted for imbalance.
- Metrics: AUROC, Sensitivity/Recall, Specificity, F1.
- Saves checkpoint + normalizer to models/cardiotwin_weights.pth

Usage:
    python src/train.py [--epochs 15] [--batch-size 128] [--lookback 72]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import (
    CardioTwinDataset,
    fit_normalizer,
    load_dataframes,
    patient_split,
)
from model import CardioTwinNet, count_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    try:
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except ValueError:
        auroc = 0.5
    return {
        "auroc": auroc,
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "n_pos": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


def train(
    epochs: int = 15,
    batch_size: int = 128,
    lookback: int = 72,
    stride: int = 12,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, float]:
    """Train CardioTwinNet and persist artifacts. Returns val metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    static_df, dynamic_df = load_dataframes(DATA_DIR)
    all_patients = sorted(static_df["patient_id"].unique().tolist())
    train_ids, val_ids = patient_split(all_patients, val_fraction=0.2, seed=seed)
    print(f"[train] patients: train={len(train_ids)} val={len(val_ids)} device={device}")

    train_static = static_df[static_df["patient_id"].isin(train_ids)]
    train_dyn = dynamic_df[dynamic_df["patient_id"].isin(train_ids)]
    normalizer = fit_normalizer(train_static, train_dyn)

    train_ds = CardioTwinDataset(static_df, dynamic_df, train_ids, lookback=lookback, stride=stride, normalizer=normalizer)
    val_ds = CardioTwinDataset(static_df, dynamic_df, val_ids, lookback=lookback, stride=stride, normalizer=normalizer)
    print(f"[train] windows: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weighting for ~3-4% positive rate.
    all_labels = np.array([float(s[2]) for s in train_ds.samples], dtype=float)
    pos_rate = float(all_labels.mean()) if len(all_labels) else 0.5
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-4)], dtype=torch.float32).to(device)
    print(f"[train] positive rate={pos_rate:.4f} pos_weight={pos_weight.item():.2f}")

    model = CardioTwinNet().to(device)
    print(f"[train] params={count_parameters(model)}")
    # Model outputs sigmoid probabilities, so element-wise BCELoss is correct.
    loss_fn = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_auroc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xs, xd, y in train_loader:
            xs, xd, y = xs.to(device), xd.to(device), y.to(device)
            optimizer.zero_grad()
            proba = model(xs, xd)
            elem = loss_fn(proba, y)
            with torch.no_grad():
                w = torch.where(y > 0.5, pos_weight, torch.ones_like(y))
            loss = (elem * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(y)
        train_loss = running / max(len(train_ds), 1)

        model.eval()
        probs: list[np.ndarray] = []
        trues: list[np.ndarray] = []
        with torch.no_grad():
            for xs, xd, y in val_loader:
                xs, xd = xs.to(device), xd.to(device)
                p = model(xs, xd).cpu().numpy().ravel()
                probs.append(p)
                trues.append(y.numpy().ravel())
        y_prob = np.concatenate(probs) if probs else np.array([])
        y_true = np.concatenate(trues).astype(int) if trues else np.array([])
        m = _metrics(y_true, y_prob)
        scheduler.step(m["auroc"])
        print(
            f"[epoch {epoch:02d}/{epochs}] loss={train_loss:.4f} "
            f"AUROC={m['auroc']:.4f} sens={m['sensitivity_recall']:.4f} "
            f"spec={m['specificity']:.4f} F1={m['f1']:.4f}"
        )
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "normalizer": {
                        "dyn_mean": normalizer.dyn_mean,
                        "dyn_std": normalizer.dyn_std,
                        "stat_min": normalizer.stat_min,
                        "stat_max": normalizer.stat_max,
                    },
                    "config": {"lookback": lookback, "stride": stride},
                    "metrics": m,
                },
                MODELS_DIR / "cardiotwin_weights.pth",
            )
            print(f"  -> saved best checkpoint (AUROC={best_auroc:.4f})")

    print(f"[train] best val AUROC={best_auroc:.4f}")
    print(f"[train] artifacts -> {MODELS_DIR / 'cardiotwin_weights.pth'}")
    final = _metrics(y_true, y_prob)
    print("[train] FINAL " + json.dumps(final, indent=2))
    return final


def main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Train CardioTwinNet.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=72)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lookback=args.lookback, stride=args.stride, lr=args.lr, seed=args.seed)


if __name__ == "__main__":
    main()
