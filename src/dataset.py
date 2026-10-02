"""PyTorch dataset pipeline for CardioTwin multimodal fusion.

Merges static EHR baselines with dynamic wearable telemetry, builds
sliding lookback windows (default 72 steps = 6 hours), normalizes
features, and returns (static_vector, dynamic_sequence, label) tensors.

Split discipline: train/val splitting must be done at the patient level
(see train.py) to avoid temporal leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

STATIC_FEATURES = [
    "age",
    "gender",
    "baseline_lvef",
    "baseline_bnp",
    "history_mi",
    "hypertension",
    "prescribed_beta_blocker",
    "prescribed_ace_inhibitor",
]

DYNAMIC_FEATURES = [
    "heart_rate",
    "hrv_rmssd",
    "resp_rate",
    "spo2",
    "steps",
]

TARGET_COLUMN = "decompensation_event_24h"


@dataclass
class Normalizer:
    """Z-score params for dynamic signals, min-max params for static."""

    dyn_mean: np.ndarray
    dyn_std: np.ndarray
    stat_min: np.ndarray
    stat_max: np.ndarray

    def transform_dynamic(self, x: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        return (x - self.dyn_mean) / np.maximum(self.dyn_std, 1e-6)

    def transform_static(self, x: np.ndarray) -> np.ndarray:
        """Apply min-max scaling to [0, 1]."""
        rng = np.maximum(self.stat_max - self.stat_min, 1e-6)
        return (x - self.stat_min) / rng


def fit_normalizer(static_df: pd.DataFrame, dynamic_df: pd.DataFrame) -> Normalizer:
    """Fit normalization statistics (call on TRAIN patients only)."""
    dyn = dynamic_df[DYNAMIC_FEATURES].to_numpy(dtype=np.float64)
    stat = static_df[STATIC_FEATURES].to_numpy(dtype=np.float64)
    return Normalizer(
        dyn_mean=dyn.mean(axis=0),
        dyn_std=dyn.std(axis=0) + 1e-6,
        stat_min=stat.min(axis=0),
        stat_max=stat.max(axis=0),
    )


class CardioTwinDataset(Dataset):
    """Sliding-window multimodal dataset.

    Each sample is a lookback window of `lookback` consecutive 5-min
    epochs for one patient. The label is the forward-horizon flag at the
    final timestep of the window (1 if decompensation within next 24h).
    """

    def __init__(
        self,
        static_df: pd.DataFrame,
        dynamic_df: pd.DataFrame,
        patient_ids: list[str],
        lookback: int = 72,
        stride: int = 12,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.lookback = lookback
        self.stride = stride
        self.normalizer = normalizer

        static_f = static_df[static_df["patient_id"].isin(patient_ids)].copy()
        self.static_lookup = {
            pid: grp[STATIC_FEATURES].to_numpy(dtype=np.float32)[0]
            for pid, grp in static_f.groupby("patient_id")
        }

        self.samples: list[tuple[np.ndarray, np.ndarray, float, str, int]] = []
        for pid in patient_ids:
            grp = dynamic_df[dynamic_df["patient_id"] == pid].sort_values("timestep")
            if len(grp) < lookback:
                continue
            dyn_mat = grp[DYNAMIC_FEATURES].to_numpy(dtype=np.float32)
            labels = grp[TARGET_COLUMN].to_numpy(dtype=np.float32)
            timesteps = grp["timestep"].to_numpy(dtype=int)
            stat_vec = self.static_lookup[pid]
            if normalizer is not None:
                dyn_mat = normalizer.transform_dynamic(dyn_mat).astype(np.float32)
                stat_vec = normalizer.transform_static(stat_vec).astype(np.float32)
            for end in range(lookback, len(grp) + 1, stride):
                window = dyn_mat[end - lookback : end]
                label = float(labels[end - 1])
                self.samples.append((stat_vec.copy(), window.copy(), label, pid, int(timesteps[end - 1])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stat_vec, window, label, _, _ = self.samples[idx]
        return (
            torch.from_numpy(stat_vec),
            torch.from_numpy(window),
            torch.tensor([label], dtype=torch.float32),
        )

    def get_meta(self, idx: int) -> tuple[str, int]:
        """Return (patient_id, end_timestep) for a sample index."""
        _, _, _, pid, ts = self.samples[idx]
        return pid, ts


def load_dataframes(data_dir: Path | str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load static and dynamic CSVs from a directory."""
    data_dir = Path(data_dir)
    static_df = pd.read_csv(data_dir / "static_ehr.csv")
    dynamic_df = pd.read_csv(data_dir / "dynamic_wearables.csv", parse_dates=["timestamp"])
    return static_df, dynamic_df


def patient_split(
    patient_ids: list[str], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Stratified-ish patient-level split (shuffled, seed-fixed)."""
    rng = np.random.default_rng(seed)
    ids = np.array(patient_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    return ids[n_val:].tolist(), ids[:n_val].tolist()
