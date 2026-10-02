"""Synthetic cohort generator for CardioTwin.

Creates a DPDP Act 2023 / HIPAA-compliant fully synthetic dataset:
  - data/static_ehr.csv       (200 patients, Synthea-style schema)
  - data/dynamic_wearables.csv (7 days @ 5-min epochs = 2016 steps/patient)

Physiology is grounded in textbook ranges (resting HR, RMSSD, RR, SpO2,
circadian sleep/wake modulation). Decompensation events are injected in
~25% of patients with gradual HR rise, HRV collapse, nocturnal tachypnea
and activity withdrawal. Labels encode a 24h forward-looking horizon so
the model learns 12-24h early warning.

Usage:
    python data/generate_synthetic_data.py [--n-patients 200] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_PATIENTS_DEFAULT = 200
SEED_DEFAULT = 42
STEPS_PER_DAY = 288  # 24h * 12 epochs/hour
N_DAYS = 7
N_STEPS = STEPS_PER_DAY * N_DAYS  # 2016
DECOMP_FRACTION = 0.25
FORWARD_HORIZON_STEPS = 288  # 24 hours
RAMP_STEPS = 216  # 18 hours of gradual decompensation

STATIC_COLUMNS = [
    "patient_id",
    "age",
    "gender",
    "baseline_lvef",
    "baseline_bnp",
    "history_mi",
    "hypertension",
    "prescribed_beta_blocker",
    "prescribed_ace_inhibitor",
]

DYNAMIC_COLUMNS = [
    "patient_id",
    "timestamp",
    "timestep",
    "heart_rate",
    "hrv_rmssd",
    "resp_rate",
    "spo2",
    "steps",
    "is_sleep",
    "decompensation_event_24h",
]


def _generate_static_cohort(n_patients: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate static EHR baselines for the cohort."""
    patient_ids = [f"PT-{i + 1:03d}" for i in range(n_patients)]
    ages = rng.integers(45, 86, size=n_patients)
    genders = rng.integers(0, 2, size=n_patients)  # 0=F, 1=M

    # LVEF: skew toward preserved EF, with a tail of reduced EF.
    lvef = np.clip(rng.normal(52.0, 11.0, size=n_patients), 25.0, 65.0).round(1)
    # BNP: log-normal-ish, higher in older / low LVEF patients.
    bnp_base = rng.lognormal(mean=5.4, sigma=0.75, size=n_patients)
    bnp_age_adj = (ages - 45) * 8.0
    bnp_lvef_adj = np.where(lvef < 40, 450.0, 0.0)
    bnp = np.clip(bnp_base + bnp_age_adj + bnp_lvef_adj, 50.0, 1800.0).round(1)

    history_mi = (rng.random(n_patients) < (0.12 + (65 - lvef) * 0.012)).astype(int)
    hypertension = (rng.random(n_patients) < 0.55).astype(int)
    # Guideline-directed therapy more likely in low LVEF / post-MI.
    bb_prob = np.where((lvef < 45) | (history_mi == 1), 0.75, 0.35)
    ace_prob = np.where((lvef < 45) | (hypertension == 1), 0.70, 0.30)
    beta_blocker = (rng.random(n_patients) < bb_prob).astype(int)
    ace_inhibitor = (rng.random(n_patients) < ace_prob).astype(int)

    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": ages,
            "gender": genders,
            "baseline_lvef": lvef,
            "baseline_bnp": bnp,
            "history_mi": history_mi,
            "hypertension": hypertension,
            "prescribed_beta_blocker": beta_blocker,
            "prescribed_ace_inhibitor": ace_inhibitor,
        }
    )


def _select_decomp_patients(static_df: pd.DataFrame, rng: np.random.Generator) -> set[str]:
    """Select ~25% of patients for decompensation, biased to high-risk."""
    n_decomp = int(len(static_df) * DECOMP_FRACTION)
    high_risk_mask = (static_df["baseline_lvef"] < 40.0) | (static_df["baseline_bnp"] > 600.0)
    high_risk_ids = static_df.loc[high_risk_mask, "patient_id"].tolist()
    all_ids = static_df["patient_id"].tolist()

    n_from_high = min(len(high_risk_ids), int(n_decomp * 0.7))
    chosen: set[str] = set()
    if n_from_high > 0:
        picked = rng.choice(high_risk_ids, size=n_from_high, replace=False)
        chosen.update(picked.tolist())
    remaining = n_decomp - len(chosen)
    pool = [pid for pid in all_ids if pid not in chosen]
    if remaining > 0:
        picked = rng.choice(pool, size=remaining, replace=False)
        chosen.update(picked.tolist())
    return chosen


def _generate_patient_telemetry(
    age: int,
    gender: int,
    lvef: float,
    bnp: float,
    beta_blocker: int,
    has_event: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]:
    """Generate 7-day 5-minute telemetry for one patient.

    Returns (hr, rmssd, rr, spo2, steps, is_sleep, onset_idx).
    """
    t = np.arange(N_STEPS)
    hour_of_day = ((t * 5) // 60) % 24
    is_sleep = ((hour_of_day >= 23) | (hour_of_day < 6)).astype(int)

    # --- Baseline heart rate ---
    base_hr = 68.0 + (age - 60) * 0.08 + (0.0 if gender == 1 else 1.5)
    if lvef < 40:
        base_hr += 6.0
    if bnp > 600:
        base_hr += 3.0
    if beta_blocker == 1:
        base_hr -= 6.0
    base_hr = float(np.clip(base_hr, 58.0, 92.0))

    # Circadian: ~-9 bpm at night, +activity bumps during day.
    circadian = -9.0 * is_sleep + 4.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)
    # Random daytime activity bursts (commute/walk analogue).
    active = ((hour_of_day >= 7) & (hour_of_day <= 21)).astype(float)
    bursts = (rng.random(N_STEPS) < 0.06).astype(float) * active * rng.uniform(8, 22, size=N_STEPS)
    hr_noise = rng.normal(0, 1.6, size=N_STEPS)
    hr = base_hr + circadian + bursts + hr_noise

    # --- HRV (RMSSD ms): higher at night, lower with age / low EF ---
    base_rmssd = 42.0 - (age - 45) * 0.35 - (0.0 if lvef >= 45 else (45 - lvef) * 0.9)
    base_rmssd = float(np.clip(base_rmssd, 12.0, 65.0))
    rmssd = base_rmssd + 8.0 * is_sleep + rng.normal(0, 2.5, size=N_STEPS)
    rmssd = np.clip(rmssd, 5.0, 110.0)

    # --- Respiratory rate ---
    base_rr = 14.5 + (0.5 if age > 70 else 0.0) + (1.0 if lvef < 40 else 0.0)
    rr = base_rr - 1.2 * is_sleep + 1.5 * (bursts > 0).astype(float) + rng.normal(0, 0.7, size=N_STEPS)
    rr = np.clip(rr, 9.0, 26.0)

    # --- SpO2 ---
    base_spo2 = 97.5 - (0.5 if age > 70 else 0.0) - (0.5 if lvef < 40 else 0.0)
    spo2 = base_spo2 + rng.normal(0, 0.4, size=N_STEPS)
    spo2 = np.clip(spo2, 90.0, 100.0)

    # --- Steps per 5-min epoch ---
    steps = np.zeros(N_STEPS)
    day_mask = is_sleep == 0
    # Sedentary most epochs, occasional walks.
    walk_prob = rng.random(N_STEPS)
    steps[day_mask] = np.where(
        walk_prob[day_mask] < 0.55,
        rng.integers(0, 25, size=int(day_mask.sum())),
        np.where(
            walk_prob[day_mask] < 0.90,
            rng.integers(25, 180, size=int(day_mask.sum())),
            rng.integers(180, 650, size=int(day_mask.sum())),
        ),
    )
    steps = steps + rng.integers(0, 3, size=N_STEPS) * day_mask
    steps = np.clip(steps, 0, 1200).astype(float)

    onset_idx: int | None = None
    if has_event:
        # Onset between start of day 5 and mid day 6 so a full 24h
        # pre-event window exists and post-event progression is visible.
        onset_idx = int(rng.integers(1200, 1600))
        hr_delta = float(rng.uniform(15.0, 25.0))
        hrv_drop = float(rng.uniform(0.40, 0.60))
        spo2_drop = float(rng.uniform(2.0, 4.0))

        ramp = np.zeros(N_STEPS)
        end_ramp = min(N_STEPS, onset_idx + RAMP_STEPS)
        prog = np.linspace(0, 1, end_ramp - onset_idx)
        ramp[onset_idx:end_ramp] = prog
        ramp[end_ramp:] = 1.0

        hr = hr + ramp * hr_delta
        rmssd = rmssd * (1.0 - ramp * hrv_drop)
        rmssd = np.clip(rmssd, 4.0, 110.0)
        # Nocturnal tachypnea accentuated; daytime RR also rises.
        rr_lift = ramp * (7.5 * is_sleep + 3.5 * (1 - is_sleep))
        rr = np.clip(rr + rr_lift, 9.0, 34.0)
        spo2 = np.clip(spo2 - ramp * spo2_drop, 84.0, 100.0)
        # Activity withdrawal toward near-zero.
        steps = steps * (1.0 - ramp * 0.92)
        hr = np.clip(hr, 45.0, 135.0)

    return hr, rmssd, rr, spo2, steps, is_sleep, onset_idx


def generate_cohort(n_patients: int = N_PATIENTS_DEFAULT, seed: int = SEED_DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the full synthetic cohort.

    Returns (static_df, dynamic_df).
    """
    rng = np.random.default_rng(seed)
    static_df = _generate_static_cohort(n_patients, rng)
    decomp_set = _select_decomp_patients(static_df, rng)

    start = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = pd.date_range(start, periods=N_STEPS, freq="5min")

    frames: list[pd.DataFrame] = []
    for _, row in static_df.iterrows():
        pid = str(row["patient_id"])
        hr, rmssd, rr, spo2, steps, is_sleep, onset_idx = _generate_patient_telemetry(
            age=int(row["age"]),
            gender=int(row["gender"]),
            lvef=float(row["baseline_lvef"]),
            bnp=float(row["baseline_bnp"]),
            beta_blocker=int(row["prescribed_beta_blocker"]),
            has_event=(pid in decomp_set),
            rng=rng,
        )
        if onset_idx is None:
            labels = np.zeros(N_STEPS, dtype=int)
        else:
            time_to_event = onset_idx - np.arange(N_STEPS)
            labels = ((time_to_event > 0) & (time_to_event <= FORWARD_HORIZON_STEPS)).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "patient_id": pid,
                    "timestamp": timestamps,
                    "timestep": np.arange(N_STEPS),
                    "heart_rate": np.round(hr, 2),
                    "hrv_rmssd": np.round(rmssd, 2),
                    "resp_rate": np.round(rr, 2),
                    "spo2": np.round(spo2, 2),
                    "steps": np.round(steps).astype(int),
                    "is_sleep": is_sleep.astype(int),
                    "decompensation_event_24h": labels,
                }
            )
        )

    dynamic_df = pd.concat(frames, ignore_index=True)
    # Enforce column order.
    dynamic_df = dynamic_df[DYNAMIC_COLUMNS]
    static_df = static_df[STATIC_COLUMNS]
    return static_df, dynamic_df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate CardioTwin synthetic cohort.")
    parser.add_argument("--n-patients", type=int, default=N_PATIENTS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    static_df, dynamic_df = generate_cohort(n_patients=args.n_patients, seed=args.seed)
    static_path = out_dir / "static_ehr.csv"
    dynamic_path = out_dir / "dynamic_wearables.csv"
    static_df.to_csv(static_path, index=False)
    dynamic_df.to_csv(dynamic_path, index=False)

    n_events = int((static_df["patient_id"].isin([])).sum())  # placeholder no-op
    n_pos = int(dynamic_df["decompensation_event_24h"].sum())
    n_event_patients = int(dynamic_df.groupby("patient_id")["decompensation_event_24h"].max().sum())
    print(f"[CardioTwin] patients={len(static_df)} timesteps/patient={N_STEPS}")
    print(f"[CardioTwin] event patients={n_event_patients} positive epochs={n_pos} ({100*n_pos/len(dynamic_df):.2f}%)")
    print(f"[CardioTwin] wrote {static_path}")
    print(f"[CardioTwin] wrote {dynamic_path}")
    _ = n_events


if __name__ == "__main__":
    main()
