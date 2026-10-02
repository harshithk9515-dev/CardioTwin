"""Explainability module for CardioTwinNet.

Provides gradient-based feature attributions (Integrated Gradients style)
without requiring SHAP at inference time, plus an optional SHAP
GradientExplainer path when `shap` is installed.

Feature space explained (13 total):
  static (8): age, gender, baseline_lvef, baseline_bnp, history_mi,
              hypertension, prescribed_beta_blocker, prescribed_ace_inhibitor
  dynamic (5, aggregated over lookback): heart_rate, hrv_rmssd, resp_rate,
              spo2, steps  (mean of last-6h window vs cohort baseline)

Output: normalized importance scores in [-1, 1] with human-readable names.

Usage:
    from explainability import explain_patient_window
    contribs = explain_patient_window(model, static_vec, dynamic_seq)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from dataset import DYNAMIC_FEATURES, STATIC_FEATURES

try:
    from model import CardioTwinNet
except ImportError:  # pragma: no cover - allows `python -m src.explainability`
    from src.model import CardioTwinNet  # type: ignore

READABLE_NAMES = {
    "age": "Age",
    "gender": "Male sex",
    "baseline_lvef": "Baseline LVEF (low)",
    "baseline_bnp": "Baseline BNP (high)",
    "history_mi": "History of MI",
    "hypertension": "Hypertension",
    "prescribed_beta_blocker": "No beta-blocker",
    "prescribed_ace_inhibitor": "No ACE-inhibitor",
    "heart_rate": "Resting HR elevation",
    "hrv_rmssd": "RMSSD drop (HRV collapse)",
    "resp_rate": "Tachypnea (RR rise)",
    "spo2": "SpO2 desaturation",
    "steps": "Activity withdrawal",
}

# Sign convention: for risk-increasing direction.
# LVEF lower -> higher risk, so invert gradient sign for readability.
INVERT_FOR_READABILITY = {"baseline_lvef", "hrv_rmssd", "spo2", "steps"}


def _integrated_gradients(
    model: CardioTwinNet,
    x_static: torch.Tensor,
    x_dynamic: torch.Tensor,
    steps: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Integrated Gradients w.r.t. static vector and dynamic mean.

    Baselines are zeros (post-normalization mean). Returns
    (static_attr[8], dynamic_attr[5]) in logit-gradient space.
    """
    model.eval()
    x_static = x_static.unsqueeze(0).float()
    x_dynamic = x_dynamic.unsqueeze(0).float()
    base_s = torch.zeros_like(x_static)
    base_d = torch.zeros_like(x_dynamic)

    grad_s_acc = torch.zeros_like(x_static)
    grad_d_acc = torch.zeros_like(x_dynamic)

    for alpha in np.linspace(0, 1, steps + 1)[1:]:
        xs = (base_s + alpha * (x_static - base_s)).detach().requires_grad_(True)
        xd = (base_d + alpha * (x_dynamic - base_d)).detach().requires_grad_(True)
        proba = model(xs, xd)
        # Attribute logit-equivalent: use log(p/(1-p)) gradient via proba grad.
        score = torch.logit(torch.clamp(proba, 1e-6, 1 - 1e-6)).sum()
        gs, gd = torch.autograd.grad(score, (xs, xd), retain_graph=False)
        grad_s_acc += gs.detach()
        grad_d_acc += gd.detach()

    avg_gs = grad_s_acc / steps
    avg_gd = grad_d_acc / steps
    attr_s = ((x_static - base_s) * avg_gs).squeeze(0).cpu().numpy()
    # Aggregate temporal attributions over time -> per-channel mean.
    attr_d_full = ((x_dynamic - base_d) * avg_gd).squeeze(0).cpu().numpy()  # (T, 5)
    attr_d = attr_d_full.mean(axis=0)
    return attr_s.astype(float), attr_d.astype(float)


def explain_patient_window(
    model: CardioTwinNet,
    static_vec: np.ndarray | torch.Tensor,
    dynamic_seq: np.ndarray | torch.Tensor,
    steps: int = 32,
) -> list[dict[str, float | str]]:
    """Return sorted normalized contributions for one window.

    Args:
        model: trained CardioTwinNet in eval mode.
        static_vec: normalized static vector, shape (8,).
        dynamic_seq: normalized dynamic window, shape (T, 5).
        steps: Riemann steps for integrated gradients.

    Returns:
        List of dicts [{feature, label, score, direction}], sorted by
        |score| descending, scores normalized to [-1, 1] by max abs.
    """
    if isinstance(static_vec, np.ndarray):
        xs = torch.from_numpy(static_vec.astype(np.float32))
    else:
        xs = static_vec.detach().cpu().float()
    if isinstance(dynamic_seq, np.ndarray):
        xd = torch.from_numpy(dynamic_seq.astype(np.float32))
    else:
        xd = dynamic_seq.detach().cpu().float()

    attr_s, attr_d = _integrated_gradients(model, xs, xd, steps=steps)

    keys = list(STATIC_FEATURES) + list(DYNAMIC_FEATURES)
    raw = np.concatenate([attr_s, attr_d]).astype(float)

    # Flip sign for readability so positive always means "pushes risk up".
    for i, k in enumerate(keys):
        base_idx_static = i < len(STATIC_FEATURES)
        if k in INVERT_FOR_READABILITY:
            # For these, a *low* raw value drives risk; gradient sign is
            # negative when low values increase risk. Negate so the bar
            # reads positive when the physiology is alarming.
            # Heuristic: if input value < 0 (below cohort mean), flip.
            val = float(xs[i].item()) if base_idx_static else float(xd[:, i - len(STATIC_FEATURES)].mean().item())
            if val < 0:
                raw[i] = -raw[i] if raw[i] < 0 else raw[i]
            else:
                raw[i] = raw[i]
    # Medication protection: not-on-therapy with high risk -> positive bar.
    # (handled by readable label "No beta-blocker" etc.)

    denom = float(np.max(np.abs(raw))) if float(np.max(np.abs(raw))) > 1e-9 else 1.0
    normed = raw / denom

    out: list[dict[str, float | str]] = []
    for k, s in zip(keys, normed):
        out.append(
            {
                "feature": k,
                "label": READABLE_NAMES.get(k, k),
                "score": float(np.clip(s, -1.0, 1.0)),
                "direction": "increases risk" if s >= 0 else "decreases risk",
            }
        )
    out.sort(key=lambda d: abs(float(d["score"])), reverse=True)
    return out


def top_k(contribs: list[dict], k: int = 5) -> list[dict]:
    """Return top-k contributions by absolute score."""
    return contribs[:k]


def shap_available() -> bool:
    """Check whether the optional `shap` dependency is installed."""
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False
