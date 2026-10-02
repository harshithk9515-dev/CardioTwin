"""CardioTwin clinician dashboard (Streamlit).

Executive-ready Digital Twin interface:
  - ABHA/ABDM-style patient selector with demographics + meds
  - Live telemetry playback (HR, HRV, SpO2, RR) with sleep shading
  - Real-time decompensation risk gauge + triage + lead-time alert
  - Explainability drawer (top-5 drivers)
  - What-If counterfactual simulator (intervention sliders)

Run:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from dataset import DYNAMIC_FEATURES, STATIC_FEATURES
from explainability import explain_patient_window
from model import CardioTwinNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT = MODELS_DIR / "cardiotwin_weights.pth"
LOOKBACK = 72

TRIAGE = {
    "normal": {"label": "Normal", "color": "#16a34a", "max": 0.30},
    "guarded": {"label": "Guarded", "color": "#d97706", "max": 0.65},
    "critical": {"label": "Critical — Impending Decompensation", "color": "#dc2626", "max": 1.01},
}


def triage_for(p: float) -> tuple[str, str]:
    """Map probability to (label, color)."""
    if p < 0.30:
        return TRIAGE["normal"]["label"], TRIAGE["normal"]["color"]
    if p < 0.65:
        return TRIAGE["guarded"]["label"], TRIAGE["guarded"]["color"]
    return TRIAGE["critical"]["label"], TRIAGE["critical"]["color"]


@st.cache_resource
def load_artifacts():
    """Load dataframes, model, and normalizer (cached)."""
    static_df = pd.read_csv(DATA_DIR / "static_ehr.csv")
    dynamic_df = pd.read_csv(DATA_DIR / "dynamic_wearables.csv", parse_dates=["timestamp"])
    if not CHECKPOINT.exists():
        return static_df, dynamic_df, None, None
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = CardioTwinNet()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = ckpt["normalizer"]
    return static_df, dynamic_df, model, norm


def normalize_static(vec: np.ndarray, norm: dict) -> np.ndarray:
    """Min-max normalize a static vector."""
    mn, mx = norm["stat_min"], norm["stat_max"]
    return (vec - mn) / np.maximum(mx - mn, 1e-6)


def normalize_dynamic(mat: np.ndarray, norm: dict) -> np.ndarray:
    """Z-score normalize a dynamic window."""
    return (mat - norm["dyn_mean"]) / np.maximum(norm["dyn_std"], 1e-6)


def predict_window(model, static_n: np.ndarray, dynamic_n: np.ndarray) -> float:
    """Single-window risk probability."""
    with torch.no_grad():
        xs = torch.from_numpy(static_n.astype(np.float32)).unsqueeze(0)
        xd = torch.from_numpy(dynamic_n.astype(np.float32)).unsqueeze(0)
        return float(model(xs, xd).item())


def risk_trajectory(model, static_n: np.ndarray, patient_dyn: pd.DataFrame, norm: dict, stride: int = 12) -> pd.DataFrame:
    """Rolling risk over the full 7-day stay (evaluated every `stride` epochs)."""
    mat = patient_dyn[DYNAMIC_FEATURES].to_numpy(dtype=float)
    mat_n = normalize_dynamic(mat, norm)
    rows = []
    for end in range(LOOKBACK, len(mat) + 1, stride):
        win = mat_n[end - LOOKBACK : end]
        p = predict_window(model, static_n, win)
        rows.append({"timestep": int(patient_dyn.iloc[end - 1]["timestep"]), "risk": p})
    return pd.DataFrame(rows)


def telemetry_figure(patient_dyn: pd.DataFrame, upto: int) -> go.Figure:
    """Multi-trace HR/HRV/SpO2 chart with sleep shading up to timestep."""
    view = patient_dyn[patient_dyn["timestep"] <= upto].tail(864)  # last 3 days
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["heart_rate"], name="HR (bpm)", line=dict(color="#dc2626", width=1.6)))
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["hrv_rmssd"], name="HRV RMSSD (ms)", yaxis="y2", line=dict(color="#2563eb", width=1.4)))
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["spo2"], name="SpO2 (%)", yaxis="y3", line=dict(color="#16a34a", width=1.2)))
    # Sleep shading.
    for _, grp in view.groupby((view["is_sleep"].diff() != 0).cumsum()):
        if int(grp["is_sleep"].iloc[0]) == 1 and len(grp) > 1:
            fig.add_vrect(x0=grp["timestamp"].iloc[0], x1=grp["timestamp"].iloc[-1], fillcolor="navy", opacity=0.07, line_width=0)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Live telemetry — HR / HRV / SpO2 (shaded = sleep)",
        legend=dict(orientation="h", y=1.08),
        yaxis=dict(title="HR bpm"),
        yaxis2=dict(title="RMSSD ms", overlaying="y", side="right"),
        yaxis3=dict(title="SpO2", overlaying="y", side="right", position=0.97, range=[82, 100]),
    )
    return fig


def risk_gauge_figure(p: float) -> go.Figure:
    """Plotly gauge for current risk."""
    label, color = triage_for(p)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%"},
            title={"text": f"Decompensation risk — {label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 65], "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="CardioTwin — Clinician Digital Twin", layout="wide")
    st.title("🫀 CardioTwin — Cardiovascular Digital Twin")
    st.caption("Synthetic-data prototype · DPDP Act 2023 / HIPAA sandbox compliant · No real patient data")

    static_df, dynamic_df, model, norm = load_artifacts()
    if model is None or norm is None:
        st.error("Model checkpoint not found. Run `python data/generate_synthetic_data.py` then `python src/train.py` first.")
        st.stop()

    # ---- Sidebar: patient + playback ----
    pids = sorted(static_df["patient_id"].unique().tolist())
    default_idx = 0
    # Prefer a high-risk patient for an impressive first demo.
    hi = static_df[(static_df["baseline_lvef"] < 40) | (static_df["baseline_bnp"] > 600)]
    if len(hi):
        default_idx = pids.index(str(hi.iloc[0]["patient_id"]))

    with st.sidebar:
        st.header("Patient")
        pid = st.selectbox("ABHA / ABDM ID", pids, index=default_idx)
        st.divider()
        st.header("What-If intervention")
        hrv_gain = st.slider("Restore HRV (RMSSD +%)", 0, 60, 30, step=5)
        hr_drop = st.slider("Lower resting HR (−bpm)", 0, 20, 15, step=1)
        rr_drop = st.slider("Relieve tachypnea (−br/min)", 0, 8, 5, step=1)
        spo2_gain = st.slider("Improve SpO2 (+%)", 0, 5, 3, step=1)

    srow = static_df[static_df["patient_id"] == pid].iloc[0]
    abha = f"ABHA-{abs(hash(pid)) % 90 + 10}-{abs(hash(pid[::-1])) % 90 + 10}-{abs(hash(pid+'X')) % 9000 + 1000}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient", f"{pid} · {abha}")
    c2.metric("Age / Sex", f"{int(srow['age'])}y · {'M' if int(srow['gender']) == 1 else 'F'}")
    c3.metric("LVEF / BNP", f"{float(srow['baseline_lvef']):.0f}% / {float(srow['baseline_bnp']):.0f}")
    meds = []
    if int(srow["prescribed_beta_blocker"]):
        meds.append("Beta-blocker")
    if int(srow["prescribed_ace_inhibitor"]):
        meds.append("ACE-I")
    hx = []
    if int(srow["history_mi"]):
        hx.append("prior MI")
    if int(srow["hypertension"]):
        hx.append("HTN")
    c4.metric("History / Meds", f"{', '.join(hx) or '—'} / {', '.join(meds) or 'none'}")

    patient_dyn = dynamic_df[dynamic_df["patient_id"] == pid].sort_values("timestep").reset_index(drop=True)
    max_ts = int(patient_dyn["timestep"].max())
    upto = st.slider("Telemetry playback (timestep, 288 = 1 day)", LOOKBACK, max_ts, max_ts, step=12)

    static_vec = srow[STATIC_FEATURES].to_numpy(dtype=float)
    static_n = normalize_static(static_vec, norm)

    # Current window ending at `upto`.
    sub = patient_dyn[patient_dyn["timestep"] <= upto].tail(LOOKBACK)
    cur_mat = sub[DYNAMIC_FEATURES].to_numpy(dtype=float)
    cur_n = normalize_dynamic(cur_mat, norm)
    risk = predict_window(model, static_n, cur_n)

    left, right = st.columns([1.4, 1])
    with left:
        st.plotly_chart(telemetry_figure(patient_dyn, upto), use_container_width=True)
        # RR strip.
        rr_view = patient_dyn[patient_dyn["timestep"] <= upto].tail(864)
        rr_fig = go.Figure()
        rr_fig.add_trace(go.Scatter(x=rr_view["timestamp"], y=rr_view["resp_rate"], name="Resp rate", line=dict(color="#7c3aed")))
        rr_fig.add_hline(y=22, line_dash="dash", line_color="red", annotation_text="tachypnea >22")
        rr_fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), title="Respiratory rate (br/min)")
        st.plotly_chart(rr_fig, use_container_width=True)
    with right:
        st.plotly_chart(risk_gauge_figure(risk), use_container_width=True)
        label, color = triage_for(risk)
        st.markdown(f"<h4 style='color:{color}'>{label} — {risk*100:.1f}%</h4>", unsafe_allow_html=True)
        # Lead-time estimate: distance to first future positive label.
        future = patient_dyn[patient_dyn["timestep"] > upto]
        hits = future[future["decompensation_event_24h"] == 1]
        gt_onset = None
        if len(hits):
            # Onset ≈ first timestep where forward label turns 0 again + horizon.
            gt_onset = int(hits["timestep"].iloc[-1]) + 1
        if risk > 0.65:
            lead_h = ((gt_onset - upto) * 5 / 60) if gt_onset else 18.5
            st.error(f"CRITICAL: Predicted decompensation within ~{lead_h:.1f}h. Recommended: clinical evaluation / diuretic adjustment.")
        elif risk > 0.30:
            st.warning("GUARDED: Rising twin risk. Consider repeat vitals, medication reconciliation, early review.")
        else:
            st.success("STABLE: Twin trajectory within normal bounds. Continue remote monitoring.")

        # Explainability drawer.
        with st.expander("🔍 Why this score? — top drivers", expanded=True):
            contribs = explain_patient_window(model, static_n, cur_n)
            top5 = contribs[:5]
            df_exp = pd.DataFrame([{"factor": d["label"], "score": d["score"]} for d in top5])
            colors = ["#dc2626" if s >= 0 else "#16a34a" for s in df_exp["score"]]
            bar = go.Figure(go.Bar(x=df_exp["score"], y=df_exp["factor"], orientation="h", marker_color=colors))
            bar.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-1, 1], title="contribution →"))
            st.plotly_chart(bar, use_container_width=True)
            st.caption("Positive = pushes risk up. Gradient-based attributions, normalized to [-1, 1].")

    # ---- What-If counterfactual ----
    st.subheader("🧪 What-If simulator — simulate intervention, re-run twin")
    st.caption("Intervention remodels the current 6h window physiology, then re-scores with the same CardioTwinNet.")
    mod = cur_mat.copy()
    idx = {f: DYNAMIC_FEATURES.index(f) for f in DYNAMIC_FEATURES}
    mod[:, idx["heart_rate"]] = np.clip(mod[:, idx["heart_rate"]] - hr_drop, 40, 140)
    mod[:, idx["hrv_rmssd"]] = mod[:, idx["hrv_rmssd"]] * (1 + hrv_gain / 100.0)
    mod[:, idx["resp_rate"]] = np.clip(mod[:, idx["resp_rate"]] - rr_drop, 8, 35)
    mod[:, idx["spo2"]] = np.clip(mod[:, idx["spo2"]] + spo2_gain, 80, 100)
    mod_n = normalize_dynamic(mod, norm)
    cf_risk = predict_window(model, static_n, mod_n)
    delta = (cf_risk - risk) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Pre-intervention risk", f"{risk*100:.1f}%")
    m2.metric("Counterfactual risk", f"{cf_risk*100:.1f}%", delta=f"{delta:+.1f} pp")
    m3.metric("Verdict", "Responder ✅" if cf_risk < risk - 0.05 else ("Neutral ➖" if cf_risk <= risk else "Non-responder ⚠️"))

    # Side-by-side trajectory: pre vs post (post applies modulation to last window ramp).
    traj = risk_trajectory(model, static_n, patient_dyn[patient_dyn["timestep"] <= upto], norm)
    comp = go.Figure()
    comp.add_trace(go.Scatter(x=traj["timestep"], y=traj["risk"] * 100, name="Observed twin trajectory", line=dict(color="#0f172a", width=2)))
    comp.add_scatter(x=[upto], y=[cf_risk * 100], mode="markers+text", name="Counterfactual now",
                     marker=dict(color="#16a34a", size=12), text=[f"{cf_risk*100:.0f}%"], textposition="top center")
    comp.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Risk trajectory vs counterfactual", yaxis=dict(title="risk %", range=[0, 100]))
    st.plotly_chart(comp, use_container_width=True)

    with st.expander("Cohort & model notes"):
        st.write(f"Checkpoints: `{CHECKPOINT.name}` · lookback={LOOKBACK} epochs (6h) · horizon=24h · device=cpu")
        st.write("All data synthetic. No PHI. For demonstration only — not a medical device.")


if __name__ == "__main__":
    main()
