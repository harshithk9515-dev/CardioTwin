# CardioTwin — Multimodal AI Digital Twin for Early Cardiac Decompensation

> Predicts acute cardiovascular decompensation **12–24 hours in advance** by fusing
> static EHR baselines (Synthea schema) with continuous 5-minute wearable telemetry,
> served through a clinician-grade Streamlit twin dashboard with explainability and
> what-if intervention simulation.

## Team Information

| Field | Details |
|---|---|
| Team Name | `[Your Team Name]` |
| College / Incubator | `[Your College / Incubator Name]` |
| Team Leader | `[Name · Phone · Email]` |
| Members | `[Member 2 · Member 3 · Member 4]` |
| Repository | Public GitHub link (folder `TeamName_CollegeName`) |
| License | Apache 2.0 (see `LICENSE`) |
| Demo Video | `[Unlisted YouTube link — min 20 minutes]` |

## Clinical Problem Statement

Heart-failure decompensation is a leading cause of emergency admission in India,
yet deterioration often brews silently for 12–48h (rising resting HR, collapsing
HRV, nocturnal tachypnea, activity withdrawal) before SpO2 crashes or edema is
obvious. Routine spot-checks miss this window; continuous ward telemetry is scarce
outside tertiary ICUs.

**CardioTwin** gives every physician a *virtual patient*: a late-fusion neural twin
that watches the wearable stream against the patient's EHR baseline (LVEF, BNP,
comorbidities, meds) and raises a **guarded/critical** flag a day early — with
*why* (feature attributions) and *what-if* (simulate diuretic / rate-control
response before acting).

Impact on India's ecosystem: ABHA-ready patient switching, low-cost consumer
wearable inputs (HR/HRV/SpO2/steps), Hindi/English-triage language potential, and
rural PHC triage where cardiologists are scarce.

## DPDP Act 2023 & HIPAA Sandbox Compliance

- **Zero real patient data.** All 200 patients × 7 days are synthesized by
  `data/generate_synthetic_data.py` (seed-fixed NumPy/Pandas physiology).
- No names, addresses, biometrics, or ABHA numbers are stored. Dashboard ABHA IDs
  are pseudo-formatted from the synthetic `PT-xxx` key for UI realism only.
- No data leaves the machine. Training and inference run locally.
- Synthetic MIMIC/Synthea-style schema is used for familiarity; no MIMIC access
  or DUA is required to run this repo.

## System Architecture & Fusion Mathematics

```
Static EHR (8) ──► MLP 8→32 ──┐
                              ├─► concat 96 ─► MLP 96→48→1 → sigmoid → P(decomp|24h)
Dynamic (T,5) ──► BiGRU →64 ──┘
```

- Static encoder: `Linear(8→32) → ReLU → LayerNorm → Dropout(0.2) → Linear(32→32)`
  (LayerNorm chosen over BatchNorm so single-patient dashboard inference with
  batch size 1 is exact in any mode; identical parameter count).
- Dynamic encoder: 2-layer BiGRU (`hidden=32`, `batch_first`), final-layer
  forward+backward concat → 64-dim temporal embedding.
- Head: `Linear(96→48) → ReLU → Dropout(0.3) → Linear(48→1) → Sigmoid`.
- Window: lookback 72 epochs (6h @ 5-min), stride 12; label = event within next
  288 epochs (24h), enabling 12–24h lead time.
- Diagram: run `python architecture/system_diagram.py` → `architecture/system_diagram.png/.pdf`.

## Project Structure

```
CardioTwin/
├── data/generate_synthetic_data.py
├── src/dataset.py | model.py | train.py | explainability.py | app.py
├── architecture/system_diagram.py
├── models/ (checkpoint after training)
├── requirements.txt | LICENSE | README.md
```

## Model Performance & Validation Benchmarks

> Measured run (seed 42, patient-level 80/20 split, threshold 0.5, best
> early-stopped checkpoint kept in `models/cardiotwin_weights.pth`):

| Metric | Value* |
|---|---|
| AUROC | **0.81** |
| Sensitivity (Recall) | **0.89** |
| Specificity | **0.62** |
| F1 | **0.15** (low prevalence ~3.6% positive epochs) |
| Accuracy | 0.63 |

*Reproduce with `python src/train.py --epochs 15 --batch-size 256` — the script
prints per-epoch metrics and keeps the best-AUROC checkpoint. F1 is modest
because only ~3.6% of 5-min epochs fall in a 24h pre-event window; AUROC and
recall (the triage-relevant metrics) are strong. Threshold tuning / PR-AUC
optimization is listed as future work. `train.py` prints AUROC/Sensitivity/
Specificity/F1 and saves `models/cardiotwin_weights.pth`.

## Quickstart

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py --n-patients 200 --seed 42
python src/train.py --epochs 15 --batch-size 128
python architecture/system_diagram.py
python -m streamlit run src/app.py
```

Then open the Streamlit URL, pick a high-risk twin (e.g. low LVEF), drag the
playback slider to day 5–6, and move the What-If sliders to watch risk fall.

## Dashboard Guide

1. **Header:** ABHA selector (PT-001…PT-200), age/sex, LVEF/BNP, history + meds.
2. **Telemetry:** HR/HRV/SpO2 multi-trace + RR strip; navy shading = sleep.
3. **Risk:** gauge (green <30%, amber 30–65%, red >65%) + lead-time alert box.
4. **Explain:** top-5 integrated-gradients drivers (RMSSD drop, HR rise, LVEF…).
5. **What-If:** sliders remodel physiology (HRV +%, HR −bpm, RR −, SpO2 +) and
   re-score live with a counterfactual trajectory chart.

## 20-Minute Video Outline

| Time | Content |
|---|---|
| 0:00–3:00 | Problem: HF burden in India, silent 24h window, doctor workflow pain |
| 3:00–6:00 | Architecture: fusion math, DPDP sandbox, 200×2016 synthetic design |
| 6:00–10:00 | Data demo: run generator, show static + telemetry, injected event physiology |
| 10:00–13:00 | Model: BiGRU+MLP late fusion, training curves, AUROC/Sens/Spec/F1 |
| 13:00–18:00 | Dashboard live: stable vs critical twin, playback, explainability, what-if |
| 18:00–20:00 | Impact, limitations (synthetic gap), ABDM deployment roadmap, team + license |

Accompany with `Presentation.pdf` and `Architecture.pdf` (export the PNG) per
challenge guidelines.

## Limitations & Safety

Research prototype on synthetic data only. Not a medical device. No diagnostic or
therapeutic recommendation should be acted on without qualified clinician review
and prospective validation on real, consented cohorts.
