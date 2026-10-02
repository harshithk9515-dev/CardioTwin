"""Render the CardioTwin system architecture diagram.

Prefers `graphviz` if installed, otherwise falls back to pure matplotlib
(no system dependencies). Output: architecture/system_diagram.png
(and .pdf for the submission) plus a print-friendly ASCII summary.

Run:
    python architecture/system_diagram.py
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def _render_matplotlib(png: Path, pdf: Path) -> None:
    """Draw a layered block diagram with matplotlib only."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    layers = [
        ("1 · DATA INGESTION\n(Synthetic Sandbox)", "#dbeafe", [
            "Synthea-schema static EHR (200 pts)",
            "Wearable telemetry 5-min × 2016",
            "HR · HRV RMSSD · RR · SpO2 · Steps",
        ]),
        ("2 · DPDP-COMPLIANT PREPROCESSING", "#fef3c7", [
            "Zero real PHI · seed-fixed synthesis",
            "Patient-level 80/20 split (no leakage)",
            "z-score (dynamic) + min-max (static)",
            "Sliding windows: 72 steps = 6h lookback",
        ]),
        ("3 · MULTIMODAL FUSION (CardioTwinNet)", "#dcfce7", [
            "Static MLP 8→32  +  BiGRU 5→64",
            "Late fusion concat → 96-dim twin state",
            "Head 96→48→1 + sigmoid = P(decomp|24h)",
            "BCELoss + Adam 1e-3 · AUROC/F1 monitoring",
        ]),
        ("4 · CLINICIAN DELIVERY", "#fee2e2", [
            "Streamlit twin dashboard + ABHA selector",
            "Live playback · risk gauge · lead-time alert",
            "Integrated-gradients explainability",
            "What-If counterfactual re-scoring",
        ]),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, len(layers) * 2.4 + 1.2)
    ax.axis("off")
    fig.suptitle("CardioTwin — System Architecture (Digital Twin for Cardiac Decompensation)", fontsize=13, fontweight="bold", y=0.97)

    y_top = len(layers) * 2.4 + 0.4
    for i, (title, color, bullets) in enumerate(layers):
        y = y_top - i * 2.4
        box = FancyBboxPatch((0.4, y - 1.9), 11.2, 2.0, boxstyle="round,pad=0.08", facecolor=color, edgecolor="#0f172a", linewidth=1.2)
        ax.add_patch(box)
        ax.text(6, y - 0.25, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(6, y - 1.05, "   •   ".join(bullets), ha="center", va="center", fontsize=7.6, wrap=True)
        if i < len(layers) - 1:
            ax.annotate("", xy=(6, y - 1.95), xytext=(6, y - 2.35), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#0f172a"))

    ax.text(6, 0.25, "Forward horizon: 24h  •  Lead goal: 12–24h  •  All data synthetic (DPDP Act 2023 / HIPAA sandbox)", ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[diagram] wrote {png}")
    print(f"[diagram] wrote {pdf}")


def _render_graphviz(png: Path, pdf: Path) -> bool:
    """Try a richer graphviz rendering. Returns True on success."""
    try:
        from graphviz import Digraph
    except ImportError:
        return False
    try:
        dot = Digraph("CardioTwin", format="png")
        dot.attr(rankdir="TB", fontsize="11", labelloc="t", label="CardioTwin — System Architecture")
        dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
        dot.node("ingest", "1 · Data Ingestion\nSynthea static + 5-min wearables", fillcolor="#dbeafe")
        dot.node("prep", "2 · DPDP Preprocessing\nnormalize + window (72) + split", fillcolor="#fef3c7")
        dot.node("model", "3 · CardioTwinNet Fusion\nMLP(32) + BiGRU(64) → 96 → p", fillcolor="#dcfce7")
        dot.node("serve", "4 · Clinician Delivery\nStreamlit + explain + what-if", fillcolor="#fee2e2")
        dot.edges([("ingest", "prep"), ("prep", "model"), ("model", "serve")])
        dot.render(str(png.with_suffix("")), cleanup=True)
        # Also export PDF.
        dot.format = "pdf"
        dot.render(str(pdf.with_suffix("")), cleanup=True)
        print(f"[diagram] wrote {png} (graphviz)")
        return True
    except Exception as exc:  # graphviz binary often missing on Windows
        print(f"[diagram] graphviz unavailable ({exc}), using matplotlib fallback")
        return False


def main() -> None:
    """Entry point."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "system_diagram.png"
    pdf = OUT_DIR / "system_diagram.pdf"
    if not _render_graphviz(png, pdf):
        _render_matplotlib(png, pdf)


if __name__ == "__main__":
    main()
