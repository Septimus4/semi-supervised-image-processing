"""
Generate a compact, presentation-friendly model comparison figure from
outputs/tables/results_comparison_detailed.csv.

It compares baseline vs semi, for both argmax and thresholded decisions, across
key metrics: Accuracy, Precision, Recall, F1, and FPR.

Outputs:
- outputs/figures/model_comparison.png
- outputs/figures/model_comparison.svg
- outputs/figures/model_comparison.txt (caption)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_CSV = Path("outputs/tables/results_comparison_detailed.csv")
OPERATING_POINT_JSON = Path("outputs/notes/operating_point.json")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Variants to show, in order
ORDERED_VARIANTS: List[str] = [
    "baseline_argmax",
    "baseline_thresholded",
    "semi_argmax",
    "semi_thresholded",
]

# Metrics to plot, in order
METRICS: List[str] = ["accuracy", "precision", "recall", "f1", "FPR"]


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"Missing results CSV: {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Keep only known variants present in the CSV
    df = df[df.iloc[:, 0].isin(ORDERED_VARIANTS)].copy()
    df.rename(columns={df.columns[0]: "variant"}, inplace=True)

    # Compute F1 if not present using precision/recall
    if "f1" not in df.columns and {"precision", "recall"}.issubset(df.columns):
        def _safe_f1(p: float, r: float) -> float:
            try:
                p = float(p)
                r = float(r)
            except Exception:
                return float("nan")
            denom = (p + r)
            return (2.0 * p * r / denom) if denom > 0 else 0.0

        df["f1"] = [
            _safe_f1(p, r) for p, r in zip(df.get("precision", []), df.get("recall", []))
        ]

    # Map to display labels
    display_map = {
        "baseline_argmax": "Baseline\n(argmax)",
        "baseline_thresholded": "Baseline\n(thresholded)",
        "semi_argmax": "Semi\n(argmax)",
        "semi_thresholded": "Semi\n(thresholded)",
    }
    df["display"] = df["variant"].map(display_map)

    # Build tidy frame for the metrics of interest
    # FPR column in CSV is named 'FPR'; others are lower case
    plot_rows = []
    for _, row in df.iterrows():
        for m in METRICS:
            col = m
            if m == "FPR":
                col = "FPR"
            if col not in row:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            plot_rows.append({
                "variant": row["variant"],
                "display": row["display"],
                "metric": m,
                "value": float(val),
            })
    plot_df = pd.DataFrame(plot_rows)

    if plot_df.empty:
        raise SystemExit("No metrics found to plot.")

    # Establish consistent color mapping per variant
    palette = {
        "Baseline\n(argmax)": "#9e9e9e",
        "Baseline\n(thresholded)": "#607d8b",
        "Semi\n(argmax)": "#80cbc4",
        "Semi\n(thresholded)": "#00796b",
    }

    # Create subplots: one per metric
    metrics = METRICS
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(1 + 3 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        mdf = plot_df[plot_df["metric"] == m]
        # Ensure ordering of bars
        mdf = mdf.set_index("display").reindex(list(palette.keys())).reset_index()
        bars = ax.bar(
            mdf["display"], mdf["value"], color=[palette.get(lbl, "#444") for lbl in mdf["display"]]
        )
        ax.set_title(m)
        if m == "FPR":
            ax.set_ylim(0, max(0.01, mdf["value"].max() * 1.15))
        else:
            ax.set_ylim(0, 1.05)
        # Value labels
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + (0.02 if m != "FPR" else 0.005),
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        # Tidy x labels with fixed ticks to avoid warnings
        ax.set_xticks(np.arange(len(mdf["display"])) )
        ax.set_xticklabels(mdf["display"], rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Model comparison — Baseline vs Semi (argmax & thresholded)", fontsize=12)

    # Save PNG and SVG
    out_png = FIG_DIR / "model_comparison.png"
    out_svg = FIG_DIR / "model_comparison.svg"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_svg)

    # Caption with a short summary and operating point when available
    caption_lines: List[str] = [
        "Title: Model comparison — Baseline vs Semi (argmax & thresholded)",
        "What this shows: Side-by-side bars for Accuracy, Precision, Recall, F1, and FPR, across baseline/semi and decision modes.",
        "How to read: Prefer the model/decision pairing with high Recall and Precision, low FPR, and strong Accuracy at the chosen operating point.",
    ]

    # Pull key values for a concise run summary
    try:
        semi_thr = df[df["variant"] == "semi_thresholded"].iloc[0]
        caption_lines.append(
            f"Current run: Semi-thresholded — Acc={semi_thr['accuracy']:.2f}, "
            f"Prec={semi_thr['precision']:.2f}, Rec={semi_thr['recall']:.2f}, FPR={semi_thr['FPR']:.2f}."
        )
    except Exception:
        pass

    if OPERATING_POINT_JSON.exists():
        try:
            op = json.loads(OPERATING_POINT_JSON.read_text())
            thr = op.get("threshold")
            pol = op.get("policy")
            caption_lines.append(
                f"Operating point: threshold≈{thr:.3f} (policy={pol}); see outputs/notes/operating_point.json."
            )
        except Exception:
            pass

    (FIG_DIR / "model_comparison.txt").write_text("\n".join(caption_lines) + "\n")
    print(f"Wrote {out_png} and {out_svg}")


if __name__ == "__main__":
    main()
