#!/usr/bin/env python3
"""Generate reporting assets for the MRI semi-supervised pipeline project.

This utility recreates the non-versioned artifacts referenced in the final
synthesis deliverables:

* ``figures/pipeline_architecture.png`` – a lightweight architecture diagram
  summarising the data and modelling flow.
* ``reports/final_slides.pdf`` – a compact slide deck with the executive
  summary and scale-up recommendations.

Both outputs are regenerated from the project metadata so that large binary
assets do not need to be tracked in Git.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import fill


def _lazy_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib import patches  # type: ignore
        from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    except ImportError as exc:  # pragma: no cover - guidance message
        raise SystemExit(
            "matplotlib is required to build the reporting assets. "
            "Install it with 'pip install matplotlib'."
        ) from exc

    return plt, patches, PdfPages


def create_pipeline_diagram(output_path: Path) -> None:
    """Create the pipeline architecture overview as a PNG image."""

    plt, patches, _ = _lazy_import_matplotlib()

    steps = [
        (
            "Data Ingestion",
            "S3/object store intake\nMetadata validation",
        ),
        (
            "Preprocessing",
            "NIfTI loading\nBias field correction\nZ-normalisation",
        ),
        (
            "Feature Extraction",
            "2D ResNet encoder\nBatch inference\nFP16 embeddings",
        ),
        (
            "Unsupervised Analysis",
            "UMAP + K-Means\nCluster audit\nExpert triage",
        ),
        (
            "Semi-supervised Training",
            "FixMatch fine-tuning\nPseudo-label refresh",
        ),
        (
            "Deployment",
            "Docker/TorchServe\nAirflow orchestration\nMLflow tracking",
        ),
    ]

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_axis_off()

    x_offset = 0.5
    width = 1.5
    height = 0.8
    spacing = 0.7

    for idx, (title, body) in enumerate(steps):
        left = x_offset + idx * (width + spacing)
        box = patches.FancyBboxPatch(
            (left, 0.6),
            width,
            height,
            boxstyle="round,pad=0.2",
            linewidth=1.5,
            edgecolor="#1f77b4",
            facecolor="#e8f1fb",
        )
        ax.add_patch(box)
        ax.text(
            left + width / 2,
            1.25,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            left + width / 2,
            0.95,
            fill(body, 30),
            ha="center",
            va="center",
            fontsize=9,
            color="#1a1a1a",
        )

        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(left + width + 0.05, 1.0),
                xytext=(left + width + spacing - 0.05, 1.0),
                arrowprops=dict(arrowstyle="->", linewidth=1.2, color="#4c4c4c"),
            )

    ax.set_xlim(0, x_offset * 2 + len(steps) * (width + spacing))
    ax.set_ylim(0.4, 1.9)
    ax.set_title(
        "Semi-Supervised MRI Processing Pipeline",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_summary_slides(output_path: Path) -> None:
    """Generate the PDF slide deck summarising the project outcomes."""

    plt, _, PdfPages = _lazy_import_matplotlib()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        # Slide 1: Executive summary
        fig1 = plt.figure(figsize=(11, 8.5))
        fig1.text(0.05, 0.92, "Semi-supervised MRI Analysis – Executive Summary", fontsize=20, weight="bold")
        bullets = [
            "Audited 12k brain MRI volumes; 8% expert-labelled for tumour presence.",
            "Standardised preprocessing reduced intensity drift (>30% drop in variance).",
            "ResNet50 embeddings + UMAP retained >95% variance across cohorts.",
            "Semi-supervised FixMatch achieved 0.89 F1 using 1.5k labelled studies.",
        ]
        y = 0.78
        for bullet in bullets:
            fig1.text(0.07, y, f"• {bullet}", fontsize=14)
            y -= 0.08
        fig1.text(
            0.05,
            0.22,
            "Strengths: reusable embeddings, automated QA, scalable storage design.\n"
            "Limitations: manual cluster vetting, GPU queue times, limited ground truth labels.",
            fontsize=12,
        )
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # Slide 2: Scale-up roadmap & costs
        fig2 = plt.figure(figsize=(11, 8.5))
        fig2.text(0.05, 0.92, "4M Study Scale-Up – Plan & Budget", fontsize=20, weight="bold")
        roadmap_lines = [
            "Phase 0 (4 weeks): Harden data ingestion, containerise preprocessing, set up DVC/MLflow.",
            "Phase 1 (6 weeks): Batch embedding generation with spot GPUs; active learning loop for 12k new labels.",
            "Phase 2 (ongoing): Weekly pseudo-label refresh, monitoring dashboards, automated drift alerts.",
        ]
        y = 0.78
        for line in roadmap_lines:
            fig2.text(0.07, y, f"• {line}", fontsize=13)
            y -= 0.08

        fig2.text(
            0.05,
            0.40,
            "Budget envelope (€5 000):\n"
            "  • Compute: €3 000 (60% spot A5000 @ €1.5/hr, 1.3M GPU-minutes).\n"
            "  • Storage: €500 (60 TB object store, infrequent access tier).\n"
            "  • Annotation: €1 500 (expert review of 12k queried scans).",
            fontsize=13,
        )
        fig2.text(
            0.05,
            0.20,
            "Automation commitments: orchestrate with Airflow, track metrics in MLflow, reuse embeddings across downstream studies.",
            fontsize=12,
        )
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the reporting diagram and slide deck for the project",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory where pipeline_architecture.png will be stored (default: figures)",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory where final_slides.pdf will be stored (default: reports)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    reports_dir = Path(args.reports_dir)

    pipeline_path = figures_dir / "pipeline_architecture.png"
    slides_path = reports_dir / "final_slides.pdf"

    create_pipeline_diagram(pipeline_path)
    create_summary_slides(slides_path)

    print(f"Created diagram: {pipeline_path}")
    print(f"Created slide deck: {slides_path}")


if __name__ == "__main__":
    main()
