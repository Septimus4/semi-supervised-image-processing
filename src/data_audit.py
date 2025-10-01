"""Data audit utilities for the MRI brain cancer dataset.

This module scans the dataset directory, samples image files, extracts
metadata, and produces the artifacts required for the exploratory audit:

- CSV containing per-file attributes for the sampled images.
- Directory summary CSV enumerating the labeled and unlabeled buckets.
- Diagnostic plots for image sizes and aspect ratios (and intensity if applicable).
- Sample image grid summarizing the sampled subset visually.
- Markdown notes capturing high-level observations.

Run directly as a script to regenerate all assets:

```
python -m src.data_audit --data-dir mri_dataset_brain_cancer_oc --sample-size 64
```
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# Paths relative to repository root
DEFAULT_DATA_DIR = Path("mri_dataset_brain_cancer_oc")
OUTPUT_TABLE_DIR = Path("outputs/tables")
OUTPUT_FIGURE_DIR = Path("outputs/figures")
OUTPUT_NOTE_PATH = Path("outputs/notes/data_audit.md")
SAMPLE_METADATA_CSV = OUTPUT_TABLE_DIR / "image_summary.csv"
DIRECTORY_SUMMARY_CSV = OUTPUT_TABLE_DIR / "directory_summary.csv"


@dataclass
class FileRecord:
    """Container for sampled file metadata."""

    bucket: str
    relative_path: str
    absolute_path: Path
    width: Optional[int]
    height: Optional[int]
    mode: Optional[str]
    image_format: Optional[str]
    byte_size: int
    readable: bool

    @property
    def aspect_ratio(self) -> Optional[float]:
        if self.width and self.height:
            return self.width / self.height
        return None


BUCKET_LABELS = {
    "avec_labels": "labeled",
    "sans_label": "unlabeled",
}


def discover_files(data_dir: Path) -> Dict[str, List[Path]]:
    """Return a mapping from bucket name to discovered files."""
    inventory: Dict[str, List[Path]] = defaultdict(list)
    for bucket_dir in BUCKET_LABELS:
        bucket_path = data_dir / bucket_dir
        if not bucket_path.exists():
            raise FileNotFoundError(f"Missing expected bucket directory: {bucket_path}")
        for file_path in sorted(bucket_path.rglob("*")):
            if file_path.is_file():
                inventory[BUCKET_LABELS[bucket_dir]].append(file_path)
    return inventory


def summarize_directory_tree(file_inventory: Dict[str, List[Path]], base_dir: Path) -> pd.DataFrame:
    """Generate a directory summary DataFrame."""
    records = []
    for bucket, files in file_inventory.items():
        counter: Counter[str] = Counter()
        for path in files:
            relative = path.relative_to(base_dir)
            parts = relative.parts
            if len(parts) > 2:
                subdir = parts[1]
            else:
                subdir = "(root)"
            counter[subdir] += 1
        if not counter:
            counter["(root)"] = 0
        for subdir, count in sorted(counter.items()):
            records.append(
                {
                    "bucket": bucket,
                    "subdirectory": subdir,
                    "file_count": count,
                }
            )
    summary_df = pd.DataFrame(records).sort_values(["bucket", "subdirectory"]).reset_index(drop=True)
    return summary_df


def sample_files(file_inventory: Dict[str, List[Path]], sample_size: int, seed: int = 42) -> List[Path]:
    """Sample files across buckets with a deterministic seed."""
    all_files: List[Path] = []
    for files in file_inventory.values():
        all_files.extend(files)
    if not all_files:
        return []
    sample_size = min(sample_size, len(all_files))
    rng = random.Random(seed)
    return rng.sample(all_files, sample_size)


def extract_metadata(sampled_paths: Iterable[Path], base_dir: Path) -> List[FileRecord]:
    """Extract metadata for the sampled files."""
    records: List[FileRecord] = []
    for path in sampled_paths:
        relative = path.relative_to(base_dir)
        bucket = BUCKET_LABELS.get(relative.parts[0], relative.parts[0])
        byte_size = path.stat().st_size
        width = height = None
        mode = image_format = None
        readable = True
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                image_format = img.format
        except (UnidentifiedImageError, OSError):
            readable = False
        records.append(
            FileRecord(
                bucket=bucket,
                relative_path=str(relative),
                absolute_path=path,
                width=width,
                height=height,
                mode=mode,
                image_format=image_format,
                byte_size=byte_size,
                readable=readable,
            )
        )
    return records


def records_to_dataframe(records: List[FileRecord]) -> pd.DataFrame:
    """Convert file records into a tidy DataFrame."""
    data = [
        {
            "bucket": record.bucket,
            "path": record.relative_path,
            "width": record.width,
            "height": record.height,
            "mode": record.mode,
            "format": record.image_format,
            "bytes": record.byte_size,
            "readable": record.readable,
            "aspect_ratio": record.aspect_ratio,
        }
        for record in records
    ]
    df = pd.DataFrame(data)
    return df.sort_values(["bucket", "path"]).reset_index(drop=True)


def ensure_output_dirs() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_sample_grid(records: List[FileRecord], output_path: Path) -> None:
    readable_records = [r for r in records if r.readable]
    if not readable_records:
        return
    cols = min(8, len(readable_records))
    rows = math.ceil(len(readable_records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.array(axes).reshape(rows, cols)
    axes_flat = axes.flatten()
    for ax, record in zip(axes_flat, readable_records):
        with Image.open(record.absolute_path) as img:
            display_img = img.convert("RGB") if img.mode != "RGB" else img
            ax.imshow(display_img)
        ax.set_title(Path(record.relative_path).name, fontsize=8)
        ax.axis("off")
    for ax in axes_flat[len(readable_records) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_histograms(df: pd.DataFrame, base_dir: Path) -> None:
    numeric_df = df.dropna(subset=["width", "height", "bytes", "aspect_ratio"])
    if numeric_df.empty:
        return
    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots()
    ax.hist(numeric_df["width"], bins=20, color="#3b7ddd")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Sample Width Distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE_DIR / "width_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(numeric_df["height"], bins=20, color="#da5b3b")
    ax.set_xlabel("Height (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Sample Height Distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE_DIR / "height_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(numeric_df["aspect_ratio"], bins=20, color="#5bda3b")
    ax.set_xlabel("Aspect Ratio (W/H)")
    ax.set_ylabel("Count")
    ax.set_title("Sample Aspect Ratio Distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE_DIR / "aspect_hist.png", dpi=200)
    plt.close(fig)

    grayscale_modes = {"1", "L", "LA", "I", "F"}
    grayscale_records = [r for _, r in df.iterrows() if r["mode"] in grayscale_modes]
    if grayscale_records:
        fig, ax = plt.subplots()
        for record in grayscale_records:
            with Image.open(base_dir / record["path"]) as img:
                arr = np.array(img.convert("L")).ravel()
            ax.hist(arr, bins=30, alpha=0.4, label=Path(record["path"]).stem)
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.set_title("Grayscale Intensity Distribution")
        if len(grayscale_records) <= 10:
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(OUTPUT_FIGURE_DIR / "intensity_hist.png", dpi=200)
        plt.close(fig)


def generate_observations(df: pd.DataFrame, dir_summary: pd.DataFrame) -> List[str]:
    observations: List[str] = []
    unreadable_count = (~df["readable"]).sum()
    if unreadable_count:
        observations.append(f"Detected {unreadable_count} unreadable files in the sample.")
    else:
        observations.append("No unreadable files detected in the sampled set.")

    modes = df["mode"].dropna().unique()
    if len(modes) == 1:
        mode = modes[0]
        observations.append(f"Sampled images share a single mode: {mode}.")
        if mode == "RGB":
            observations.append("Convert to a single grayscale channel if downstream models expect MRI intensity inputs.")
        elif mode in {"L", "1"}:
            observations.append("Grayscale inputs align with typical MRI pipelines; ensure channel handling stays consistent.")
    elif len(modes) > 1:
        observations.append(f"Mixed image modes detected ({', '.join(modes)}); harmonize channels before training.")

    size_counts = df.dropna(subset=["width", "height"]).groupby(["width", "height"]).size()
    if not size_counts.empty:
        dominant_size, dominant_count = size_counts.idxmax(), size_counts.max()
        total = len(df)
        width, height = dominant_size
        observations.append(
            f"Most sampled images are {width}x{height} ({dominant_count}/{total}); standardize other files to this resolution."
        )

    observations.append("Normalize pixel intensities to [0, 1] and consider per-image standardization for contrast stability.")

    labeled_rows = dir_summary[dir_summary["bucket"] == "labeled"]["file_count"].sum()
    if labeled_rows:
        observations.append("Verify labeled subdirectories align with metadata before splitting into train/val sets.")
    return observations


def write_markdown_report(
    df: pd.DataFrame,
    dir_summary: pd.DataFrame,
    observations: List[str],
    data_dir: Path,
    output_path: Path,
) -> None:
    summary_stats = (
        df[["width", "height", "bytes"]]
        .dropna()
        .astype(int)
        .describe()
        .round(2)
    )
    markdown_lines = ["# Data Audit Notes", ""]

    markdown_lines.append("## Directory Structure")
    markdown_lines.append("")

    for bucket in ["labeled", "unlabeled"]:
        bucket_rows = dir_summary[dir_summary["bucket"] == bucket]
        total = int(bucket_rows["file_count"].sum())
        bucket_dir = data_dir / ("avec_labels" if bucket == "labeled" else "sans_label")
        markdown_lines.append(f"- **{bucket}**: {total} files under `{bucket_dir}`")
        for _, row in bucket_rows.iterrows():
            if row["subdirectory"] != "(root)":
                prefix = "avec_labels" if bucket == "labeled" else "sans_label"
                markdown_lines.append(f"  - `{prefix}/{row['subdirectory']}`: {int(row['file_count'])} files")
    markdown_lines.append("")

    if not summary_stats.empty:
        markdown_lines.append("## Sample Summary Statistics")
        markdown_lines.append("")
        markdown_lines.append(summary_stats.to_markdown())
        markdown_lines.append("")

    modes = ", ".join(sorted(df["mode"].dropna().unique())) or "None"
    markdown_lines.append("### Image Modes")
    markdown_lines.append("")
    markdown_lines.append(f"- {modes}")
    markdown_lines.append("")

    unreadable = df[~df["readable"]]
    markdown_lines.append("### Unreadable Files")
    markdown_lines.append("")
    if unreadable.empty:
        markdown_lines.append("- None detected in sample.")
    else:
        for path in unreadable["path"].tolist():
            markdown_lines.append(f"- {path}")
    markdown_lines.append("")

    markdown_lines.append("## Observations & Considerations")
    markdown_lines.append("")
    for observation in observations:
        markdown_lines.append(f"- {observation}")
    markdown_lines.append("")

    markdown_lines.append("## Generated Artifacts")
    markdown_lines.append("")
    markdown_lines.extend(
        [
            "- Sample grid: `outputs/figures/sample_grid.png`",
            "- Width histogram: `outputs/figures/width_hist.png`",
            "- Height histogram: `outputs/figures/height_hist.png`",
            "- Aspect ratio histogram: `outputs/figures/aspect_hist.png`",
        ]
    )
    if (OUTPUT_FIGURE_DIR / "intensity_hist.png").exists():
        markdown_lines.append("- Intensity histogram: `outputs/figures/intensity_hist.png`")
    markdown_lines.append("- Sample metadata: `outputs/tables/image_summary.csv`")
    markdown_lines.append("- Directory summary: `outputs/tables/directory_summary.csv`")
    markdown_lines.append("")

    markdown_lines.append("## Reproduction")
    markdown_lines.append("")
    markdown_lines.append("Run `python -m src.data_audit` from the repository root to regenerate these artifacts.")
    markdown_lines.append("")

    output_path.write_text("\n".join(markdown_lines) + "\n")


def audit_dataset(data_dir: Path, sample_size: int, seed: int) -> None:
    ensure_output_dirs()
    file_inventory = discover_files(data_dir)
    dir_summary = summarize_directory_tree(file_inventory, data_dir)
    dir_summary.to_csv(DIRECTORY_SUMMARY_CSV, index=False)

    sampled_paths = sample_files(file_inventory, sample_size, seed=seed)
    records = extract_metadata(sampled_paths, data_dir)
    df = records_to_dataframe(records)
    df.to_csv(SAMPLE_METADATA_CSV, index=False)

    save_sample_grid(records, OUTPUT_FIGURE_DIR / "sample_grid.png")
    save_histograms(df, data_dir)

    observations = generate_observations(df, dir_summary)
    write_markdown_report(df, dir_summary, observations, data_dir, OUTPUT_NOTE_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRI dataset audit script")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=64,
        help="Number of files to sample across both buckets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    audit_dataset(args.data_dir, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
