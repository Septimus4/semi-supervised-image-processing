"""Dimensionality reduction and clustering workflow for Task 3.

This module consumes the standardized feature matrix generated in Task 2
and performs the exploratory analysis requested for Task 3:

* Validate that the standardized features retain zero mean and unit variance
  for both labeled and unlabeled subsets.
* Produce PCA, t-SNE, and UMAP embeddings (persisted for reproducibility).
* Cluster the PCA-reduced representation with K-Means and DBSCAN, sweeping
  hyper-parameters while guarding against label leakage.
* Evaluate the resulting partitions on the labeled subset with ARI/NMI while
  also tracking the global silhouette score for model selection.
* Materialize visualizations, tables, and a short report under the
  ``outputs`` directory tree.

Run the module directly as a script to execute the pipeline::

    python -m src.clustering --features-npz outputs/features/standardized_features.npz

The default CLI arguments mirror the acceptance criteria listed in Task 3 but
can be customized for ad-hoc experimentation. See ``--help`` for details.

Notes (how to read these clusters):
- We reduce dimensionality first (PCA) to denoise and speed up clustering.
    The number of components is chosen to hit a target explained variance, a
    common and principled heuristic students can reason about.
- We use both internal (silhouette) and external (ARI/NMI on labeled subset)
    metrics to select among configurations. External metrics are only computed
    where ground-truth labels exist to avoid leakage.
- Visualisations plot both clusters and known labels on top of 2D embeddings
    (PCA, t-SNE, UMAP) to build intuition—clusters that align with labels are a
    good sign that the features capture class structure.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
import umap  # type: ignore

# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureBundle:
    """Container describing the standardized feature matrix."""

    features: np.ndarray
    paths: np.ndarray
    is_labeled: np.ndarray
    labels: np.ndarray
    scaler_mean: Optional[np.ndarray]
    scaler_scale: Optional[np.ndarray]

    @property
    def labeled_mask(self) -> np.ndarray:
        return self.is_labeled.astype(bool)

    @property
    def unlabeled_mask(self) -> np.ndarray:
        return ~self.labeled_mask


@dataclass(frozen=True)
class EmbeddingResult:
    name: str
    data: np.ndarray
    params: Dict[str, object]


@dataclass(frozen=True)
class ClusteringResult:
    method: str
    space: str
    labels: np.ndarray
    params: Dict[str, object]
    ari: float
    nmi: float
    silhouette: float
    noise_rate: float
    seed: int


@dataclass(frozen=True)
class PCAResults:
    cluster_space: EmbeddingResult
    pca_2d: EmbeddingResult
    pca_tsne_init: EmbeddingResult


# ----------------------------------------------------------------------------
# Loading utilities
# ----------------------------------------------------------------------------


def load_feature_bundle(npz_path: Path) -> FeatureBundle:
    """Load the standardized feature matrix produced during Task 2."""

    if not npz_path.exists():
        raise FileNotFoundError(f"Standardized feature bundle not found: {npz_path}")

    payload = np.load(npz_path, allow_pickle=True)
    required_keys = {"features", "paths", "is_labeled", "labels"}
    missing = sorted(required_keys - set(payload.files))
    if missing:
        raise KeyError(
            "Feature bundle missing required arrays: " + ", ".join(missing)
        )

    features = np.asarray(payload["features"], dtype=np.float32)
    paths = np.asarray(payload["paths"], dtype=str)
    is_labeled = np.asarray(payload["is_labeled"], dtype=bool)
    labels = np.asarray(payload["labels"], dtype=object).astype(str)
    labels = np.where(is_labeled, labels, "")
    scaler_mean = (
        np.asarray(payload["scaler_mean"], dtype=np.float32)
        if "scaler_mean" in payload
        else None
    )
    scaler_scale = (
        np.asarray(payload["scaler_scale"], dtype=np.float32)
        if "scaler_scale" in payload
        else None
    )

    if features.ndim != 2:
        raise ValueError("`features` must be a 2D array of shape [N, D].")
    if paths.shape[0] != features.shape[0]:
        raise ValueError("`paths` must align with the first dimension of `features`.")
    if is_labeled.shape[0] != features.shape[0]:
        raise ValueError(
            "`is_labeled` must align with the first dimension of `features`."
        )
    if labels.shape[0] != features.shape[0]:
        raise ValueError("`labels` must align with the first dimension of `features`.")

    return FeatureBundle(
        features=features,
        paths=paths,
        is_labeled=is_labeled,
        labels=labels,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )


# ----------------------------------------------------------------------------
# Standardization checks
# ----------------------------------------------------------------------------


def summarize_standardization(features: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    subset = features[mask]
    if subset.size == 0:
        return {"mean_abs_mean": float("nan"), "mean_std": float("nan")}
    mean_abs_mean = float(np.mean(np.abs(np.mean(subset, axis=0))))
    mean_std = float(np.mean(np.std(subset, axis=0)))
    return {"mean_abs_mean": mean_abs_mean, "mean_std": mean_std}


def validate_standardization(bundle: FeatureBundle) -> Dict[str, Dict[str, float]]:
    stats = {
        "labeled": summarize_standardization(bundle.features, bundle.labeled_mask),
        "unlabeled": summarize_standardization(bundle.features, bundle.unlabeled_mask),
    }
    if bundle.scaler_mean is not None:
        stats["scaler_mean_abs_max"] = {
            "value": float(np.max(np.abs(bundle.scaler_mean)))
        }
    if bundle.scaler_scale is not None:
        stats["scaler_scale_mean"] = {"value": float(np.mean(bundle.scaler_scale))}
    return stats


# ----------------------------------------------------------------------------
# Dimensionality reduction
# ----------------------------------------------------------------------------


def run_pca(
    features: np.ndarray,
    variance_target: float,
    tsne_dim: int,
    seed: int,
) -> PCAResults:
    n_samples, n_features = features.shape
    max_components = min(n_samples, n_features)
    logging.info(
        "Fitting PCA with up to %s components (samples=%s, features=%s)",
        max_components,
        n_samples,
        n_features,
    )
    full_pca = PCA(n_components=max_components, random_state=seed, svd_solver="full")
    projected = full_pca.fit_transform(features)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    cluster_components = int(np.searchsorted(cumulative, variance_target) + 1)
    cluster_components = max(2, min(cluster_components, projected.shape[1]))
    cluster_space = projected[:, :cluster_components]
    logging.info(
        "Selected %s PCA components to reach %.2f%% explained variance",
        cluster_components,
        cumulative[cluster_components - 1] * 100,
    )
    pca_2d = projected[:, :2]
    tsne_components = min(tsne_dim, projected.shape[1])
    pca_tsne_init = projected[:, :tsne_components]

    return PCAResults(
        cluster_space=EmbeddingResult(
            name="pca_cluster",
            data=cluster_space,
            params={
                "variance_target": variance_target,
                "components": cluster_components,
            },
        ),
        pca_2d=EmbeddingResult(
            name="pca_2d",
            data=pca_2d,
            params={"components": 2},
        ),
        pca_tsne_init=EmbeddingResult(
            name="pca_tsne_init",
            data=pca_tsne_init,
            params={"components": tsne_components},
        ),
    )


def run_tsne(
    base: EmbeddingResult,
    perplexities: Sequence[float],
    seed: int,
) -> List[EmbeddingResult]:
    results: List[EmbeddingResult] = []
    for perplexity in perplexities:
        logging.info("Running t-SNE (perplexity=%s)", perplexity)
        tsne = TSNE(
            n_components=2,
            perplexity=float(perplexity),
            init="pca",
            random_state=seed,
            learning_rate="auto",
            max_iter=1000,
            metric="euclidean",
        )
        embedding = np.asarray(tsne.fit_transform(base.data))
        results.append(
            EmbeddingResult(
                name=f"tsne_perp{int(perplexity)}",
                data=embedding,
                params={"perplexity": float(perplexity), "seed": seed},
            )
        )
    return results


def run_umap(
    base: EmbeddingResult,
    neighbor_values: Sequence[int],
    min_dists: Sequence[float],
    seed: int,
) -> List[EmbeddingResult]:
    results: List[EmbeddingResult] = []
    for n_neighbors in neighbor_values:
        for min_dist in min_dists:
            logging.info("Running UMAP (n_neighbors=%s, min_dist=%.2f)", n_neighbors, min_dist)
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                random_state=seed,
                metric="euclidean",
            )
            embedding = np.asarray(reducer.fit_transform(base.data))
            results.append(
                EmbeddingResult(
                    name=f"umap_nn{int(n_neighbors)}_md{min_dist:.2f}",
                    data=embedding,
                    params={
                        "n_neighbors": int(n_neighbors),
                        "min_dist": float(min_dist),
                        "seed": seed,
                    },
                )
            )
    return results


# ----------------------------------------------------------------------------
# Clustering utilities
# ----------------------------------------------------------------------------


def compute_external_metrics(
    bundle: FeatureBundle,
    predicted: np.ndarray,
) -> Tuple[float, float]:
    mask = bundle.labeled_mask
    if np.count_nonzero(mask) == 0:
        return float("nan"), float("nan")
    true_labels = bundle.labels[mask]
    pred_labels = predicted[mask]
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return float(ari), float(nmi)


def compute_silhouette(space: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    try:
        return float(silhouette_score(space, labels))
    except ValueError:
        return float("nan")


def evaluate_kmeans(
    space: EmbeddingResult,
    bundle: FeatureBundle,
    k_values: Sequence[int],
    n_init: int,
    seed: int,
) -> List[ClusteringResult]:
    results: List[ClusteringResult] = []
    for k in k_values:
        if k < 2:
            continue
        logging.info("Fitting K-Means with k=%s", k)
        model = KMeans(
            n_clusters=int(k),
            n_init=int(n_init),
            random_state=seed,
        )
        labels = model.fit_predict(space.data)
        ari, nmi = compute_external_metrics(bundle, labels)
        silhouette = compute_silhouette(space.data, labels)
        results.append(
            ClusteringResult(
                method="kmeans",
                space=space.name,
                labels=labels,
                params={"k": int(k), "n_init": int(n_init)},
                ari=ari,
                nmi=nmi,
                silhouette=silhouette,
                noise_rate=0.0,
                seed=seed,
            )
        )
    return results


def evaluate_dbscan(
    space: EmbeddingResult,
    bundle: FeatureBundle,
    eps_values: Sequence[float],
    min_samples_values: Sequence[int],
    seed: int,
    scope: str = "all",
) -> List[ClusteringResult]:
    """Evaluate DBSCAN over provided grids and optional scope.

    scope: all | labeled | unlabeled
    If scope != all, DBSCAN is fitted only on the selected subset and labels
    for the other subset are set to -1. Silhouette is computed on the subset
    used for fitting to avoid degeneracy when non-fitted points are marked noise.
    """
    if scope not in {"all", "labeled", "unlabeled"}:
        raise ValueError("scope must be one of: all, labeled, unlabeled")

    if scope == "labeled":
        mask = bundle.labeled_mask
    elif scope == "unlabeled":
        mask = bundle.unlabeled_mask
    else:
        mask = np.ones(space.data.shape[0], dtype=bool)

    sub_space = space.data[mask]
    results: List[ClusteringResult] = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            logging.info(
                "Fitting DBSCAN (scope=%s) with eps=%.3f, min_samples=%s", scope, eps, min_samples
            )
            model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
            sub_labels = model.fit_predict(sub_space)
            labels_full = np.full(space.data.shape[0], -1, dtype=int)
            labels_full[mask] = sub_labels
            ari, nmi = compute_external_metrics(bundle, labels_full)
            silhouette = compute_silhouette(sub_space, sub_labels)
            noise_rate = float(np.mean(sub_labels == -1))
            results.append(
                ClusteringResult(
                    method="dbscan",
                    space=f"{space.name}:{scope}",
                    labels=labels_full,
                    params={"eps": float(eps), "min_samples": int(min_samples), "scope": scope},
                    ari=ari,
                    nmi=nmi,
                    silhouette=silhouette,
                    noise_rate=noise_rate,
                    seed=seed,
                )
            )
    return results

def auto_eps_from_kdistance(space: np.ndarray, min_samples: int, quantile: float = 0.98) -> float:
    """Select eps from the quantile of the k-distance curve (simple elbow heuristic)."""
    nn = NearestNeighbors(n_neighbors=int(min_samples))
    nn.fit(space)
    distances, _ = nn.kneighbors(space)
    kth_dist = np.sort(distances[:, -1])
    idx = int(np.clip(round(quantile * (len(kth_dist) - 1)), 0, len(kth_dist) - 1))
    eps = float(kth_dist[idx])
    return eps


def choose_best(results: Sequence[ClusteringResult]) -> Optional[ClusteringResult]:
    if not results:
        return None
    sorted_results = sorted(
        results,
        key=lambda r: (
            np.nan_to_num(r.ari, nan=-1.0),
            np.nan_to_num(r.nmi, nan=-1.0),
            np.nan_to_num(r.silhouette, nan=-1.0),
        ),
        reverse=True,
    )
    return sorted_results[0]


# ----------------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------------


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_embedding_npz(root: Path, result: EmbeddingResult) -> None:
    ensure_directory(root)
    target = root / f"{result.name}.npz"
    np.savez_compressed(target, embedding=result.data, params_json=json.dumps(result.params, sort_keys=True))


def plot_embedding(
    embedding: EmbeddingResult,
    bundle: FeatureBundle,
    cluster_labels: np.ndarray,
    labeled_title: str,
    output_path: Path,
    dbscan_noise_rate: Optional[float] = None,
) -> None:
    ensure_directory(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    plot_clusters(axes[0], embedding, cluster_labels)
    axes[0].set_title(f"{embedding.name} — clusters")
    plot_labels(axes[1], embedding, bundle)
    axes[1].set_title(labeled_title)
    if dbscan_noise_rate is not None and not np.isnan(dbscan_noise_rate):
        fig.suptitle(f"DBSCAN noise rate: {dbscan_noise_rate:.2%}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_clusters(ax: Axes, embedding: EmbeddingResult, labels: np.ndarray) -> None:
    unique = np.unique(labels)
    for cluster_id in unique:
        mask = labels == cluster_id
        count = int(np.sum(mask))
        if cluster_id == -1:
            label = f"noise (n={count})"
        else:
            label = f"cluster {cluster_id} (n={count})"
        ax.scatter(
            embedding.data[mask, 0],
            embedding.data[mask, 1],
            s=12,
            alpha=0.8,
            label=label,
        )
    ax.legend(loc="best", fontsize="small", frameon=False)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")


def plot_labels(ax: Axes, embedding: EmbeddingResult, bundle: FeatureBundle) -> None:
    unlabeled_mask = bundle.unlabeled_mask
    ax.scatter(
        embedding.data[unlabeled_mask, 0],
        embedding.data[unlabeled_mask, 1],
        s=8,
        color="lightgray",
        alpha=0.4,
        label="unlabeled",
    )
    labeled_mask = bundle.labeled_mask
    labeled_embeddings = embedding.data[labeled_mask]
    labeled_labels = bundle.labels[labeled_mask]
    unique_labels = np.unique(labeled_labels)
    for label in unique_labels:
        mask = labeled_labels == label
        ax.scatter(
            labeled_embeddings[mask, 0],
            labeled_embeddings[mask, 1],
            s=20,
            alpha=0.9,
            label=str(label),
        )
    ax.legend(loc="best", fontsize="small", frameon=False)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")


def plot_k_distance(
    space: EmbeddingResult,
    min_samples: int,
    output_path: Path,
) -> None:
    ensure_directory(output_path.parent)
    logging.info(
        "Rendering k-distance plot for min_samples=%s (points=%s)",
        min_samples,
        space.data.shape[0],
    )
    nn = NearestNeighbors(n_neighbors=int(min_samples))
    nn.fit(space.data)
    distances, _ = nn.kneighbors(space.data)
    kth_dist = np.sort(distances[:, -1])
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(np.arange(kth_dist.size), kth_dist)
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"{min_samples}-NN distance")
    ax.set_title("DBSCAN k-distance curve")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------


def write_metrics_table(results: Sequence[ClusteringResult], output_path: Path) -> pd.DataFrame:
    ensure_directory(output_path.parent)
    rows = []
    for result in results:
        rows.append(
            {
                "method": result.method,
                "space": result.space,
                "params_json": json.dumps(result.params, sort_keys=True),
                "ARI": result.ari,
                "NMI": result.nmi,
                "silhouette": result.silhouette,
                "noise_rate": result.noise_rate,
                "seed": result.seed,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def write_assignments_table(
    bundle: FeatureBundle,
    kmeans_result: ClusteringResult,
    dbscan_result: Optional[ClusteringResult],
    pca_results: PCAResults,
    tsne_choice: EmbeddingResult,
    umap_choice: EmbeddingResult,
    output_path: Path,
) -> pd.DataFrame:
    ensure_directory(output_path.parent)
    data = {
        "path": bundle.paths,
        "cluster_kmeans": kmeans_result.labels,
        "cluster_dbscan": dbscan_result.labels if dbscan_result else np.full_like(
            kmeans_result.labels, fill_value=-1
        ),
        "pca_dim": pca_results.cluster_space.data.shape[1],
        "tsne_id": tsne_choice.name,
        "umap_id": umap_choice.name,
        "is_labeled": bundle.is_labeled,
        "true_label": bundle.labels,
    }
    frame = pd.DataFrame(data)
    frame.to_csv(output_path, index=False)
    return frame


def write_report(
    output_path: Path,
    standardization_stats: Dict[str, Dict[str, float]],
    kmeans_best: ClusteringResult,
    dbscan_best: Optional[ClusteringResult],
) -> None:
    ensure_directory(output_path.parent)
    lines = ["# Clustering Analysis Report", ""]
    lines.append("## Standardization Checks")
    for subset, stats in standardization_stats.items():
        formatted = ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
        lines.append(f"- {subset}: {formatted}")
    lines.append("")

    lines.append("## Best K-Means Configuration")
    lines.append(
        f"- Params: {json.dumps(kmeans_best.params, sort_keys=True)}"
    )
    lines.append(
        f"- ARI={kmeans_best.ari:.4f}, NMI={kmeans_best.nmi:.4f}, silhouette={kmeans_best.silhouette:.4f}"
    )
    lines.append("")

    if dbscan_best is not None:
        lines.append("## Best DBSCAN Configuration")
        lines.append(
            f"- Params: {json.dumps(dbscan_best.params, sort_keys=True)}"
        )
        lines.append(
            f"- ARI={dbscan_best.ari:.4f}, NMI={dbscan_best.nmi:.4f}, silhouette={dbscan_best.silhouette:.4f}, noise_rate={dbscan_best.noise_rate:.4f}"
        )
        lines.append("")
    else:
        lines.append("## Best DBSCAN Configuration")
        lines.append("- No viable DBSCAN configuration identified.")
        lines.append("")

    lines.append("## Notes")
    lines.append(
        "- ARI/NMI computed on labeled subset only; silhouette on full PCA space."
    )
    lines.append("- See tables and figures under `outputs/` for further details.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3 clustering pipeline")
    parser.add_argument(
        "--features-npz",
        type=Path,
        required=True,
        help="Path to the standardized feature bundle (.npz).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for generated artifacts.",
    )
    parser.add_argument(
        "--variance-target",
        type=float,
        default=0.9,
        help="Explained variance threshold for PCA cluster space.",
    )
    parser.add_argument(
        "--tsne-dim",
        type=int,
        default=50,
        help="Number of PCA components fed into t-SNE and UMAP.",
    )
    parser.add_argument(
        "--tsne-perplexities",
        type=float,
        nargs="*",
        default=[10.0, 30.0, 50.0],
        help="Perplexities to evaluate for t-SNE.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        nargs="*",
        default=[15, 30, 50],
        help="n_neighbors values to evaluate for UMAP.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        nargs="*",
        default=[0.0, 0.1],
        help="min_dist values to evaluate for UMAP.",
    )
    parser.add_argument(
        "--kmeans-range",
        type=int,
        nargs="*",
        default=list(range(2, 11)),
        help="Candidate cluster counts for K-Means.",
    )
    parser.add_argument(
        "--kmeans-n-init",
        type=int,
        default=10,
        help="Number of initializations for each K-Means run.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        nargs="*",
        default=[0.5, 0.75, 1.0, 1.25],
        help="Candidate epsilon values for DBSCAN.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        nargs="*",
        default=[5, 10, 15],
        help="Candidate min_samples values for DBSCAN.",
    )
    parser.add_argument(
        "--dbscan-scope",
        type=str,
        default="all",
        choices=["all", "labeled", "unlabeled"],
        help="Run DBSCAN on: all points, labeled subset only, or unlabeled subset only.",
    )
    parser.add_argument(
        "--dbscan-auto",
        action="store_true",
        help="Enable simple auto-selection of eps via k-distance quantile (98th percentile). Overrides --dbscan-eps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible embeddings and clustering.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    bundle = load_feature_bundle(args.features_npz)
    stats = validate_standardization(bundle)
    logging.info("Standardization summary: %s", stats)

    pca_results = run_pca(bundle.features, args.variance_target, args.tsne_dim, args.seed)

    embedding_dir = args.output_root / "features" / "dimensionality_reduction"
    save_embedding_npz(embedding_dir, pca_results.cluster_space)
    save_embedding_npz(embedding_dir, pca_results.pca_2d)
    save_embedding_npz(embedding_dir, pca_results.pca_tsne_init)

    tsne_results = run_tsne(pca_results.pca_tsne_init, args.tsne_perplexities, args.seed)
    for result in tsne_results:
        save_embedding_npz(embedding_dir, result)
    umap_results = run_umap(pca_results.pca_tsne_init, args.umap_neighbors, args.umap_min_dist, args.seed)
    for result in umap_results:
        save_embedding_npz(embedding_dir, result)

    kmeans_results = evaluate_kmeans(
        pca_results.cluster_space,
        bundle,
        args.kmeans_range,
        args.kmeans_n_init,
        args.seed,
    )
    # DBSCAN with optional scope and auto-eps
    dbscan_eps_grid = args.dbscan_eps
    if args.dbscan_auto:
        # choose eps per min_samples using the selected scope mask
        if args.dbscan_scope == "labeled":
            mask = bundle.labeled_mask
        elif args.dbscan_scope == "unlabeled":
            mask = bundle.unlabeled_mask
        else:
            mask = np.ones(pca_results.cluster_space.data.shape[0], dtype=bool)
        sub_space = pca_results.cluster_space.data[mask]

        figures_dir = args.output_root / "figures"
        # Save k-distance plots for each min_samples
        for ms in args.dbscan_min_samples:
            plot_k_distance(
                EmbeddingResult(name=f"pca_cluster:{args.dbscan_scope}", data=sub_space, params={}),
                int(ms),
                figures_dir / f"kdist_plot_{args.dbscan_scope}_ms{int(ms)}.png",
            )
        # Build eps grid around the auto pick for each min_samples
        dbscan_eps_grid = []
        for ms in args.dbscan_min_samples:
            base_eps = auto_eps_from_kdistance(sub_space, int(ms), quantile=0.98)
            dbscan_eps_grid.extend([max(1e-6, base_eps * f) for f in (0.8, 1.0, 1.2)])
        # Deduplicate and sort
        dbscan_eps_grid = sorted(set(float(e) for e in dbscan_eps_grid))

    dbscan_results = evaluate_dbscan(
        pca_results.cluster_space,
        bundle,
        dbscan_eps_grid,
        args.dbscan_min_samples,
        args.seed,
        scope=args.dbscan_scope,
    )
    all_results = kmeans_results + dbscan_results

    metrics_path = args.output_root / "tables" / "metrics_clustering.csv"
    metrics_frame = write_metrics_table(all_results, metrics_path)
    logging.info("Wrote metrics table to %s", metrics_path)

    best_kmeans = choose_best(kmeans_results)
    if best_kmeans is None:
        raise RuntimeError("K-Means sweep produced no viable solutions.")
    best_dbscan = choose_best(dbscan_results)

    assignments_path = args.output_root / "tables" / "cluster_assignments.csv"
    tsne_choice = tsne_results[0] if tsne_results else pca_results.pca_2d
    umap_choice = umap_results[0] if umap_results else pca_results.pca_2d
    assignments_frame = write_assignments_table(
        bundle,
        best_kmeans,
        best_dbscan,
        pca_results,
        tsne_choice,
        umap_choice,
        assignments_path,
    )
    logging.info("Wrote cluster assignments to %s", assignments_path)

    figures_dir = args.output_root / "figures"
    noise_rate = best_dbscan.noise_rate if best_dbscan is not None else None
    plot_embedding(
        pca_results.pca_2d,
        bundle,
        best_kmeans.labels,
        "PCA 2D — labeled overlay",
        figures_dir / "pca2d_clusters.png",
        dbscan_noise_rate=noise_rate,
    )
    if tsne_results:
        plot_embedding(
            tsne_choice,
            bundle,
            best_kmeans.labels,
            "t-SNE 2D — labeled overlay",
            figures_dir / "tsne2d_clusters.png",
            dbscan_noise_rate=noise_rate,
        )
    if umap_results:
        plot_embedding(
            umap_choice,
            bundle,
            best_kmeans.labels,
            "UMAP 2D — labeled overlay",
            figures_dir / "umap2d_clusters.png",
            dbscan_noise_rate=noise_rate,
        )
    if best_dbscan is not None:
        dbscan_min_samples = int(cast(int, best_dbscan.params.get("min_samples", 5)))
        # plot k-distance for the scope used in best dbscan
        scope = str(best_dbscan.params.get("scope", args.dbscan_scope))
        if scope == "labeled":
            mask = bundle.labeled_mask
        elif scope == "unlabeled":
            mask = bundle.unlabeled_mask
        else:
            mask = np.ones(pca_results.cluster_space.data.shape[0], dtype=bool)
        sub_space = EmbeddingResult(
            name=f"pca_cluster:{scope}",
            data=pca_results.cluster_space.data[mask],
            params={},
        )
        plot_k_distance(
            sub_space,
            dbscan_min_samples,
            figures_dir / f"kdist_plot_{scope}.png",
        )

    report_path = args.output_root / "notes" / "clustering_report.md"
    write_report(report_path, stats, best_kmeans, best_dbscan)
    logging.info("Wrote clustering report to %s", report_path)

    logging.info(
        "Artifacts generated: %s rows in assignments, %s rows in metrics",
        assignments_frame.shape[0],
        metrics_frame.shape[0],
    )


if __name__ == "__main__":
    main()
