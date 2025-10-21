# Evaluation and Operating Points

This guide explains the evaluation artifacts, how to read them, and how to
choose a decision threshold to achieve high recall (sensitivity) with
controlled false positives.

## Artifacts produced

- Confusion matrices (test set):
  - `outputs/figures/confusion_matrix_baseline.png` (argmax)
  - `outputs/figures/confusion_matrix_baseline_thresholded.png`
  - `outputs/figures/confusion_matrix_semi.png` (argmax)
  - `outputs/figures/confusion_matrix_semi_thresholded.png`
- Curves:
  - ROC curves: `outputs/figures/roc_curves.png`
  - Precision–Recall (PR) curves: `outputs/figures/pr_curves.png`
- Summary tables:
  - Thresholded summary: `outputs/tables/results_comparison.csv`
  - Detailed metrics (TP, FP, TN, FN, TPR, FPR, etc.) for both decisions: 
    `outputs/tables/results_comparison_detailed.csv`
- Aggregate comparison plot:
  - Grouped bars of TPR/FPR/TNR/precision/accuracy: 
    `outputs/figures/metrics_comparison.png`

## Threshold selection for high recall

When you pass `--target-recall`, the training script selects a decision
threshold on the validation set and applies it to the test set.

- The selection rule chooses the highest threshold whose validation recall
  meets or exceeds `--target-recall`. This minimizes false positives while
  satisfying the recall constraint.
- If perfect recall (1.0) is unattainable on validation, the script will fall
  back to the smallest threshold (maximizing recall) and record the threshold
  used in the results.

You can compare the following decision modes:
- Argmax — standard predicted class (no threshold tuning)
- Thresholded — tuned for target recall

Both are included in `results_comparison_detailed.csv` and visualized in
`metrics_comparison.png`.

## Threshold sweep utility

Use the `threshold_sweep` tool to compute confusion-derived metrics across a
range of thresholds on the test set, then pick the best operating point for
your clinical needs.

Run (semi-supervised model by default):
```bash
source .venv/bin/activate
python -m src.threshold_sweep \
  --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
  --model semi \
  --device cuda \
  --num-workers 8
```
Outputs:
- CSV with per-threshold metrics: `outputs/tables/threshold_sweep_semi.csv`
- A JSON summary is printed to stdout with the highest threshold that achieves
  TPR ≈ 1.0, including TP, FP, TN, FN, FPR, precision, and accuracy.

Example (from a sample run; values vary by split):
- Best threshold for TPR=1.0: ~0.068
- Confusion at this threshold: TP=10, FP=2, TN=8, FN=0
- FPR=0.20, Precision≈0.83, Accuracy=0.90

## Interpreting ROC and PR curves

- ROC curve (TPR vs FPR) summarizes ranking performance independent of a single
  threshold. Higher AUC indicates better separability. Prefer the model whose
  curve dominates (up/left).
- PR curve is often more informative under class imbalance and when recall is
  the priority; higher Average Precision (AP) indicates better precision across
  recall levels.

## Choosing an operating point

- If you need “catch everything,” set `--target-recall 1.0` and use the
  threshold selected by the validation procedure, or take the sweep’s highest
  threshold achieving TPR=1.0 on test.
- If a small number of misses is acceptable to reduce false positives, try
  `--target-recall 0.98` or `0.995`, or pick a higher threshold from the sweep
  that reduces FPR/raises precision while maintaining acceptable recall.

Record the final threshold alongside the checkpoint if you plan to deploy; we
can add a small inference script that uses the saved threshold for consistent
behavior.
