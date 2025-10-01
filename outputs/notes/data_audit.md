# Data Audit Notes

## Directory Structure

- **labeled**: 100 files under `mri_dataset_brain_cancer_oc/avec_labels`
  - `avec_labels/cancer`: 50 files
  - `avec_labels/normal`: 50 files
- **unlabeled**: 1406 files under `mri_dataset_brain_cancer_oc/sans_label`

## Sample Summary Statistics

|       |   width |   height |    bytes |
|:------|--------:|---------:|---------:|
| count |      64 |       64 |    64    |
| mean  |     512 |      512 | 25364.7  |
| std   |       0 |        0 |  6925.88 |
| min   |     512 |      512 | 13879    |
| 25%   |     512 |      512 | 20701.2  |
| 50%   |     512 |      512 | 23619.5  |
| 75%   |     512 |      512 | 28154    |
| max   |     512 |      512 | 51406    |

### Image Modes

- RGB

### Unreadable Files

- None detected in sample.

## Observations & Considerations

- No unreadable files detected in the sampled set.
- Sampled images share a single mode: RGB.
- Convert to a single grayscale channel if downstream models expect MRI intensity inputs.
- Most sampled images are 512x512 (64/64); standardize other files to this resolution.
- Normalize pixel intensities to [0, 1] and consider per-image standardization for contrast stability.
- Verify labeled subdirectories align with metadata before splitting into train/val sets.

## Generated Artifacts

- Sample grid: `outputs/figures/sample_grid.png`
- Width histogram: `outputs/figures/width_hist.png`
- Height histogram: `outputs/figures/height_hist.png`
- Aspect ratio histogram: `outputs/figures/aspect_hist.png`
- Sample metadata: `outputs/tables/image_summary.csv`
- Directory summary: `outputs/tables/directory_summary.csv`

## Reproduction

Run `python -m src.data_audit` from the repository root to regenerate these artifacts.

