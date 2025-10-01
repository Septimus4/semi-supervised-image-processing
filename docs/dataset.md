# Dataset

Place the dataset at the repository root under `mri_dataset_brain_cancer_oc/`.

```
mri_dataset_brain_cancer_oc/
├── avec_labels/        # labeled images grouped by class
└── sans_label/         # unlabeled study images
```

- `avec_labels/` is expected to follow an ImageFolder layout: one subfolder per class, images inside.
- `sans_label/` is a flat directory of images without labels.

Tip: To quickly check counts:
```bash
find mri_dataset_brain_cancer_oc/avec_labels -type f | wc -l
find mri_dataset_brain_cancer_oc/sans_label -type f | wc -l
```
