# Feature Extraction Summary

- Backbone: torchvision.resnet18 (ResNet18_Weights.IMAGENET1K_V1)
- Layer: global average pooled features (512-D)
- Input spec: resize 256 → center crop 224, RGB conversion, ImageNet normalization
- Batch size: 32
- Device: cuda
- Total images processed: 1506
- Failed decodes: 0
- Mean per-image latency (s): 0.0026
- Median per-image latency (s): 0.0026

## Sanity Check Statistics

- Mean of |dimension means|: 0.884986
- Mean of dimension standard deviations: 0.581787

## Nearest Neighbor Spot Check

| Query | Neighbor | Cosine |
| --- | --- | --- |
| sans_label/04dd2b78-0aa3-4364-b762-21a6e71d3658.jpg | avec_labels/normal/b9b56b95-43de-4c35-8362-400e9271af71.jpg | 1.0000 |
| sans_label/b0573368-2b4a-4bb5-a2f2-fd4c106cbd6a.jpg | sans_label/2e671412-32fd-4136-9844-f7053b0696c9.jpg | 0.9308 |
| sans_label/db87f1fa-f718-4bf4-821f-689d5d6e72a1.jpg | sans_label/33a6d838-67db-4ad2-a67a-b4d4b641ec8a.jpg | 0.8776 |
| sans_label/a694549b-164f-4ab6-8c6e-821661302d50.jpg | sans_label/dd2eb4e7-90a3-4801-ab47-b94dee64eab2.jpg | 0.9454 |
| sans_label/63ca22a2-f45f-44d9-9aae-00f4d9bd795a.jpg | sans_label/5434d6b2-b0dd-42d7-832c-ae465e64ddc2.jpg | 1.0000 |
| sans_label/65664115-d6c0-4d61-8708-1e8e36f1c193.jpg | sans_label/bb7be2dc-ad86-4c83-8350-c711317d5fb0.jpg | 0.9396 |
| sans_label/05e808fc-a00c-4f1e-9682-1f73915f6902.jpg | sans_label/ecd494ef-9aed-48e7-a2a3-99c91b46bfed.jpg | 1.0000 |
| sans_label/c38f8369-c2b1-41e3-a5ab-fd4c0cdafd3e.jpg | avec_labels/cancer/c6acea0d-0c31-4ce8-b0f8-b4a610a9477d.jpg | 0.8950 |

## Decode Failures

None
