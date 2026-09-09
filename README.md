# swin-segmentation-pipeline

Swin **semantic-segmentation** pipeline for DIMER — NATIVE ml-worker vision
pipeline umbrella and specification.

## Upstream alignment

This pipeline corresponds to the **Semantic Segmentation** downstream task of
the original Microsoft Swin Transformer project.

- **Upstream project:** [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- **Official semantic-segmentation implementation:** [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)
- **Upstream task:** Semantic Segmentation
- **Canonical benchmark:** ADE20K
- **Reference architecture:** Swin backbone + UPerNet semantic-segmentation head
- **Pipeline task boundary:** image → per-pixel class mask
- **Primary metrics:** mIoU and per-class IoU
- **Planned implementation:** Swin + UPerNet through an allowlisted, digest-pinned model catalog

This repository is **not** an instance-segmentation pipeline. Instance masks are
part of the upstream Swin detection lineage through architectures such as Mask
R-CNN; this repo is reserved for the dedicated per-pixel semantic-segmentation
task.

The canonical Swin pipeline family is:

1. **Image Classification** → `swin-classification-pipeline`
2. **Semantic Segmentation** → `swin-segmentation-pipeline`
3. **Object Detection** → `swin-detection-pipeline`

See issue #1 for the NATIVE task, dataset, model, and conformance specification.
