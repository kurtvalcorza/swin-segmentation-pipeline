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
- **Reference architecture:** Swin-T backbone + UPerNet semantic-segmentation head
- **Pipeline task boundary:** image → per-pixel class mask
- **Primary metrics:** mIoU and per-class IoU
- **Canonical v1 checkpoint:** official `upernet_swin_tiny_patch4_window7_512x512.pth` release asset

This repository is **not** an instance-segmentation pipeline. Instance masks are
part of the upstream Swin detection lineage through architectures such as Mask
R-CNN; this repo is reserved for the dedicated per-pixel semantic-segmentation
task.

## Builder status

The pipeline-level specification is frozen and the repository now carries an
executable, fail-closed scaffold. It deliberately does **not** emit a
`pipeline-manifest.json` or release artifact yet, because the reviewed
`ml-worker` contract does not contain the semantic-segmentation task profile or
a raster-mask representation profile, and the task-specific validator and
finetuner releases do not yet exist.

Repository-owned build artifacts:

- `spec/pipeline-surface.json` — machine-readable task/model/contract surface and blockers;
- `provenance/open-weights.json` — exact official source revision, config blob, release asset identity, and checkpoint-digest state;
- `scripts/verify_scaffold.py` — verifies that the scaffold remains internally consistent and fail-closed;
- `.github/workflows/verify-scaffold.yml` — runs the verifier on pull requests and pushes to `main`.

Run locally:

```sh
python scripts/verify_scaffold.py
```

A passing scaffold check means the **specification is internally consistent**;
it does not mean the pipeline is runtime-qualified or releasable.

## Canonical Swin family

1. **Image Classification** → `swin-classification-pipeline`
2. **Semantic Segmentation** → `swin-segmentation-pipeline`
3. **Object Detection** → `swin-detection-pipeline`

See issue #1 for the full NATIVE task, dataset, model, provenance, and
conformance specification.
