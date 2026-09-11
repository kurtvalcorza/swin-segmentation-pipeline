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

**Lifecycle (DIMER Pipeline Specification 1.0):** `scaffold` — declared machine-readably in
`spec/pipeline-surface.json` (`dimerPipelineSpec`: `lifecycle_status`, intended
`implementation_topology` `COMPOSED-WORKERS`, `capability_modes` `GRADIENT-ADAPTATION`). The
canonical checkpoint `upernet_swin_tiny_patch4_window7_512x512.pth` carries `redistribution_status: unknown` in
`provenance/open-weights.json` (`weightLicensing`), so DIMER hosting is **BLOCKED** until an
authoritative upstream weight-licence determination is recorded (LIC2/LIC7).
`scripts/verify_scaffold.py` refuses any lifecycle above `scaffold` while blockers exist, any
declared components/release, and any non-blocked hosting for an `unknown`/`prohibited` status.

`MODEL_CARD.md` (DIMER Model Card Specification 1.0) is a **scaffold-lifecycle card**: it states
the intended model, boundaries, factors, metrics, risks and prohibited uses, records that this
repository measures nothing and distributes no weights, and lists only the mitigations that
actually exist here (provenance pinning, fail-closed lifecycle, hosting gate, no `.pth`
deserialisation).

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

## Cloud verification & open-weights digest (Kaggle)

The scaffold and official upstream weights were verified in an isolated Kaggle cloud container:

- **Kernel:** [`kurtvalcorza/swin-segmentation-verify`](https://www.kaggle.com/code/kurtvalcorza/swin-segmentation-verify)
- **Status:** `KernelWorkerStatus.COMPLETE` (Exit Code 0)
- **Scaffold Verifier (`scripts/verify_scaffold.py`):** `PASS` (internally consistent and fail-closed, 0.17s)
- **Official Checkpoint Asset:** `upernet_swin_tiny_patch4_window7_512x512.pth` (240,144,566 bytes in 16.72s) from official release `SwinTransformer/storage@v1.0.1` (asset ID `34862982`)
- **Authoritative SHA-256:** `c26408bb5ddee935dcc709ad7815fa039f2db2b41134cd702da7da277d1a7d89` (recorded in `provenance/open-weights.json`)
- **Deserialized State Dict Inspection:** 271 parameter tensors, 59,983,487 parameters (189 backbone, 74 decode head)

## Canonical Swin family

1. **Image Classification** → `swin-classification-pipeline`
2. **Semantic Segmentation** → `swin-segmentation-pipeline`
3. **Object Detection** → `swin-detection-pipeline`

See issue #1 for the full NATIVE task, dataset, model, provenance, and
conformance specification.
