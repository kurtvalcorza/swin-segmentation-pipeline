# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/swin-segmentation-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/swin-segmentation-pipeline/blob/main/tutorials/swin_segmentation_task_inference.ipynb)
[![Python 3.10 required](https://img.shields.io/badge/Python-3.10%20required-3776ab?style=flat&logo=python&logoColor=white)](../README.md)
[![Checkpoint](https://img.shields.io/badge/Checkpoint-upernet__swin__tiny__patch4__window7__512x512-ffcc4d?style=flat)](https://github.com/SwinTransformer/storage/releases/tag/v1.0.1)
[![Upstream](https://img.shields.io/badge/Upstream-microsoft%2FSwin--Transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/Swin-Transformer)
[![arXiv](https://img.shields.io/badge/arXiv-2103.14030-b31b1b.svg)](https://arxiv.org/abs/2103.14030)
[![Model released](https://img.shields.io/badge/Model%20released-2021--04--12-6f42c1?style=flat)](https://github.com/SwinTransformer/storage/releases/tag/v1.0.1)
[![Sample eval](https://img.shields.io/badge/Sample%20eval-mIoU%200.387%20%7C%20pixel%20acc%200.707-2ea44f?style=flat)](../MODEL_CARD.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `swin_segmentation_task_inference.ipynb` | `TASK-INFERENCE` | Verified pretrained Swin-T + UPerNet semantic segmentation; ADE20K tutorial mIoU/per-class IoU; machine-readable masks/metrics/provenance | CPU / CPython 3.10 Jupyter | Optional image, gated off by default | **release-grade** — exact committed notebook passed clean GitHub-hosted execution on 2026-09-11 |
| `swin_segmentation_scaffold_smoke_colab.ipynb` | `SMOKE` | Scaffold lifecycle/model-card/provenance checks | CPU / Python 3.11+ | — | **Engineering-only** |

## Supported user-facing capability

The repository now exposes a narrow public task-inference API, `dimer_swin_segmentation.DimerSwinSegmenter`, plus the `dimer-swin-segment` CLI. It wraps the pinned MMSegmentation 1.2.2 **Swin-T + UPerNet** ADE20K inference path, verifies the exact checkpoint size and SHA-256 before `.pth` deserialization, validates image inputs, emits 2-D ADE20K class-index masks, and exports provenance.

This does **not** remove the existing gradient-adaptation blockers. The canonical DIMER semantic-segmentation task/representation contract, validator/finetuner workers, artifact-serving composition, and accelerator qualification for training are still future work. The release-grade tutorial therefore uses `TASK-INFERENCE`, not `E2E`, and makes no fine-tuning claim.

## Model identity and source distinction

The task-inference runtime uses the official OpenMMLab MMSegmentation distribution associated with `swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512`: checkpoint size 240,154,742 bytes and SHA-256 `e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064`. This runtime source is recorded separately from the Microsoft/SwinTransformer release asset frozen in `provenance/open-weights.json` for the future adaptation lineage; the tutorial does not claim the two files are byte-identical.

The upstream `.pth` format is code-capable serialization. Digest verification proves that the bytes match the pinned distribution; it does not prove publisher authenticity or make an otherwise untrusted checkpoint safe.

## Tutorial evidence and limits

The default labelled sample is two immutable ADE20K validation fixtures from `hf-internal-testing/fixtures_ade20k` at commit `850d349e5038f291284e7999fcacbedc0922534b`. The notebook preserves those sample identities, applies the ADE20K `reduce_zero_label` convention used by the pinned MMSegmentation config, and reports aggregate **mIoU**, per-class IoU, and pixel accuracy as **sample/tutorial metrics only**. A constant-majority-pixel reference is included as a trivial sample baseline. The upstream full-ADE20K mIoU is displayed only as upstream-reported context and is not represented as notebook-measured performance.

The notebook writes semantic-mask PNGs, `predictions.json`, `metrics.json`, `per-class-iou.csv`, and `provenance.json`. BYOD inference is optional and disabled by default. The task runtime exports class indices, not calibrated per-pixel uncertainty.

## Release verification

Clean execution evidence for the committed `TASK-INFERENCE` notebook:

- **Date:** 2026-09-11 UTC
- **PR head tested:** `21bb78bfe2c53820490042b0523ee5dd0c7becd1`
- **GitHub Actions run:** `verify-task-tutorial` run `34575117856`, conclusion **success**
- **Environment:** GitHub-hosted Ubuntu 24.04.5, CPython 3.10.21; CPU runtime
- **Effective model stack:** torch 2.1.2+cpu; MMSegmentation 1.2.2; MMCV 2.1.0; MMEngine 0.10.7; NumPy 1.26.4; OpenCV 4.10.0.84
- **Checkpoint:** 240,154,742 bytes; SHA-256 `e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064`; verified before loading
- **Default sample:** two immutable ADE20K validation fixtures, 510,803 valid labelled pixels
- **Tutorial metrics observed:** aggregate sample mIoU `0.3869899942`; pixel accuracy `0.7065385286`; 11 classes with non-zero union; constant-majority (`sky`) sample baseline mIoU `0.0539564666`
- **Exports asserted:** two semantic-mask PNGs, `predictions.json`, `metrics.json`, `per-class-iou.csv`, `provenance.json`

The registry promotion in this commit changes documentation only; the notebook bytes exercised by the recorded run are unchanged. `.github/workflows/verify-task-tutorial.yml` remains the regression gate and re-executes the committed notebook whenever the notebook, runtime, registry, or workflow changes.

Static JSON/compile checks are not treated as execution evidence. The older `verify-smoke-notebook.yml` remains useful engineering coverage but is not evidence for the task tutorial.

## Recorded SHOULD deviation

**G16 / model-card link:** the current `MODEL_CARD.md` is a scaffold-lifecycle card written for the future gradient-adaptation pipeline and still describes that capability as having no runtime. The task-inference tutorial deliberately does not present that card as documentation for the newly implemented pretrained inference surface. This is a documentation follow-up, not a `TASK-INFERENCE` execution blocker; until the model card is revised, this registry, `spec/task-inference-surface.json`, and the runtime `MODEL_SPEC` are the durable sources for the supported inference capability and exact checkpoint identity.
