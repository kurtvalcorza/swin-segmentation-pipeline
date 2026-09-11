# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `swin_segmentation_task_inference.ipynb` | `TASK-INFERENCE` | Verified pretrained Swin-T + UPerNet semantic segmentation; ADE20K tutorial mIoU/per-class IoU; machine-readable masks/metrics/provenance | CPU / CPython 3.10 Jupyter | Optional image, gated off by default | **Candidate** — real task runtime implemented; promote only after exact-revision clean notebook execution passes |
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

`.github/workflows/verify-task-tutorial.yml` is the promotion gate for `swin_segmentation_task_inference.ipynb`. It must execute the **committed notebook** top-to-bottom from a fresh Python 3.10 Jupyter environment, exercise the repository API, download and verify the model, evaluate the fixed labelled sample, and assert the machine-readable outputs. Static JSON/compile checks are kept separate from execution evidence.

The older `verify-smoke-notebook.yml` remains useful engineering coverage but is not evidence for the task tutorial.
