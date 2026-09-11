# swin-segmentation-pipeline

Swin **semantic-segmentation** pipeline for DIMER — pretrained task inference is implemented; the composed-worker gradient-adaptation pipeline remains a scaffold.

## Upstream alignment

This repository corresponds to the **Semantic Segmentation** downstream task of the original Microsoft Swin Transformer project.

- **Upstream project:** [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- **Official Swin segmentation lineage:** [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)
- **Upstream task:** Semantic Segmentation
- **Canonical benchmark:** ADE20K
- **Reference architecture:** Swin-T backbone + UPerNet
- **DIMER task boundary:** RGB image → one ADE20K class index per pixel
- **Primary metrics:** mIoU, per-class IoU, pixel accuracy

This repository is **not** an instance-segmentation pipeline. Instance masks belong to the Mask R-CNN lineage in the sibling detection repository.

## Public pretrained task-inference runtime

The repository exposes a real, public, CPU-capable pretrained inference path:

- **Python API:** `dimer_swin_segmentation.DimerSwinSegmenter`
- **CLI:** `dimer-swin-segment`
- **Package:** `dimer-swin-segmentation` 0.1.0
- **Qualified runtime:** CPython 3.10; torch 2.1.2; MMSegmentation 1.2.2; MMCV 2.1.0; MMEngine 0.10.7; NumPy 1.26.4; OpenCV 4.10.0.84
- **Model distribution:** OpenMMLab Swin-T + UPerNet ADE20K distribution
- **Checkpoint:** 240,154,742 bytes; SHA-256 `e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064`
- **Input validation:** readable image, positive dimensions, maximum 64,000,000 pixels
- **Outputs:** two-dimensional `uint8` semantic class-index mask over the 150 ADE20K classes
- **Checkpoint handling:** exact size and SHA-256 are verified before the upstream `.pth` file is deserialized

The `.pth` format is code-capable PyTorch serialization. Digest verification proves byte identity against the pinned distribution; it does not prove publisher authenticity or make an otherwise untrusted checkpoint safe.

Machine-readable capability metadata is in [`spec/task-inference-surface.json`](spec/task-inference-surface.json).

### Supported installation contract

The OpenMMLab stack includes binary wheels selected from the OpenMMLab/PyTorch indexes, so **`pip install .` by itself is not a qualified runtime installation**. `pyproject.toml` packages the DIMER wrapper and CLI; it intentionally does not imply that standard PyPI dependency resolution reproduces the tested binary stack.

The supported public bootstrap is the release-grade notebook [`tutorials/swin_segmentation_task_inference.ipynb`](tutorials/swin_segmentation_task_inference.ipynb), which installs the exact qualified Python 3.10 CPU dependency graph before installing this repository package with `--no-deps`. A standalone caller may reproduce those same pinned commands. Changing Python, torch, MMCV, MMEngine or MMSegmentation versions is outside the qualified runtime until revalidated. The API also checks the OpenMMLab package versions at startup and fails closed on drift.

### Release-grade tutorial

[`tutorials/swin_segmentation_task_inference.ipynb`](tutorials/swin_segmentation_task_inference.ipynb) is the DIMER Notebook Specification 1.0 **`TASK-INFERENCE` release-grade tutorial**. It exercises the repository API rather than reimplementing model inference, evaluates two immutable labelled ADE20K validation fixtures, handles ADE20K ignore-label semantics, reports mIoU/per-class IoU/pixel accuracy against a constant-majority baseline, includes gated BYOD, and exports semantic PNGs plus machine-readable predictions, metrics and provenance.

Clean GitHub-hosted execution of the exact committed notebook passed on 2026-09-11. The two-image tutorial sample measured mIoU `0.3869899942` and pixel accuracy `0.7065385286` across 510,803 valid labelled pixels, versus a constant-majority sample baseline mIoU `0.0539564666`. These are **small-sample tutorial metrics**, not reproduction of the upstream full-ADE20K benchmark and not production-fitness evidence. See [`tutorials/README.md`](tutorials/README.md).

## Gradient-adaptation builder status

The separate DIMER **gradient-adaptation** design remains a scaffold. The reviewed `ml-worker` contract still lacks the semantic-segmentation task profile and raster-mask representation required for the composed-worker training path, and task-specific validator/finetuner releases and accelerator qualification do not yet exist. No gradient-adaptation composition or release manifest is emitted.

**Lifecycle for that composed-worker surface (DIMER Pipeline Specification 1.0):** `scaffold` — declared machine-readably in `spec/pipeline-surface.json` with intended `implementation_topology: COMPOSED-WORKERS` and `capability_modes: [GRADIENT-ADAPTATION]`. The canonical Microsoft/SwinTransformer checkpoint recorded for that future adaptation lineage carries `redistribution_status: unknown` in `provenance/open-weights.json`, so DIMER hosting remains **BLOCKED** until an authoritative weight-licence determination is recorded.

`scripts/verify_scaffold.py` continues to refuse any lifecycle above `scaffold` while adaptation blockers exist, any declared adaptation components/release, and any non-blocked DIMER hosting for an `unknown`/`prohibited` weight status. The pretrained `TASK-INFERENCE` capability does **not** satisfy or remove those adaptation blockers.

`MODEL_CARD.md` distinguishes the implemented pretrained inference surface from the still-scaffolded gradient-adaptation design. The inference runtime uses the separately pinned OpenMMLab distribution recorded in `spec/task-inference-surface.json`; the adaptation lineage remains governed by `spec/pipeline-surface.json` and `provenance/open-weights.json`.

Repository-owned artifacts include:

- `spec/task-inference-surface.json` — implemented pretrained inference capability and runtime/model/output/evaluation contract;
- `src/dimer_swin_segmentation/` — public inference API and CLI;
- `tutorials/swin_segmentation_task_inference.ipynb` — release-grade `TASK-INFERENCE` tutorial;
- `.github/workflows/verify-task-tutorial.yml` — exact-notebook clean execution gate;
- `spec/pipeline-surface.json` — future gradient-adaptation task/model/contract surface and blockers;
- `provenance/open-weights.json` — Microsoft/SwinTransformer adaptation-lineage provenance;
- `scripts/verify_scaffold.py` — fail-closed adaptation-scaffold verifier;
- `.github/workflows/verify-scaffold.yml` — scaffold verification CI.

## Cloud verification & adaptation-lineage open-weights digest

The pre-existing scaffold and Microsoft/SwinTransformer release asset were separately verified in an isolated Kaggle cloud container:

- **Kernel:** [`kurtvalcorza/swin-segmentation-verify`](https://www.kaggle.com/code/kurtvalcorza/swin-segmentation-verify)
- **Status:** `KernelWorkerStatus.COMPLETE` (Exit Code 0)
- **Scaffold Verifier:** `PASS`
- **Microsoft/SwinTransformer checkpoint asset:** `upernet_swin_tiny_patch4_window7_512x512.pth`, 240,144,566 bytes, `SwinTransformer/storage@v1.0.1`, asset ID `34862982`
- **SHA-256:** `c26408bb5ddee935dcc709ad7815fa039f2db2b41134cd702da7da277d1a7d89`

That asset is part of the frozen future gradient-adaptation lineage and is **not claimed to be byte-identical** to the OpenMMLab task-inference checkpoint used by `DimerSwinSegmenter`.

## Canonical Swin family

1. **Image Classification** → `swin-classification-pipeline`
2. **Semantic Segmentation** → `swin-segmentation-pipeline`
3. **Object Detection** → `swin-detection-pipeline`

See issue #1 for the original NATIVE task, dataset, model, provenance, and conformance specification.
