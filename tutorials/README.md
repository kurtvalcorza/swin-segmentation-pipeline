# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | Release status |
|---|---|---|---|---|
| `swin_segmentation_scaffold_smoke_colab.ipynb` | `SMOKE` | Scaffold lifecycle, model-card, blocker, and open-weight provenance verification | CPU / Python 3.11+ | **Engineering-only** — clean execution is CI-gated, but this does not satisfy the primary release-grade tutorial requirement |

## Why this remains a smoke notebook

The repository lifecycle is explicitly `scaffold` and the current surface is `BLOCKED_PENDING_CONTRACT_AND_WORKERS`. The semantic-segmentation task profile and raster-mask representation are absent from the reviewed contract, validator/finetuner releases do not exist, accelerator qualification is pending, no composition/release is emitted, and DIMER hosting of the upstream checkpoint is blocked while its redistribution status remains `unknown`.

Publishing an `E2E` or `TASK-INFERENCE` notebook before a real semantic-segmentation runtime exists would misrepresent the implementation. This notebook therefore limits itself to the repository-owned verifier plus lifecycle/model-card/provenance assertions. It does not download or deserialize the upstream `.pth` checkpoint.

The notebook is anchored to the stacked lifecycle + model-card revision used by PRs #5/#6. PR #4 should merge only after those prerequisite changes land (or after the anchor is repointed to their final `main` commit).

## Clean-runtime verification

`.github/workflows/verify-smoke-notebook.yml` executes the committed notebook top-to-bottom on Python 3.12 in a fresh GitHub-hosted Ubuntu/Jupyter environment whenever the notebook, registry, verifier, lifecycle/provenance files, model card, or workflow changes. A green run is execution evidence for the `SMOKE` profile only; it is not semantic-segmentation task-runtime evidence and does not make this repository tutorial-ready for end users.

The previous Kaggle execution record targeted the pre-lifecycle scaffold revision and is intentionally not carried forward as evidence for this notebook revision.

## Upgrade gate

A release-grade replacement requires the semantic-segmentation task profile and raster-mask representation, validator/finetuner releases, accelerator qualification, the real production-facing DIMER API, image/mask input validation, mIoU/per-class IoU evaluation, machine-readable masks/metrics/provenance, and clean-runtime execution evidence for the exact release revision.
