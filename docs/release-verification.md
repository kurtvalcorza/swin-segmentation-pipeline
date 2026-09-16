# Release verification

`tutorials/swin_segmentation_colab.ipynb` (`E2E`, standalone) is a **Candidate** until the exact provenance-refreshed notebook blob executes top-to-bottom in a clean supported runtime and an integrator explicitly promotes it. Static validation, unit tests and a successful run of an earlier carrier do not transfer execution evidence to a different notebook blob.

## Automatic coverage

`.github/workflows/ci.yml` runs `ruff`, the offline unit suite, `tools/validate_release_assets.py` and `tools/build_notebook.py --check` on Python 3.10. The gates validate Notebook Specification 2.0 metadata, code-cell compilation, output-free source, required learner guidance, local module/manifest/pin parity, generated-byte parity, forbidden trust-boundary patterns and cross-document release status.

The standalone provenance gate additionally requires `metadata.dimer.generated_from.revision` to be a committed 40-hex revision containing the exact bytes of all three carried modules:

- `src/dimer_swin_segmentation/metrics.py`
- `src/dimer_swin_segmentation/samples.py`
- `src/dimer_swin_segmentation/runtime.py`

These are source/provenance checks, not clean-runtime execution evidence. `.github/workflows/verify-tutorial.yml` provides the supported execution gate: it copies only the committed notebook into a scratch directory, executes it with `nbconvert` on a GitHub-hosted CPython 3.10 CPU runner, and asserts the five machine-readable outputs including the safely reloaded adapter.

## Executor paths

| Path | Runtime | Release role |
|---|---|---|
| GitHub Actions `verify-tutorial` | Ubuntu, CPython 3.10, CPU; exact notebook copied to a scratch directory | Supported clean-runtime promotion evidence when tied to the exact commit and notebook blob |
| Local Jupyter | Linux CPython 3.10, CPU | Supported user path when the exact pins and clean-carrier conditions are reproduced |
| Default Google Colab | Python 3.11+ | Unsupported: the pinned OpenMMLab wheel set requires Python 3.10 |
| Default Kaggle notebook | Python 3.11+ | Unsupported for the same reason |

## Supported release verification procedure

Before changing the status from `Candidate` to `Release-grade`:

1. Resolve the exact commit and Git blob of `tutorials/swin_segmentation_colab.ipynb`; confirm CI and generated-byte parity are green.
2. Execute that blob from a scratch directory with no repository checkout on its path, a clean `weights/` directory, CPython 3.10 and CPU.
3. Keep `USE_BYOD_IMAGE = False` and `USE_BYOD_DATASET = False`; run all 16 code cells without implementation edits.
4. Confirm `NOTEBOOK_SOURCE.repository_revision` equals `metadata.dimer.generated_from.revision` and the runtime matches `tools/pins.txt`.
5. Confirm the checkpoint size/SHA-256 before deserialization, the 24 generated records and 18/6 split, dynamic three-class re-heading, frozen backbone, bounded AdamW loop, `sample-sanity` evaluation, unseen-scene inference, adapter export and `weights_only=True` reload with exact mask equality.
6. Confirm these outputs exist and are non-empty: `swin_segmentation_input_manifest.json`, `swin_segmentation_evaluation_report.json`, `swin_segmentation_result.json`, `swin_segmentation_class_coverage.csv`, and `swin-segmentation-adapter-v1.pt`.
7. Record the commit, notebook blob, workflow/run identifier, runtime, device, wall time, result and relevant warnings below. Never record credentials.

A known failure or an execution record for a different carrier blocks promotion.

## Recorded executions

Notebook identity is the Git blob of `tutorials/swin_segmentation_colab.ipynb`, resolved with `git rev-parse <commit>:tutorials/swin_segmentation_colab.ipynb`.

### E2E standalone carrier (Notebook Specification 2.0)

| Date (UTC) | Commit / notebook blob | Executor | Path | Wall | Outcome | Qualification |
|---|---|---|---|---:|---|---|
| 2026-09-15 | PR #10 head `5161303faa38ce70a8f61797b0a451a6e6c31646` / `2a13f3835b58c04743f4070a55a966c0dc701358` | GitHub Actions run `34974178073`, job `104397600851`; CPython 3.10, CPU | Scratch-directory default E2E path | 557 s notebook step (572 s job) | **PASSED** — 16/16 code cells, output assertions, `sample-sanity`, adapter reload exact-mask match | **Historical, non-qualifying for the refreshed carrier** — embedded source revision `79e83e8` did not contain the carried E2E modules |

The successful run shows that the PR #10 implementation executed end to end, but its false source-revision label breaks the immutable provenance claim. It cannot qualify the regenerated notebook.

### Superseded task-inference carriers — audit trail only

| Date (UTC) | Commit / notebook blob | Executor | Outcome |
|---|---|---|---|
| 2026-09-14 | `40c2cc3` / `32dd1b002204` | GitHub Actions run `34770237499`; CPython 3.10, CPU | Passed 17/17 code cells in 75.0 s; synthetic inference verdict `not-measurable` |
| 2026-09-11 | PR head `21bb78bfe2c53820490042b0523ee5dd0c7becd1` | GitHub Actions run `34575117856`; CPython 3.10, CPU | Passed repository-installing carrier; two ADE20K fixtures produced mIoU `0.3869899942` and pixel accuracy `0.7065385286` |

These records belong to replaced task-inference notebooks and are not E2E adaptation evidence.

## Current status

The source-bound E2E carrier is **Candidate**. Its static gates can pass locally after regeneration, but no clean supported-runtime execution is yet recorded for its exact notebook blob. Promotion remains pending that run and explicit integrator review.
