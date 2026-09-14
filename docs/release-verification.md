# Release verification

`tutorials/swin_segmentation_task_inference.ipynb` (`TASK-INFERENCE`, standalone) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence
under DIMER Notebook Specification 1.1. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

`.github/workflows/ci.yml` runs `ruff`, the offline unit suite (`tests/test_snapshot.py`, `tests/test_role_helpers.py`,
`tests/test_metrics.py`, `tests/test_notebook_parity.py` — OpenMMLab stubbed, the metrics are pure NumPy, no weights, no model),
`tools/validate_release_assets.py` and `tools/build_notebook.py --check` on Python 3.10. The validator checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE` profile, the notebook-spec
  version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`, `standalone: true` and
  `generated_from` (repository, revision, the two carried modules, their joined SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install, repository import, worker process or
  subprocess on the primary path (the generator-owned install cell excepted); one cell tagged `embedded_module` per
  carried module (`src/dimer_swin_segmentation/metrics.py`, then `runtime.py`) equal to the module after the generator's
  documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to
  `tools/pins.txt`; the notebook byte-identical to `tools/build_notebook.py` output; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cells (and repeated in the inline manifest, which
  the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same
  identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `DimerSwinSegmenter.from_pretrained(weights_dir=...)`, `validate_inputs`, `predict`, `evaluation_report`),
  the Python 3.10 assertion, the ceiling print (`MAX_PIXELS`, class count), the pinned ADE20K fixture commit and
  digests, the exports, the learner-facing statements (no per-pixel confidence, no shipped threshold, the
  `reduce_zero_label` convention, the `.pth` trust boundary, `not-measurable` / `sample-sanity` verdicts) and the
  gated-off `USE_BYOD` / `USE_ADE20K_FIXTURES` defaults listed in the validator; forbidden patterns (credential-in-URL,
  any `git clone` / `github.com/kurtvalcorza` / repository import, a mutable `revision='main'`, direct `mmseg` / `mmcv` /
  `mmengine` / `torchvision` / `transformers` / `huggingface_hub` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

These are source/provenance and unit checks. They are **not** execution evidence. The fleet CI venv used for the
lane gates has no OpenMMLab stack (`mmseg`, `mmcv`, `mmengine` are stubbed in the unit suite; the NumPy metrics
are tested for real), so the package's real loader path is exercised only by a notebook execution.

`.github/workflows/verify-task-tutorial.yml` executes the committed notebook with `nbconvert` on a GitHub-hosted
Python 3.10 runner from a scratch directory that contains only the notebook (no repository checkout on the notebook's
path) and asserts the four exports. A green run there is a clean-runtime execution of the default path and is the
promotion evidence to record below; a red run blocks release.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| GitHub Actions `verify-task-tutorial` (supported clean-room path) | `ubuntu-latest`, `actions/setup-python` 3.10, CPU; `nbconvert` from a scratch directory | The qualified Python 3.10 CPU runtime; a green run is promotion evidence once recorded here with the notebook blob |
| Jupyter on a CPython 3.10 kernel (user path) | Any Linux host with Python 3.10; CPU | The runtime the tutorial is written for; the notebook asserts the interpreter version |
| Google Colab | Default Colab runtimes ship Python 3.11+ | **Unsupported**: the OpenMMLab wheels exist for Python 3.10 only and the pinned install fails; the badge is kept for the file location, not as a supported executor |
| Kaggle CLI kernel | Kaggle images ship Python 3.11+ | **Unsupported** for the same reason |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. run that exact notebook revision in a clean CPython 3.10 CPU runtime with **no repository checkout** on the
   notebook's path and a clean `weights/` directory (the `verify-task-tutorial` workflow does this);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults for the
   sample path: `USE_BYOD = False`, `USE_ADE20K_FIXTURES = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `tools/pins.txt`: torch 2.1.2+cpu, mmcv 2.1.0, mmengine 0.10.7, mmsegmentation 1.2.2, numpy 1.26.4);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` (PyPI plus the PyTorch CPU index and the OpenMMLab mmcv
     find-links) with no GitHub access;
   - the two carried module cells execute (define `DimerSwinSegmenter`, `semantic_iou`, `validate_inputs`,
     `evaluation_report`) with no import of the repository package;
   - pinned checkpoint acquisition through the package: the inline `MANIFEST` is asserted against the module identity
     and written to `weights/swin-t-upernet-ade20k/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports
     the one manifest entry on a clean runtime, `verify_snapshot` returns the manifest dict, and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` reports `source == 'local-snapshot'`;
   - Section 4 asserts Python 3.10 and prints `mmseg 1.2.2`, `mmcv 2.1.0`, `mmengine 0.10.7`, 150 classes;
   - the synthetic 512×384 scene is generated in code with its SHA-256 printed;
   - `validate_inputs` writes `outputs/swin_segmentation_task_inference_input_manifest.json` (verdict `accepted`, one
     recorded rejection finding from the missing-file probe);
   - segmentation through `predict(path)` per image, each mask saved as
     `outputs/swin_segmentation_task_inference_<image>_semantic.png`;
   - `evaluation_report` writes `outputs/swin_segmentation_task_inference_evaluation_report.json` with verdict
     `not-measurable` on the synthetic sample (no ground truth), stated as such;
   - `outputs/swin_segmentation_task_inference_result.json` and `outputs/swin_segmentation_task_inference_class_coverage.csv`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, checkpoint digest, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, mmseg/mmcv/mmengine, device), model
   identifier and immutable revision, whether the weights directory was clean, outcome, produced outputs, and any
   warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

The gated `USE_ADE20K_FIXTURES` path (two labelled ADE20K validation fixtures, `semantic_iou` + `majority_class_baseline`)
is not part of the default-path gate; a separate run with the gate enabled may be recorded as additional evidence.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/swin_segmentation_task_inference.ipynb` (verify with
`git rev-parse <commit>:tutorials/swin_segmentation_task_inference.ipynb`).

### Standalone carrier (Notebook Specification 1.1) — clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `40c2cc3` / `32dd1b002204` | GitHub Actions `verify-task-tutorial` run `34770237499`, Ubuntu 24.04.5, CPython 3.10.19, CPU | Default sample path | 75.0 s | **PASSED** — 17/17 code cells executed cleanly, checkpoint verified & loaded, 4 outputs generated, evaluation verdict `not-measurable` on synthetic sample |

### Previous carrier (Notebook Specification 1.0, repository-installing) — audit trail only

| Date (UTC) | Commit | Executor | Path exercised | Outcome |
|---|---|---|---|---|
| 2026-09-11 | PR head `21bb78bfe2c53820490042b0523ee5dd0c7becd1` | GitHub Actions `verify-task-tutorial` run `34575117856`, Ubuntu 24.04.5, CPython 3.10.21, CPU | repository clone + pinned install, two ADE20K validation fixtures (510,803 valid labelled pixels) | success — aggregate mIoU `0.3869899942`, pixel accuracy `0.7065385286`, 11 classes with non-zero union; constant-majority (`sky`) baseline mIoU `0.0539564666` |

That run exercised a notebook that cloned this repository and evaluated the ADE20K fixtures by default; it is
evidence for that carrier and for the OpenMMLab inference path, not for the standalone notebook above.

## Current status

No clean-runtime execution of the standalone notebook has been recorded yet; the run is **pending**. Static validation
(`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code cell, and the offline
unit suite passed on the tutorial source at the candidate revision, which is necessary but not sufficient. The
registry status remains **Candidate** until a reviewer confirms a recorded run against the notebook blob under review
and an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real fetch from
`download.openmmlab.com` into a fresh `weights/swin-t-upernet-ade20k/` has not been executed on this carrier);
`verify_snapshot` was executed once over the real local checkpoint on the builder's workstation (OK); the OpenMMLab
loader and the single-command pinned install (`--extra-index-url` PyTorch CPU +
`--find-links` OpenMMLab mmcv, in place of the previous notebook's separate `pip`/`mim` steps) have been validated
statically only — wheel availability for every pin was checked against the indexes, resolution has not been run; and
the standalone carrier itself — executing the carried module cells in a runtime that has no repository checkout — has
been validated statically (parity PASS, carrier probe) but never run.
