# Release verification

`tutorials/swin_segmentation_task_inference.ipynb` (`TASK-INFERENCE`, standalone) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence
under DIMER Notebook Specification 2.2. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

`.github/workflows/ci.yml` runs `ruff`, the offline unit suite (`tests/test_snapshot.py`, `tests/test_role_helpers.py`,
`tests/test_metrics.py`, `tests/test_notebook_parity.py` — OpenMMLab stubbed, the metrics are pure NumPy, no weights, no model),
`tools/validate_release_assets.py` and `tools/build_notebook.py --check` on Python 3.10. The validator checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE` profile, the notebook-spec
  version and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, `standalone: true` and
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

Notebook identity is the Git blob id of the notebook file (verify with `git rev-parse <commit>:<path>`). The current notebook is
`tutorials/swin_segmentation_colab.ipynb` (`E2E`); `tutorials/swin_segmentation_task_inference.ipynb` was removed in `af73863`
and its row below is history.

### Current notebook `swin_segmentation_colab.ipynb` (E2E) — GitHub Actions runs

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-24 | PR head `ffff5c9` (merged as `4d5f4bb`) / `2a13f3835b58` | GitHub Actions `verify-tutorial` run [`35997457508`](https://github.com/kurtvalcorza/swin-segmentation-pipeline/actions/runs/35997457508) (job `107625689260`), Ubuntu, CPython 3.10 kernel, CPU | Default path (in-kernel pinned install; synthetic adaptation dataset) | 10 min 8 s (job) | **PASSED** — the workflow's assertions held: input manifest accepted with a recorded refusal, evaluation verdict `sample-sanity`, checkpoint SHA-256 `e380ad3e…6064`, `mmsegmentation` 1.2.2, Python 3.10.x, exact reload mask match. The workflow printed no metric values, so none are recorded; the 2026-10-05 fixes make it print and upload them. |
| 2026-10-10 (11:11:12 UTC start) | `36f8f08573bf384372aead0e0a6c618e5affc094` / `3cce5e9c2ecf3ed8d14581e2a78ff4889d222e16` (`NOTEBOOK_SOURCE.repository_revision` `42810a87de14`, `module_sha256` `f9e4d755d1c7…`, generator `build_notebook.py/2.1-swd`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-swin-36f8f08-29da`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated uv-managed CPython 3.10.18 (45 locked packages, setup 12 s), `torch 2.1.2+cpu`, MMSegmentation 1.2.2, MMCV 2.1.0, MMEngine 0.10.7, device `cpu` (by design; the T4 is unused), `restarted: false` | Default synthetic path, every form field at its default (BYOD off) | 1027.7 s | **PASSED** — one pass, no restart, 0 errors; 18/18 code cells in order (cells 4–6, the carried modules, print nothing); 1 checkpoint file digest-verified at `c685fe6767c4`; demonstration scene `not-measurable` (no ground truth); 24 records, 18/6 split, empty-dataset probe rejected; random-head mIoU 0.2857; loss 7.486 → 0.919 → 0.602; adapted mIoU 0.9140 / pixel accuracy 0.9777 beside majority 0.1842 / 0.5527 and colour 0.9754 / 0.9950 (the notebook notes the adapted model does not beat the colour baseline); per-class IoU 0.963 / 0.990 / 0.790; scene 99 mIoU 0.9655; adapter 271 tensors, fresh reload `exact_mask_match` true. Evidence in `docs/execution-evidence/2026-10-10-36f8f08/`: executed notebook SHA-256 `279dce578ce148798b1ce040ab953fb06518c6e9666099fc7b14bf77b00714b1`, `run_summary.json` `e7014265bc30359c5cf2e8656464c97d31fac6631df2eec8ff9d23e671d31f32`, `exec.log` `36cfd8e47f8433f91839642dd4c3c5ac0733fcefcf7c9d4e8ef705870f1a8877`. Not exercised: BYOD, the optional activity, a browser Run all |

The 2026-10-05 review fixes (isolated uv environment with a managed CPython 3.10.18, BYOD dataset branch, colour baseline, corrected masks) change the notebook; the regenerated blob is recorded in the 2026-10-10 Colab T4 row above.

### Removed notebook `swin_segmentation_task_inference.ipynb` (TASK-INFERENCE) — history

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

The current notebook's earlier blob `2a13f383` passed the GitHub Actions run above; the regenerated blob `3cce5e9c2ecf` (commit `36f8f08`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 VM on 2026-10-10 (Colab CLI 0.7.4 sequential execution, 18/18 code cells, 1027.7 s, CPU execution by design; adapted mIoU 0.914 beside majority 0.184 and colour 0.975, scene 99 mIoU 0.966, fresh reload exact). Static validation
(`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code cell, and the offline
unit suite passed on the tutorial source at the candidate revision, which is necessary but not sufficient. The
registry status remains **Candidate** until a reviewer confirms a recorded run against the notebook blob under review
and an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh: the
2026-09-24 run executed the checkpoint fetch, the OpenMMLab loader, the adaptation and the reload with pip in a Python 3.10
kernel; the current revision installs the same pins with `uv` into a uv-managed CPython 3.10.18 from the 45-package
hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`; MMCV 2.1.0 is the prebuilt
cp310 manylinux wheel from the OpenMMLab page, torch the `+cpu` wheel from the PyTorch CPU index), re-heads a fresh
model copy, adds a BYOD dataset branch and a colour baseline, and corrects the synthetic masks; all of that ran in the 2026-10-10 Colab T4 row above. A hosted run must record `restarted: false`, the library versions, the adaptation metrics beside both baselines, and a
Colab run before any entry point calls Colab supported.
