---
license: mit
model_card_spec: "1.1"
pipeline_tag: image-segmentation
task: "Semantic Segmentation"
pipeline_spec: "1.0"
base_model: SwinTransformer/storage upernet_swin_tiny_patch4_window7_512x512.pth (release v1.0.1, asset 34862982)
date_published: "2021-04-12"
date_published_source: "GitHub release SwinTransformer/storage v1.0.1 published_at 2021-04-12"
base_model_sha256: c26408bb5ddee935dcc709ad7815fa039f2db2b41134cd702da7da277d1a7d89
base_model_weights_license: MIT
pipeline_id: org.valcorza.swin-segmentation
lifecycle_status: scaffold
implementation_topology: COMPOSED-WORKERS
capability_modes:
  - GRADIENT-ADAPTATION
task_profile_candidate: core.task.vision.semantic-segmentation
task_inference_status: release-grade
task_inference_surface: spec/task-inference-surface.json
---

# Swin-T + UPerNet — Pretrained Inference and Notebook-Local Segmentation Adaptation

[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-microsoft%2FSwin--Transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/Swin-Transformer)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2103.14030-b31b1b.svg)](https://arxiv.org/abs/2103.14030)
[![Code license: MIT](https://img.shields.io/badge/Code%20license-MIT-yellow.svg)](LICENSE)
[![Weights license: MIT](https://img.shields.io/badge/Weights%20license-MIT-yellow.svg)](https://github.com/SwinTransformer/storage/blob/main/LICENSE)

[![Checkpoint](https://img.shields.io/badge/Checkpoint-upernet__swin__tiny__patch4__window7__512x512-ffcc4d?style=flat)](spec/task-inference-surface.json)
[![Model released](https://img.shields.io/badge/Model%20released-2021--04--12-6f42c1?style=flat)](https://github.com/SwinTransformer/storage/releases/tag/v1.0.1)
[![Sample eval](https://img.shields.io/badge/Sample%20eval-mIoU%200.387%20%7C%20pixel%20acc%200.707-2ea44f?style=flat)](tutorials/README.md)
[![Python 3.10 required](https://img.shields.io/badge/Python-3.10%20required-3776ab?style=flat&logo=python&logoColor=white)](README.md)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook demonstrating end-to-end execution (Notebook Spec 2.0) — pretrained inference, dataset contract validation, dynamic re-heading, frozen-backbone fine-tuning, held-out evaluation, and artifact export/reload:

- **End-to-End Adaptation Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/swin-segmentation-pipeline/blob/main/tutorials/swin_segmentation_colab.ipynb) [`swin_segmentation_colab.ipynb`](https://github.com/kurtvalcorza/swin-segmentation-pipeline/blob/main/tutorials/swin_segmentation_colab.ipynb)  
  *Pretrained ADE20K semantic segmentation, dataset validation (`core.dataset.vision.raster-mask`), dynamic re-heading to custom classes (`background`, `road`, `structure`), bounded fine-tuning with frozen backbone, held-out mIoU and pixel accuracy evaluation against majority baseline, and portable adapter artifact export and reload.*

> [!NOTE]
> Runs on the default CPU runtime; the shipped OpenMMLab runtime is CPU-only and requires Python 3.10.

---

#### Description

This repository has **three distinct capability surfaces**. First, it ships pretrained `TASK-INFERENCE` through `dimer_swin_segmentation.DimerSwinSegmenter`, the `dimer-swin-segment` CLI and `spec/task-inference-surface.json`; that path verifies the pinned OpenMMLab checkpoint and emits ADE20K class-index masks. Second, `tutorials/swin_segmentation_colab.ipynb` carries the repository package into one Python 3.10 runtime and implements notebook-local `E2E` adaptation: validate image/mask records, re-head the decode and auxiliary heads, freeze the Swin-T backbone, fine-tune with AdamW, evaluate a held-out split, export a classifier-head adapter and reload it safely. Third, the separate DIMER `COMPOSED-WORKERS` / `GRADIENT-ADAPTATION` surface remains a Pipeline Specification 1.0 `scaffold`; the `lifecycle_status: scaffold` front matter applies only to that worker composition, not to pretrained inference or notebook-local adaptation.

#### Intended Use and Limitations

The supported implementation today includes pretrained ADE20K-150 inference and bounded notebook-local adaptation to an ordered custom class vocabulary. The default adaptation path uses 24 deterministic generated scenes with a disjoint 18/6 split; optional BYOD accepts operator-controlled image/mask records through the same validator. Only the decode and auxiliary segmentation heads are re-headed and optimized while the Swin-T backbone remains frozen. The resulting `.pt` adapter requires the exact pinned base architecture and is not a standalone production model. The upstream checkpoint is code-capable PyTorch serialization: digest verification establishes byte identity before MMSegmentation loads it, not publisher authentication. Neither surface provides calibrated pixel uncertainty, production serving or the still-missing composed-worker release. The standalone E2E notebook remains **Candidate** pending a clean execution of the provenance-refreshed carrier and an explicit promotion decision.

###### Primary Intended Uses

Implemented uses are exploratory pretrained scene parsing and supervised transfer-learning instruction. The E2E notebook demonstrates a bounded local path from aligned RGB images and class-index masks through validation, split construction, head replacement, frozen-backbone optimization, held-out evaluation, adapter export and fresh reload. The generated `background` / `road` / `structure` data is intentionally simple and suitable only for execution checks. Operators may supply representative data through the bounded BYOD path, but they remain responsible for labels, rights, leakage control and domain validity. The notebook-local path must not be represented as the unavailable DIMER composed-worker adaptation release, a full-network fine-tuner, a production service or evidence of real-world segmentation quality.

###### Primary Intended Users

Intended users are machine-learning engineers, imaging analysts, researchers, educators and DIMER integrators who can operate the pinned Python 3.10/OpenMMLab environment and interpret dense masks. They should understand semantic versus instance segmentation, class-index rasters, mIoU, class imbalance, split leakage, image/mask registration and distribution shift. `pip install .` alone is not a qualified OpenMMLab bootstrap because MMCV/PyTorch binary wheels depend on specialized indexes; the notebook carries the tested install sequence. BYOD users additionally own dataset rights, class definitions, mask quality, representative sampling and artifact governance. This repository is not a zero-context consumer tool or autonomous decision system.

###### Out-of-scope use cases

This repository does not implement instance, panoptic, depth or video segmentation. Pretrained inference emits one ADE20K class index per pixel; adapted inference emits one class from the caller's ordered target vocabulary. Neither path exposes calibrated per-pixel probabilities, abstention or boundary confidence. Notebook-local adaptation does not unfreeze the Swin-T backbone, publish a general DIMER artifact, provide a remote worker or create a production serving composition. The canonical DIMER semantic-segmentation task profile, representation profile, validator and finetuner worker releases, accelerator qualification and composed release remain absent. Multispectral and medical imagery, safety-critical perception, autonomous consequential decisions, surveillance/profiling, coercive land-rights adjudication, unlawful discrimination and deceptive presentation of generated masks remain outside scope.

#### Factors

Behavior depends on scene content, optics, resolution, color rendering, lighting, compression, object scale, occlusion and distance from ADE20K's web/consumer photographic distribution. Adaptation adds mask alignment, class-index encoding, class balance, split construction, initialization, sample order, optimizer settings and annotation quality. The tutorial controls dataset generation, splitting, re-heading and optimization seeds, but a single deterministic synthetic split cannot characterize geographic, seasonal, sensor, demographic or rare-class variability. Operators must treat reproducible execution, local sample fit and real-domain validity as separate questions; improvement on generated shapes does not establish transfer to photographs or operational imagery.

###### Groups

ADE20K is a scene-parsing dataset rather than a group-labelled human dataset, but its imagery includes people and private/public environments and its class list includes `person`. The upstream model can therefore behave differently across social, geographic or environmental contexts even though this repository has not conducted group-disaggregated evaluation. The two tutorial fixtures are not suitable for fairness measurement and must not be represented as such. Where segmentation outputs touch people, neighborhoods, property, mobility, occupation or other proxies connected to sensitive characteristics, operators should perform representative labelled evaluation and governance review before use. They should inspect per-group or context-specific IoU and error patterns where such grouping is lawful and appropriate. No measured disparity in this repository should be interpreted as evidence of parity because the necessary study has not been performed. Uses designed to infer or act on protected characteristics are outside intended scope.

###### Instrumentation

The inference API validates readable images, positive dimensions and the 64,000,000-pixel ceiling before pinned MMSegmentation preprocessing. The notebook-local dataset validator accepts RGB images or paths plus two-dimensional class-index masks, rejects missing fields, dimension mismatch, invalid class indices and excessive image size, and never interpolates mask labels. Camera optics, compression, color management, sensor noise and resizing still influence predictions, while annotation tools, boundary conventions and image/mask registration can introduce systematic training noise. These repository-local checks support the tutorial representation identifier `core.dataset.vision.raster-mask`; they are not a released ML Worker Contract validator or substitute for operator review of data rights and semantics.

###### Environment

The supported notebook environment is CPython 3.10 with torch 2.1.2+cpu, MMSegmentation 1.2.2, MMCV 2.1.0, MMEngine 0.10.7, NumPy 1.26.4 and the other exact pins carried from `tools/pins.txt`. The API fails closed on OpenMMLab version drift and verifies the 240,154,742-byte checkpoint at SHA-256 `e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064` before deserialization. CPU is the supported E2E path; GPU precision, VRAM and repeatability are not qualified. GitHub Actions run `34974178073` completed the pre-refresh carrier from a scratch directory, but its embedded source-revision label was stale. The provenance-refreshed notebook therefore remains Candidate until that exact blob is executed and recorded.

#### Metrics

The E2E notebook reports mIoU, per-class IoU and pixel accuracy on the held-out synthetic split, plus a constant-majority baseline and pre/post-adaptation comparison. These measures fit single-label semantic segmentation: mIoU reduces domination by large classes, per-class IoU exposes which target class fails and pixel accuracy remains an intentionally secondary coverage measure. The 2026-09-15 scratch execution asserted a `sample-sanity` evaluation and exact artifact reload, but its exact numeric outputs were not retained in the GitHub log and are not reported here. Historical mIoU `0.3869899942` and pixel accuracy `0.7065385286` belong only to the replaced two-fixture inference carrier. Neither result is a benchmark or acceptance criterion.

###### Performance Measures

The notebook evaluates through the repository's public `DimerSwinSegmenter.evaluate` API over six held-out generated records. `semantic_iou` accumulates intersections and unions across labelled pixels and returns mIoU, pixel accuracy and per-class IoU; `majority_class_baseline` scores the constant majority-mask predictor on the same records. A pre-adaptation report makes improvement attributable to the bounded training stage rather than the architecture alone. The tutorial does not measure corruption robustness, boundary F-score, calibration, latency distributions, memory ceilings, geographic transfer, multispectral behavior or group-level parity. Deployment evaluation requires a much larger independent and representative labelled set, per-class inspection and domain-specific failure costs.

###### Decision thresholds

Both pretrained and adapted prediction use per-pixel argmax over model logits; every output pixel receives one class index and the package exposes no calibrated confidence, abstention policy or universal acceptance threshold. The adapted path changes only the ordered target vocabulary and trainable heads, not this decision rule. A high logit or softmax value must not be interpreted as a calibrated probability of correctness. Applications needing uncertainty-aware masks must design and validate calibration, abstention and boundary policy on representative labelled data. The tutorial introduces no hidden cutoff, and the composed-worker scaffold defines no deployment acceptance rule.

###### Approaches to uncertainty and variability

The default adaptation result comes from one deterministic 18/6 split and one seeded training run; it reports no standard deviation, confidence interval, cross-validation or bootstrap estimate. Dataset generation, split order, head initialization and minibatch order are seeded, while platform kernels and numerical libraries can still vary. Six constructed validation scenes are far too few to estimate real-domain generalization, and per-pixel logits are not calibrated uncertainty estimates. Version pins, digest checks and exact reload comparison reduce software drift but do not address sampling uncertainty or distribution shift. Real evaluation should use independent representative data, repeated seeds and class-stratified uncertainty analysis.

#### Ethical considerations and biases

Dense scene parsing can affect people indirectly even when output classes are generic. ADE20K contains web/consumer scenes that may reflect geographic, cultural and socioeconomic skews, and the upstream model may encode those patterns without a group audit. Fine-tuning on operator images can amplify sampling bias, annotation bias, private-data exposure or discriminatory proxies rather than cure them. A visually coherent mask can look authoritative, increasing automation-bias risk under distribution shift. Pinned identity, digest checks, dataset validation, runtime checks and safe adapter reload reduce technical drift but do not eliminate social-context risks. Consequential uses require independent validation, human oversight and governance; surveillance, medical diagnosis, coercive property decisions and discriminatory profiling remain outside scope.

###### Data

The pretrained checkpoint was trained upstream on ADE20K, approximately twenty thousand scene-parsing images over 150 classes collected from broader web/scene sources; this repository does not redistribute that corpus. The default adaptation data consists of 24 deterministic generated scenes and exact programmatic masks over three teaching classes, not people or real habitats. Optional BYOD image/mask records stay inside the notebook runtime and are not sent to a DIMER service, but a hosted notebook still processes them on third-party infrastructure. Users must not upload confidential, personal, proprietary or restricted imagery without authorization and remain responsible for consent, rights, split leakage, class semantics, annotation quality and representativeness.

###### Human Life

This repository is not intended for autonomous or materially consequential decisions involving health, criminal justice, employment, housing, credit, insurance, education, immigration, public benefits, policing or physical safety. General segmentation and easy local fine-tuning can still be repurposed for medical delineation, surveillance or land-use decisions affecting rights and livelihoods; those uses require domain-specific validation and governance beyond anything demonstrated here. A successful synthetic tutorial run does not establish clinical, legal, safety or human-rights fitness. Any proposed human-impact use needs representative data, accountable human authority, error analysis, uncertainty policy and applicable regulatory clearance; profiling, discriminatory and coercive uses remain unacceptable regardless of measured accuracy.

###### Mitigations

Implemented mitigations include exact OpenMMLab version checks, image and raster-mask validation, a pixel ceiling, checkpoint size/SHA-256 verification before code-capable `.pth` deserialization, deterministic split and training seeds, a frozen backbone, bounded epochs and batches, majority-baseline comparison, adapter-only export and `weights_only=True` reload with exact-mask verification. The notebook writes an input manifest, evaluation report, result/provenance JSON, class-coverage CSV and adapter. A new validation gate binds `generated_from.revision` to the exact committed source modules carried by the notebook. Separately, `scripts/verify_scaffold.py` refuses promotion of the composed-worker surface while task, representation, worker and accelerator blockers remain. These controls do not replace domain validation, data governance, fairness review, access control or legal assessment.

###### Risks and harms

Primary risks are systematic misclassification, boundary errors, rare-class failure, silent distribution shift and misleading pixel accuracy on imbalanced masks. Fine-tuning adds risks from misregistered or mislabeled masks, train/validation leakage, class imbalance, private-data exposure, overfitting and a custom vocabulary that users may misinterpret. Because every pixel receives a class, visually coherent but incorrect masks can invite automation bias. The `.pth` trust boundary remains code-capable even after exact-byte verification. MIT redistribution of the adaptation-lineage checkpoint requires preservation of its copyright and permission notice. Easy inference and adaptation can lower barriers to surveillance, property analysis or profiling; tutorial evidence must not bypass governance or domain validation.

###### Use cases

Unacceptable uses include biometric or demographic profiling, surveillance or social scoring; unlawful discrimination or segmentation-derived proxies for eligibility in employment, housing, credit, insurance, education, healthcare or public services; autonomous medical diagnosis or safety-critical control; coercive property or law-enforcement decisions based solely on generated masks; deceptive presentation of masks as verified measurements; and any use violating rights in supplied imagery or upstream terms. Acceptable experimentation must distinguish pretrained ADE20K inference, notebook-local bounded adaptation and the unavailable composed-worker release. Users must not represent Candidate tutorial execution as production serving, calibrated confidence, benchmark validation or DIMER worker qualification.

---

## Model details

| Item | Value |
|---|---|
| Pipeline id | `org.valcorza.swin-segmentation` |
| Implemented capability | pretrained `TASK-INFERENCE` plus notebook-local `E2E` frozen-backbone head adaptation through package `dimer-swin-segmentation` 0.1.0 |
| Inference model | OpenMMLab Swin-T + UPerNet, MMSegmentation 1.2.2; checkpoint 240,154,742 bytes, SHA-256 `e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064` |
| Inference outputs | 2-D ADE20K class-index mask over 150 classes |
| Release tutorial | `tutorials/swin_segmentation_colab.ipynb` (`E2E`, standalone 2.0, Candidate) |
| Tutorial evidence | PR #10 scratch execution passed 16/16 code cells and output assertions, but carried stale source provenance; refreshed carrier rerun pending |
| Notebook-local adaptation | 24 generated scenes, 18/6 split, three re-headed classes, frozen Swin-T backbone, bounded AdamW, mIoU/pixel/per-class evaluation, adapter export/reload |
| Composed-worker adaptation lifecycle | `scaffold`, intended `COMPOSED-WORKERS` / `GRADIENT-ADAPTATION` |
| Adaptation lineage checkpoint | Microsoft/SwinTransformer `upernet_swin_tiny_patch4_window7_512x512.pth`, SHA-256 `c26408bb5ddee935dcc709ad7815fa039f2db2b41134cd702da7da277d1a7d89` |
| Adaptation weight hosting | `redistribution_status: permitted`, `license: MIT`, `dimer_hosting: CLEARED` |
| Composed-worker blockers | semantic-segmentation task profile, representation profile, validator release, finetuner release, accelerator qualification and release composition |

## Immutable provenance

- **Inference model identity (fleet snapshot scheme):** `MODEL_ID` `open-mmlab/mmsegmentation:swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512` — the config recipe inside the pinned MMSegmentation 1.2.2 package; `MODEL_REVISION` `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8` is the `v1.2.2` release-tag commit of `open-mmlab/mmsegmentation` (the config source; the checkpoint itself has no git revision); `MODEL_KEY` `swin-t-upernet-ade20k`; weights licence Apache-2.0 (OpenMMLab).
- **Snapshot manifest:** `weights/swin-t-upernet-ade20k/dimer-base-manifest.json` pins the single checkpoint file `upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth` at 240,154,742 bytes, SHA-256 `e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064` (equal to `MODEL_SPEC`); `verify_snapshot` refuses any manifest that disagrees with `MODEL_SPEC`, and `stage_missing_files` fetches only an absent checkpoint from the pinned `download.openmmlab.com` URL. The `.pth` is code-capable serialization deserialized by the pinned MMSegmentation loader after the digest check; see `docs/WEIGHTS.md`.
- **Standalone tutorial:** `tutorials/swin_segmentation_colab.ipynb` carries `src/dimer_swin_segmentation/metrics.py`, `samples.py` and `runtime.py` plus the manifest and runtime pins inline under Notebook Specification 2.0. Its `generated_from.revision` is validated against the committed bytes of all carried modules; release status remains Candidate until the refreshed notebook blob passes and records the supported clean-runtime execution gate.

## References

- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. arXiv:2103.14030.
- Xiao, T. et al. *Unified Perceptual Parsing for Scene Understanding.* ECCV 2018. arXiv:1807.10221.
- Zhou, B. et al. *Scene Parsing through ADE20K Dataset.* CVPR 2017.
- Microsoft Swin Transformer and Swin semantic-segmentation repositories.
- OpenMMLab MMSegmentation 1.2.2 model distribution used by the implemented task-inference runtime.
- `spec/task-inference-surface.json`, `spec/pipeline-surface.json`, `provenance/open-weights.json`, and `tutorials/README.md` in this repository.
