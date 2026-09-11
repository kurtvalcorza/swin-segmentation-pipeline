---
license: mit
model_card_spec: "1.0"
pipeline_spec: "1.0"
base_model: SwinTransformer/storage upernet_swin_tiny_patch4_window7_512x512.pth (release v1.0.1, asset 34862982)
base_model_sha256: c26408bb5ddee935dcc709ad7815fa039f2db2b41134cd702da7da277d1a7d89
base_model_weights_license: unknown — not yet determined from an authoritative upstream statement; DIMER hosting BLOCKED
pipeline_id: org.valcorza.swin-segmentation
lifecycle_status: scaffold
implementation_topology: COMPOSED-WORKERS
capability_modes:
  - GRADIENT-ADAPTATION
task_profile_candidate: core.task.vision.semantic-segmentation
---

# Swin Semantic Segmentation Pipeline (org.valcorza.swin-segmentation) — scaffold, no packaged version

###### Description

This repository is a **scaffold**: it freezes the task, architecture lineage, checkpoint provenance and contract blockers for a future DIMER semantic-segmentation pipeline, and ships a verifier for those frozen claims. **It contains no runtime.** Nothing here loads a model, validates a dataset, trains, or predicts, and no version has been packaged. The card exists so that the intended model and its boundaries are stated before implementation, and so that a reader cannot mistake the scaffold for a runnable pipeline.

The intended model is the official **Swin-T + UPerNet** semantic-segmentation checkpoint from the Microsoft Swin Transformer project — *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (Liu et al., 2021, arXiv:2103.14030) with the UPerNet decode head (Xiao et al., 2018, arXiv:1807.10221) — trained on ADE20K at 512×512 and published as `upernet_swin_tiny_patch4_window7_512x512.pth` in the `SwinTransformer/storage` GitHub release `v1.0.1` (asset 34862982, 240,144,566 bytes, SHA-256 `c26408bb…`). A Swin transformer is a hierarchical vision transformer computing self-attention inside shifted local windows; UPerNet fuses its multi-scale features through a feature-pyramid and pyramid-pooling head into one class logit per pixel. At inference the network maps an RGB image to a per-pixel class mask; the intended adaptation is **full-network gradient fine-tuning** on the operator's own image/mask pairs, mirroring the classification sibling.

What this repository adds today is the pinned provenance (`provenance/open-weights.json`), the machine-readable task surface and blockers (`spec/pipeline-surface.json`), the fail-closed verifier (`scripts/verify_scaffold.py`) and a `SMOKE` notebook. The validator and finetuner workers, the `ml-worker` task profile and raster-mask representation, and any evaluation are **not implemented**.

#### Intended Use and Limitations

Everything in this container describes design intent for a pipeline that does not yet exist; no use is currently supported.

###### Primary Intended Uses

**Intended task (not yet implemented):** single-label-per-pixel semantic segmentation by supervised fine-tuning. Input would be an image/mask dataset in a raster-mask representation (`core.dataset.vision.raster-mask`, not yet present on the reviewed contract branch); output would be a per-pixel class mask at the input resolution, with mIoU and per-class IoU as primary metrics, and a deployable artifact for DIMER serving.

**Application domains envisioned:** dense labelling of scenes where every pixel belongs to one class — land-cover and crop tiles from aerial or satellite imagery at ~512×512, indoor/outdoor scene parsing, material or defect maps in inspection imagery. The upstream ADE20K training gives a scene-parsing prior over 150 everyday classes; the intended pipeline would replace that label space with the operator's.

**Role in a larger system:** the second of three Swin task pipelines for DIMER, reusing the validator/finetuner split, the digest-pinned catalog and the artifact contract established by `swin-classification-pipeline`. Until the workers exist, the repository's only role is to hold the frozen specification and refuse to be treated as anything more.

###### Primary Intended Users

The intended users of the eventual pipeline are **machine-learning engineers and geospatial or imaging analysts operating DIMER deployments**, in an internal enterprise or research setting where the operator controls the data, the GPU and the downstream use of masks.

They are assumed to understand pixel-level annotation and its cost, the difference between semantic and instance segmentation, how a train/validation split of image–mask pairs is formed, why IoU rather than pixel accuracy is the headline metric on imbalanced classes, and that ADE20K pretraining transfers to photographs far better than to multispectral or medical rasters.

Today the repository has only one user role: **a maintainer or reviewer** checking that the scaffold's claims stay internally consistent (`python scripts/verify_scaffold.py`) and that the checkpoint identity has not drifted. It is not for hobbyist or self-service use, and it currently offers nothing an end user can run.

###### Out-of-scope use cases

Capability boundaries:

- Not for any inference, training or evaluation **today** — the repository has no runtime; the `SMOKE` notebook only runs the verifier.
- Not for instance segmentation or object detection: instance masks belong to the Mask R-CNN lineage of the sibling `swin-detection-pipeline`, and are explicitly `OUT_OF_SCOPE` here.
- Not for image classification (see `swin-classification-pipeline`), panoptic segmentation, depth estimation, or video segmentation.

Input boundaries (intended, not enforced yet):

- Not for datasets without pixel-aligned masks whose class indices are frozen by a label map; a raster-mask validator does not exist yet, so nothing validates them.
- Not for imagery whose semantics depend on bands beyond RGB; the upstream checkpoint consumes 3-channel input.
- Not for resolution regimes far from the 512×512 training crop without the operator's own re-evaluation.

Decision boundaries:

- Not for autonomous decisions affecting people — medical image segmentation, security, land-rights adjudication from satellite masks — with or without a human in the loop; nothing has been validated.
- Not for redistribution or hosting of the upstream checkpoint through DIMER: its `redistribution_status` is `unknown` and `dimer_hosting` is `BLOCKED` in `provenance/open-weights.json`.

#### Factors

No behaviour has been measured by this repository; the subsections record what is known about the upstream model and what the eventual pipeline will have to consider.

###### Groups

The intended pipeline is **not human-centric by design** — it labels pixels by operator-defined classes — but ADE20K, the upstream training set, contains scenes with people and its class list includes `person`; the upstream authors publish no group-level audit, so the pretrained features are **not group-audited**. No evaluation of any kind has been performed here.

The obligation transfers to the eventual operator: where masks touch people or correlate with people's characteristics (street scenes, indoor scenes with occupants), the operator must run a group-disaggregated evaluation — per-group IoU on a labelled holdout they control — before deployment. The pipeline will not perform that audit; the scaffold records the obligation so it is not forgotten when the runtime arrives.

###### Instrumentation

Training and evaluation data for the eventual pipeline will be **operator-supplied image/mask pairs**; the repository will know nothing about the camera, sensor, orthorectification or annotation tool that produced them. Upstream, ADE20K images are consumer and web photographs of mixed provenance, annotated by hand.

Instrument characteristics that will reach the model: spatial resolution (resize to the training crop), colour encoding (RGB), compression artefacts, and — critically for segmentation — the **registration between image and mask** and the annotation tool's boundary conventions. Instrument error propagates as label noise at object boundaries and as distribution shift when sensor or season changes; the intended validator must at minimum refuse image/mask size mismatches and undeclared class indices, and must never resample discrete masks with interpolation that invents classes (Pipeline Spec §21.11). Nothing detects any of this today.

###### Environment

**Operating environment (intended).** Fine-tuning a Swin-T + UPerNet at 512×512 requires an NVIDIA GPU; the sibling classification finetuner's fail-closed `cuda:0` policy is intended to carry over. No qualification packet exists for this repository — `ACCELERATOR_QUALIFICATION_PENDING` is an open blocker — so no VRAM, precision or runtime envelope is claimed. The upstream checkpoint is a PyTorch `.pth` pickle that requires a trusted, task-runtime loader (`torch.load` is code-capable serialization); the repository's own tooling never deserialises it and its notebook says so. The verified environment for the scaffold itself is a Kaggle CPU kernel (Python 3.12.13, Linux 6.12.90) where `verify_scaffold.py` passed.

**Data environment (intended).** The eventual model will assume inference imagery is drawn from the same distribution as the operator's training tiles — same sensor, resolution, season and class semantics — and photographs or photograph-like renderings in the RGB domain ADE20K covers. Degradation under shift is silent: a segmentation network keeps emitting confident masks on out-of-distribution imagery. None of this is measured here.

#### Metrics

No metric is computed by this repository. The subsections state what the eventual pipeline is specified to report and what the upstream authors reported, kept separate.

###### Performance Measures

**Measured by this repository: nothing.** The scaffold has no evaluation code, and its verifier proves only internal consistency of the specification and provenance files.

**Specified for the eventual pipeline** (`spec/pipeline-surface.json → metrics`): **mean IoU** (intersection-over-union averaged over classes — the headline segmentation metric because it is insensitive to class frequency and penalises both false-positive and false-negative pixels) and **per-class IoU** (which reveals the classes the mean hides). Pixel accuracy is deliberately not primary: on imbalanced masks it is dominated by background. Dice and boundary metrics may be added when the evaluation contract is written.

**Reported upstream, not reproduced here:** the official checkpoint's ADE20K validation mIoU is 44.51 single-scale and 45.81 with multi-scale + flip testing (`provenance/open-weights.json → canonicalV1.reportedMetrics`, from the upstream task repository's model table). These are the upstream authors' numbers on their benchmark; this repository has not executed the model and makes no performance claim.

###### Decision thresholds

The intended default decision rule is a per-pixel **`argmax` over class logits** — every pixel receives exactly one class, even where the network is uncertain — with an `ignore_index` for unlabelled pixels in training masks, following the upstream UPerNet convention. No probability threshold, abstention or minimum-confidence mask is intended to ship: the outputs would be uncalibrated softmax maps, and calibration and any confidence-based masking are the deploying operator's to design on labelled holdout data from their domain, weighing over-segmentation against missed regions.

No acceptance threshold was set because nothing has been trained. When the finetuner exists, its publication rule (publish whatever the validation metrics are, and record them) is intended to match the classification sibling. All of the above is design intent recorded here so it can be checked against the implementation when it lands.

###### Approaches to uncertainty and variability

**Nothing is estimated by this repository**, so there is no estimation procedure, dispersion or seed policy to report yet. The upstream mIoU figures quoted above are single numbers from the upstream authors' evaluation protocol; they carry no dispersion and are not this repository's evidence.

For the eventual pipeline the intended design is the classification sibling's: a single validator-frozen validation holdout, explicit seeds for initialisation and data order, `reproducibility: REEXECUTABLE` because GPU kernel selection remains non-deterministic, and softmax maps declared uncalibrated. Anything stronger — repeated runs, confidence intervals, calibration — would be the operator's to add. Until the runtime exists, this section's honest content is that there are no numbers and therefore no uncertainty statement to make about them.

#### Ethical considerations and biases

No external board has reviewed this scaffold and no testing with any population has occurred; nothing below implies otherwise.

###### Data

**Upstream training data.** The canonical checkpoint was trained on **ADE20K** (Zhou et al.), roughly 20,000 hand-annotated scene images across 150 classes, drawn from the SUN and Places databases — web and consumer photographs collected without individual consent, containing people and private interiors. The upstream authors publish the class list but do not enumerate the images' provenance beyond their source databases; whether the corpus contains personal or sensitive imagery cannot be ruled out, and its licence terms have not yet been assessed by this repository.

**Fine-tuning data** would be operator-supplied and, by the intended policy (`runtimeNetworkFetch: DENY`), would never leave the worker's filesystem.

**What this repository distributes:** JSON provenance and specification files, a verifier, a smoke notebook and this card. It distributes **no weights** and **no sample data**. The checkpoint's identity is recorded by size and SHA-256 (verified once in an isolated Kaggle container, `evidence/open-weights-kaggle.json`) but its bytes are not stored here, and DIMER hosting is **BLOCKED** while `redistribution_status` is `unknown`.

**Operator obligation.** The eventual pipeline will perform no audit of the operator's imagery or masks for personal data, consent, licensing or confidentiality; fine-tuned artifacts derived from that data inherit its governance.

###### Human Life

The intended pipeline is **not intended** for decisions in health, safety, criminal justice, employment, credit, housing or any other matter central to human life, and **nothing has been validated** for any purpose: this repository has never executed the model.

Foreseeable sensitive uses of a general segmentation model — medical image delineation, surveillance footage parsing, land-use decisions from satellite masks that affect people's property or subsistence — would be admissible only with an independent domain validation study on representative data, a human decision-maker reviewing every consequential mask, the operator's own calibration and group-disaggregated evaluation, and whatever regulatory clearance the domain requires. None of that is provided or implied here, and the scaffold's blockers mean none of it can even begin yet.

###### Mitigations

Only mechanisms that exist in this repository are listed; the intended worker-level controls are named as intent, not as mitigations.

*Supply-chain integrity (implemented).* The canonical checkpoint is pinned to an official GitHub release asset (`SwinTransformer/storage@v1.0.1`, asset 34862982) with recorded size and SHA-256, verified against the downloaded bytes in an isolated Kaggle container and recorded in `evidence/open-weights-kaggle.json`; the reference config is pinned by task-repository commit (`d70598c0…`) and git blob id. `scripts/verify_scaffold.py` fails if the evidence digest or size disagrees with the provenance, if any revision is not an immutable 40-hex id, or if the source-of-record policy is weakened.

*Fail-closed lifecycle (implemented).* The surface declares `lifecycle_status: scaffold` and five blockers; the verifier refuses `candidate`/`release` while blockers exist, refuses any declared components or release, and requires the composition to be recorded as `NOT_EMITTED`. No pipeline manifest or release is emitted (Pipeline Spec §33).

*Hosting gate (implemented).* `redistribution_status: unknown` with `dimer_hosting: BLOCKED`; the verifier refuses `permitted` without a licence and source, and refuses an unblocked hosting flag for `unknown`/`prohibited`.

*Serialization trust boundary (implemented as a refusal).* Nothing in this repository deserialises the `.pth` checkpoint; the smoke notebook records the digest and stops, stating that PyTorch checkpoints are code-capable.

*Intended, not implemented:* image/mask validation, `ignore_index` handling, no-interpolation of discrete masks, digest-pinned catalog with typed off-catalog refusal, fail-closed GPU, seeded training, content-addressed artifacts with fresh reload — all specified by analogy to `swin-classification-pipeline` and all absent here.

###### Risks and harms

- **Scaffold mistaken for a pipeline.** Mechanism: a reader or tool treats the passing verifier or the notebook as runtime qualification. Bearer: the operator who plans a deployment on it. Mitigated by the lifecycle field, the verifier's refusals and this card, but the risk is the reason those exist.
- **Unlicensed redistribution.** Mechanism: the checkpoint is mirrored before its weight licence is determined. Bearer: the project and upstream rights-holders. Currently blocked by `redistribution_status: unknown`; realised if someone bypasses the gate.
- **Code-capable checkpoint.** Mechanism: `torch.load` of the `.pth` executes pickled objects. Bearer: whoever runs an eventual loader on a tampered file. The SHA-256 pin mitigates substitution, not the trust boundary itself.
- **Silent distribution shift (future).** Mechanism: confident masks on imagery unlike the training tiles. Bearer: whoever acts on the masks and any people depicted or affected. Likely under normal use over time once a runtime exists.
- **Boundary label noise and class imbalance (future).** Mechanism: annotation-boundary conventions and rare classes depress per-class IoU while mIoU looks acceptable. Bearer: operator making go/no-go decisions on one number.
- **Inherited pretraining bias (future).** Mechanism: ADE20K's scene and demographic skews encoded in the backbone. Bearer: under-represented data subjects. Unmeasured.
- **Automation bias (future).** Mechanism: reviewers accept machine masks without inspection. Bearer: data subjects and third parties.

###### Use cases

Uses the developers consider unacceptable for the eventual pipeline, even where it would work:

1. **Surveillance, biometric or demographic profiling, and social scoring** — segmenting people or their belongings in footage to track, identify or characterise individuals or groups.
2. **Unlawful discrimination** — using segmentation-derived measures (property extent, land use, dwelling type) as proxies for eligibility in credit, insurance, housing, employment, education or healthcare access.
3. **Deceptive or manipulative applications** — presenting uncalibrated masks as verified measurements to people affected by them, or fabricating or altering imagery evidence.
4. **Uses prohibited by upstream terms** — the weight licence is not yet determined and ADE20K's own terms have not been assessed; until both are recorded, any redistribution of the checkpoint through DIMER is prohibited by this repository's hosting gate, and any deployment must honour the operator's data licences and DIMER platform terms.

---

## Model details

| Item | Value |
|---|---|
| Pipeline id | `org.valcorza.swin-segmentation` — **no packaged version**; `spec/pipeline-surface.json` status `BLOCKED_PENDING_CONTRACT_AND_WORKERS` |
| Lifecycle / topology / mode | `scaffold` / intended `COMPOSED-WORKERS` / intended `GRADIENT-ADAPTATION` (`dimerPipelineSpec`) |
| Candidate task profile | `core.task.vision.semantic-segmentation` — not present on the reviewed `ml-worker` branch `build/dimer-v1-freeze` |
| Candidate dataset representation | `core.dataset.vision.raster-mask` — not present on the reviewed branch |
| Canonical checkpoint | `upernet_swin_tiny_patch4_window7_512x512.pth`, `SwinTransformer/storage` release `v1.0.1`, asset 34862982, 240,144,566 bytes, SHA-256 `c26408bb5ddee935dcc709ad7815fa039f2db2b41134cd702da7da277d1a7d89` (verified: `evidence/open-weights-kaggle.json`) |
| Reference implementation | `SwinTransformer/Swin-Transformer-Semantic-Segmentation` @ `d70598c00d37855e404e58415e9aac47340ece02`, config `configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py` (blob `aae70e2a…`) |
| Weight licence / hosting | `redistribution_status: unknown`, `dimer_hosting: BLOCKED` (`provenance/open-weights.json → canonicalV1.weightLicensing`) |
| Upstream reported metrics (not reproduced) | ADE20K val mIoU 44.51 (single-scale), 45.81 (multi-scale + flip) |
| Blockers | `CONTRACT_SEMANTIC_SEGMENTATION_TASK_PROFILE_MISSING`, `CONTRACT_RASTER_MASK_REPRESENTATION_MISSING`, `VALIDATOR_WORKER_RELEASE_MISSING`, `FINETUNER_WORKER_RELEASE_MISSING`, `ACCELERATOR_QUALIFICATION_PENDING` |
| Verifier | `python scripts/verify_scaffold.py` — internal consistency and fail-closed lifecycle only; not runtime evidence |

## References

- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. arXiv:2103.14030.
- Xiao, T. et al. *Unified Perceptual Parsing for Scene Understanding.* ECCV 2018. arXiv:1807.10221.
- Zhou, B. et al. *Scene Parsing through ADE20K Dataset.* CVPR 2017.
- Microsoft Swin Transformer: https://github.com/microsoft/Swin-Transformer · Semantic segmentation task repository: https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
- Sibling pipelines: `swin-classification-pipeline` (candidate), `swin-detection-pipeline` (scaffold)
- Smoke notebook: `tutorials/swin_segmentation_scaffold_smoke_colab.ipynb` (`SMOKE`, engineering-only)
