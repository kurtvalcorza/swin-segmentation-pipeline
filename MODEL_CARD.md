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
task_inference_status: release-grade
task_inference_surface: spec/task-inference-surface.json
---

# Swin Semantic Segmentation Pipeline (org.valcorza.swin-segmentation) — release-grade pretrained inference; gradient adaptation scaffold

###### Description

This repository has **two separate capability surfaces**. It now ships an implemented pretrained `TASK-INFERENCE` runtime for Swin-T + UPerNet through `dimer_swin_segmentation.DimerSwinSegmenter`, the `dimer-swin-segment` CLI, `spec/task-inference-surface.json`, and `tutorials/swin_segmentation_task_inference.ipynb`. The runtime uses the pinned OpenMMLab MMSegmentation 1.2.2 distribution, verifies exact checkpoint size and SHA-256 before deserialization, validates input images, and emits a two-dimensional semantic class-index mask over the 150 ADE20K classes. Separately, the intended composed-worker `GRADIENT-ADAPTATION` pipeline remains a DIMER Pipeline Specification 1.0 `scaffold`: the DIMER semantic-segmentation task profile, raster-mask representation, validator/finetuner worker releases, accelerator qualification, composition and release manifest are not yet implemented. The `lifecycle_status: scaffold` front matter refers to that adaptation composition and must not be read as denying the existence of the qualified pretrained inference runtime.

#### Intended Use and Limitations

The supported capability today is pretrained semantic-segmentation inference in the frozen ADE20K 150-class label space. Users can run the repository-owned API/CLI in the qualified Python 3.10/OpenMMLab environment or follow the release-grade notebook to generate class-index masks and reproduce a small labelled tutorial evaluation. The runtime does not train on user data, produce an adapted DIMER artifact or provide a production serving composition. The upstream `.pth` checkpoint is code-capable PyTorch serialization; the runtime verifies the pinned bytes before MMSegmentation loads it, but digest identity is not publisher authentication. Output masks do not include calibrated uncertainty. The two-image tutorial mIoU is execution/evaluation sanity evidence only, not a full ADE20K benchmark reproduction or production validation. DIMER hosting of the separate Microsoft/SwinTransformer adaptation-lineage checkpoint remains blocked while its redistribution status is `unknown`.

###### Primary Intended Uses

The implemented use is **pretrained semantic-segmentation inference** on RGB photographs or photograph-like scenes whose semantics are plausibly represented by ADE20K. Appropriate technical uses include integration testing, model exploration, qualitative scene parsing, pipeline prototyping and controlled research evaluation. The release notebook additionally demonstrates label-aware evaluation using two immutable ADE20K validation fixtures, reporting aggregate mIoU, per-class IoU and pixel accuracy against a constant-majority baseline. Optional BYOD permits new-image inference but does not manufacture a metric without ground truth. A future intended use is supervised full-network gradient adaptation on operator-controlled image/mask pairs, but that remains design intent only until DIMER defines the task/representation contracts and ships validator/finetuner workers and accelerator qualification. The pretrained runtime should not be described as a substitute for that missing fine-tuning surface or as a production application.

###### Primary Intended Users

Current users are machine-learning engineers, geospatial/imaging analysts, researchers and DIMER integrators who can operate a pinned Python environment and interpret dense semantic masks. They should understand the difference between semantic and instance segmentation, class-index rasters, ignore labels, mIoU, class imbalance, image/mask registration and distribution shift. The supported installation contract is the exact dependency bootstrap encoded in the release-grade notebook; `pip install .` alone is not a qualified OpenMMLab installation because MMCV/PyTorch binary wheels depend on specialized package indexes. The future gradient-adaptation capability targets the same technical audience but adds responsibility for mask quality, label maps, split integrity, GPU execution and artifact governance. This repository is not intended as a zero-context consumer tool or as an autonomous decision system. Users are expected to replace tutorial fixtures with domain-representative labelled data before making any deployment claim.

###### Out-of-scope use cases

This repository is not an instance-segmentation, object-detection, panoptic-segmentation, depth-estimation or video-segmentation pipeline. The current inference model exposes only one ADE20K class index per output pixel; it does not provide calibrated per-pixel probability, abstention or boundary confidence. The public runtime is not a fine-tuner and cannot replace ADE20K's taxonomy with an operator's custom classes. The repository does not yet provide the canonical DIMER semantic-segmentation task profile, raster-mask validator, finetuner worker, adapted artifact format or production serving composition. Multispectral imagery, medical segmentation, safety-critical perception and imagery far outside the upstream photographic domain are not qualified. Autonomous consequential decisions, surveillance/profiling, land-rights adjudication, unlawful discrimination and deceptive use of generated masks are unacceptable. Redistribution or DIMER hosting of the adaptation-lineage checkpoint is also out of scope until licensing is authoritatively resolved.

#### Factors

Model behavior depends on scene content, camera characteristics, spatial resolution, color rendering, lighting, compression, object scale, occlusion and how closely imagery resembles ADE20K's web/consumer photographic domain. Semantic segmentation is especially sensitive to image scale and boundaries because every pixel receives a class index. Evaluation is also sensitive to ground-truth registration, ignore-label handling and label-map correctness. The exact-head tutorial demonstrates that the frozen runtime produces masks and meaningful metrics on two known ADE20K fixtures, but those fixtures cannot characterize geographic, seasonal, sensor, demographic or rare-class variability. Future gradient adaptation adds further factors such as mask annotation quality, class imbalance, train/validation split construction, random seeds, augmentation, GPU kernels and training duration. Those training factors are not part of the implemented inference surface. Operators should treat runtime reproducibility, domain validity and statistical representativeness as separate questions rather than inferring one from another.

###### Groups

ADE20K is a scene-parsing dataset rather than a group-labelled human dataset, but its imagery includes people and private/public environments and its class list includes `person`. The upstream model can therefore behave differently across social, geographic or environmental contexts even though this repository has not conducted group-disaggregated evaluation. The two tutorial fixtures are not suitable for fairness measurement and must not be represented as such. Where segmentation outputs touch people, neighborhoods, property, mobility, occupation or other proxies connected to sensitive characteristics, operators should perform representative labelled evaluation and governance review before use. They should inspect per-group or context-specific IoU and error patterns where such grouping is lawful and appropriate. No measured disparity in this repository should be interpreted as evidence of parity because the necessary study has not been performed. Uses designed to infer or act on protected characteristics are outside intended scope.

###### Instrumentation

The inference API validates that an input path is a readable image with positive dimensions and no more than 64,000,000 pixels, then delegates preprocessing to the pinned MMSegmentation configuration. Camera optics, compression, color management, sensor noise, resizing and aspect ratio can influence results. The release notebook uses two fixed ADE20K image/mask fixtures and explicitly handles ADE20K's raw-zero ignore convention before computing IoU. For future adaptation, the annotation instrument becomes equally important: image/mask alignment, class-index encoding, boundary conventions and discrete-mask interpolation rules can introduce systematic label noise. A proper DIMER raster-mask validator should eventually reject image/mask dimension mismatches and undeclared class values and should never resample class indices with interpolation that invents labels. Those validator guarantees are not claimed today. The implemented boundary is image validation plus the controlled tutorial evaluation path; training-data validation remains a blocker for the future adaptation composition.

###### Environment

The qualified task-inference environment is CPython 3.10 with torch 2.1.2, MMSegmentation 1.2.2, MMCV 2.1.0, MMEngine 0.10.7, NumPy 1.26.4, OpenCV 4.10.0.84 and the required MMSegmentation text dependencies. CPU inference is supported and is the path exercised in clean GitHub-hosted notebook execution. The API verifies the pinned OpenMMLab versions at startup and fails closed on drift. Because MMCV and PyTorch rely on platform-specific binary wheels, `pip install .` is not represented as a complete environment bootstrap; the release-grade notebook contains the qualified install sequence. The runtime verifies the 240,154,742-byte checkpoint and SHA-256 `e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064` before deserialization. The future training environment is not qualified: GPU/VRAM requirements, precision policy, worker containers and repeatability remain open.

#### Metrics

The implemented tutorial reports mean intersection-over-union (mIoU), per-class IoU and pixel accuracy because they are appropriate for single-label semantic segmentation. On the two immutable ADE20K validation fixtures, the exact committed notebook measured aggregate mIoU `0.3869899942` and pixel accuracy `0.7065385286` across 510,803 valid labelled pixels, with 11 classes having nonzero union. It also computed a constant-majority sample baseline mIoU of `0.0539564666`. These values are **small-sample tutorial measurements** and have large sampling uncertainty; they are not reproduction of the upstream ADE20K benchmark and are not acceptance criteria. Upstream full-dataset figures remain upstream-reported context unless independently reproduced. For the future gradient-adaptation pipeline, mIoU and per-class IoU remain intended validation metrics, but no training evaluation exists until the missing DIMER workers and representation contract are implemented.

###### Performance Measures

The release-grade notebook evaluates the actual repository runtime, not a notebook-local substitute. It predicts semantic masks for the fixed fixtures, applies the documented ignore-label conversion, accumulates intersections/unions across valid pixels, writes per-class IoU, and compares the model against a deliberately weak constant-majority baseline. This demonstrates that the supported inference path is executable and evaluable end to end. It does not measure robustness to corruption, boundary F-score, calibration, latency distributions, memory ceilings, geographic transfer, seasonal transfer, multispectral behavior or group-level parity. Pixel accuracy is reported as supplementary because frequent classes can dominate it; mIoU is the primary aggregate metric. Any deployment evaluation should use a substantially larger representative labelled set, inspect per-class behavior and define failure costs appropriate to the real application. The tutorial's measured values should remain attached to the exact fixture/runtime identity recorded in provenance rather than copied as general model specifications.

###### Decision thresholds

The current semantic output rule is effectively per-pixel class selection from the upstream model's logits; every non-ignored output pixel receives one ADE20K class index. The public runtime does not expose a calibrated confidence threshold, abstention policy or universal operating point. A high logit or softmax value should not be interpreted as a calibrated probability that a pixel label is correct. Applications that need uncertainty-aware masking must design and validate that behavior separately on representative labelled data. The tutorial therefore evaluates class-index masks directly and does not introduce a hidden confidence cutoff. The future gradient-adaptation pipeline likewise has no deployment threshold or acceptance rule because training and artifact publication are not implemented. If adaptation is later added, its validation contract should state how ignore labels, class maps, thresholds or abstention are handled. No consequential decision should rely on undocumented threshold behavior.

###### Approaches to uncertainty and variability

The exact-head tutorial proves re-execution of the frozen code/model path on fixed fixtures; it does not provide a statistical uncertainty estimate. Two images are far too few to estimate confidence intervals for generalization, and the resulting mIoU can move sharply if fixture composition changes. Per-pixel outputs are not calibrated uncertainty estimates, and no ensemble, bootstrap or repeated-run dispersion is reported. Version pins, immutable model identity, digest verification and clean CI reduce software/supply-chain variability, but they do not eliminate domain shift. Real variability includes scene type, resolution, geographic context, sensor/camera characteristics, class frequency and annotation boundary choices. Future training will add random-seed, sampling and GPU-kernel variability that must be separately documented when the gradient-adaptation surface exists. Operators should distinguish deterministic execution from statistical reliability and should use representative validation sets, repeated studies where necessary, and domain-specific acceptance criteria before relying on masks operationally.

#### Ethical considerations and biases

Dense scene parsing can affect people indirectly even when the output classes are generic. ADE20K contains web/consumer scenes that may reflect geographic, cultural and socioeconomic skews, and the upstream model may encode those patterns without an accompanying group audit. A segmentation mask can look authoritative because it covers every pixel, which increases automation-bias risk under distribution shift. Technical integrity controls in this repository—pinned model identity, pre-deserialization digest checks, runtime version checks, narrow output semantics and exact-notebook execution evidence—do not eliminate social-context risks. Consequential applications require independent validation, human oversight and governance. Uses involving surveillance, medical diagnosis, land/property adjudication, discriminatory proxies or coercive action are outside intended scope. Operators are also responsible for rights and privacy in input imagery. Fine-tuning on private operator data, when eventually supported, could amplify dataset bias or governance problems rather than cure them automatically.

###### Data

The pretrained model was trained upstream on ADE20K, a scene-parsing dataset with roughly twenty thousand training images and 150 semantic classes collected from broader web/scene datasets. This repository does not redistribute the training corpus. The release tutorial retrieves two immutable public ADE20K validation fixtures solely to exercise evaluation and records their provenance; they are not a representative benchmark sample. Optional BYOD stays within the notebook runtime and is not sent to an inference service. Users must not upload confidential, sensitive or restricted imagery to hosted notebook environments without authorization. The future gradient-adaptation design is intended to consume operator-controlled image/mask pairs, but the canonical DIMER raster-mask representation and validator do not yet exist. Therefore the repository makes no current claim that arbitrary training masks are aligned, legally usable, free of leakage, correctly indexed or sufficiently representative. Those requirements remain operator obligations and future worker responsibilities.

###### Human Life

This repository is not intended for autonomous or materially consequential decisions involving health, criminal justice, employment, housing, credit, insurance, education, immigration, public benefits, policing or physical safety. General scene segmentation can still be repurposed for sensitive tasks such as medical delineation, surveillance parsing or land-use decisions that affect rights and livelihoods; those uses require domain-specific validation and governance beyond anything demonstrated here. The successful tutorial run and sample mIoU do not establish clinical, legal, safety or human-rights fitness. Any proposed human-impact use would need representative data, accountable human decision authority, error analysis, uncertainty/threshold policy and applicable regulatory clearance; some profiling, discriminatory and coercive uses remain unacceptable regardless of accuracy. The future availability of fine-tuning would not itself make such applications appropriate. Better task performance cannot substitute for legitimate purpose, rights-respecting data practices and accountable decision processes.

###### Mitigations

Implemented inference mitigations include a repository-owned API/CLI, exact OpenMMLab runtime version checks, image validation, an operational pixel ceiling, exact checkpoint size/SHA-256 verification before code-capable `.pth` deserialization, a narrow 150-class semantic-mask contract, provenance export and exact-notebook clean execution. The tutorial distinguishes sample measurements from upstream benchmark claims, handles ignore labels explicitly and compares against a simple baseline. The adaptation scaffold retains separate lifecycle controls: `scripts/verify_scaffold.py` refuses promotion while task/representation/worker/accelerator blockers remain, and `provenance/open-weights.json` keeps DIMER hosting blocked while adaptation-lineage redistribution is `unknown`. These mechanisms reduce accidental drift, substitution and overclaiming. They do not replace domain validation, data governance, fairness review, access control, audit logging, user training or legal assessment. Operators must add those controls according to the application context.

###### Risks and harms

Primary technical risks are systematic misclassification, boundary errors, rare-class failure, silent distribution shift and misleading pixel accuracy on imbalanced scenes. Because every pixel receives a class, users may over-trust visually coherent but incorrect masks. ADE20K-derived representations can also encode geographic or contextual biases that are unmeasured here. The upstream `.pth` trust boundary remains code-capable even though exact bytes are verified before loading. A distinct licensing risk applies to DIMER redistribution of the future adaptation-lineage Microsoft checkpoint, so hosting remains blocked. The existence of an easy inference API can lower the barrier to sensitive surveillance, property analysis or profiling; explicit scope and governance are therefore essential. Future fine-tuning could add harms from mislabeled masks, class imbalance, private-data leakage or dataset bias if operator data is poorly governed. The repository's tutorial evidence should not be used to bypass those assessments.

###### Use cases

Unacceptable uses include biometric/demographic profiling, surveillance or social scoring; unlawful discrimination or using segmentation-derived measures as proxies for eligibility in employment, housing, credit, insurance, education, healthcare or public services; autonomous medical diagnosis or safety-critical control without independently validated domain systems; coercive land/property or law-enforcement decisions based solely on generated masks; deceptive presentation of masks as verified measurements; and any use that violates rights in the input imagery or upstream terms. DIMER redistribution of checkpoint bytes is also prohibited while the relevant weight-licence determination is unresolved. Acceptable technical experimentation must preserve the distinction between pretrained ADE20K inference and the unavailable gradient-adaptation pipeline. No user should represent the existence of the release-grade tutorial as proof that task-specific training, production serving, calibration or deployment qualification exists. Those capabilities require their own contracts and evidence.

---

## Model details

| Item | Value |
|---|---|
| Pipeline id | `org.valcorza.swin-segmentation` |
| Implemented capability | pretrained `TASK-INFERENCE`, `spec/task-inference-surface.json`, package `dimer-swin-segmentation` 0.1.0 |
| Inference model | OpenMMLab Swin-T + UPerNet, MMSegmentation 1.2.2; checkpoint 240,154,742 bytes, SHA-256 `e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064` |
| Inference outputs | 2-D ADE20K class-index mask over 150 classes |
| Release tutorial | `tutorials/swin_segmentation_task_inference.ipynb` (`TASK-INFERENCE`, release-grade) |
| Tutorial evidence | 2 ADE20K fixtures; mIoU `0.3869899942`, pixel accuracy `0.7065385286`, majority baseline mIoU `0.0539564666`; tutorial-only evidence |
| Adaptation lifecycle | `scaffold`, intended `COMPOSED-WORKERS` / `GRADIENT-ADAPTATION` |
| Adaptation lineage checkpoint | Microsoft/SwinTransformer `upernet_swin_tiny_patch4_window7_512x512.pth`, SHA-256 `c26408bb5ddee935dcc709ad7815fa039f2db2b41134cd702da7da277d1a7d89` |
| Adaptation weight hosting | `redistribution_status: unknown`, `dimer_hosting: BLOCKED` |
| Adaptation blockers | semantic-segmentation task profile, raster-mask representation, validator release, finetuner release, accelerator qualification |

## References

- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. arXiv:2103.14030.
- Xiao, T. et al. *Unified Perceptual Parsing for Scene Understanding.* ECCV 2018. arXiv:1807.10221.
- Zhou, B. et al. *Scene Parsing through ADE20K Dataset.* CVPR 2017.
- Microsoft Swin Transformer and Swin semantic-segmentation repositories.
- OpenMMLab MMSegmentation 1.2.2 model distribution used by the implemented task-inference runtime.
- `spec/task-inference-surface.json`, `spec/pipeline-surface.json`, `provenance/open-weights.json`, and `tutorials/README.md` in this repository.
