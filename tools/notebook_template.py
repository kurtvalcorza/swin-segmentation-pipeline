"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install (from `tools/pins.txt`), the
embedded package modules (`metrics.py`, `runtime.py`), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "dimer_swin_segmentation",
    "repo_name": "swin-segmentation-pipeline",
    "stem": "swin_segmentation_task_inference",
    "notebook_name": "swin_segmentation_task_inference.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "DimerSwinSegmenter",
    "weights_key": "swin-t-upernet-ade20k",
    "modules": ["metrics.py", "runtime.py"],
    "entry_module": "runtime.py",
    "pins_file": "tools/pins.txt",
    "model_host": {
        "name": "the OpenMMLab checkpoint host (`download.openmmlab.com`)",
        "reference_url": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth",
        "revision_label": "MMSegmentation release-tag commit",
    },
    "runtime_imports": ["torch", "numpy"],
    "title": "Swin-T + UPerNet (ADE20K) — DIMER semantic-segmentation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/swin-segmentation-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/swin-segmentation-pipeline/blob/main/tutorials/swin_segmentation_task_inference.ipynb",
        ),
        (
            "Python 3.10 required",
            "https://img.shields.io/badge/Python-3.10%20required-3776ab?style=flat&logo=python&logoColor=white",
            "https://github.com/kurtvalcorza/swin-segmentation-pipeline/blob/main/README.md",
        ),
        (
            "Checkpoint",
            "https://img.shields.io/badge/OpenMMLab-upernet__swin--t__ade20k__512x512__160k-ffcc4d?style=flat",
            "https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2/configs/swin",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-microsoft%2FSwin--Transformer-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/microsoft/Swin-Transformer",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2103.14030-b31b1b.svg", "https://arxiv.org/abs/2103.14030"),
    ],
    "capability": "pretrained ADE20K-150 semantic segmentation (one class index per pixel) using the pinned OpenMMLab `swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512` checkpoint through the repository's `DimerSwinSegmenter` API",
    "intro": (
        "At inference the pinned Swin-T backbone + UPerNet decode head maps one RGB image to one ADE20K class index "
        "(`0..149`) per pixel; the public API returns that 2-D `uint8` mask and the classes present, and exposes no "
        "per-pixel confidence. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or "
        "preprocessing fitting happens in this notebook — the upstream OpenMMLab checkpoint supplies the weights and "
        "the pinned MMSegmentation 1.2.2 package supplies the config and test pipeline, and the carried package adds "
        "snapshot verification, runtime-version checks, input validation, a fixed output contract and the "
        "`semantic_iou`, `majority_class_baseline`, `validate_inputs` and `evaluation_report` helpers.\n\n"
        "**Trust boundary (MOD12).** The checkpoint is a code-capable PyTorch `.pth` serialization. The carried "
        "`verify_snapshot` re-hashes it against the inline manifest (and against the digest the package has always "
        "pinned in `MODEL_SPEC`) before the pinned MMSegmentation loader deserializes it inside `mmengine`; the "
        "deserialization call is upstream's, not the package's, and is **not** a `weights_only` load. A matching "
        "digest proves byte identity with the pinned OpenMMLab distribution, not publisher authenticity — run this "
        "notebook only where that pinned source is trusted.\n\n"
        "The default sample is a synthetic scene generated in code (no download, no ground truth), so its mask is "
        "demonstration (plumbing) evidence, not a correctness or benchmark claim. A gated option fetches two labelled "
        "ADE20K validation fixtures from the Hugging Face Hub at an immutable dataset commit instead and evaluates "
        "mean IoU on them."
    ),
    "learning_objectives": (
        "install the pinned Python 3.10 OpenMMLab CPU runtime, read what the carried package guarantees, resolve and "
        "digest-verify the immutable OpenMMLab checkpoint, generate a synthetic default input (or opt into the labelled "
        "ADE20K fixtures / a BYOD image) and validate it into an input manifest, run segmentation through the public "
        "API, read a class-index mask and its class coverage correctly, produce an evaluation report that is "
        "`sample-sanity` with `semantic_iou` only when ground-truth masks exist and `not-measurable` otherwise, and "
        "export class-index masks plus machine-readable results and provenance."
    ),
    "exclusions": (
        "instance or panoptic segmentation, object detection, depth, open-vocabulary segmentation, image "
        "classification, or any training or fine-tuning. The label space is fixed to the 150 ADE20K categories; "
        "pixels of other things still receive an ADE20K label, and the API exposes no per-pixel uncertainty."
    ),
    "prerequisites": [
        "- **Runtime:** a **CPython 3.10** Jupyter kernel on Linux (the notebook asserts `sys.version_info[:2] == (3, 10)` and stops otherwise). The qualified OpenMMLab stack — torch 2.1.2 (CPU build), MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation 1.2.2, NumPy 1.26.4 — has prebuilt wheels for Python 3.10 only; `pip` cannot change the interpreter, so a Python 3.11+ kernel (including current default Colab runtimes) is unsupported and the pinned install fails there. CPU is the default and only qualified path; no GPU is required. The pinned torch/mmcv wheels are the largest downloads of the run.",
        "- **Knowledge:** basic Python and image handling; what a per-pixel class map and an intersection-over-union metric are.",
        "- **Data:** the default sample is a deterministic 512×384 synthetic scene (gradient background plus flat-coloured shapes) generated in code, so nothing is downloaded and there is no ground truth. Two optional gates are off by default so the sample path runs top-to-bottom without interaction: `USE_ADE20K_FIXTURES` fetches two public ADE20K validation fixtures (`ADE_val_00000001`, `ADE_val_00000002`: two JPEG images and their PNG annotation maps, ~94 KB in total) from the Hugging Face dataset `hf-internal-testing/fixtures_ade20k` at the immutable commit `850d349e…`, verifies each file's SHA-256, and enables mean-IoU evaluation; `USE_BYOD` uploads one image file decodable by Pillow (PNG/JPEG/WebP and similar, at most 64 megapixels). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Confirm the qualified runtime\n\n"
                "The carried package fails closed on version drift: `verify_runtime_versions` (called when the model "
                "was constructed above) compares the installed `mmseg`, `mmcv` and `mmengine` distributions with the "
                "versions pinned in `MODEL_SPEC`, and this cell additionally asserts the Python 3.10 interpreter the "
                "OpenMMLab wheels were built for. Look for a dictionary reporting Python 3.10.x, `torch` 2.1.2+cpu, "
                "MMSegmentation 1.2.2, MMCV 2.1.0, MMEngine 0.10.7, 150 classes, the verified checkpoint file name and "
                "the `local-snapshot` source."
            ),
            "code": (
                "import sys\n\n"
                "if sys.version_info[:2] != (3, 10):\n"
                "    raise RuntimeError(f'Python 3.10 is required by the qualified OpenMMLab runtime (see Prerequisites); this kernel is {{sys.version.split()[0]}}. Use a Python 3.10 kernel.')\n"
                "print({{'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__, **pipe.versions, 'classes': len(pipe.classes), 'checkpoint': pipe.checkpoint.name, 'source': pipe.source, 'device': pipe.device}})"
            ),
        },
        {
            "md": (
                "## 5. Generate the synthetic sample, or opt into the ADE20K fixtures / BYOD\n\n"
                "The default sample is **synthetic**: a deterministic 512×384 scene (a red→green gradient background "
                "with three flat-coloured shapes) drawn in code and written to `sample/`, so it needs no download and "
                "its SHA-256 is printed for the record. It depicts no ADE20K scene, so it has **no ground truth**: "
                "whatever mask the model returns is a sanity check that the input contract, preprocessing and forward "
                "pass work, not a correctness measurement. Two gates are off by default. `USE_ADE20K_FIXTURES` fetches "
                "two real labelled ADE20K validation examples from the Hugging Face Hub at an immutable dataset commit, "
                "refuses any file whose SHA-256 differs from the recorded digest, checks that each annotation map has "
                "the image's shape and raw labels in `0..150`, and converts the raw labels with the carried "
                "`ade20k_raw_to_indices` (`0` → ignore, `1..150` → `0..149`, the pinned config's "
                "`reduce_zero_label=True` convention) — no split is invented or changed. `USE_BYOD` uploads one image; "
                "BYOD has no ground truth unless you supply a matching annotation map yourself. Look for a dictionary "
                "naming the sample kind, the image files, their digests and whether ground truth exists."
            ),
            "code": (
                "import hashlib\n"
                "from pathlib import Path\n\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "USE_ADE20K_FIXTURES = False  # @param {{type:\"boolean\"}}\n"
                "FIXTURES_COMMIT = '850d349e5038f291284e7999fcacbedc0922534b'\n"
                "FIXTURES_BASE = f'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/{{FIXTURES_COMMIT}}'\n"
                "FIXTURES_SHA256 = {{\n"
                "    'ADE_val_00000001.jpg': '99f7af15a7bd66f3d2ad8b98f1b41941c27407a35d303f9b70724cb231a36098',\n"
                "    'ADE_val_00000001.png': '7724c8b985ba9978e968fed231d74fb9e72abd4179f3c4b58bb87c525efb9ae7',\n"
                "    'ADE_val_00000002.jpg': 'ce373a18513e5a357d7fd2d2f70e1db2249417635068e108cd85ddca08195f30',\n"
                "    'ADE_val_00000002.png': 'db1497cd6acd98c61178fbff5611da0604877f55cb7b72655d4e4ecac386904f',\n"
                "}}\n"
                "sample_dir = Path('sample')\n"
                "sample_dir.mkdir(exist_ok=True)\n"
                "ground_truth = None\n"
                "if USE_BYOD and USE_ADE20K_FIXTURES:\n"
                "    raise ValueError('Enable at most one of USE_BYOD and USE_ADE20K_FIXTURES.')\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    upload_name = next(iter(uploaded))\n"
                "    image_path = sample_dir / Path(upload_name).name\n"
                "    image_path.write_bytes(uploaded[upload_name])\n"
                "    image_paths = [image_path]\n"
                "    sample_kind = 'BYOD'\n"
                "elif USE_ADE20K_FIXTURES:\n"
                "    import urllib.request\n\n"
                "    for name, expected in FIXTURES_SHA256.items():\n"
                "        target = sample_dir / name\n"
                "        if not target.exists():\n"
                "            urllib.request.urlretrieve(f'{{FIXTURES_BASE}}/{{name}}', target)\n"
                "        observed = hashlib.sha256(target.read_bytes()).hexdigest()\n"
                "        if observed != expected:\n"
                "            raise RuntimeError(f'{{name}}: digest {{observed}} != pinned {{expected}}; refusing the fixture')\n"
                "    image_paths = [sample_dir / 'ADE_val_00000001.jpg', sample_dir / 'ADE_val_00000002.jpg']\n"
                "    ground_truth = []\n"
                "    for path in image_paths:\n"
                "        with Image.open(path) as opened:\n"
                "            width, height = opened.size\n"
                "        with Image.open(path.with_suffix('.png')) as annotation:\n"
                "            raw = numpy.array(annotation)\n"
                "        if raw.shape != (height, width):\n"
                "            raise RuntimeError(f'{{path.name}}: annotation shape {{raw.shape}} != image shape {{(height, width)}}')\n"
                "        ground_truth.append(ade20k_raw_to_indices(raw))\n"
                "    sample_kind = 'ADE20K-fixtures'\n"
                "else:\n"
                "    # Deterministic synthetic scene: no randomness, so no seed is needed and the digest is stable.\n"
                "    width, height = 512, 384\n"
                "    ramp = numpy.linspace(0.0, 255.0, width)\n"
                "    red = numpy.tile(ramp, (height, 1))\n"
                "    green = numpy.tile(numpy.linspace(0.0, 255.0, height)[:, None], (1, width))\n"
                "    blue = (red + green) / 2.0\n"
                "    array = numpy.rint(numpy.stack([red, green, blue], axis=-1)).astype(numpy.uint8)\n"
                "    scene = Image.fromarray(array, mode='RGB')\n"
                "    draw = ImageDraw.Draw(scene)\n"
                "    draw.rectangle([40, 240, 220, 360], fill=(20, 20, 20))\n"
                "    draw.ellipse([300, 60, 460, 220], fill=(240, 240, 240))\n"
                "    draw.polygon([(260, 370), (330, 250), (400, 370)], fill=(30, 90, 200))\n"
                "    image_path = sample_dir / 'synthetic_scene_512x384.png'\n"
                "    scene.save(image_path)\n"
                "    image_paths = [image_path]\n"
                "    sample_kind = 'synthetic'\n"
                "image_names = [path.name for path in image_paths]\n"
                "sample_sha256 = {{path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in image_paths}}\n"
                "print({{'sample_kind': sample_kind, 'images': image_names, 'sha256': sample_sha256, 'ground_truth': None if ground_truth is None else [list(mask.shape) for mask in ground_truth]}})"
            ),
        },
        {
            "md": (
                "## 6. Validate the input → input manifest\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks `predict` "
                "applies — per image, `validate_image` (the file exists, Pillow can decode it, positive dimensions, "
                "at most `MAX_PIXELS` = 64,000,000 pixels) — and returns an **input manifest** naming the schema and "
                "ceilings, each input's observed path, size and mode, and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a "
                "path that does not exist and records the package's own error message as a finding. Inside the "
                "package every accepted image goes through the pinned config's MMSegmentation test pipeline (resize "
                "to the 512 scale, normalise); nothing is dropped or altered by the package itself, and the returned "
                "mask has the input image's height and width."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MAX_PIXELS': MAX_PIXELS, 'pixels': INPUT_SCHEMA['pixels'], 'classes': len(pipe.classes)}}}})\n"
                "input_manifest = validate_inputs(image_paths, names=image_names)\n"
                "# Demonstrate rejection on an input that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(sample_dir / 'does-not-exist.png')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'missing-file-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 7. Segment\n\n"
                "`predict` runs one image through the pinned MMSegmentation inference path and returns a "
                "`SegmentationResult`: a 2-D `uint8` `mask` of ADE20K class indices (`0..149`, the per-pixel argmax "
                "of the decode head) with the image's height and width, plus the sorted `classes_present`. The mask "
                "is a hard decision — the API exposes **no per-pixel confidence or calibrated uncertainty**, and the "
                "package ships no threshold. Inference is deterministic given the same weights, device and library "
                "versions (`model.eval()`, no sampling); CPU kernel choices can flip near-tied pixels. Each mask is "
                "saved as a PNG class-index raster (`outputs/{stem}_<image>_semantic.png`, value = class index). "
                "Look for the per-image shape, the number of classes present and the top classes by pixel coverage; "
                "on the synthetic scene expect a handful of large flat regions with arbitrary labels."
            ),
            "code": (
                "results = [pipe.predict(path) for path in image_paths]\n"
                "coverage = []\n"
                "for path, result in zip(image_paths, results, strict=True):\n"
                "    mask_path = result.save_mask(f'outputs/{stem}_{{path.stem}}_semantic.png')\n"
                "    values, counts = numpy.unique(result.mask, return_counts=True)\n"
                "    order = numpy.argsort(counts)[::-1]\n"
                "    for class_id, pixels in zip(values[order].tolist(), counts[order].tolist(), strict=True):\n"
                "        coverage.append({{'image_id': result.image_id, 'class_id': int(class_id), 'class_name': pipe.classes[int(class_id)], 'pixels': int(pixels), 'fraction': float(pixels / result.mask.size)}})\n"
                "    print({{**result.summary(), 'mask_file': mask_path.name}})\n"
                "for row in coverage[:8]:\n"
                "    print(f\"{{row['image_id']:<24}} class {{row['class_id']:>3}} {{row['class_name']:<20}} {{row['fraction']:.3f}} of pixels\")"
            ),
        },
        {
            "md": (
                "## 8. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the package's public evaluation stage and always produces a report. When "
                "ground-truth masks exist (the `USE_ADE20K_FIXTURES` path) it carries `semantic_iou` — the repository's "
                "metric helper: mean IoU over the classes present in the sample (union > 0), pixel accuracy over the "
                "labelled pixels, and per-class IoU aggregated across the images — with the verdict `sample-sanity` and "
                "the `majority_class_baseline` (a constant predictor of the sample's most frequent class, derived from "
                "the same two images, so a descriptive reference rather than an independent benchmark): a two-image "
                "tutorial metric with high sampling variance, not comparable to the upstream full-ADE20K mIoU of 44.41 "
                "that `MODEL_SPEC` records as upstream-reported context. On the synthetic default sample (and on BYOD "
                "without an annotation map) no metric exists, so the verdict is `not-measurable` and the report states "
                "what would make the task measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, ground_truth, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{key: value for key, value in report.items() if key != 'per_class'}}, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No ground-truth masks were supplied, so semantic_iou is not computed; the masks above are sanity evidence only.')\n"
                "else:\n"
                "    for row in sorted(report['per_class'], key=lambda row: row['union_pixels'], reverse=True)[:10]:\n"
                "        print(f\"class {{row['class_id']:>3}} {{pipe.classes[row['class_id']]:<20}} IoU {{row['iou']:.3f}}  union {{row['union_pixels']}} px\")"
            ),
        },
        {
            "md": (
                "## 9. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves each mask's summary and file name, the per-image class coverage, the "
                "evaluation report (including per-class IoU when measured), the input manifest, the sample identity and "
                "digests, the notebook's source (repository, revision, embedded module digest, generator), the model "
                "identifier, the immutable revision label, the checkpoint digest the package verified, and the runtime "
                "identity (Python, `torch`, `mmseg`, `mmcv`, `mmengine`, device). The class coverage is also written as "
                "CSV with explicit `image_id`, `class_id`, `class_name`, `pixels` and `fraction` columns, ordered by "
                "coverage within each image, so the mask semantics survive downstream use. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "payload = {{\n"
                "    'predictions': [{{**result.summary(), 'mask_file': f'{stem}_{{Path(result.image_id).stem}}_semantic.png'}} for result in results],\n"
                "    'class_coverage': coverage,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'images': image_names, 'sha256': sample_sha256, 'ground_truth': ground_truth is not None}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'checkpoint_sha256': MODEL_SPEC['checkpoint_sha256'],\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'numpy': numpy.__version__,\n"
                "        **pipe.versions,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_class_coverage.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image_id', 'class_id', 'class_name', 'pixels', 'fraction'])\n"
                "    for row in coverage:\n"
                "        writer.writerow([row['image_id'], row['class_id'], row['class_name'], row['pixels'], f\"{{row['fraction']:.6f}}\"])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Each pixel receives one class from the fixed 150-category ADE20K label space by per-pixel argmax; the API "
        "exposes **no per-pixel confidence** and the package ships no threshold. On the synthetic scene the mask is "
        "meaningless by construction and the evaluation report says `not-measurable`; a `semantic_iou` value from the "
        "two ADE20K fixtures is tutorial evidence for those images and must not be generalized to a domain, camera, "
        "resolution or scene composition. Scenes outside ADE20K's indoor/outdoor photographic distribution, thin "
        "structures, class boundaries and domain shifts (medical, aerial, line art) all degrade results in ways the "
        "package does not detect. The package provides no instance, panoptic, detection, depth, classification, or "
        "training capability, and the `.pth` checkpoint remains a code-capable serialization whose digest check fixes "
        "the bytes, not the author.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned OpenMMLab checkpoint, assert the qualified Python 3.10 / MMSegmentation "
        "1.2.2 runtime, validate the demonstrated input, execute the public segmentation path, and emit the shown "
        "machine-readable outputs and class-index masks in the tested runtime — without the repository being "
        "reachable. It does **not** establish benchmark superiority, reproduction of the upstream ADE20K result, "
        "calibrated per-pixel uncertainty, safety for high-consequence decisions, or production fitness on an unseen "
        "domain.\n\n"
        "**Next experiments:** enable `USE_ADE20K_FIXTURES` to see the report switch to `sample-sanity` with "
        "`semantic_iou` (mean IoU, pixel accuracy, per-class IoU) against the `majority_class_baseline`; enable "
        "`USE_BYOD` with a photograph from your own domain and inspect which classes dominate the coverage table "
        "before investing in annotation; annotate a small holdout from that domain with ADE20K indices and compare its "
        "mean IoU with the fixture value.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/swin-segmentation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/swin-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/swin-segmentation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Pinned checkpoint (OpenMMLab host): https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth\n"
        "- Config source (MMSegmentation v1.2.2, `configs/swin`): https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2/configs/swin\n"
        "- ADE20K fixtures (Hugging Face dataset, immutable commit): https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/tree/850d349e5038f291284e7999fcacbedc0922534b\n"
        "- Upstream project: https://github.com/microsoft/Swin-Transformer\n"
        "- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows: https://arxiv.org/abs/2103.14030\n"
        "- Unified Perceptual Parsing for Scene Understanding (UPerNet): https://arxiv.org/abs/1807.10221\n"
        "- Scene Parsing through ADE20K Dataset: https://arxiv.org/abs/1608.05442"
    ),
}
