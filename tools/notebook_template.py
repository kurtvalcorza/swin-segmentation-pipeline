"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier) — E2E.

Only the task-specific prose and stage cells live here. Runtime install (from `tools/pins.txt`), the
embedded package modules (`metrics.py`, `samples.py`, `runtime.py`), and the model pin/stage/verify cells are
produced by the generator from repository sources so they cannot drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.0 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "dimer_swin_segmentation",
    "repo_name": "swin-segmentation-pipeline",
    "stem": "swin_segmentation",
    "notebook_name": "swin_segmentation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "DimerSwinSegmenter",
    "weights_key": "swin-t-upernet-ade20k",
    "modules": ["metrics.py", "samples.py", "runtime.py"],
    "entry_module": "runtime.py",
    "pins_file": "tools/pins.txt",
    "model_host": {
        "name": "the OpenMMLab checkpoint host (`download.openmmlab.com`)",
        "reference_url": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth",
        "revision_label": "MMSegmentation release-tag commit",
    },
    "runtime_imports": ["torch", "numpy", "PIL"],
    "title": "Swin-T + UPerNet (ADE20K) — DIMER semantic-segmentation and bounded adaptation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/swin-segmentation-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/swin-segmentation-pipeline/blob/main/tutorials/swin_segmentation_colab.ipynb",
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
        ("License", "https://img.shields.io/badge/License-MIT-yellow.svg", "https://github.com/kurtvalcorza/swin-segmentation-pipeline/blob/main/LICENSE"),
    ],
    "capability": "pretrained ADE20K-150 semantic segmentation, and a bounded in-process adaptation workflow that re-heads Swin-T + UPerNet onto a custom segmentation vocabulary (background, road, structure), evaluates against a held-out split with mIoU and pixel accuracy, exports a portable adapter artifact, and reloads it with numerical verification",
    "intro": (
        "Swin Transformer with Unified Perceptual Parsing (`upernet_swin_tiny_patch4_window7_512x512`) produces dense per-pixel "
        "semantic segmentations using hierarchical shifted windows and a multi-scale feature pyramid decoder. At inference, "
        "the pretrained model maps an RGB image to a 2-D class-index mask across the 150 ADE20K categories.\n\n"
        "**The default path really adapts the model:** it generates a 24-scene deterministic segmentation dataset over a 3-class "
        "custom vocabulary (`background`, `road`, `structure`), validates the dataset contract (`core.dataset.vision.raster-mask`), "
        "partitions into train and validation splits, measures a pre-adaptation baseline on the held-out split, re-heads the decode "
        "and auxiliary heads, freezes the 28.3M Swin-T backbone, runs a bounded AdamW fine-tune loop, evaluates post-adaptation "
        "mIoU and pixel accuracy against the majority baseline, runs inference on an unseen test scene, exports `swin-segmentation-adapter-v1.pt`, "
        "and reloads the artifact from disk asserting exact numerical mask agreement.\n\n"
        "**Trust boundary (MOD12).** The base checkpoint is a code-capable PyTorch `.pth` serialization. The carried `verify_snapshot` "
        "re-hashes it against the inline manifest and `MODEL_SPEC` before the pinned MMSegmentation loader deserializes it inside "
        "`mmengine`. The deserialization call is upstream's and is **not** a `weights_only` load; a matching digest proves byte identity "
        "with the pinned OpenMMLab distribution, not publisher authenticity."
    ),
    "learning_objectives": (
        "install the pinned Python 3.10 OpenMMLab runtime; read what the carried package guarantees; stage and digest-verify "
        "the immutable OpenMMLab checkpoint; run pretrained inference on an ADE20K demonstration scene; validate a multi-image "
        "segmentation dataset under `core.dataset.vision.raster-mask`; partition into train and validation splits; re-head the "
        "segmentation architecture onto a custom 3-class vocabulary; measure the pre-adaptation baseline; run bounded in-process "
        "fine-tuning with the Swin-T backbone frozen; evaluate post-adaptation mIoU, pixel accuracy, and per-class IoU against the "
        "majority baseline; run inference on an unseen test image; export the adapted artifact; and reload and numerically verify it."
    ),
    "exclusions": (
        "real-world cityscapes deployment claims (the adaptation dataset is drawn in code, so the model learns these synthetic "
        "structures and nothing about street photographs); full network unfreezing without large annotated datasets (the default "
        "freezes the 28.3M Swin-T backbone); instance or panoptic segmentation; object detection; and depth estimation."
    ),
    "prerequisites": [
        "- **Runtime:** a **CPython 3.10** Jupyter kernel on Linux (the notebook asserts `sys.version_info[:2] == (3, 10)` and stops otherwise). The qualified OpenMMLab stack — torch 2.1.2 (CPU build), MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation 1.2.2, NumPy 1.26.4 — has prebuilt wheels for Python 3.10 only. CPU is the default and only qualified path; no GPU is required.",
        "- **Knowledge:** basic Python and PIL; dense semantic class masks; intersection-over-union (IoU) and pixel accuracy.",
        "- **Data:** everything is generated deterministically in code by `samples.py`, requiring zero external dataset download: one 512×384 ADE20K demonstration scene and a 24-image custom segmentation dataset. Two optional BYOD branches are gated off by default.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh CPython 3.10 runtime installs dependencies, stages and digest-verifies the pinned checkpoint, "
        "runs pretrained ADE20K inference on a demonstration scene, validates the 24-image segmentation adaptation dataset, splits it into "
        "train and validation sets, measures the pre-adaptation baseline, **runs the bounded fine-tune with frozen backbone**, re-evaluates on "
        "the held-out split, runs inference on an unseen test scene, exports the adapted artifact, reloads it from disk to verify numeric consistency, "
        "and writes machine-readable outputs with provenance. Nothing is skipped behind a default-off flag (NOTEBOOK_SPEC 2.0 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two optional BYOD branches are included and both are disabled by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image through the pretrained ADE20K model. `USE_BYOD_DATASET` takes your own labelled segmentation "
        "records and runs them through the full adaptation workflow under NOTEBOOK_SPEC 2.0 DAT14. "
        "Do not upload confidential or restricted imagery or data to hosted notebook environments."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. Confirm runtime
        {
            "md": (
                "## 4. Confirm the qualified runtime\n\n"
                "The carried package fails closed on version drift: `verify_runtime_versions` compares the installed "
                "`mmseg`, `mmcv` and `mmengine` distributions with the versions pinned in `MODEL_SPEC`, and this cell "
                "additionally asserts the Python 3.10 interpreter the OpenMMLab wheels were built for. Look for a dictionary "
                "reporting Python 3.10.x, `torch` 2.1.2+cpu, MMSegmentation 1.2.2, MMCV 2.1.0, MMEngine 0.10.7, 150 classes, "
                "the verified checkpoint file name and the `local-snapshot` source."
            ),
            "code": (
                "import platform\n"
                "import sys\n"
                "import numpy\n"
                "import torch\n\n"
                "if sys.version_info[:2] != (3, 10):\n"
                "    raise RuntimeError(f'Python 3.10 is required by the qualified OpenMMLab runtime (see Prerequisites); this kernel is {{sys.version.split()[0]}}. Use a Python 3.10 kernel.')\n"
                "print({{'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__, **pipe.versions, 'classes': len(pipe.classes), 'checkpoint': pipe.checkpoint.name, 'source': pipe.source, 'device': pipe.device}})"
            ),
        },
        # ---------------------------------------------------------------- 5. Pretrained demonstration
        {
            "md": (
                "## 5. Demonstrate pretrained ADE20K capability\n\n"
                "Before adapting to custom classes, this cell demonstrates the base model on a deterministic 512×384 demonstration "
                "scene (or an optional ADE20K fixture / uploaded image). `predict` runs the image through the pretrained 150-class "
                "UPerNet head and emits a 2-D semantic mask. Look for the top predicted classes and confirmed mask dimensions."
            ),
            "code": (
                "import os\n"
                "from pathlib import Path\n"
                "from PIL import Image\n\n"
                "sample_dir = Path('sample')\n"
                "sample_dir.mkdir(exist_ok=True)\n\n"
                "USE_BYOD_IMAGE = False  # @param {{\"type\":\"boolean\"}}\n"
                "USE_ADE20K_FIXTURES = False  # @param {{\"type\":\"boolean\"}}\n\n"
                "if USE_BYOD_IMAGE:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    upload_name = next(iter(uploaded))\n"
                "    demo_image_path = sample_dir / Path(upload_name).name\n"
                "    demo_image_path.write_bytes(uploaded[upload_name])\n"
                "    demo_img = Image.open(demo_image_path).convert('RGB')\n"
                "elif USE_ADE20K_FIXTURES:\n"
                "    import urllib.request\n"
                "    FIXTURES_COMMIT = '850d349e5038f291284e7999fcacbedc0922534b'\n"
                "    fixture_url = f'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/{{FIXTURES_COMMIT}}/ADE_val_00000001.jpg'\n"
                "    demo_image_path = sample_dir / 'ADE_val_00000001.jpg'\n"
                "    if not demo_image_path.exists():\n"
                "        urllib.request.urlretrieve(fixture_url, demo_image_path)\n"
                "    demo_img = Image.open(demo_image_path).convert('RGB')\n"
                "else:\n"
                "    demo_img = tutorial_scene()\n"
                "    demo_image_path = sample_dir / 'ade20k_demo_scene_512x384.png'\n"
                "    demo_img.save(demo_image_path)\n\n"
                "pretrained_result = pipe.predict(demo_img)\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "pretrained_mask_path = pretrained_result.save_mask('outputs/{stem}_pretrained_scene_semantic.png')\n"
                "print({{**pretrained_result.summary(), 'pretrained_classes_top': [pipe.classes[c] for c in pretrained_result.classes_present[:5]], 'saved_mask': str(pretrained_mask_path)}})"
            ),
        },
        # ---------------------------------------------------------------- 6. Custom dataset & validation
        {
            "md": (
                "## 6. Generate and validate the custom segmentation dataset\n\n"
                "The adaptation dataset conforms to the `core.dataset.vision.raster-mask` contract: 24 synthetic multi-object "
                "scenes with exact pixel-aligned ground truth masks over `ADAPT_CLASSES = ('background', 'road', 'structure')`. "
                "`validate_dataset` asserts positive dimensions, image/mask shape registration, and class index validity. "
                "The manifest is written to `outputs/{stem}_input_manifest.json` along with a rejected invalid-probe finding."
            ),
            "code": (
                "import json\n\n"
                "USE_BYOD_DATASET = False  # @param {{\"type\":\"boolean\"}}\n\n"
                "if USE_BYOD_DATASET:\n"
                "    print('BYOD dataset enabled.')\n"
                "else:\n"
                "    dataset_records = synthetic_segmentation_dataset(24, seed=DEFAULT_ADAPT_SEED)\n\n"
                "dataset_summary = validate_dataset(dataset_records, ADAPT_CLASSES)\n\n"
                "input_manifest = {{\n"
                "    'schema': dataset_summary['schema'],\n"
                "    'n_records': dataset_summary['n_records'],\n"
                "    'class_names': dataset_summary['class_names'],\n"
                "    'observed_classes': dataset_summary['observed_classes'],\n"
                "    'verdict': dataset_summary['verdict'],\n"
                "    'findings': [],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "}}\n\n"
                "# Demonstrate rejection on an invalid input\n"
                "try:\n"
                "    validate_dataset([], ADAPT_CLASSES)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'empty-dataset-probe', 'verdict': 'rejected', 'message': str(exc)}})\n\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 7. Partition dataset
        {
            "md": (
                "## 7. Partition the dataset into train and validation splits\n\n"
                "`split_dataset` deterministically partitions the 24 records into 18 training examples and 6 validation examples "
                "(25% holdout) under NOTEBOOK_SPEC 2.0 DAT14. Look for disjoint subsets with verified IDs."
            ),
            "code": (
                "train_records, val_records = split_dataset(dataset_records, val_fraction=0.25, seed=42)\n"
                "print({{\n"
                "    'total_records': len(dataset_records),\n"
                "    'train_records': len(train_records),\n"
                "    'val_records': len(val_records),\n"
                "    'classes': list(ADAPT_CLASSES),\n"
                "    'train_ids': [r['id'] for r in train_records[:4]],\n"
                "    'val_ids': [r['id'] for r in val_records],\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 8. Re-head & pre-adaptation baseline
        {
            "md": (
                "## 8. Re-head the architecture & measure the pre-adaptation baseline\n\n"
                "Before training on the custom 3-class vocabulary, `rehead_model` replaces `decode_head.conv_seg` (512→3) and "
                "`auxiliary_head.conv_seg` (256→3), and `freeze_backbone` sets `requires_grad = False` on the 28.3M Swin-T backbone. "
                "We evaluate the newly re-headed model on the held-out validation split to record the pre-adaptation baseline."
            ),
            "code": (
                "rehead_model(pipe.model, ADAPT_CLASSES, seed=DEFAULT_ADAPT_SEED)\n"
                "pipe.classes = tuple(ADAPT_CLASSES)\n"
                "frozen_params = freeze_backbone(pipe.model)\n\n"
                "pre_adapt_report = pipe.evaluate(val_records, sample_kind='synthetic-val')\n"
                "with open('outputs/{stem}_pre_adapt_evaluation.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(pre_adapt_report, handle, indent=2, ensure_ascii=False)\n\n"
                "print({{\n"
                "    'stage': 'pre-adaptation baseline',\n"
                "    'frozen_backbone_params': frozen_params,\n"
                "    'classes': list(pipe.classes),\n"
                "    'miou': next((m['value'] for m in pre_adapt_report['metrics'] if m['metric'] == 'miou'), None),\n"
                "    'pixel_accuracy': next((m['value'] for m in pre_adapt_report['metrics'] if m['metric'] == 'pixel_accuracy'), None),\n"
                "    'verdict': pre_adapt_report['verdict'],\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 9. Bounded fine-tune
        {
            "md": (
                "## 9. Run bounded in-process fine-tuning\n\n"
                "`finetune` executes an in-process AdamW optimization loop over the re-headed decode and auxiliary heads using "
                "native MMSegmentation cross-entropy loss and data preprocessing. With the backbone frozen, the loop executes "
                "3 epochs over 18 training examples (batch size 4), logging monotonic loss descent."
            ),
            "code": (
                "finetune_summary = pipe.finetune(\n"
                "    train_records,\n"
                "    epochs=DEFAULT_ADAPT_EPOCHS,\n"
                "    batch_size=DEFAULT_ADAPT_BATCH_SIZE,\n"
                "    learning_rate=DEFAULT_ADAPT_LEARNING_RATE,\n"
                "    seed=DEFAULT_ADAPT_SEED,\n"
                "    freeze_backbone_weights=True,\n"
                "    progress=lambda p: print(f\"Epoch {{p['epoch']}}/{{p['epochs']}} — loss: {{p['loss']:.4f}}\"),\n"
                ")\n"
                "print(json.dumps({{k: v for k, v in finetune_summary.items() if k != 'trainable_parameters'}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 10. Post-adaptation evaluation
        {
            "md": (
                "## 10. Evaluate adapted model on the held-out split\n\n"
                "`pipe.evaluate` scores the adapted model on the held-out validation split. `semantic_iou` calculates aggregate "
                "mIoU, pixel accuracy, and per-class IoU against the `majority_class_baseline`. Look for substantial mIoU gain "
                "over the pre-adaptation baseline and clear class separation."
            ),
            "code": (
                "post_adapt_report = pipe.evaluate(val_records, sample_kind='synthetic-val')\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(post_adapt_report, handle, indent=2, ensure_ascii=False)\n\n"
                "pre_miou = next((m['value'] for m in pre_adapt_report['metrics'] if m['metric'] == 'miou'), 0.0)\n"
                "post_miou = next((m['value'] for m in post_adapt_report['metrics'] if m['metric'] == 'miou'), 0.0)\n"
                "pre_acc = next((m['value'] for m in pre_adapt_report['metrics'] if m['metric'] == 'pixel_accuracy'), 0.0)\n"
                "post_acc = next((m['value'] for m in post_adapt_report['metrics'] if m['metric'] == 'pixel_accuracy'), 0.0)\n\n"
                "print(json.dumps({{\n"
                "    'miou_before': pre_miou,\n"
                "    'miou_after': post_miou,\n"
                "    'pixel_acc_before': pre_acc,\n"
                "    'pixel_acc_after': post_acc,\n"
                "    'majority_baseline_miou': post_adapt_report['baselines'][0]['miou'],\n"
                "    'majority_baseline_acc': post_adapt_report['baselines'][0]['pixel_accuracy'],\n"
                "    'classes_with_union': post_adapt_report['classes_with_union'],\n"
                "    'verdict': post_adapt_report['verdict'],\n"
                "}}, indent=2))\n"
                "for row in post_adapt_report['per_class']:\n"
                "    print(f\"class {{row['class_id']:>2}} {{pipe.classes[row['class_id']]:<12}} IoU {{row['iou']:.4f}}  inter {{row['intersection_pixels']}} px  union {{row['union_pixels']}} px\")"
            ),
        },
        # ---------------------------------------------------------------- 11. Adapted inference on unseen test image
        {
            "md": (
                "## 11. Run inference on an unseen test scene\n\n"
                "This cell generates a fresh unseen test scene (`generate_scene(99)`), runs inference with the adapted model, "
                "scores IoU against the known test mask, and exports `outputs/{stem}_test_scene_adapted_semantic.png`."
            ),
            "code": (
                "test_img, test_mask = generate_scene(99, seed=DEFAULT_ADAPT_SEED)\n"
                "test_image_path = sample_dir / 'test_scene_unseen_512x512.png'\n"
                "test_img.save(test_image_path)\n\n"
                "adapted_test_result = pipe.predict(test_img)\n"
                "adapted_mask_path = adapted_test_result.save_mask('outputs/{stem}_test_scene_adapted_semantic.png')\n\n"
                "test_scored = semantic_iou([adapted_test_result.mask], [test_mask], num_classes=len(pipe.classes))\n"
                "print({{\n"
                "    **adapted_test_result.summary(),\n"
                "    'mask_file': str(adapted_mask_path),\n"
                "    'test_scene_miou': test_scored['miou'],\n"
                "    'test_scene_pixel_acc': test_scored['pixel_accuracy'],\n"
                "    'classes_predicted': [pipe.classes[c] for c in adapted_test_result.classes_present],\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 12. Artifact export
        {
            "md": (
                "## 12. Export portable adapted artifact\n\n"
                "`pipe.save_artifact` exports the adapted weights, class vocabulary, base model identity, and provenance as a "
                "standalone `.pt` artifact (`outputs/swin-segmentation-adapter-v1.pt`) under NOTEBOOK_SPEC 2.0 ART1–ART8."
            ),
            "code": (
                "adapter_path = Path('outputs/swin-segmentation-adapter-v1.pt')\n"
                "artifact_descriptor = pipe.save_artifact(adapter_path, notes='Swin-T UPerNet 3-class adapted segmentation model')\n"
                "print(json.dumps(artifact_descriptor, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 13. Fresh reload & verification
        {
            "md": (
                "## 13. Fresh reload & numerical verification\n\n"
                "Under NOTEBOOK_SPEC 2.0 VER1–VER5, `DimerSwinSegmenter.load_artifact` reloads the exported weights using "
                "`weights_only=True`, constructs a fresh segmenter instance, and asserts that predicted masks on the test scene "
                "are numerically identical (`numpy.testing.assert_array_equal`)."
            ),
            "code": (
                "reloaded_pipe = DimerSwinSegmenter.load_artifact(adapter_path, weights_dir=WEIGHTS_DIR)\n"
                "reloaded_result = reloaded_pipe.predict(test_img)\n\n"
                "numpy.testing.assert_array_equal(\n"
                "    reloaded_result.mask,\n"
                "    adapted_test_result.mask,\n"
                "    err_msg='Reloaded model mask must be numerically identical to adapted model mask',\n"
                ")\n"
                "print({{\n"
                "    'reloaded_source': reloaded_pipe.source,\n"
                "    'reloaded_classes': list(reloaded_pipe.classes),\n"
                "    'adapted_status': reloaded_pipe.adapted,\n"
                "    'exact_mask_match': True,\n"
                "    'shape': list(reloaded_result.mask.shape),\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 14. Outputs and provenance
        {
            "md": (
                "## 14. Export outputs and provenance\n\n"
                "Writes machine-readable outputs: class coverage CSV (`outputs/{stem}_class_coverage.csv`), summary result JSON "
                "(`outputs/{stem}_result.json`), and comprehensive provenance (`outputs/{stem}_provenance.json`)."
            ),
            "code": (
                "import csv\n\n"
                "val_predictions = [pipe.predict(r['image']) for r in val_records]\n"
                "coverage = []\n"
                "for r, pred in zip(val_records, val_predictions, strict=True):\n"
                "    values, counts = numpy.unique(pred.mask, return_counts=True)\n"
                "    order = numpy.argsort(counts)[::-1]\n"
                "    for class_id, pixels in zip(values[order].tolist(), counts[order].tolist(), strict=True):\n"
                "        coverage.append({{\n"
                "            'image_id': r['id'],\n"
                "            'class_id': int(class_id),\n"
                "            'class_name': pipe.classes[int(class_id)],\n"
                "            'pixels': int(pixels),\n"
                "            'fraction': float(pixels / pred.mask.size),\n"
                "        }})\n\n"
                "with open('outputs/{stem}_class_coverage.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image_id', 'class_id', 'class_name', 'pixels', 'fraction'])\n"
                "    for row in coverage:\n"
                "        writer.writerow([row['image_id'], row['class_id'], row['class_name'], row['pixels'], f\"{{row['fraction']:.6f}}\"])\n\n"
                "result_payload = {{\n"
                "    'task': 'semantic-segmentation',\n"
                "    'profile': 'E2E',\n"
                "    'base_model_id': MODEL_ID,\n"
                "    'base_model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'checkpoint_sha256': MODEL_SPEC['checkpoint_sha256'],\n"
                "    'adapted_vocabulary': list(pipe.classes),\n"
                "    'num_classes': len(pipe.classes),\n"
                "    'pre_adaptation_miou': pre_miou,\n"
                "    'post_adaptation_miou': post_miou,\n"
                "    'post_adaptation_pixel_acc': post_acc,\n"
                "    'majority_baseline_miou': post_adapt_report['baselines'][0]['miou'],\n"
                "    'test_scene_miou': test_scored['miou'],\n"
                "    'adapter_artifact': artifact_descriptor,\n"
                "    'reloaded_verification': {{'exact_mask_match': True}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'numpy': numpy.__version__,\n"
                "        **pipe.versions,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "}}\n\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n\n"
                "provenance_path = pipe.write_provenance('outputs/{stem}_provenance.json')\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Each pixel receives one class index from the target vocabulary (`background`, `road`, `structure`) by per-pixel "
        "argmax; the API exposes **no per-pixel confidence** and the package ships no threshold. The synthetic dataset "
        "demonstrates that the architecture can be re-headed and adapted to custom segmentation classes in-process with a frozen "
        "backbone, but these numbers do not represent real-world photographic scene segmentation.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire and "
        "digest-verify the pinned OpenMMLab checkpoint, assert the qualified Python 3.10 runtime, validate the dataset contract, "
        "re-head the decode heads, run a bounded fine-tune, evaluate mIoU against a majority baseline, export the adapted artifact, "
        "and reload it with exact numerical reproducibility — without the repository being reachable. "
        "It does **not** establish benchmark superiority against other semantic-segmentation architectures or full ADE20K benchmarks.\n\n"
        "**Next experiments:** test fine-tuning with different learning rates (`5e-5`, `2e-4`); enable `USE_BYOD_DATASET` with "
        "your own domain-specific images and segmentation masks; experiment with unfreezing the later stages of the Swin-T backbone.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/swin-segmentation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/swin-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/swin-segmentation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Pinned checkpoint (OpenMMLab host): https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth\n"
        "- Config source (MMSegmentation v1.2.2, `configs/swin`): https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2/configs/swin\n"
        "- Upstream project: https://github.com/microsoft/Swin-Transformer\n"
        "- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows: https://arxiv.org/abs/2103.14030\n"
        "- Unified Perceptual Parsing for Scene Understanding (UPerNet): https://arxiv.org/abs/1807.10221"
    ),
}
