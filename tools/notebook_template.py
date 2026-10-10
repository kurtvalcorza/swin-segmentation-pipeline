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
    # SWS-M1/m4 (NOTEBOOK_SPEC 2.2 §5): the qualified OpenMMLab stack needs CPython 3.10, which pip cannot provide. Section 1
    # has uv provision a managed CPython 3.10.18, installs the hash lock compiled from tools/pins.txt there
    # (--require-hashes --only-binary :all:, with the PyTorch CPU index and the OpenMMLab find-links page the pins name)
    # and routes every later cell to a persistent worker in it. The kernel's own Python does not matter.
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.10.18",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
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
            "Open In Colab (verification pending)",
            "https://img.shields.io/badge/Colab-verification%20pending-lightgrey?style=flat&logo=googlecolab",
            "https://colab.research.google.com/github/kurtvalcorza/swin-segmentation-pipeline/blob/main/tutorials/swin_segmentation_colab.ipynb",
        ),
        (
            "Isolated Python 3.10",
            "https://img.shields.io/badge/Python-3.10%20(isolated%2C%20uv--managed)-3776ab?style=flat&logo=python&logoColor=white",
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
        "partitions into train and validation splits, re-heads the decode and auxiliary heads of a fresh copy of the pretrained model "
        "(the pretrained 150-class model stays available), freezes the 28.3M Swin-T backbone, runs a bounded AdamW fine-tune loop, "
        "evaluates post-adaptation mIoU and pixel accuracy against a majority baseline and a non-learned colour baseline, runs "
        "inference on an unseen test scene, exports `swin-segmentation-adapter-v1.pt`, "
        "and reloads the artifact from disk asserting exact numerical mask agreement.\n\n"
        "**Trust boundary (MOD12).** The base checkpoint is a code-capable PyTorch `.pth` serialization. The carried `verify_snapshot` "
        "re-hashes it against the inline manifest and `MODEL_SPEC` before the pinned MMSegmentation loader deserializes it inside "
        "`mmengine`. The deserialization call is upstream's and is **not** a `weights_only` load; a matching digest proves byte identity "
        "with the pinned OpenMMLab distribution, not publisher authenticity.\n\n"
        "**Read the default adaptation result for what it is.** The three synthetic classes differ by colour (blue sky, grey road, "
        "red structures), so a rule that labels each pixel by its nearest class colour already scores close to the ceiling. The "
        "default run therefore shows that re-heading, fine-tuning, export and reload work end to end; it cannot show what the "
        "pretrained Swin features add. Section 10 prints that colour baseline beside the network."
    ),
    "learning_objectives": (
        "install the pinned Python 3.10 OpenMMLab runtime into an isolated environment; read what the carried package guarantees; stage and digest-verify "
        "the immutable OpenMMLab checkpoint; run pretrained inference on an ADE20K demonstration scene; validate a multi-image "
        "segmentation dataset under `core.dataset.vision.raster-mask`; partition into train and validation splits; re-head the "
        "segmentation architecture of a fresh model copy onto a custom 3-class vocabulary; run bounded in-process "
        "fine-tuning with the Swin-T backbone frozen; evaluate post-adaptation mIoU, pixel accuracy, and per-class IoU against a "
        "majority and a non-learned colour baseline, and say what that comparison can and cannot show; run inference on an unseen test image; export the adapted artifact; and reload and numerically verify it."
    ),
    "exclusions": (
        "real-world cityscapes deployment claims (the adaptation dataset is drawn in code, so the model learns these synthetic "
        "structures and nothing about street photographs); full network unfreezing without large annotated datasets (the default "
        "freezes the 28.3M Swin-T backbone); instance or panoptic segmentation; object detection; and depth estimation."
    ),
    "guided_opening": [
        (
            "## How to use this notebook\n\n"
            "**Who this notebook is for.** Learners who can run notebook cells in order and read short Python, and who want to see how a "
            "pretrained semantic-segmentation model is run, adapted to a new label set with a frozen backbone, measured honestly and "
            "exported with provenance. No prior experience with MMSegmentation or Swin Transformers is assumed: each term is explained "
            "where it is first needed and again in the glossary below. No GPU is needed. The **Prerequisites** give the details.\n\n"
            "**Running it.** Choose *Run all* (in Colab: *Runtime → Run all*). The default path needs no edit, no upload, no account, no "
            "token and no runtime restart. Section 1 builds the isolated Python 3.10 environment — the torch, MMCV and MMSegmentation wheels "
            "make it the slowest step — and a second Run all in the same runtime reuses it. You can also run one cell at a time with "
            "*Shift + Enter*.\n\n"
            "**Where the code runs.** The first two code cells run in the notebook kernel: they build the environment and start one "
            "Python 3.10 process inside it. Every later cell is sent to that process, so the OpenMMLab stack runs on the interpreter its "
            "wheels were built for, whatever Python the kernel itself has. Printed output and errors come back to the notebook as usual, "
            "and variables persist from cell to cell.\n\n"
            "**Two kinds of cell.** *Infrastructure cells* (Sections 1–3: the isolated install and router, the carried package and the "
            "checkpoint staging) are collapsed and labelled **Infrastructure**. *Learner cells* (Sections 4–14) are the workflow.\n\n"
            "**Two models.** `pipe` is the pretrained 150-class ADE20K model and is never changed. Section 8 builds `adapt_pipe`, a fresh "
            "copy loaded again from the verified checkpoint, and re-heads that one; Sections 9–14 use `adapt_pipe`. Re-running Section 8 "
            "therefore always starts the adaptation from the same pretrained weights and the same seeded head.\n\n"
            "**Form controls.** `USE_BYOD_IMAGE`, `BYOD_IMAGE_PATH` and `USE_ADE20K_FIXTURES` (Section 5); `USE_BYOD_DATASET` and "
            "`BYOD_DATASET_PATH` (Section 6); `LEARNING_RATE` (Section 9). Leave them at their defaults for the first run.\n\n"
            "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is produced and "
            "how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
            "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an "
            "expectation; after it, **What to notice** describes normal output and a collapsed **Check your reasoning** block gives a worked "
            "answer. No metric values of a hosted run are recorded for this notebook, so the answers describe the shape of a normal "
            "result; the baseline numbers they quote were computed offline from the default data, which needs no model."
        ),
        (
            "## The task: Input → Model/System → Output\n\n"
            "| Stage | Input | Model / system | Output |\n"
            "|---|---|---|---|\n"
            "| **Pretrained inference** | one RGB image | MMSegmentation 1.2.2 test pipeline (resize to the 512 scale, normalise) → Swin-T backbone → UPerNet decode head (150 classes) | a 2-D class-index mask the size of the image |\n"
            "| **Adaptation** | 18 training image/mask pairs over 3 classes | a fresh model copy: new 3-class `conv_seg` layers in the decode and auxiliary heads, backbone frozen, AdamW + cross-entropy | an adapted model and a `.pt` adapter artifact |\n"
            "| **Evaluation** | 6 held-out pairs | the adapted model, a majority-class constant, a colour nearest-centroid rule | mIoU, pixel accuracy, per-class IoU |\n"
            "| **Reload** | the artifact file | `load_artifact` with `weights_only=True` | a mask identical to the in-memory model's |\n\n"
            "## Roadmap\n\n"
            "| Section | Tag | What happens | What you read |\n"
            "|---|---|---|---|\n"
            "| 1–3 | [Engineering] | isolated Python 3.10 environment; carried package; checkpoint staged and verified | versions, digests |\n"
            "| 4. Runtime | [Engineering] | interpreter and library versions asserted | the versions |\n"
            "| 5. Pretrained model | [Concept] | the 150-class model on a demonstration scene (or ADE20K fixtures / your image) | top classes, or mIoU on fixtures |\n"
            "| 6. Dataset | [Evaluation practice] | 24 synthetic scenes (or your dataset) validated | the manifest and a refusal |\n"
            "| 7. Split | [Evaluation practice] | 18 train / 6 validation | the ids |\n"
            "| 8. Re-head | [Concept] | a fresh copy re-headed onto 3 classes, backbone frozen | parameter counts, the random-head reference |\n"
            "| 9. Fine-tune | [Concept] | 3 epochs of AdamW on the heads | the loss per epoch |\n"
            "| 10. Evaluate | [Evaluation practice] | adapted model vs majority and colour baselines | the principal result and its limits |\n"
            "| 11. Unseen scene | [Concept] | one new scene | its mIoU |\n"
            "| 12–13. Export and reload | [Engineering] | artifact written, reloaded, masks compared | `exact_mask_match` |\n"
            "| 14. Outputs | [Engineering] | CSV, JSON and provenance | the file list |\n"
            "| 15. Activity (optional) | [Concept] | change the learning rate | your comparison |\n"
            "| Troubleshooting, Glossary, Conclusion | — | recovery, terms, your notes | when needed |"
        ),
        (
            "<details>\n<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
            "| Term | Meaning in this notebook |\n"
            "|---|---|\n"
            "| **Semantic segmentation** | Labelling every pixel with one class; objects of the same class are not told apart. |\n"
            "| **Class-index mask** | A 2-D array the size of the image whose value at each pixel is a class number; `255` means \"ignore\". |\n"
            "| **ADE20K-150** | The 150 scene categories the pretrained model was trained on. |\n"
            "| **UPerNet** | A decoder that fuses the backbone's features at several scales into one per-pixel prediction. |\n"
            "| **Decode head / auxiliary head** | The main per-pixel classifier and a helper classifier used during training; each ends in a `conv_seg` layer with one output per class. |\n"
            "| **Re-heading** | Replacing the final `conv_seg` layers so the model predicts a new set of classes; the new layers start random. |\n"
            "| **Frozen backbone** | The Swin-T feature extractor's weights are not updated during fine-tuning; only the heads learn. |\n"
            "| **IoU / mIoU** | For one class, overlap of predicted and true pixels divided by their union; mIoU is the mean over classes present. |\n"
            "| **Pixel accuracy** | The share of pixels labelled correctly; dominated by large classes. |\n"
            "| **Majority baseline** | Paint every pixel with the most frequent class. |\n"
            "| **Colour baseline** | Label each pixel with the class whose mean training colour is nearest; no learning, no context. |\n"
            "| **Random-head reference** | The re-headed model before any training; it says nothing about the pretrained features. |\n"
            "| **`IGNORE_INDEX`** | The mask value `255`: pixels that are not scored or trained on. |\n"
            "| **`.pth` trust boundary** | The checkpoint is a PyTorch pickle that can run code when loaded; the digest check fixes its bytes, not its author. |\n"
            "| **Adapter artifact** | The `.pt` file holding the adapted weights and class names, reloaded with `weights_only=True`. |\n"
            "| **Isolated environment** | A separate Python 3.10 built from the hash-locked pins, in which every learner cell runs. |\n\n"
            "</details>"
        ),
    ],
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime; the kernel's own Python version does not matter. The qualified OpenMMLab stack — torch 2.1.2 (CPU build), MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation 1.2.2, NumPy 1.26.4 — has prebuilt wheels for Python 3.10 only, so Section 1 has `uv` provision a managed **CPython 3.10.18**, installs the hash-locked pins there and runs every later cell in that interpreter (Section 4 asserts it). The recorded runtime is a Python 3.10 Jupyter kernel on Linux (GitHub Actions); Google Colab (Python 3.12 kernel) is expected to work the same way but **no Colab run has been recorded yet**. CPU is the default and only qualified path; no GPU is required.",
        "- **External package indexes:** the isolated install reads the PyTorch CPU index (`download.pytorch.org`) and the OpenMMLab wheel page (`download.openmmlab.com`) besides PyPI.",
        "- **Knowledge:** basic Python and PIL; dense semantic class masks; intersection-over-union (IoU) and pixel accuracy.",
        "- **Data:** the default path generates everything deterministically in code by `samples.py` (no dataset download): one 512×384 demonstration scene and a 24-image custom segmentation dataset with exact masks. `USE_ADE20K_FIXTURES` (off) fetches two ADE20K validation images with their annotations from a pinned Hugging Face commit, verified by SHA-256, and scores the pretrained model on them. Two BYOD branches are off by default: `USE_BYOD_IMAGE` (one image, by `BYOD_IMAGE_PATH` or the Colab upload) and `USE_BYOD_DATASET` (a folder or `.zip` with `images/`, `masks/` and `classes.txt`; see Section 6). Do not upload confidential or restricted imagery to a hosted runtime unless you are authorised to process it there; uploads stay in this runtime.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime builds an isolated environment with a uv-managed CPython 3.10.18 and the "
        "hash-locked pins (the kernel's own Python and packages are left alone, so no restart is needed), stages and digest-verifies the pinned checkpoint, "
        "runs pretrained ADE20K inference on a demonstration scene, validates the 24-image segmentation adaptation dataset, splits it into "
        "train and validation sets, re-heads a fresh copy of the pretrained model, **runs the bounded fine-tune with frozen backbone**, evaluates on "
        "the held-out split beside a majority and a colour baseline, runs inference on an unseen test scene, exports the adapted artifact, reloads it from disk to verify numeric consistency, "
        "and writes machine-readable outputs with provenance. Nothing is skipped behind a default-off flag, and no runtime restart is needed "
        "(NOTEBOOK_SPEC 2.2 §5, RUN7, FT2). The recorded runs are GitHub Actions runs in a Python 3.10 kernel; no Colab run is recorded yet."
    ),
    "byod": (
        "Two optional BYOD branches are included and both are disabled by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image (by `BYOD_IMAGE_PATH`, or the Colab upload) through the pretrained ADE20K model, after "
        "`validate_inputs`. `USE_BYOD_DATASET` takes a folder or `.zip` (`BYOD_DATASET_PATH`, or the Colab upload) holding `images/`, "
        "`masks/` (single-channel PNG class-index rasters, `255` = ignore) and `classes.txt`, validates it, and runs it through the same "
        "split, re-head, fine-tune, evaluation, export and reload cells as the synthetic dataset (DAT14). "
        "Do not upload confidential or restricted imagery or data to hosted notebook environments."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. Confirm runtime
        {
            "md": (
                "## 4. Confirm the qualified runtime · [Engineering]\n\n"
                "The carried package fails closed on version drift: `verify_runtime_versions` compares the installed "
                "`mmseg`, `mmcv` and `mmengine` distributions with the versions pinned in `MODEL_SPEC`, and this cell "
                "additionally asserts the Python 3.10 interpreter the OpenMMLab wheels were built for — the isolated environment's, "
                "not the kernel's. Look for a dictionary reporting Python 3.10.x, `torch` 2.1.2+cpu, MMSegmentation 1.2.2, MMCV 2.1.0, "
                "MMEngine 0.10.7, 150 classes, the verified checkpoint file name and the `local-snapshot` source."
            ),
            "code": (
                "import platform\n"
                "import sys\n"
                "import numpy\n"
                "import torch\n\n"
                "if sys.version_info[:2] != (3, 10):\n"
                "    raise RuntimeError(f'Python 3.10 is required by the qualified OpenMMLab runtime (see Prerequisites); this interpreter is {{sys.version.split()[0]}}. Run the Section 1 cells first: they provide a managed Python 3.10 whatever the kernel runs.')\n"
                "print({{'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__, **pipe.versions, 'classes': len(pipe.classes), 'checkpoint': pipe.checkpoint.name, 'source': pipe.source, 'device': pipe.device}})"
            ),
        },
        # ---------------------------------------------------------------- 5. Pretrained demonstration
        {
            "md": (
                "## 5. Demonstrate the pretrained ADE20K model · [Concept]\n\n"
                "Before adapting anything, this cell runs the unchanged 150-class model (`pipe`). By default it uses a deterministic "
                "512×384 demonstration scene (a gradient with three flat shapes). That scene has **no ground truth**, so the cell only "
                "lists the top predicted classes: the pretrained model's quality is **not measured** on the default path.\n\n"
                "`USE_ADE20K_FIXTURES = True` measures it instead: the cell fetches two ADE20K validation images and their annotation "
                "maps from the dataset `hf-internal-testing/fixtures_ade20k` at the immutable commit `850d349e…`, refuses any file whose "
                "SHA-256 differs from the recorded digest, converts the raw labels with the carried `ade20k_raw_to_indices` (`0` → "
                "ignore, `1..150` → `0..149`, the pinned config's convention) and reports mIoU and pixel accuracy beside the majority "
                "baseline — two images, so a sanity check, not a benchmark.\n\n"
                "`USE_BYOD_IMAGE = True` runs one image of your own through the pretrained model after `validate_inputs` has checked it "
                "(decodable, positive size, at most 64 megapixels): set `BYOD_IMAGE_PATH`, or leave it empty in Colab for the upload "
                "dialog, which must receive exactly one file.\n\n"
                "**Predict before running:** which ADE20K classes will the model name for a sky-blue-to-green gradient with a dark "
                "rectangle, a white ellipse and a blue triangle?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import os\n"
                "from pathlib import Path\n"
                "from PIL import Image\n\n"
                "sample_dir = Path('sample')\n"
                "sample_dir.mkdir(exist_ok=True)\n"
                "os.makedirs('outputs', exist_ok=True)\n\n"
                "USE_BYOD_IMAGE = False  # @param {{\"type\":\"boolean\"}}\n"
                "BYOD_IMAGE_PATH = ''  # @param {{\"type\":\"string\"}}\n"
                "USE_ADE20K_FIXTURES = False  # @param {{\"type\":\"boolean\"}}\n"
                "FIXTURES_COMMIT = '850d349e5038f291284e7999fcacbedc0922534b'\n"
                "FIXTURES_SHA256 = {{\n"
                "    'ADE_val_00000001.jpg': '99f7af15a7bd66f3d2ad8b98f1b41941c27407a35d303f9b70724cb231a36098',\n"
                "    'ADE_val_00000001.png': '7724c8b985ba9978e968fed231d74fb9e72abd4179f3c4b58bb87c525efb9ae7',\n"
                "    'ADE_val_00000002.jpg': 'ce373a18513e5a357d7fd2d2f70e1db2249417635068e108cd85ddca08195f30',\n"
                "    'ADE_val_00000002.png': 'db1497cd6acd98c61178fbff5611da0604877f55cb7b72655d4e4ecac386904f',\n"
                "}}\n"
                "if USE_BYOD_IMAGE and USE_ADE20K_FIXTURES:\n"
                "    raise ValueError('Enable at most one of USE_BYOD_IMAGE and USE_ADE20K_FIXTURES.')\n"
                "pretrained_ground_truth = None\n"
                "if USE_BYOD_IMAGE:\n"
                "    if BYOD_IMAGE_PATH:\n"
                "        demo_image_path = Path(BYOD_IMAGE_PATH)\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError as exc:\n"
                "            raise RuntimeError('There is no upload dialog outside Google Colab: set BYOD_IMAGE_PATH to an image file in this runtime.') from exc\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one image file (received {{len(uploaded)}}); run the cell again, or set BYOD_IMAGE_PATH.')\n"
                "        upload_name = next(iter(uploaded))\n"
                "        demo_image_path = sample_dir / Path(upload_name).name\n"
                "        demo_image_path.write_bytes(uploaded[upload_name])\n"
                "    byod_image_manifest = validate_inputs(demo_image_path)  # refuses an undecodable or oversized file, naming it\n"
                "    demo_images = [Image.open(demo_image_path).convert('RGB')]\n"
                "    demo_kind = 'BYOD image'\n"
                "elif USE_ADE20K_FIXTURES:\n"
                "    import urllib.request\n\n"
                "    for name, expected in FIXTURES_SHA256.items():\n"
                "        target = sample_dir / name\n"
                "        if not target.exists():\n"
                "            urllib.request.urlretrieve(f'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/{{FIXTURES_COMMIT}}/{{name}}', target)\n"
                "        observed = hashlib.sha256(target.read_bytes()).hexdigest()\n"
                "        if observed != expected:\n"
                "            raise RuntimeError(f'{{name}}: digest {{observed}} != pinned {{expected}}; refusing the fixture (delete it and run the cell again)')\n"
                "    demo_images, pretrained_ground_truth = [], []\n"
                "    for stem in ('ADE_val_00000001', 'ADE_val_00000002'):\n"
                "        image = Image.open(sample_dir / f'{{stem}}.jpg').convert('RGB')\n"
                "        raw = numpy.array(Image.open(sample_dir / f'{{stem}}.png'))\n"
                "        if raw.shape != (image.height, image.width):\n"
                "            raise RuntimeError(f'{{stem}}: annotation shape {{raw.shape}} != image shape {{(image.height, image.width)}}')\n"
                "        demo_images.append(image)\n"
                "        pretrained_ground_truth.append(ade20k_raw_to_indices(raw))\n"
                "    demo_kind = 'ADE20K fixtures'\n"
                "else:\n"
                "    demo_images = [tutorial_scene()]\n"
                "    demo_images[0].save(sample_dir / 'ade20k_demo_scene_512x384.png')\n"
                "    demo_kind = 'synthetic demonstration scene (no ground truth)'\n\n"
                "pretrained_results = [pipe.predict(image) for image in demo_images]\n"
                "pretrained_mask_path = pretrained_results[0].save_mask('outputs/{stem}_pretrained_scene_semantic.png')\n"
                "pretrained_report = evaluation_report(pretrained_results, pretrained_ground_truth, sample_kind=demo_kind)\n"
                "with open('outputs/{stem}_pretrained_evaluation.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(pretrained_report, handle, indent=2, ensure_ascii=False)\n"
                "print({{'sample': demo_kind, **pretrained_results[0].summary(), 'pretrained_classes_top': [pipe.classes[c] for c in pretrained_results[0].classes_present[:5]], 'saved_mask': str(pretrained_mask_path)}})\n"
                "print({{'pretrained_evaluation': pretrained_report['verdict'], 'metrics': pretrained_report.get('metrics'), 'baselines': pretrained_report.get('baselines')}})"
            ),
        },
        # ---------------------------------------------------------------- 6. Custom dataset & validation
        {
            "md": (
                "**What to notice (Section 5):** the mask has the image's height and width, the top classes are ADE20K names, and the "
                "evaluation verdict is `not-measurable` on the default scene.\n\n"
                "<details><summary>Check your reasoning</summary>The model answers in its 150-class vocabulary whatever it is shown, so "
                "a gradient with shapes gets labels such as sky, wall or floor — plausible-sounding, unverifiable, because the scene has "
                "no ground truth. That is why the report says `not-measurable`: a confident label is not evidence. The ADE20K fixtures "
                "give a measured answer on two real images; the earlier carrier of this notebook recorded mIoU 0.387 on them against a "
                "majority baseline of 0.054 (see `docs/release-verification.md`).</details>\n\n"
                "## 6. Generate (or bring) and validate the segmentation dataset · [Evaluation practice]\n\n"
                "The adaptation dataset follows the `core.dataset.vision.raster-mask` contract. By default it is 24 synthetic scenes "
                "with exact, pixel-aligned masks over `ADAPT_CLASSES = ('background', 'road', 'structure')`. `validate_dataset` "
                "checks positive dimensions, image/mask shape agreement and that every mask value is a valid class index or `255` "
                "(`IGNORE_INDEX`); a refusal names the record and the rule.\n\n"
                "**Your own dataset (`USE_BYOD_DATASET = True`).** Set `BYOD_DATASET_PATH` to a folder or a `.zip` already in the "
                "runtime, or leave it empty in Colab for the upload dialog (exactly one `.zip`). The layout is `images/` (PNG or JPEG), "
                "`masks/` (one **single-channel PNG** per image, same file stem, each pixel a class index `0..N-1`, `255` = ignore) and "
                "`classes.txt` (one class name per line, in index order; or `classes.json`, a list). It may sit inside one wrapper "
                "folder; `__MACOSX/` and `.DS_Store` are skipped. Limits: 2–500 images, each side 32–4096 px, 2–254 classes. A mask "
                "holding a class index outside `0..N-1` is refused, naming the record. The flag never falls back to other data: if it "
                "is on and nothing is supplied, the cell stops.\n\n"
                "**Predict before running:** the probe validates an empty dataset. Will that be refused, and with what message?"
            ),
            "code": (
                "USE_BYOD_DATASET = False  # @param {{\"type\":\"boolean\"}}\n"
                "BYOD_DATASET_PATH = ''  # @param {{\"type\":\"string\"}}\n\n"
                "dataset_records = None  # never reuse records left from an earlier run of this cell\n"
                "if USE_BYOD_DATASET:\n"
                "    if BYOD_DATASET_PATH:\n"
                "        byod_source = Path(BYOD_DATASET_PATH)\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError as exc:\n"
                "            raise RuntimeError('USE_BYOD_DATASET is on but BYOD_DATASET_PATH is empty, and there is no upload dialog outside Google Colab: set BYOD_DATASET_PATH to a folder or .zip.') from exc\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'USE_BYOD_DATASET is on: upload exactly one .zip (received {{len(uploaded)}} files), or set BYOD_DATASET_PATH.')\n"
                "        upload_name = next(iter(uploaded))\n"
                "        byod_source = sample_dir / Path(upload_name).name\n"
                "        byod_source.write_bytes(uploaded[upload_name])\n"
                "    if not byod_source.exists():\n"
                "        raise FileNotFoundError(f'BYOD_DATASET_PATH {{byod_source}} does not exist in this runtime.')\n"
                "    skipped_metadata = []\n"
                "    if byod_source.is_file():\n"
                "        if byod_source.suffix.lower() != '.zip':\n"
                "            raise ValueError(f'{{byod_source.name}}: supply a folder or a .zip with images/, masks/ and classes.txt')\n"
                "        skipped_metadata = extract_byod_zip(byod_source, sample_dir / 'byod_dataset')\n"
                "        byod_source = sample_dir / 'byod_dataset'\n"
                "    dataset_records, adapt_classes = read_byod_dataset(byod_source)\n"
                "    dataset_kind = 'BYOD'\n"
                "else:\n"
                "    dataset_records = synthetic_segmentation_dataset(24, seed=DEFAULT_ADAPT_SEED)\n"
                "    adapt_classes = ADAPT_CLASSES\n"
                "    dataset_kind = 'synthetic'\n"
                "    skipped_metadata = []\n\n"
                "dataset_summary = validate_dataset(dataset_records, adapt_classes)\n\n"
                "input_manifest = {{\n"
                "    'schema': dataset_summary['schema'],\n"
                "    'dataset_kind': dataset_kind,\n"
                "    'n_records': dataset_summary['n_records'],\n"
                "    'class_names': dataset_summary['class_names'],\n"
                "    'observed_classes': dataset_summary['observed_classes'],\n"
                "    'skipped_archive_metadata': skipped_metadata,\n"
                "    'verdict': dataset_summary['verdict'],\n"
                "    'findings': [],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "}}\n\n"
                "# Demonstrate rejection on an invalid input\n"
                "try:\n"
                "    validate_dataset([], adapt_classes)\n"
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
                "## 7. Partition the dataset into train and validation splits · [Evaluation practice]\n\n"
                "`split_dataset` deterministically partitions the records with a 25 % holdout (18 training and 6 validation scenes on "
                "the default path). Look for disjoint subsets with their ids."
            ),
            "code": (
                "train_records, val_records = split_dataset(dataset_records, val_fraction=0.25, seed=42)\n"
                "print({{\n"
                "    'total_records': len(dataset_records),\n"
                "    'train_records': len(train_records),\n"
                "    'val_records': len(val_records),\n"
                "    'classes': list(adapt_classes),\n"
                "    'train_ids': [r['id'] for r in train_records[:4]],\n"
                "    'val_ids': [r['id'] for r in val_records],\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 8. Re-head a fresh copy
        {
            "md": (
                "## 8. Re-head a fresh copy of the pretrained model · [Concept]\n\n"
                "The adaptation never touches `pipe`. This cell loads a second model, `adapt_pipe`, from the same verified checkpoint, "
                "then `rehead_model` replaces its `decode_head.conv_seg` (512 → classes) and `auxiliary_head.conv_seg` (256 → classes) "
                "with new layers initialised from `DEFAULT_ADAPT_SEED`, and `freeze_backbone` sets `requires_grad = False` on the "
                "28.3M-parameter Swin-T backbone. Re-running this cell always starts again from the pretrained weights and the same head.\n\n"
                "The re-headed model is then scored on the validation split. This **random-head reference is not a baseline**: the new "
                "head has never seen a label, so it predicts noise and any training beats it. The baselines that say something are in "
                "Section 10."
            ),
            "code": (
                "adapt_pipe = DimerSwinSegmenter.from_pretrained(weights_dir=WEIGHTS_DIR)  # a fresh copy; `pipe` keeps the 150-class head\n"
                "rehead_model(adapt_pipe.model, adapt_classes, seed=DEFAULT_ADAPT_SEED)\n"
                "adapt_pipe.classes = tuple(adapt_classes)\n"
                "frozen_params = freeze_backbone(adapt_pipe.model)\n\n"
                "pre_adapt_report = adapt_pipe.evaluate(val_records, sample_kind=f'{{dataset_kind}}-val')\n"
                "with open('outputs/{stem}_pre_adapt_evaluation.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(pre_adapt_report, handle, indent=2, ensure_ascii=False)\n\n"
                "print({{\n"
                "    'stage': 'random-head reference (not a baseline)',\n"
                "    'frozen_backbone_params': frozen_params,\n"
                "    'classes': list(adapt_pipe.classes),\n"
                "    'pretrained_pipe_classes': len(pipe.classes),\n"
                "    'miou': next((m['value'] for m in pre_adapt_report['metrics'] if m['metric'] == 'miou'), None),\n"
                "    'pixel_accuracy': next((m['value'] for m in pre_adapt_report['metrics'] if m['metric'] == 'pixel_accuracy'), None),\n"
                "    'verdict': pre_adapt_report['verdict'],\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 9. Bounded fine-tune
        {
            "md": (
                "## 9. Run bounded in-process fine-tuning · [Concept]\n\n"
                "`finetune` runs an in-process AdamW loop over the re-headed decode and auxiliary heads with MMSegmentation's own "
                "cross-entropy loss and data preprocessing. With the backbone frozen, it runs 3 epochs over the training split "
                "(batch size 4) and prints the mean loss per epoch. Three short epochs need not fall monotonically; read the overall "
                "direction.\n\n"
                "**Predict before running:** will the loss fall a little, by about half, or by most of its first value over three "
                "epochs?"
            ),
            "code": (
                "LEARNING_RATE = DEFAULT_ADAPT_LEARNING_RATE  # @param {{\"type\":\"number\"}}\n\n"
                "finetune_summary = adapt_pipe.finetune(\n"
                "    train_records,\n"
                "    epochs=DEFAULT_ADAPT_EPOCHS,\n"
                "    batch_size=DEFAULT_ADAPT_BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
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
                "**What to notice (Section 9):** one loss line per epoch and the summary with `epoch_losses` and `final_loss`.\n\n"
                "<details><summary>Check your reasoning</summary>No hosted loss values are recorded for this notebook, so read your "
                "own. With a random head the first epoch's loss is high; on this easy task it usually falls steeply within three "
                "epochs, because a few colour-like features from the frozen backbone are enough to separate the classes.</details>\n\n"
                "## 10. Evaluate the adapted model beside two baselines · [Evaluation practice]\n\n"
                "`adapt_pipe.evaluate` scores the adapted model on the held-out validation split with `semantic_iou` (mIoU, pixel "
                "accuracy, per-class IoU) and the `majority_class_baseline`. This cell adds a second, stronger baseline, "
                "`colour_centroid_baseline`: one mean colour per class fitted on the training masks, and each validation pixel labelled "
                "with the nearest mean colour — no learning and no spatial context.\n\n"
                "**What this comparison can and cannot show.** On the default synthetic data the three classes differ by colour, so the "
                "colour baseline scores close to the ceiling (computed offline on the default split: mIoU 0.975, pixel accuracy 0.995; "
                "majority baseline mIoU 0.184, pixel accuracy 0.553). An adapted mIoU near 1.0 here shows that the re-head, training, "
                "export and reload work; it **cannot** show what the pretrained Swin features add over a colour rule. On your own "
                "dataset, where classes are not separable by colour, the same two baselines make the network's gain readable.\n\n"
                "**Predict before running:** will the adapted model clearly beat the colour baseline on the default data, roughly "
                "match it, or fall short?"
            ),
            "code": (
                "post_adapt_report = adapt_pipe.evaluate(val_records, sample_kind=f'{{dataset_kind}}-val')\n"
                "colour_baseline = colour_centroid_baseline(train_records, val_records, len(adapt_classes))\n"
                "post_adapt_report['baselines'].append({{'id': colour_baseline['id'], 'miou': colour_baseline['miou'], 'pixel_accuracy': colour_baseline['pixel_accuracy'], 'fitted_on': 'train'}})\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(post_adapt_report, handle, indent=2, ensure_ascii=False)\n\n"
                "pre_miou = next((m['value'] for m in pre_adapt_report['metrics'] if m['metric'] == 'miou'), 0.0)\n"
                "post_miou = next((m['value'] for m in post_adapt_report['metrics'] if m['metric'] == 'miou'), 0.0)\n"
                "pre_acc = next((m['value'] for m in pre_adapt_report['metrics'] if m['metric'] == 'pixel_accuracy'), 0.0)\n"
                "post_acc = next((m['value'] for m in post_adapt_report['metrics'] if m['metric'] == 'pixel_accuracy'), 0.0)\n"
                "run_history = globals().get('run_history', [])\n"
                "run_history.append({{'learning_rate': LEARNING_RATE, 'dataset': dataset_kind, 'miou_after': post_miou, 'pixel_acc_after': post_acc, 'final_loss': finetune_summary.get('final_loss')}})\n\n"
                "print(json.dumps({{\n"
                "    'random_head_miou': pre_miou,\n"
                "    'miou_after': post_miou,\n"
                "    'pixel_acc_after': post_acc,\n"
                "    'majority_baseline_miou': post_adapt_report['baselines'][0]['miou'],\n"
                "    'majority_baseline_acc': post_adapt_report['baselines'][0]['pixel_accuracy'],\n"
                "    'colour_baseline_miou': colour_baseline['miou'],\n"
                "    'colour_baseline_acc': colour_baseline['pixel_accuracy'],\n"
                "    'classes_with_union': post_adapt_report['classes_with_union'],\n"
                "    'verdict': post_adapt_report['verdict'],\n"
                "}}, indent=2))\n"
                "for row in post_adapt_report['per_class']:\n"
                "    print(f\"class {{row['class_id']:>2}} {{adapt_pipe.classes[row['class_id']]:<12}} IoU {{row['iou']:.4f}}  inter {{row['intersection_pixels']}} px  union {{row['union_pixels']}} px\")\n"
                "print('run history (one row per evaluation in this session):')\n"
                "for index, row in enumerate(run_history):\n"
                "    print(index, row)\n"
                "if post_miou <= colour_baseline['miou']:\n"
                "    print('NOTE: the adapted model does not beat the colour baseline; this split cannot show what the learned features add.')"
            ),
        },
        # ---------------------------------------------------------------- 11. Adapted inference on unseen test image
        {
            "md": (
                "**What to notice (Section 10):** the adapted mIoU against both baselines, the per-class IoU lines, the run history "
                "and, if it applies, the NOTE.\n\n"
                "<details><summary>Check your reasoning</summary>Roughly match it. The colour rule is already near the ceiling on these "
                "scenes, so even a perfect network could only tie it within a few hundredths; the small remaining errors of the colour "
                "rule sit on outlines and roof edges whose colours differ from the class mean. The honest sentence is: \"the pipeline "
                "adapts and reproduces end to end; this data cannot separate learned features from colour\".</details>\n\n"
                "## 11. Run inference on an unseen test scene · [Concept]\n\n"
                "On the default path this cell draws a fresh scene that is in neither split (`generate_scene(99)`), runs the adapted "
                "model, scores IoU against its known mask and exports `outputs/{stem}_test_scene_adapted_semantic.png`. With your own "
                "dataset there is no generator for your domain, so the cell uses the first validation record instead and says so.\n\n"
                "**Predict before running:** will the unseen scene score about the same mIoU as the validation split, or clearly "
                "less?"
            ),
            "code": (
                "if dataset_kind == 'BYOD':\n"
                "    test_img, test_mask = val_records[0]['image'], numpy.asarray(val_records[0]['mask'])\n"
                "    test_origin = f\"validation record {{val_records[0]['id']}} (no unseen generator for BYOD data)\"\n"
                "else:\n"
                "    test_img, test_mask = generate_scene(99, seed=DEFAULT_ADAPT_SEED)\n"
                "    test_origin = 'generate_scene(99): in neither split'\n"
                "test_image_path = sample_dir / 'test_scene.png'\n"
                "test_img.save(test_image_path)\n\n"
                "adapted_test_result = adapt_pipe.predict(test_img)\n"
                "adapted_mask_path = adapted_test_result.save_mask('outputs/{stem}_test_scene_adapted_semantic.png')\n\n"
                "test_scored = semantic_iou([adapted_test_result.mask], [test_mask], num_classes=len(adapt_pipe.classes))\n"
                "print({{\n"
                "    **adapted_test_result.summary(),\n"
                "    'test_origin': test_origin,\n"
                "    'mask_file': str(adapted_mask_path),\n"
                "    'test_scene_miou': test_scored['miou'],\n"
                "    'test_scene_pixel_acc': test_scored['pixel_accuracy'],\n"
                "    'classes_predicted': [adapt_pipe.classes[c] for c in adapted_test_result.classes_present],\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 12. Artifact export
        {
            "md": (
                "**What to notice (Section 11):** `test_origin`, the scene's mIoU and pixel accuracy, and the classes predicted.\n\n"
                "<details><summary>Check your reasoning</summary>About the same: scene 99 is drawn by the same generator with the "
                "same colours, so it is new but not different — the colour baseline scores mIoU 0.973 on it (computed offline). "
                "A held-out scene from the same generator tests memorisation of particular scenes, not generalisation to new "
                "imagery.</details>\n\n"
                "## 12. Export the portable adapted artifact · [Engineering]\n\n"
                "`adapt_pipe.save_artifact` exports the adapted weights, class vocabulary, base model identity and provenance as a "
                "standalone `.pt` artifact (`outputs/swin-segmentation-adapter-v1.pt`)."
            ),
            "code": (
                "adapter_path = Path('outputs/swin-segmentation-adapter-v1.pt')\n"
                "artifact_descriptor = adapt_pipe.save_artifact(adapter_path, notes=f'Swin-T UPerNet {{len(adapt_classes)}}-class adapted segmentation model ({{dataset_kind}} data)')\n"
                "print(json.dumps(artifact_descriptor, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 13. Fresh reload & verification
        {
            "md": (
                "## 13. Fresh reload & numerical verification · [Engineering]\n\n"
                "`DimerSwinSegmenter.load_artifact` reloads the exported weights with `weights_only=True`, constructs a fresh segmenter "
                "and the cell asserts that its mask on the test scene is identical to the adapted model's "
                "(`numpy.testing.assert_array_equal`)."
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
                "## 14. Export outputs and provenance · [Engineering]\n\n"
                "Writes machine-readable outputs: class coverage CSV (`outputs/{stem}_class_coverage.csv`), summary result JSON "
                "(`outputs/{stem}_result.json`, now with both baselines, the per-epoch losses and the pretrained-stage verdict) and "
                "the provenance record (`outputs/{stem}_provenance.json`)."
            ),
            "code": (
                "import csv\n\n"
                "val_predictions = [adapt_pipe.predict(r['image']) for r in val_records]\n"
                "coverage = []\n"
                "for r, pred in zip(val_records, val_predictions, strict=True):\n"
                "    values, counts = numpy.unique(pred.mask, return_counts=True)\n"
                "    order = numpy.argsort(counts)[::-1]\n"
                "    for class_id, pixels in zip(values[order].tolist(), counts[order].tolist(), strict=True):\n"
                "        coverage.append({{\n"
                "            'image_id': r['id'],\n"
                "            'class_id': int(class_id),\n"
                "            'class_name': adapt_pipe.classes[int(class_id)],\n"
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
                "    'dataset_kind': dataset_kind,\n"
                "    'adapted_vocabulary': list(adapt_pipe.classes),\n"
                "    'num_classes': len(adapt_pipe.classes),\n"
                "    'pretrained_stage': {{'sample': demo_kind, 'verdict': pretrained_report['verdict'], 'metrics': pretrained_report.get('metrics')}},\n"
                "    'random_head_miou': pre_miou,\n"
                "    'post_adaptation_miou': post_miou,\n"
                "    'post_adaptation_pixel_acc': post_acc,\n"
                "    'majority_baseline_miou': post_adapt_report['baselines'][0]['miou'],\n"
                "    'colour_baseline': {{'miou': colour_baseline['miou'], 'pixel_accuracy': colour_baseline['pixel_accuracy']}},\n"
                "    'epoch_losses': finetune_summary.get('epoch_losses'),\n"
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
                "provenance_path = adapt_pipe.write_provenance('outputs/{stem}_provenance.json')\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## 15. Activity: change one thing — the learning rate · [Concept]\n\n"
        "Optional; **Predict → Change one thing → Run → Observe → Explain**.\n\n"
        "1. **Predict:** with `LEARNING_RATE = 2e-4` (or `5e-5`) instead of the default, will the adapted mIoU on the validation split "
        "rise, fall or stay about the same? Will the final loss change more than the mIoU?\n"
        "2. **Change:** in Section 9 set `LEARNING_RATE`. Change nothing else.\n"
        "3. **Run:** select the **Section 8** cell and choose *Runtime → Run after* (Sections 8–14). Section 8 reloads the pretrained "
        "model and re-heads it with the same seed, so both runs start from identical weights; running Section 9 alone would continue "
        "training the already-adapted heads instead.\n"
        "4. **Observe:** Section 10 prints the run history — one row per evaluation, the default first — side by side.\n"
        "5. **Explain:** why can the loss move while the mIoU barely does on this data?\n\n"
        "<details><summary>Check your reasoning</summary>The mIoU is close to its ceiling and the colour baseline is already near it, "
        "so a different learning rate mostly changes how confidently the heads separate the colours (the loss), not which class wins "
        "at each pixel (the mIoU). A large difference in mIoU would point to a learning rate too high for three epochs. Only data "
        "that colour cannot separate — your own dataset — makes this comparison informative.</details>\n\n"
        "## Interpretation and limits\n\n"
        "Each pixel receives one class index from the target vocabulary (`background`, `road`, `structure` on the default path) by "
        "per-pixel argmax; the API exposes **no per-pixel confidence** and the package ships no threshold. The synthetic dataset "
        "demonstrates that the architecture can be re-headed and adapted to custom segmentation classes in-process with a frozen "
        "backbone. Because its classes are separable by colour, a non-learned colour rule scores close to the ceiling too, so the default "
        "adaptation numbers do not show what the pretrained features contribute, and they do not represent real-world photographic "
        "scene segmentation. The pretrained 150-class model is measured only when `USE_ADE20K_FIXTURES` is on, on two images.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire and "
        "digest-verify the pinned OpenMMLab checkpoint, provision and assert the qualified Python 3.10 runtime in an isolated "
        "environment, validate the dataset contract, re-head a fresh copy of the model, run a bounded fine-tune, evaluate mIoU against "
        "a majority and a colour baseline, export the adapted artifact, and reload it with exact numerical reproducibility — without "
        "the repository being reachable. "
        "It does **not** establish benchmark superiority against other semantic-segmentation architectures or full ADE20K benchmarks.\n\n"
        "**Next experiments:** the Section 15 activity; enable `USE_ADE20K_FIXTURES` to measure the pretrained model; enable "
        "`USE_BYOD_DATASET` with your own images and masks, ideally classes that colour alone cannot separate, and compare the adapted "
        "model with both baselines; experiment with unfreezing the later stages of the Swin-T backbone.\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a Windows or macOS kernel, or an ARM machine | Use a Linux x86_64 Jupyter kernel or Google Colab. |\n"
        "| Section 1 fails while downloading `uv`, or `The pinned uv wheel failed its size/SHA-256 check` | a network failure, or an altered download | Run the Section 1 install cell again; never edit the digest. |\n"
        "| `CalledProcessError` from `uv pip install` in Section 1 | the PyTorch CPU index, the OpenMMLab wheel page or PyPI was unreachable | Run the cell again later; never loosen a pin. |\n"
        "| `holds Python …, not 3.10.18` in Section 1 | an older `dimer_isolated_env/` folder | Delete that folder (or start a fresh runtime) and run the install cell again. |\n"
        "| `Python 3.10 is required …` in Section 4 | Section 1 was skipped, so the cell ran in the kernel | Run Sections 1–3 first, or choose *Run all*. |\n"
        "| The checkpoint download fails in Section 3, or `sha256 … != manifest` | `download.openmmlab.com` unreachable, or a changed file | Run the Section 3 cell again; a mismatch is never loaded. |\n"
        "| `RuntimeError` about `mmseg`, `mmcv` or `mmengine` versions | runtime-version drift | Delete `dimer_isolated_env/` and run Section 1 again. |\n"
        "| `…: digest … != pinned …; refusing the fixture` in Section 5 | a partial or changed ADE20K fixture download | Delete the file under `sample/` and run the cell again. |\n"
        "| `There is no upload dialog outside Google Colab` / `Upload exactly one …` | BYOD without a path outside Colab, or a cancelled or multi-file upload | Set `BYOD_IMAGE_PATH` / `BYOD_DATASET_PATH`, or run the cell again and choose one file. |\n"
        "| `USE_BYOD_DATASET is on but BYOD_DATASET_PATH is empty …` | the flag is on and no data was supplied | Set `BYOD_DATASET_PATH`, or turn the flag off. |\n"
        "| `… needs images/ and masks/ folders`, `no mask named …`, `classes.txt` errors | the BYOD layout differs from Section 6 | Fix the layout: `images/`, `masks/` (same stems, single-channel PNG), `classes.txt`. |\n"
        "| `Record …: mask contains class index … outside valid range` | a mask value is not a class in `classes.txt` (and not `255`) | Fix that mask or add the class; the message names the record. |\n"
        "| `Reloaded model mask must be numerically identical …` | the artifact changed on disk, or a nondeterministic kernel | Re-run Sections 12–13; do not ship the artifact if it repeats. |\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. The pretrained model on the default scene named ___; it was / was not measured (`USE_ADE20K_FIXTURES`: mIoU ___).\n"
        "2. After fine-tuning, validation mIoU was ___ against ___ for the colour baseline and ___ for the majority baseline.\n"
        "3. What this comparison can show, and what it cannot: ___.\n"
        "4. The reloaded artifact reproduced the masks exactly: ___ (yes/no).\n"
        "5. Before trusting an adapted model on my own imagery I would need ___.\n\n"
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
