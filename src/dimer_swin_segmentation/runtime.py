from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .metrics import majority_class_baseline, semantic_iou

MODEL_SPEC = {
    "runtime_id": "swin-t-upernet-ade20k-mmseg-v1.2.2",
    "architecture": "Swin-T + UPerNet",
    "task": "semantic-segmentation",
    "dataset": "ADE20K",
    "mmsegmentation_version": "1.2.2",
    "mmcv_version": "2.1.0",
    "mmengine_version": "0.10.7",
    "torch_version": "2.1.2",
    "config": "swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py",
    "config_source_revision": "open-mmlab/mmsegmentation@v1.2.2",
    "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth",
    "checkpoint_size_bytes": 240154742,
    "checkpoint_sha256": "e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064",
    "upstream_reported_miou": 44.41,
    "num_classes": 150,
}

ARTIFACT_FORMAT = "dimer_swin_segmentation_adapter_v1"
DEFAULT_ADAPT_EPOCHS = 3
DEFAULT_ADAPT_BATCH_SIZE = 4
DEFAULT_ADAPT_LEARNING_RATE = 1e-4
DEFAULT_ADAPT_SEED = 20260915


@dataclass(frozen=True)
class SegmentationResult:
    image_id: str
    mask: np.ndarray
    classes_present: tuple[int, ...]

    def summary(self) -> dict:
        return {
            "image_id": self.image_id,
            "shape": list(self.mask.shape),
            "classes_present": list(self.classes_present),
            "num_classes_present": len(self.classes_present),
        }

    def save_mask(self, path: str | os.PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.mask.astype(np.uint8), mode="L").save(target)
        return target


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _version(dist: str) -> str:
    return importlib.metadata.version(dist)


# ---------------------------------------------------------------------------------------------
# Fleet snapshot scheme (DIMER standalone carrier). The identity constants below name the OpenMMLab
# distribution: MODEL_ID is the config recipe inside the pinned package, MODEL_REVISION the upstream
# git commit of that package's release tag (the config source), and the manifest pins the checkpoint
# bytes. The checkpoint host is download.openmmlab.com, not the Hugging Face Hub, so the staging
# downloader is the pinned URL in MODEL_SPEC rather than hf_hub_download.
# ---------------------------------------------------------------------------------------------
MODEL_ID = "open-mmlab/mmsegmentation:swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512"
MODEL_REVISION = "c685fe6767c4cadf6b051983ca6208f1b9d1ccb8"
MODEL_LICENSE = "Apache-2.0"
MODEL_KEY = "swin-t-upernet-ade20k"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth"
MAX_PIXELS = 64_000_000  # validate_image ceiling


def verify_snapshot(path: str | os.PathLike | None = None) -> dict:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    entries = {entry["path"]: entry for entry in manifest.get("files", [])}
    pinned = entries.get(WEIGHTS_FILE)
    if pinned is None or pinned["sha256"] != MODEL_SPEC["checkpoint_sha256"] or pinned["bytes"] != MODEL_SPEC["checkpoint_size_bytes"]:
        raise ValueError(f"manifest entry for {WEIGHTS_FILE} does not match the checkpoint digest pinned in MODEL_SPEC")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _openmmlab_download(relative_path: str, root: Path) -> None:
    """Fetch the pinned OpenMMLab checkpoint into the snapshot directory (the only manifest entry)."""
    if relative_path != WEIGHTS_FILE:
        raise ValueError(f"no pinned download source for {relative_path}")
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="dimer-swin-", suffix=".pth", dir=root)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(MODEL_SPEC["checkpoint_url"], timeout=120) as response, temporary.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def stage_missing_files(
    path: str | os.PathLike | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the checkpoint). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them from {MODEL_SPEC['checkpoint_url']}"
        )
    fetch = downloader or _openmmlab_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def verify_runtime_versions() -> dict[str, str]:
    expected = {
        "mmsegmentation": MODEL_SPEC["mmsegmentation_version"],
        "mmcv": MODEL_SPEC["mmcv_version"],
        "mmengine": MODEL_SPEC["mmengine_version"],
    }
    actual = {name: _version(name) for name in expected}
    for name, expected_version in expected.items():
        if actual[name] != expected_version:
            raise RuntimeError(
                f"Unsupported {name} {actual[name]}; this runtime is qualified for {expected_version}."
            )
    return actual


def resolve_packaged_config() -> Path:
    import mmseg

    config = (
        Path(mmseg.__file__).resolve().parent
        / ".mim"
        / "configs"
        / MODEL_SPEC["config"]
    )
    if not config.is_file():
        raise RuntimeError(
            f"The pinned MMSegmentation package does not contain the expected config: {config}"
        )
    return config


def acquire_verified_checkpoint(cache_dir: str | os.PathLike = ".dimer-models") -> Path:
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / Path(MODEL_SPEC["checkpoint_url"]).name

    def valid(path: Path) -> bool:
        return (
            path.is_file()
            and path.stat().st_size == MODEL_SPEC["checkpoint_size_bytes"]
            and _sha256(path) == MODEL_SPEC["checkpoint_sha256"]
        )

    if valid(target):
        return target
    if target.exists():
        target.unlink()

    fd, temporary_name = tempfile.mkstemp(prefix="dimer-swin-segmentation-", suffix=".pth", dir=root)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(MODEL_SPEC["checkpoint_url"], timeout=120) as response, temporary.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if temporary.stat().st_size != MODEL_SPEC["checkpoint_size_bytes"]:
            raise RuntimeError(
                f"Checkpoint size mismatch: got {temporary.stat().st_size}, expected {MODEL_SPEC['checkpoint_size_bytes']}."
            )
        digest = _sha256(temporary)
        if digest != MODEL_SPEC["checkpoint_sha256"]:
            raise RuntimeError(
                f"Checkpoint SHA-256 mismatch: got {digest}, expected {MODEL_SPEC['checkpoint_sha256']}."
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def validate_image(path: str | os.PathLike, *, max_pixels: int = 64_000_000) -> dict:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise ValueError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
            mode = image.mode
    except Exception as exc:
        raise ValueError(f"Input is not a readable image: {image_path}: {exc}") from exc
    if width < 1 or height < 1:
        raise ValueError(f"Image dimensions must be positive; got {width}x{height}.")
    if width * height > max_pixels:
        raise ValueError(
            f"Image has {width * height:,} pixels; ceiling is {max_pixels:,}. Resize before inference."
        )
    return {"path": str(image_path), "width": width, "height": height, "mode": mode}


INPUT_SCHEMA: dict = {
    "input": "one or more image files readable by Pillow (any mode), given by path",
    "pixels": [1, MAX_PIXELS],
    "classes": "150 ADE20K categories in MMSegmentation order (or custom classes when adapted)",
    "preprocessing": "MMSegmentation test pipeline of the pinned config (resize to 512-scale, normalise); nothing is altered by this module",
}


def validate_inputs(
    images: str | os.PathLike | Iterable[str | os.PathLike],
    *,
    names: Iterable[str] | None = None,
) -> dict:
    """Validation stage: return the input manifest (schema, per-image observations, verdict)."""
    paths = [images] if isinstance(images, (str, os.PathLike)) else list(images)
    observed = [validate_image(p) for p in paths]
    ids = list(names) if names is not None else [Path(o["path"]).name for o in observed]
    if len(ids) != len(observed):
        raise ValueError("names must have one entry per image")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": ids[i], **o} for i, o in enumerate(observed)],
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    results: Iterable[SegmentationResult | dict],
    ground_truth: Iterable[np.ndarray] | None = None,
    *,
    sample_kind: str = "synthetic",
    num_classes: int | None = None,
) -> dict:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``ground_truth`` (one 2-D map of class indices ``0..num_classes-1`` per result, ``255`` = ignore)
    the report carries ``semantic_iou`` (mean IoU over classes present, pixel accuracy, per-class IoU)
    and the ``majority_class_baseline`` with verdict ``sample-sanity``. Without ground truth the verdict
    is ``not-measurable``.
    """
    items = list(results)
    rows = [r.summary() if isinstance(r, SegmentationResult) else dict(r) for r in items]
    eval_num_classes = num_classes if num_classes is not None else MODEL_SPEC["num_classes"]
    base = {
        "task": f"semantic-segmentation ({eval_num_classes} classes)",
        "score_semantics": "argmax class per pixel; no per-pixel confidence is exposed",
        "sample_kind": sample_kind,
        "n_images": len(rows),
        "num_classes": eval_num_classes,
        "context": {
            "upstream_reported_miou": MODEL_SPEC["upstream_reported_miou"],
            "note": "upstream full-ADE20K mIoU as reported by OpenMMLab; not measured here",
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if ground_truth is None:
        return {
            **base,
            "metrics": [],
            "baselines": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth masks were supplied for the evaluated images",
            "needs": (
                "Ground-truth indexed masks (indices 0..num_classes-1, 255 = ignore) scored with "
                "semantic_iou (mean IoU over classes present, pixel accuracy, per-class IoU) against "
                "the majority_class_baseline; a labelled set from the deployment domain for any generalisable claim."
            ),
        }
    if not all(isinstance(item, SegmentationResult) for item in items):
        raise TypeError("ground_truth evaluation needs SegmentationResult inputs (the predicted masks)")
    references = [np.asarray(reference) for reference in ground_truth]
    scored = semantic_iou([item.mask for item in items], references, num_classes=eval_num_classes)
    baseline = majority_class_baseline(references, num_classes=eval_num_classes)
    estimation = (
        f"sample of {scored['n_images']} image(s) / {scored['valid_pixels']} labelled pixels, "
        "aggregate IoU, no dispersion estimate"
    )
    return {
        **base,
        "metrics": [
            {"id": "semantic_iou", "metric": "miou", "value": scored["miou"], "estimation": estimation},
            {"id": "semantic_iou", "metric": "pixel_accuracy", "value": scored["pixel_accuracy"], "estimation": estimation},
        ],
        "per_class": scored["per_class"],
        "classes_with_union": scored["classes_with_union"],
        "baselines": [
            {
                "id": "majority_class_baseline",
                "majority_class_id": baseline["majority_class_id"],
                "miou": baseline["miou"],
                "pixel_accuracy": baseline["pixel_accuracy"],
                "note": "constant predictor of the sample's most frequent class; derived from the same sample",
            }
        ],
        "verdict": "sample-sanity",
        "reason": f"{scored['n_images']} labelled image(s); evaluated locally",
        "needs": "a representative labelled holdout from the deployment domain for any generalisable mIoU claim",
    }


def rehead_model(model: Any, class_names: Sequence[str], *, seed: int = DEFAULT_ADAPT_SEED) -> tuple[int, int]:
    """Re-head the UPerNet decode head and auxiliary head onto a custom class vocabulary."""
    num_classes = len(class_names)
    if num_classes < 1:
        raise ValueError("class_names must contain at least 1 class")

    if hasattr(model, "dataset_meta") and isinstance(model.dataset_meta, dict):
        model.dataset_meta["classes"] = tuple(class_names)

    in_decode = 512
    in_aux = 256

    decode_head = getattr(model, "decode_head", None)
    if decode_head is not None and hasattr(decode_head, "conv_seg"):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        in_decode = decode_head.conv_seg.in_channels
        new_conv = nn.Conv2d(in_decode, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.constant_(new_conv.bias, 0.0)
        decode_head.conv_seg = new_conv
        decode_head.num_classes = num_classes
        decode_head.out_channels = num_classes

    aux_head = getattr(model, "auxiliary_head", None)
    if aux_head is not None and hasattr(aux_head, "conv_seg"):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        in_aux = aux_head.conv_seg.in_channels
        new_conv_aux = nn.Conv2d(in_aux, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(new_conv_aux.weight, mode="fan_out", nonlinearity="relu")
        if new_conv_aux.bias is not None:
            nn.init.constant_(new_conv_aux.bias, 0.0)
        aux_head.conv_seg = new_conv_aux
        aux_head.num_classes = num_classes
        aux_head.out_channels = num_classes

    return in_decode, in_aux


def freeze_backbone(model: Any) -> int:
    """Freeze all parameters in the model backbone; return total frozen parameters."""
    frozen = 0
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "parameters"):
        for p in backbone.parameters():
            p.requires_grad = False
            frozen += p.numel()
    return frozen


class DimerSwinSegmenter:
    """Public semantic-segmentation task-inference and E2E adaptation API for Swin-T UPerNet."""

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike = ".dimer-models",
        device: str = "cpu",
        checkpoint: str | os.PathLike | None = None,
        source: str = "openmmlab-cache",
        class_names: Sequence[str] | None = None,
        freeze_backbone_weights: bool = False,
        seed: int = DEFAULT_ADAPT_SEED,
    ):
        versions = verify_runtime_versions()
        checkpoint = Path(checkpoint) if checkpoint is not None else acquire_verified_checkpoint(cache_dir)
        config = resolve_packaged_config()
        from mmseg.apis import init_model

        self.model = init_model(str(config), str(checkpoint), device=device)
        self.device = device
        self.checkpoint = checkpoint
        self.config = config
        self.versions = versions
        self.source = source
        self.adapted = False
        self.seed = seed
        self.base_state_digest = _sha256(self.checkpoint) if self.checkpoint.is_file() else ""

        if class_names is not None:
            rehead_model(self.model, class_names, seed=seed)
            if freeze_backbone_weights:
                freeze_backbone(self.model)
            self.classes = tuple(class_names)
        else:
            self.classes = tuple(self.model.dataset_meta.get("classes", ()))
            if len(self.classes) != MODEL_SPEC["num_classes"]:
                raise RuntimeError(
                    f"Expected {MODEL_SPEC['num_classes']} ADE20K classes, got {len(self.classes)}."
                )

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str = "cpu",
        weights_dir: str | os.PathLike | None = None,
        allow_download: bool = False,
        class_names: Sequence[str] | None = None,
        freeze_backbone: bool = False,
        seed: int = DEFAULT_ADAPT_SEED,
    ) -> DimerSwinSegmenter:
        """Load from the fleet snapshot directory, verify digest, and optionally re-head for custom classes."""
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no snapshot manifest at {root}; use DimerSwinSegmenter(cache_dir=...) for the cache path"
            )
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        return cls(
            device=device,
            checkpoint=root / WEIGHTS_FILE,
            source="local-snapshot",
            class_names=class_names,
            freeze_backbone_weights=freeze_backbone,
            seed=seed,
        )

    def predict(self, image: str | os.PathLike | Image.Image | np.ndarray) -> SegmentationResult:
        if isinstance(image, Image.Image):
            info = {"path": "in_memory_image.png", "width": image.width, "height": image.height, "mode": image.mode}
            img_input = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            info = {"path": "in_memory_array.png", "width": image.shape[1], "height": image.shape[0], "mode": "RGB"}
            img_input = image
        else:
            info = validate_image(image)
            img_input = info["path"]

        from mmseg.apis import inference_model

        sample = inference_model(self.model, img_input)
        mask = sample.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.uint8, copy=False)
        if mask.ndim != 2:
            raise RuntimeError(f"Expected a 2-D semantic mask; got shape {mask.shape}.")
        classes_present = tuple(int(x) for x in np.unique(mask))
        return SegmentationResult(
            image_id=Path(info["path"]).name,
            mask=mask,
            classes_present=classes_present,
        )

    def finetune(
        self,
        records: Sequence[dict[str, Any]],
        *,
        epochs: int = DEFAULT_ADAPT_EPOCHS,
        batch_size: int = DEFAULT_ADAPT_BATCH_SIZE,
        learning_rate: float = DEFAULT_ADAPT_LEARNING_RATE,
        seed: int = DEFAULT_ADAPT_SEED,
        freeze_backbone_weights: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded in-process fine-tuning on ``records`` using MMSegmentation loss and AdamW."""
        import torch

        from .samples import validate_dataset

        validate_dataset(records, self.classes, epochs=epochs)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
            raise ValueError(f"learning_rate must be a number, got {learning_rate!r}")
        if not 0.0 < float(learning_rate) <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate!r}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)

        if freeze_backbone_weights:
            freeze_backbone(self.model)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)

        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample

        n_records = len(records)
        epoch_losses: list[float] = []

        for epoch in range(epochs):
            self.model.train()
            order = rng.permutation(n_records)
            running_loss = 0.0
            n_batches = 0

            for start in range(0, n_records, batch_size):
                batch_indices = order[start : start + batch_size]
                batch_records = [records[i] for i in batch_indices]

                img_tensors = []
                data_samples = []
                for rec in batch_records:
                    img_raw = rec["image"]
                    if isinstance(img_raw, (str, Path)):
                        with Image.open(img_raw) as opened:
                            arr = np.array(opened.convert("RGB"))
                    elif isinstance(img_raw, Image.Image):
                        arr = np.array(img_raw.convert("RGB"))
                    else:
                        arr = np.asarray(img_raw)

                    mask_raw = rec["mask"]
                    if isinstance(mask_raw, (str, Path)):
                        with Image.open(mask_raw) as opened_m:
                            mask_arr = np.array(opened_m)
                    elif isinstance(mask_raw, Image.Image):
                        mask_arr = np.array(mask_raw)
                    else:
                        mask_arr = np.asarray(mask_raw)

                    h, w = arr.shape[:2]
                    img_t = torch.from_numpy(arr).permute(2, 0, 1).to(self.device).float()
                    mask_t = torch.from_numpy(mask_arr).long().unsqueeze(0).to(self.device)

                    sample = SegDataSample()
                    sample.gt_sem_seg = PixelData(data=mask_t)
                    sample.set_metainfo({"img_shape": (h, w), "ori_shape": (h, w), "pad_shape": (h, w)})

                    img_tensors.append(img_t)
                    data_samples.append(sample)

                batch_data = self.model.data_preprocessor(
                    {"inputs": img_tensors, "data_samples": data_samples}, training=True
                )
                losses = self.model(**batch_data, mode="loss")
                parsed_loss, _ = self.model.parse_losses(losses)

                optimizer.zero_grad(set_to_none=True)
                parsed_loss.backward()
                optimizer.step()

                running_loss += float(parsed_loss.detach().cpu())
                n_batches += 1

            mean_epoch_loss = running_loss / max(1, n_batches)
            epoch_losses.append(mean_epoch_loss)
            if progress is not None:
                progress({"epoch": epoch + 1, "epochs": epochs, "loss": mean_epoch_loss})

        self.model.eval()
        self.adapted = True

        total_params = sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, "parameters") else 0
        trainable_params = sum(p.numel() for p in trainable)

        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "seed": seed,
            "freeze_backbone_weights": freeze_backbone_weights,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "device": self.device,
            "classes": list(self.classes),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def evaluate(
        self,
        records: Sequence[dict[str, Any]],
        *,
        sample_kind: str = "validation",
    ) -> dict[str, Any]:
        """Evaluate the model on a sequence of labelled dataset records."""
        results = [self.predict(rec["image"]) for rec in records]
        references = [np.asarray(rec["mask"]) for rec in records]
        return evaluation_report(
            results,
            references,
            sample_kind=sample_kind,
            num_classes=len(self.classes),
        )

    def save_artifact(self, path: str | os.PathLike, *, notes: str | None = None) -> dict[str, Any]:
        """Write adapted weights, vocabulary, and provenance as a portable .pt artifact."""
        import torch

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_key": MODEL_KEY,
            "checkpoint_sha256": MODEL_SPEC["checkpoint_sha256"],
            "class_names": list(self.classes),
            "adapted": self.adapted,
            "base_state_digest": self.base_state_digest,
            "notes": notes or "",
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
        }
        torch.save(payload, target)
        return {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
            "format": ARTIFACT_FORMAT,
            "class_names": list(self.classes),
            "tensors": len(payload["state_dict"]),
            "base_state_digest": self.base_state_digest,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @classmethod
    def load_artifact(
        cls,
        path: str | os.PathLike,
        *,
        weights_dir: str | os.PathLike | None = None,
        device: str = "cpu",
    ) -> DimerSwinSegmenter:
        """Rebuild an adapted segmentation pipeline from an exported .pt artifact."""
        import torch

        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Artifact not found: {target}")
        payload = torch.load(target, map_location="cpu", weights_only=True)
        if payload.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {payload.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if payload.get("model_id") != MODEL_ID or payload.get("model_revision") != MODEL_REVISION:
            raise ValueError(
                f"artifact built on {payload.get('model_id')}@{payload.get('model_revision')}, "
                f"package pins {MODEL_ID}@{MODEL_REVISION}"
            )
        if payload.get("model_key") != MODEL_KEY:
            raise ValueError(f"artifact model_key {payload.get('model_key')!r} != {MODEL_KEY!r}")

        names = tuple(payload["class_names"])
        pipe = cls.from_pretrained(
            device=device,
            weights_dir=weights_dir,
            class_names=names,
        )
        pipe.model.load_state_dict(payload["state_dict"], strict=True)
        pipe.adapted = True
        pipe.source = f"artifact:{target.name}"
        return pipe

    def provenance(self) -> dict:
        import platform

        import torch

        return {
            "runtime": MODEL_SPEC,
            "effective": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                **self.versions,
                "device": self.device,
                "source": self.source,
                "adapted": self.adapted,
                "classes": list(self.classes),
                "num_classes": len(self.classes),
                "checkpoint_path": str(self.checkpoint),
                "checkpoint_sha256": _sha256(self.checkpoint),
            },
            "output_semantics": "Per-pixel class index in [0, num_classes - 1]. No calibrated per-pixel uncertainty is exported.",
        }

    def write_provenance(self, path: str | os.PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.provenance(), indent=2) + "\n")
        return target
