from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

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


class DimerSwinSegmenter:
    """Public semantic-segmentation task-inference API for the pinned Swin-T UPerNet model.

    The upstream `.pth` checkpoint is code-capable PyTorch serialization. This
    runtime verifies exact size and SHA-256 before MMSegmentation deserializes
    it. Digest verification establishes byte identity, not publisher authenticity.
    """

    def __init__(self, *, cache_dir: str | os.PathLike = ".dimer-models", device: str = "cpu"):
        versions = verify_runtime_versions()
        checkpoint = acquire_verified_checkpoint(cache_dir)
        config = resolve_packaged_config()
        from mmseg.apis import init_model

        self.model = init_model(str(config), str(checkpoint), device=device)
        self.device = device
        self.checkpoint = checkpoint
        self.config = config
        self.versions = versions
        self.classes = tuple(self.model.dataset_meta.get("classes", ()))
        if len(self.classes) != MODEL_SPEC["num_classes"]:
            raise RuntimeError(
                f"Expected {MODEL_SPEC['num_classes']} ADE20K classes, got {len(self.classes)}."
            )

    def predict(self, image: str | os.PathLike) -> SegmentationResult:
        info = validate_image(image)
        from mmseg.apis import inference_model

        sample = inference_model(self.model, info["path"])
        mask = sample.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.uint8, copy=False)
        if mask.ndim != 2:
            raise RuntimeError(f"Expected a 2-D semantic mask; got shape {mask.shape}.")
        classes_present = tuple(int(x) for x in np.unique(mask))
        return SegmentationResult(
            image_id=Path(info["path"]).name,
            mask=mask,
            classes_present=classes_present,
        )

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
                "classes": list(self.classes),
                "checkpoint_path": str(self.checkpoint),
                "checkpoint_sha256": _sha256(self.checkpoint),
            },
            "output_semantics": "Per-pixel ADE20K class index in [0, 149]. No calibrated per-pixel uncertainty is exported by this runtime.",
        }

    def write_provenance(self, path: str | os.PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.provenance(), indent=2) + "\n")
        return target
