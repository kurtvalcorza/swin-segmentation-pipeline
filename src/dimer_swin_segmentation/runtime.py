from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

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
    """Check a local snapshot against its manifest; raise naming the first mismatch.

    The checkpoint entry must also carry the digest MODEL_SPEC has always pinned, so the manifest
    cannot silently re-point the runtime at different bytes.
    """
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
    "classes": "150 ADE20K categories in MMSegmentation order",
    "preprocessing": "MMSegmentation test pipeline of the pinned config (resize to 512-scale, normalise); nothing is altered by this module",
}


def validate_inputs(
    images: str | os.PathLike | Iterable[str | os.PathLike],
    *,
    names: Iterable[str] | None = None,
) -> dict:
    """Validation stage: return the input manifest (schema, per-image observations, verdict).

    Rejection is reported by raising exactly as ``predict`` would (``validate_image`` for each image);
    a caller that wants the finding recorded catches the exception and stores ``str(exc)`` under
    ``findings``.
    """
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
) -> dict:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``ground_truth`` (one 2-D map of model class indices ``0..149`` per result, in the same order,
    ``255`` = ignore — see ``metrics.ade20k_raw_to_indices``) the report carries ``semantic_iou`` (mean
    IoU over the classes present, pixel accuracy, per-class IoU) and the ``majority_class_baseline`` with
    the verdict ``sample-sanity``; ``ground_truth`` then requires ``SegmentationResult`` inputs (the
    masks). Without it the verdict is ``not-measurable`` (EVAL9) and the report says what labelled data
    would make the task measurable.
    """
    items = list(results)
    rows = [r.summary() if isinstance(r, SegmentationResult) else dict(r) for r in items]
    base = {
        "task": "ADE20K-150 semantic segmentation",
        "score_semantics": "argmax class per pixel; no per-pixel confidence is exposed",
        "sample_kind": sample_kind,
        "n_images": len(rows),
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
                "ADE20K-indexed ground-truth masks (model indices 0..149, 255 = ignore) for the evaluated images, "
                "scored with semantic_iou (mean IoU over classes present, pixel accuracy, per-class IoU) against the "
                "majority_class_baseline; a labelled set from the deployment domain for any generalisable claim"
            ),
        }
    if not all(isinstance(item, SegmentationResult) for item in items):
        raise TypeError("ground_truth evaluation needs SegmentationResult inputs (the predicted masks)")
    references = [np.asarray(reference) for reference in ground_truth]
    scored = semantic_iou([item.mask for item in items], references, num_classes=MODEL_SPEC["num_classes"])
    baseline = majority_class_baseline(references, num_classes=MODEL_SPEC["num_classes"])
    estimation = (
        f"single labelled sample of {scored['n_images']} image(s) / {scored['valid_pixels']} labelled pixels, "
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
        "reason": f"{scored['n_images']} labelled image(s) from the tutorial sample; not a benchmark",
        "needs": "a representative labelled holdout from the deployment domain for any generalisable mIoU claim",
    }


class DimerSwinSegmenter:
    """Public semantic-segmentation task-inference API for the pinned Swin-T UPerNet model.

    The upstream `.pth` checkpoint is code-capable PyTorch serialization. This
    runtime verifies exact size and SHA-256 before MMSegmentation deserializes
    it. Digest verification establishes byte identity, not publisher authenticity.
    """

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike = ".dimer-models",
        device: str = "cpu",
        checkpoint: str | os.PathLike | None = None,
        source: str = "openmmlab-cache",
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
    ) -> DimerSwinSegmenter:
        """Load from the fleet snapshot directory: stage absent manifest entries (only with
        ``allow_download=True``, from the pinned OpenMMLab URL), re-hash every entry against the
        manifest and MODEL_SPEC, then deserialise the checkpoint through the pinned OpenMMLab loader.
        The ``.pth`` is code-capable PyTorch serialization: digest verification fixes the bytes, not
        the author — see the class docstring.
        """
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no snapshot manifest at {root}; use DimerSwinSegmenter(cache_dir=...) for the cache path"
            )
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        return cls(device=device, checkpoint=root / WEIGHTS_FILE, source="local-snapshot")

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
                "source": self.source,
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
