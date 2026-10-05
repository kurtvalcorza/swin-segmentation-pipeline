"""Deterministic in-code sample data: ADE20K demonstration scene and the custom adaptation dataset.

Nothing here is downloaded and nothing needs external dependencies beyond Pillow and numpy,
so the tutorial's default path has zero network dataset dependencies and the same generators
are exercised by the repository's unit tests.

Two separate label vocabularies live here and must not be conflated:

* ``tutorial_scene`` returns an RGB scene for demonstrating the pretrained ADE20K 150-class model.
* ``synthetic_segmentation_dataset`` returns records labelled with ``ADAPT_CLASSES``, a three-class
  vocabulary (``background``, ``road``, ``structure``) that represents a custom target deployment domain.
  A model fine-tuned on this dataset predicts only these three classes.
"""

from __future__ import annotations

import math
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .metrics import IGNORE_INDEX, semantic_iou

ADE20K_SCENE_SIZE = (512, 384)
ADAPT_SCENE_SIZE = (512, 512)

# The custom adaptation vocabulary. Deliberately distinct from ADE20K indices.
# A model re-headed and fine-tuned on this dataset predicts these three classes.
ADAPT_CLASSES: tuple[str, ...] = ("background", "road", "structure")


def tutorial_scene(
    width: int = ADE20K_SCENE_SIZE[0], height: int = ADE20K_SCENE_SIZE[1]
) -> Image.Image:
    """The ADE20K demonstration scene: deterministic gradient background plus flat shapes.

    Matches the fixed tutorial fixture used to demonstrate pretrained inference.
    """
    ramp = np.linspace(0.0, 255.0, width)
    red = np.tile(ramp, (height, 1))
    green = np.tile(np.linspace(0.0, 255.0, height)[:, None], (1, width))
    blue = (red + green) / 2.0
    array = np.rint(np.stack([red, green, blue], axis=-1)).astype(np.uint8)
    scene = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(scene)
    draw.rectangle([40, int(height * 0.625), 220, int(height * 0.9375)], fill=(20, 20, 20))
    draw.ellipse([300, int(height * 0.156), 460, int(height * 0.573)], fill=(240, 240, 240))
    draw.polygon(
        [(260, int(height * 0.963)), (330, int(height * 0.651)), (400, int(height * 0.963))],
        fill=(30, 90, 200),
    )
    return scene


def generate_scene(
    index: int,
    *,
    width: int = ADAPT_SCENE_SIZE[0],
    height: int = ADAPT_SCENE_SIZE[1],
    seed: int = 20260915,
) -> tuple[Image.Image, np.ndarray]:
    """Generate a single synthetic image and its exact pixel-aligned ground truth mask.

    Classes:
      0: background (sky / environment)
      1: road (traversable ground plane)
      2: structure (geometric buildings / block entities)
    """
    rng = np.random.default_rng(seed + index * 101)

    # Base background (class 0)
    bg_r = int(rng.integers(120, 160))
    bg_g = int(rng.integers(170, 210))
    bg_b = int(rng.integers(210, 250))
    img = Image.new("RGB", (width, height), (bg_r, bg_g, bg_b))
    d = ImageDraw.Draw(img)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Road / ground plane (class 1)
    road_top = int(height * rng.uniform(0.55, 0.65))
    road_color = int(rng.integers(60, 90))
    d.rectangle([0, road_top, width, height], fill=(road_color, road_color, road_color))
    mask[road_top:height, :] = 1

    # Add 1 to 3 structures (class 2) seated on or above the ground plane
    n_structures = int(rng.integers(1, 4))
    for s_idx in range(n_structures):
        sw = max(4, int(rng.integers(max(4, int(width * 0.12)), max(5, int(width * 0.28)))))
        sh = max(4, int(rng.integers(max(4, int(height * 0.15)), max(5, int(height * 0.35)))))
        low_x = int(width * 0.05 + s_idx * width * 0.28)
        high_x = int(min(width - sw - 2, (s_idx + 1) * width * 0.32))
        if high_x <= low_x:
            sx = max(2, min(low_x, max(2, width - sw - 2)))
        else:
            sx = int(rng.integers(low_x, high_x))
        sx = max(2, min(sx, max(2, width - sw - 2)))
        sy = road_top - int(sh * rng.uniform(0.6, 0.9))
        sy = max(5, min(sy, road_top - 5))
        sy_bottom = min(height - 2, sy + sh)

        st_r = int(rng.integers(160, 220))
        st_g = int(rng.integers(30, 80))
        st_b = int(rng.integers(30, 80))
        d.rectangle([sx, sy, sx + sw, sy_bottom], fill=(st_r, st_g, st_b), outline=(255, 255, 255), width=2)
        # PIL rectangles include both end coordinates, so the mask does too: every drawn pixel is labelled.
        mask[sy : sy_bottom + 1, sx : sx + sw + 1] = 2

        # Optional roof triangle
        if rng.random() > 0.4:
            peak_y = max(5, sy - int(sh * 0.3))
            peak_x = sx + sw // 2
            roof_pts = [(sx - 4, sy), (peak_x, peak_y), (sx + sw + 4, sy)]
            d.polygon(roof_pts, fill=(max(0, st_r - 40), max(0, st_g - 20), max(0, st_b - 20)))
            # Compute triangle mask using polygon rasterization
            roof_img = Image.new("L", (width, height), 0)
            roof_draw = ImageDraw.Draw(roof_img)
            roof_draw.polygon(roof_pts, fill=1)
            roof_arr = np.array(roof_img, dtype=bool)
            mask[roof_arr] = 2

    return img, mask


def synthetic_segmentation_dataset(
    n_samples: int = 24,
    *,
    width: int = ADAPT_SCENE_SIZE[0],
    height: int = ADAPT_SCENE_SIZE[1],
    seed: int = 20260915,
) -> list[dict[str, Any]]:
    """Produce a deterministic multi-image segmentation adaptation dataset.

    Returns a list of dicts with keys:
    - ``id``: unique string identifier
    - ``image``: PIL.Image in RGB mode
    - ``mask``: 2D uint8 numpy array with pixel values in 0..len(ADAPT_CLASSES)-1
    """
    records = []
    for i in range(n_samples):
        img, mask = generate_scene(i, width=width, height=height, seed=seed)
        records.append({
            "id": f"scene_{i:03d}",
            "image": img,
            "mask": mask,
        })
    return records


def split_dataset(
    records: Sequence[dict[str, Any]],
    *,
    val_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically partition records into train and validation splits."""
    n = len(records)
    if n < 2:
        raise ValueError(f"Need at least 2 records to partition; got {n}.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")

    n_val = max(1, math.floor(n * val_fraction))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    val_idx = set(indices[:n_val])

    train_split = [r for i, r in enumerate(records) if i not in val_idx]
    val_split = [r for i, r in enumerate(records) if i in val_idx]
    return train_split, val_split


def validate_dataset(
    records: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    *,
    epochs: int = 1,
    max_pixels: int = 64_000_000,
) -> dict[str, Any]:
    """Validate that ``records`` adhere to the ``core.dataset.vision.raster-mask`` contract.

    Raises ValueError on schema violations, shape mismatches, or invalid class indices.
    """
    if not records:
        raise ValueError("Dataset cannot be empty; at least 1 record required.")
    if not class_names:
        raise ValueError("class_names cannot be empty.")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")

    num_classes = len(class_names)
    observed_classes: set[int] = set()
    validated_records = []

    for idx, record in enumerate(records):
        rec_id = str(record.get("id", f"sample_{idx}"))
        if "image" not in record or "mask" not in record:
            raise ValueError(f"Record {rec_id} missing 'image' or 'mask' field.")

        img_raw = record["image"]
        if isinstance(img_raw, (str, Path)):
            img_path = Path(img_raw)
            if not img_path.is_file():
                raise ValueError(f"Record {rec_id}: image file not found: {img_path}")
            with Image.open(img_path) as opened:
                img = opened.convert("RGB")
        elif isinstance(img_raw, Image.Image):
            img = img_raw.convert("RGB")
        else:
            raise TypeError(f"Record {rec_id}: unexpected image type {type(img_raw)}.")

        width, height = img.size
        if width < 1 or height < 1:
            raise ValueError(f"Record {rec_id}: dimensions must be positive; got {width}x{height}.")
        if width * height > max_pixels:
            raise ValueError(f"Record {rec_id}: {width * height} pixels exceeds ceiling {max_pixels}.")

        mask_raw = record["mask"]
        if isinstance(mask_raw, (str, Path)):
            mask_path = Path(mask_raw)
            if not mask_path.is_file():
                raise ValueError(f"Record {rec_id}: mask file not found: {mask_path}")
            with Image.open(mask_path) as opened_m:
                mask = np.array(opened_m)
        elif isinstance(mask_raw, Image.Image):
            mask = np.array(mask_raw)
        elif isinstance(mask_raw, np.ndarray):
            mask = mask_raw
        else:
            raise TypeError(f"Record {rec_id}: unexpected mask type {type(mask_raw)}.")

        if mask.ndim != 2:
            raise ValueError(f"Record {rec_id}: mask must be 2-D; got shape {mask.shape}.")
        if mask.shape != (height, width):
            raise ValueError(
                f"Record {rec_id}: mask shape {mask.shape} != image height/width ({height}, {width})."
            )

        unique_vals = np.unique(mask)
        for val in unique_vals:
            if val != IGNORE_INDEX:
                if val < 0 or val >= num_classes:
                    raise ValueError(
                        f"Record {rec_id}: mask contains class index {val} "
                        f"outside valid range 0..{num_classes - 1}."
                    )
                observed_classes.add(int(val))

        validated_records.append({
            "id": rec_id,
            "width": width,
            "height": height,
            "classes_present": [int(v) for v in unique_vals if v != IGNORE_INDEX],
        })

    return {
        "n_records": len(records),
        "class_names": list(class_names),
        "observed_classes": sorted(observed_classes),
        "schema": "core.dataset.vision.raster-mask",
        "verdict": "accepted",
    }



# ------------------------------------------------------------------------------------------------------------
# A non-learned baseline (SWS-M4) and the BYOD dataset reader (SWS-M2)
# ------------------------------------------------------------------------------------------------------------


def _rgb(image: Any) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return np.asarray(opened.convert("RGB"), dtype=np.float32)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def colour_centroid_baseline(
    train_records: Sequence[dict[str, Any]],
    eval_records: Sequence[dict[str, Any]],
    num_classes: int,
    *,
    stride: int = 4,
) -> dict[str, Any]:
    """A non-learned baseline: one mean RGB colour per class, fitted on the training masks (every
    ``stride``-th pixel), and every evaluated pixel labelled with the class of the nearest mean colour. If it
    scores as well as the adapted network, the evaluation cannot show what the learned features contribute."""
    sums = np.zeros((num_classes, 3), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for record in train_records:
        rgb = _rgb(record["image"])[::stride, ::stride].reshape(-1, 3)
        labels = np.asarray(record["mask"])[::stride, ::stride].reshape(-1).astype(np.int64)
        keep = labels != IGNORE_INDEX
        np.add.at(sums, labels[keep], rgb[keep])
        counts += np.bincount(labels[keep], minlength=num_classes)[:num_classes]
    seen = counts > 0
    if not seen.any():
        raise ValueError("colour_centroid_baseline needs labelled training pixels")
    centroids = np.where(seen[:, None], sums / np.maximum(counts, 1)[:, None], np.inf)
    predictions, references = [], []
    for record in eval_records:
        rgb = _rgb(record["image"])
        distance = ((rgb[:, :, None, :] - centroids[None, None, :, :]) ** 2).sum(axis=-1)
        predictions.append(distance.argmin(axis=-1).astype(np.int64))
        references.append(np.asarray(record["mask"]).astype(np.int64))
    scored = semantic_iou(predictions, references, num_classes=num_classes)
    return {
        "id": "colour_nearest_centroid",
        "miou": scored["miou"],
        "pixel_accuracy": scored["pixel_accuracy"],
        "per_class": scored["per_class"],
        "centroids_rgb": [
            [round(float(v), 1) for v in row] if seen[i] else None for i, row in enumerate(centroids)
        ],
        "fitted_on": "train",
    }


BYOD_MAX_RECORDS = 500
BYOD_SIDE_RANGE = (32, 4096)
BYOD_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
BYOD_CLASS_FILES = ("classes.txt", "classes.json")


def extract_byod_zip(
    archive: str | Path, destination: str | Path, *, max_expanded_bytes: int = 2 * 1024**3
) -> list[str]:
    """Extract a BYOD dataset zip member by member (no ``extractall``): absolute or ``..`` paths and symlinks
    are refused; operating-system metadata (``__MACOSX/``, ``._*``, ``.DS_Store``) is skipped and returned."""
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    skipped: list[str] = []
    expanded = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = info.filename
            member = PurePosixPath(name)
            if "\\" in name or member.is_absolute() or ".." in member.parts:
                raise ValueError(f"BYOD zip member {name!r} has an unsafe path; refusing the archive")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"BYOD zip member {name!r} is a symlink; refusing the archive")
            if "__MACOSX" in member.parts or member.name.startswith("._") or member.name == ".DS_Store":
                skipped.append(name)
                continue
            if info.is_dir():
                continue
            expanded += info.file_size
            if expanded > max_expanded_bytes:
                raise ValueError(f"BYOD zip expands beyond {max_expanded_bytes} bytes; split it")
            target = destination / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    return skipped


def read_byod_dataset(root: str | Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Read a user segmentation dataset: ``images/`` (PNG or JPEG), ``masks/`` (single-channel PNG
    class-index rasters with the same file stem as their image; ``255`` = ignore) and ``classes.txt`` (one
    class name per line, in index order) or ``classes.json`` (a list of names), at the folder root or inside
    one wrapper folder.

    Returns ``(records, class_names)`` with images and masks loaded in memory. Every refusal names the file
    and the rule; ``validate_dataset`` then checks shapes and class indices (for example a mask holding class
    7 with three classes is refused, naming the record)."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"BYOD dataset {root} is not a folder (supply a folder or a .zip)")
    if not (root / "images").is_dir():
        children = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "__MACOSX"))]
        if len(children) == 1 and (children[0] / "images").is_dir():
            root = children[0]
        else:
            raise ValueError(
                f"BYOD dataset {root} needs images/ and masks/ folders "
                "(at the top level or inside one wrapper folder)"
            )
    if not (root / "masks").is_dir():
        raise ValueError(f"BYOD dataset {root} has images/ but no masks/ folder")
    class_file = next((root / name for name in BYOD_CLASS_FILES if (root / name).is_file()), None)
    if class_file is None:
        raise ValueError(
            f"BYOD dataset {root} needs classes.txt (one class name per line) "
            "or classes.json (a list of names)"
        )
    if class_file.suffix == ".json":
        import json

        names = json.loads(class_file.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"{class_file.name}: must be a JSON list of class-name strings")
    else:
        names = [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(f"{class_file.name}: needs at least 2 distinct class names, found {names}")
    if len(names) > 254:
        raise ValueError(f"{class_file.name}: at most 254 classes (255 is the ignore index)")
    images = sorted(p for p in (root / "images").iterdir() if p.is_file() and not p.name.startswith("."))
    stray = [p.name for p in images if p.suffix.lower() not in BYOD_IMAGE_SUFFIXES]
    if stray:
        raise ValueError(f"images/ holds files that are not PNG or JPEG: {stray}")
    if not 2 <= len(images) <= BYOD_MAX_RECORDS:
        raise ValueError(
            f"images/ holds {len(images)} images; 2..{BYOD_MAX_RECORDS} are required "
            "(a train and a validation split are made)"
        )
    masks = {p.stem: p for p in (root / "masks").iterdir() if p.is_file() and not p.name.startswith(".")}
    records = []
    for path in images:
        mask_path = masks.get(path.stem)
        if mask_path is None:
            raise ValueError(f"{path.name}: no mask named {path.stem}.png in masks/")
        if mask_path.suffix.lower() != ".png":
            raise ValueError(f"masks/{mask_path.name}: masks must be PNG class-index rasters")
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            with Image.open(mask_path) as opened_mask:
                if opened_mask.mode not in ("L", "P", "I;16", "I"):
                    raise ValueError(
                        f"masks/{mask_path.name}: mode {opened_mask.mode}; "
                        "a mask must be a single-channel class-index PNG"
                    )
                mask = np.array(opened_mask).astype(np.int64)
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"{path.name}: cannot be decoded ({type(exc).__name__}: {exc})") from exc
        width, height = image.size
        if not (BYOD_SIDE_RANGE[0] <= min(width, height) and max(width, height) <= BYOD_SIDE_RANGE[1]):
            low, high = BYOD_SIDE_RANGE
            raise ValueError(f"{path.name}: {width}x{height} px; each side must be within {low}..{high} px")
        stored = mask.astype(np.uint8) if mask.max(initial=0) <= 255 else mask
        records.append({"id": path.stem, "image": image, "mask": stored})
    unused = sorted(set(masks) - {p.stem for p in images})
    if unused:
        raise ValueError(f"masks/ holds masks with no image: {unused}")
    return records, tuple(names)
