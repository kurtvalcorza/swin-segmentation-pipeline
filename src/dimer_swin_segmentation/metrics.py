"""Semantic-segmentation metric helpers carried by the standalone tutorial (NOTEBOOK_SPEC 1.1 EVAL2).

`semantic_iou` is the repository's segmentation metric: per-class intersection/union aggregated over
every (prediction, reference) pair, mean IoU over the classes that occur in the sample (union > 0),
and pixel accuracy over the labelled pixels. `majority_class_baseline` scores the constant predictor
that paints every pixel with the sample's most frequent reference class — a descriptive reference
derived from the same tiny sample, not an independent benchmark. `ade20k_raw_to_indices` maps a raw
ADE20K annotation (`0` = unlabelled, `1..150` = classes) onto the model's class indices `0..149` with
`0` sent to the ignore index, the same `reduce_zero_label=True` convention the pinned MMSegmentation
config uses. Pure NumPy; no model logic lives here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

ADE20K_NUM_CLASSES = 150
IGNORE_INDEX = 255


def ade20k_raw_to_indices(raw: np.ndarray, *, num_classes: int = ADE20K_NUM_CLASSES) -> np.ndarray:
    """Raw ADE20K labels (0 = ignore, 1..num_classes) -> model indices 0..num_classes-1, ignore -> 255."""
    array = np.asarray(raw)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D label map, got shape {array.shape}")
    if array.min() < 0 or array.max() > num_classes:
        observed = f"{int(array.min())}..{int(array.max())}"
        raise ValueError(f"raw ADE20K labels must lie in 0..{num_classes}: {observed}")
    out = np.full(array.shape, IGNORE_INDEX, dtype=np.uint16)
    labelled = array > 0
    out[labelled] = array[labelled].astype(np.uint16) - 1
    return out


def _pairs(
    predictions: Sequence[np.ndarray], references: Sequence[np.ndarray], ignore_index: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions but {len(references)} references")
    if not predictions:
        raise ValueError("at least one prediction/reference pair is required")
    out = []
    for index, (prediction, reference) in enumerate(zip(predictions, references, strict=True)):
        pred = np.asarray(prediction)
        ref = np.asarray(reference)
        if pred.ndim != 2 or pred.shape != ref.shape:
            raise ValueError(
                f"pair {index}: prediction {pred.shape} and reference {ref.shape} must be equal 2-D shapes"
            )
        out.append((pred.astype(np.int64), ref.astype(np.int64), ref != ignore_index))
    return out


def semantic_iou(
    predictions: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
    *,
    num_classes: int = ADE20K_NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    """Aggregate per-class IoU, mean IoU over classes present (union > 0) and pixel accuracy.

    `predictions` and `references` are equal-length sequences of equal-shape 2-D class-index maps
    (model indices `0..num_classes-1`); reference pixels equal to `ignore_index` are excluded.
    """
    intersections = np.zeros(num_classes, dtype=np.int64)
    unions = np.zeros(num_classes, dtype=np.int64)
    correct = 0
    valid_pixels = 0
    for pred, ref, valid in _pairs(predictions, references, ignore_index):
        if valid.any() and (pred[valid].min() < 0 or pred[valid].max() >= num_classes):
            raise ValueError(f"prediction indices must lie in 0..{num_classes - 1}")
        if valid.any() and ref[valid].max() >= num_classes:
            raise ValueError(f"reference indices must lie in 0..{num_classes - 1} or equal ignore_index")
        correct += int(((pred == ref) & valid).sum())
        valid_pixels += int(valid.sum())
        pred_hist = np.bincount(pred[valid], minlength=num_classes)[:num_classes]
        ref_hist = np.bincount(ref[valid], minlength=num_classes)[:num_classes]
        both = np.bincount(pred[valid & (pred == ref)], minlength=num_classes)[:num_classes]
        intersections += both
        unions += pred_hist + ref_hist - both
    present = np.flatnonzero(unions > 0)
    per_class = [
        {
            "class_id": int(class_id),
            "intersection_pixels": int(intersections[class_id]),
            "union_pixels": int(unions[class_id]),
            "iou": float(intersections[class_id] / unions[class_id]),
        }
        for class_id in present
    ]
    return {
        "miou": float(np.mean([row["iou"] for row in per_class])) if per_class else float("nan"),
        "pixel_accuracy": float(correct / valid_pixels) if valid_pixels else float("nan"),
        "valid_pixels": valid_pixels,
        "classes_with_union": len(per_class),
        "n_images": len(predictions),
        "per_class": per_class,
    }


def majority_class_baseline(
    references: Sequence[np.ndarray],
    *,
    num_classes: int = ADE20K_NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    """Score the constant predictor painting every pixel with the sample's most frequent reference class."""
    refs = [np.asarray(reference).astype(np.int64) for reference in references]
    if not refs:
        raise ValueError("at least one reference is required")
    counts = np.zeros(num_classes, dtype=np.int64)
    for ref in refs:
        valid = ref != ignore_index
        counts += np.bincount(ref[valid], minlength=num_classes)[:num_classes]
    if not counts.any():
        raise ValueError("references contain no labelled pixels")
    majority = int(counts.argmax())
    constant = [np.full(ref.shape, majority, dtype=np.int64) for ref in refs]
    scored = semantic_iou(constant, refs, num_classes=num_classes, ignore_index=ignore_index)
    return {"majority_class_id": majority, "miou": scored["miou"], "pixel_accuracy": scored["pixel_accuracy"]}
