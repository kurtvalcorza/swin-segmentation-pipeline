"""Offline tests for the carried segmentation metric helpers (pure NumPy; no OpenMMLab stack needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dimer_swin_segmentation import (
    MODEL_ID,
    MODEL_REVISION,
    ade20k_raw_to_indices,
    evaluation_report,
    majority_class_baseline,
    semantic_iou,
)
from dimer_swin_segmentation.metrics import IGNORE_INDEX
from dimer_swin_segmentation.runtime import SegmentationResult


def test_ade20k_raw_to_indices_reduces_zero_label_and_marks_ignore() -> None:
    raw = np.array([[0, 1], [150, 7]], dtype=np.uint8)
    out = ade20k_raw_to_indices(raw)
    assert out.tolist() == [[IGNORE_INDEX, 0], [149, 6]]
    with pytest.raises(ValueError, match="0..150"):
        ade20k_raw_to_indices(np.array([[151]]))
    with pytest.raises(ValueError, match="2-D"):
        ade20k_raw_to_indices(np.zeros(4))


def test_semantic_iou_aggregates_per_class_over_pairs_and_ignores_255() -> None:
    ref_a = np.array([[0, 0], [1, IGNORE_INDEX]])
    # class 0: inter 1 / union 2; class 1: inter 1 / union 2; the ignored pixel is excluded
    pred_a = np.array([[0, 1], [1, 1]])
    ref_b = np.array([[2, 2]])
    pred_b = np.array([[2, 2]])  # class 2: inter 2 / union 2
    out = semantic_iou([pred_a, pred_b], [ref_a, ref_b], num_classes=4)
    per_class = {row["class_id"]: row for row in out["per_class"]}
    assert set(per_class) == {0, 1, 2}
    assert (per_class[0]["intersection_pixels"], per_class[0]["union_pixels"]) == (1, 2)
    assert (per_class[1]["intersection_pixels"], per_class[1]["union_pixels"]) == (1, 2)
    assert per_class[2]["iou"] == 1.0
    assert out["miou"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert out["pixel_accuracy"] == pytest.approx(4 / 5)
    assert (out["valid_pixels"], out["classes_with_union"], out["n_images"]) == (5, 3, 2)


def test_semantic_iou_rejects_shape_and_index_mismatches() -> None:
    with pytest.raises(ValueError, match="equal 2-D shapes"):
        semantic_iou([np.zeros((2, 2))], [np.zeros((2, 3))])
    with pytest.raises(ValueError, match="predictions but"):
        semantic_iou([np.zeros((2, 2))], [])
    with pytest.raises(ValueError, match="prediction indices"):
        semantic_iou([np.full((2, 2), 150)], [np.zeros((2, 2))])
    assert math.isnan(semantic_iou([np.zeros((2, 2))], [np.full((2, 2), IGNORE_INDEX)])["miou"])


def test_majority_class_baseline_scores_the_constant_predictor() -> None:
    ref = np.array([[3, 3, 3], [5, IGNORE_INDEX, 3]])
    out = majority_class_baseline([ref], num_classes=8)
    assert out["majority_class_id"] == 3
    assert out["pixel_accuracy"] == pytest.approx(4 / 5)
    assert out["miou"] == pytest.approx((4 / 5 + 0.0) / 2)  # class 3 IoU 4/5, class 5 IoU 0/1
    with pytest.raises(ValueError, match="no labelled pixels"):
        majority_class_baseline([np.full((2, 2), IGNORE_INDEX)])


def test_evaluation_report_sample_sanity_with_masks() -> None:
    mask = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    result = SegmentationResult("a.jpg", mask, (1, 2))
    reference = np.array([[1, 1], [2, IGNORE_INDEX]])
    report = evaluation_report([result], [reference], sample_kind="ADE20K-fixture")
    assert report["verdict"] == "sample-sanity"
    assert [(m["id"], m["metric"], m["value"]) for m in report["metrics"]] == [
        ("semantic_iou", "miou", 1.0),
        ("semantic_iou", "pixel_accuracy", 1.0),
    ]
    assert report["baselines"][0]["id"] == "majority_class_baseline"
    assert report["baselines"][0]["majority_class_id"] == 1
    assert report["classes_with_union"] == 2
    assert report["context"]["upstream_reported_miou"] == 44.41
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    with pytest.raises(TypeError, match="SegmentationResult"):
        evaluation_report([result.summary()], [reference])
