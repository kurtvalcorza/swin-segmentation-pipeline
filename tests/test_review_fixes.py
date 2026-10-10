"""Regression tests for the 2026-10-05 notebook review findings (SWS-M1..M5, SWS-m1..m4).

They need only CI's lightweight dependencies (NumPy, Pillow, pytest): the carried data helpers run on real synthetic
data, the generated notebook is checked statically, and the notebook's own dataset cell is executed with the carried
helpers (no model). No OpenMMLab, no weights; none of this is model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import dimer_swin_segmentation as pkg
from dimer_swin_segmentation import (
    ADAPT_CLASSES,
    colour_centroid_baseline,
    generate_scene,
    read_byod_dataset,
    split_dataset,
    synthetic_segmentation_dataset,
    validate_dataset,
    validate_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"sws_fix_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[str]:
    return [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# ---- SWS-M1 / m4 -------------------------------------------------------------------------------------------------


def test_sws_M1_first_code_cell_provisions_python_310_before_any_install(notebook: dict) -> None:
    first = _code(notebook)[0]
    assert "MANAGED_PYTHON = '3.10.18'" in first and '"--managed-python"' in first
    assert first.index("isolated_version != MANAGED_PYTHON") < first.index('"pip", "install"')
    for path in (NOTEBOOK, ROOT / "README.md", ROOT / "tutorials" / "README.md"):
        text = path.read_text(encoding="utf-8")
        assert "colab-badge.svg" not in text and ("verification pending" in text or "verification%20pending" in text), path.name


def test_sws_m4_no_kernel_install_and_no_restart_instruction(notebook: dict) -> None:
    assert "Restart the runtime" not in NOTEBOOK.read_text(encoding="utf-8")
    sources = _code(notebook)
    assert not any("[sys.executable, '-m', 'pip'" in s for s in sources)
    assert sum("# dimer: kernel cell" in s for s in sources) == 2


# ---- SWS-M2: the BYOD dataset branch -----------------------------------------------------------------------------


def _byod_zip(tmp_path: Path, *, bad_class: bool = False, wrapper: str = "mydata/") -> Path:
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{wrapper}classes.txt", "background\nroad\nstructure\n")
        for i in range(4):
            image, mask = generate_scene(i, width=64, height=48)
            if bad_class and i == 2:
                mask = mask.copy()
                mask[0, 0] = 7
            for name, picture in ((f"images/s{i}.png", image), (f"masks/s{i}.png", Image.fromarray(mask.astype(np.uint8), mode="L"))):
                buffer = __import__("io").BytesIO()
                picture.save(buffer, format="PNG")
                bundle.writestr(f"{wrapper}{name}", buffer.getvalue())
        bundle.writestr("__MACOSX/._mydata", b"x")
    return archive


def _dataset_cell(notebook: dict) -> str:
    return next(s for s in _code(notebook) if "USE_BYOD_DATASET = False" in s)


def _run_dataset_cell(notebook: dict, tmp_path: Path, monkeypatch, *, path: str = "", previous=None) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    namespace = {name: getattr(pkg, name) for name in pkg.__all__}
    namespace.update({"Path": Path, "json": json, "numpy": np, "sample_dir": tmp_path / "sample", "dataset_records": previous})
    (tmp_path / "sample").mkdir(exist_ok=True)
    source = _dataset_cell(notebook).replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True").replace("BYOD_DATASET_PATH = ''", f"BYOD_DATASET_PATH = {path!r}")
    exec(source, namespace)  # noqa: S102 - the notebook's own cell
    return namespace


def test_sws_M2_byod_zip_is_read_validated_and_used(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    archive = _byod_zip(tmp_path)
    namespace = _run_dataset_cell(notebook, tmp_path, monkeypatch, path=str(archive))
    assert namespace["dataset_kind"] == "BYOD" and [r["id"] for r in namespace["dataset_records"]] == ["s0", "s1", "s2", "s3"]
    assert tuple(namespace["adapt_classes"]) == ("background", "road", "structure")
    manifest = json.loads((tmp_path / "outputs" / f"{TEMPLATE['stem']}_input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_kind"] == "BYOD" and manifest["skipped_archive_metadata"] == ["__MACOSX/._mydata"]


def test_sws_M2_flag_on_with_nothing_supplied_stops_and_never_reuses_old_records(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    stale = synthetic_segmentation_dataset(2, width=32, height=32)
    with pytest.raises(RuntimeError, match="USE_BYOD_DATASET is on but BYOD_DATASET_PATH is empty"):
        _run_dataset_cell(notebook, tmp_path, monkeypatch, previous=stale)


def test_sws_M2_mask_with_class_7_is_refused_naming_the_record(tmp_path: Path) -> None:
    from dimer_swin_segmentation import extract_byod_zip

    extract_byod_zip(_byod_zip(tmp_path, bad_class=True), tmp_path / "out")
    records, classes = read_byod_dataset(tmp_path / "out")
    with pytest.raises(ValueError, match=r"Record s2: mask contains class index 7 outside valid range 0\.\.2"):
        validate_dataset(records, classes)


def test_sws_M2_layout_refusals_name_the_file(tmp_path: Path) -> None:
    (tmp_path / "d" / "images").mkdir(parents=True)
    (tmp_path / "d" / "masks").mkdir()
    (tmp_path / "d" / "classes.txt").write_text("a\nb\n", encoding="utf-8")
    for i in range(2):
        Image.new("RGB", (40, 40)).save(tmp_path / "d" / "images" / f"x{i}.png")
    Image.new("L", (40, 40)).save(tmp_path / "d" / "masks" / "x0.png")
    with pytest.raises(ValueError, match=r"x1\.png: no mask named x1\.png in masks/"):
        read_byod_dataset(tmp_path / "d")


# ---- SWS-M3: the pretrained pipeline is never re-headed ------------------------------------------------------------


def test_sws_M3_adaptation_uses_a_fresh_copy(notebook: dict) -> None:
    code = "\n".join(_code(notebook))
    assert "rehead_model(pipe.model" not in code and "pipe.classes = tuple" not in code.replace("adapt_pipe.classes = tuple", "")
    rehead_cell = next(s for s in _code(notebook) if "rehead_model(" in s and "def rehead_model" not in s)
    assert rehead_cell.index("adapt_pipe = DimerSwinSegmenter.from_pretrained(weights_dir=WEIGHTS_DIR)") < rehead_cell.index("rehead_model(adapt_pipe.model, adapt_classes, seed=DEFAULT_ADAPT_SEED)")
    for marker in ("adapt_pipe.finetune(", "adapted_test_result = adapt_pipe.predict(test_img)", "adapt_pipe.save_artifact(", "val_predictions = [adapt_pipe.predict("):
        assert marker in code
    demo = next(s for s in _code(notebook) if "pretrained_results = [pipe.predict(image) for image in demo_images]" in s)
    assert "pipe.classes[c]" in demo
    assert "select the **Section 8** cell and choose *Runtime → Run after*" in _markdown(notebook)


# ---- SWS-M4: honest labels and a non-learned baseline -------------------------------------------------------------


def test_sws_M4_masks_match_the_drawn_pixels_exactly() -> None:
    scenes = [generate_scene(i) for i in range(24)] + [generate_scene(99)]
    for image, mask in scenes:
        rgb = np.asarray(image).astype(int)
        labels = np.full(mask.shape, 2)
        labels[(rgb == rgb[0, 0]).all(-1)] = 0
        labels[(rgb == rgb[-1, 0]).all(-1)] = 1
        assert int((labels != mask).sum()) == 0


def test_sws_M4_colour_baseline_is_reported_and_near_the_ceiling() -> None:
    train, val = split_dataset(synthetic_segmentation_dataset(24), val_fraction=0.25, seed=42)
    baseline = colour_centroid_baseline(train, val, len(ADAPT_CLASSES))
    assert baseline["id"] == "colour_nearest_centroid" and baseline["miou"] > 0.95 and baseline["fitted_on"] == "train"


def test_sws_M4_notebook_states_what_the_comparison_can_and_cannot_show(notebook: dict) -> None:
    md = _markdown(notebook)
    assert "**What this comparison can and cannot show.**" in md and "random-head reference is not a baseline" in md
    assert "Look for substantial mIoU gain" not in md and "monotonic loss descent" not in md
    assert "colour_baseline = colour_centroid_baseline(train_records, val_records, len(adapt_classes))" in "\n".join(_code(notebook))


# ---- SWS-M5: the guided layer ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    ["## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model/System → Output", "## Roadmap", "<strong>Glossary</strong>", "**Re-heading**", "**`IGNORE_INDEX`**", "**Predict before running:**", "<summary>Check your reasoning</summary>", "## 15. Activity: change one thing", "## Troubleshooting", "## Conclusion (your notes)"],
)
def test_sws_M5_guided_layer_marker_is_present(notebook: dict, marker: str) -> None:
    assert marker in _markdown(notebook)


def test_sws_M5_infrastructure_cells_are_collapsed(notebook: dict) -> None:
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    learner_start = next(i for i, c in enumerate(code) if "if sys.version_info[:2] != (3, 10):" in c["source"])
    for cell in code[:learner_start]:
        assert cell["metadata"].get("cellView") == "form", cell["source"][:60]
    assert _markdown(notebook).count("**Predict before running:**") >= 5


# ---- SWS-m2 / m3 ---------------------------------------------------------------------------------------------------


def test_sws_m2_byod_image_is_validated_and_the_upload_guarded(notebook: dict, tmp_path: Path) -> None:
    demo = next(s for s in _code(notebook) if "USE_BYOD_IMAGE = False" in s)
    assert "BYOD_IMAGE_PATH = ''" in demo and "byod_image_manifest = validate_inputs(demo_image_path)" in demo
    assert "if len(uploaded) != 1:" in demo and "next(iter(uploaded))" in demo
    text = tmp_path / "notes.png"
    text.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Input is not a readable image"):
        validate_inputs(text)


def test_sws_m3_ade20k_fixtures_are_digest_pinned_and_scored(notebook: dict) -> None:
    demo = next(s for s in _code(notebook) if "USE_ADE20K_FIXTURES = False" in s)
    import re

    assert len(re.findall(r"'ADE_val_0000000\d\.(?:jpg|png)': '[0-9a-f]{64}'", demo)) == 4 and "if observed != expected:" in demo
    assert "pretrained_report = evaluation_report(pretrained_results, pretrained_ground_truth, sample_kind=demo_kind)" in demo
    assert "the pretrained model's quality is **not measured** on the default path" in _markdown(notebook)


