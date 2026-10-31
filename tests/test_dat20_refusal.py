"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

import dimer_swin_segmentation as pkg
from test_review_fixes import _byod_zip, _run_dataset_cell

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/swin_segmentation_colab.ipynb"
BYOD_CELL = "swin_segmentation-22"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


NB_DATA = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
DATASET_CELL = next(c.get("id") for c in _code_cells() if "USE_BYOD_DATASET = False" in _source(c))
EARLIER_DATASET = {"dataset_summary": "old", "input_manifest": "old", "adapt_pipe": "old", "post_adapt_report": "old", "result_payload": "old"}
EARLIER_IMAGE = {"demo_images": "old", "demo_kind": "old", "pretrained_results": "old", "pretrained_report": "old"}
DATASET_FILES = ["swin_segmentation_input_manifest.json", "swin_segmentation_evaluation_report.json", "swin_segmentation_result.json",
                 "swin-segmentation-adapter-v2.safetensors"]
IMAGE_FILES = ["swin_segmentation_pretrained_scene_semantic.png", "swin_segmentation_pretrained_evaluation.json"]


def _seed(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir(exist_ok=True)
    for name in DATASET_FILES + IMAGE_FILES:
        (out / name).write_text("{}", encoding="utf-8")


def _left(tmp_path: Path) -> list:
    return sorted(p.name for p in (tmp_path / "outputs").iterdir())


def _dataset_cases(tmp_path: Path) -> dict:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    bad = _byod_zip(tmp_path, bad_class=True).rename(tmp_path / "badclass.zip")
    return {
        "folder without images/ (the scripted /etc run)": ("etc", "BYOD dataset"),
        "corrupt .zip (zipfile.BadZipFile)": ("fake.zip", "File is not a zip file"),
        "not a folder or .zip": ("notes.txt", "supply a folder or a .zip"),
        "mask class out of range (validate_dataset)": (bad.name, "mask contains class index 7"),
        "missing path": ("absent", "does not exist in this runtime"),
    }


@pytest.mark.parametrize("case", ["folder without images/ (the scripted /etc run)", "corrupt .zip (zipfile.BadZipFile)", "not a folder or .zip",
                                  "mask class out of range (validate_dataset)", "missing path"])
def test_dataset_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    file_name, rule = _dataset_cases(tmp_path)[case]
    _seed(tmp_path)
    with pytest.raises(ValueError) as caught:
        _run_dataset_cell(NB_DATA, tmp_path, monkeypatch, path=str(tmp_path / file_name), previous=[{"stale": True}])
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "run this cell again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert _left(tmp_path) == sorted(IMAGE_FILES)  # the image section's outputs stay


def test_dataset_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    _seed(tmp_path)
    (tmp_path / "sample").mkdir()
    ns = {name: getattr(pkg, name) for name in pkg.__all__}
    ns.update(Path=Path, json=json, zipfile=zipfile, sample_dir=tmp_path / "sample", **EARLIER_DATASET, **EARLIER_IMAGE)
    exec(compile(_cell(BYOD_CELL)[_cell(BYOD_CELL).index("class BYODRefusal(ValueError):"):_cell(BYOD_CELL).index("# What an earlier run of this section left")], "<helpers>", "exec"), ns)
    source = _cell(DATASET_CELL).replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True", 1).replace("BYOD_DATASET_PATH = ''", f"BYOD_DATASET_PATH = {str(tmp_path / 'fake.zip')!r}", 1)
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    with pytest.raises(ValueError):
        exec(compile(source, "<dataset>", "exec"), ns)
    assert not set(EARLIER_DATASET) & set(ns) and "dataset_records" not in ns
    assert set(EARLIER_IMAGE) <= set(ns)


def test_dataset_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run_dataset_cell(NB_DATA, tmp_path, monkeypatch)
    assert _refusal_line(caught.value).startswith("Refused: (no file): USE_BYOD_DATASET is on but BYOD_DATASET_PATH is empty")


def test_dataset_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    archive = _byod_zip(tmp_path)
    _seed(tmp_path)
    monkeypatch.setattr(pkg, "read_byod_dataset", lambda _root: (_ for _ in ()).throw(RuntimeError("internal bug")))
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run_dataset_cell(NB_DATA, tmp_path, monkeypatch, path=str(archive))
    assert not getattr(caught.value, "dimer_refusal", False)
    assert _left(tmp_path) == sorted(DATASET_FILES + IMAGE_FILES)  # nothing removed


def test_good_byod_zip_still_loads(monkeypatch, tmp_path):
    ns = _run_dataset_cell(NB_DATA, tmp_path, monkeypatch, path=str(_byod_zip(tmp_path)))
    assert ns["dataset_kind"] == "BYOD" and len(ns["dataset_records"]) == 4


@pytest.mark.skipif(not Path("/etc").is_dir() or Path("/etc/images").exists(), reason="needs a POSIX /etc, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD_DATASET=True;BYOD_DATASET_PATH='/etc': a folder present in every fresh runtime."""
    with pytest.raises(ValueError) as caught:
        _run_dataset_cell(NB_DATA, tmp_path, monkeypatch, path="/etc")
    assert _refusal_line(caught.value).startswith("Refused: etc: BYOD dataset /etc")


# --- image section --------------------------------------------------------------------------------------------------


def _image(monkeypatch, tmp_path, path: str, *, fixtures: bool = False, ns_extra: dict | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    ns = {name: getattr(pkg, name) for name in pkg.__all__}
    ns.update(Path=Path, json=json, Image=Image, numpy=__import__("numpy"), **EARLIER_IMAGE, **(ns_extra or {}))
    source = _cell(BYOD_CELL).replace("USE_BYOD_IMAGE = False", "USE_BYOD_IMAGE = True", 1).replace("BYOD_IMAGE_PATH = ''", f"BYOD_IMAGE_PATH = {path!r}", 1)
    if fixtures:
        source = source.replace("USE_ADE20K_FIXTURES = False", "USE_ADE20K_FIXTURES = True", 1)
    exec(compile(source, "<image>", "exec"), ns)
    return ns


@pytest.mark.parametrize(("make", "file_name", "rule"), [
    (lambda p: (p / "hostname").write_text("runtime\n", encoding="utf-8"), "hostname", "Input is not a readable image"),
    (lambda p: None, "nope.png", "Image does not exist"),
])
def test_image_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, make, file_name, rule):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    make(tmp_path)
    ns = {}
    with pytest.raises(ValueError) as caught:
        ns = _image(monkeypatch, tmp_path, str(tmp_path / file_name))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert not ns
    assert _left(tmp_path) == sorted(DATASET_FILES)  # the dataset section's outputs stay


def test_image_switch_conflict_and_missing_upload_dialog_are_refusals(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _image(monkeypatch, tmp_path, "", fixtures=True)
    assert _refusal_line(caught.value).startswith("Refused: (no file): Enable at most one of USE_BYOD_IMAGE and USE_ADE20K_FIXTURES")
    with pytest.raises(ValueError) as caught:
        _image(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith("Refused: (no file): There is no upload dialog outside Google Colab")


@pytest.mark.parametrize(("uploaded", "file_name"), [({}, "(no file)"), ({"a.png": b"1", "b.png": b"2"}, "a.png, b.png")])
def test_cancelled_or_multiple_image_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    (tmp_path / "sample").mkdir()
    with pytest.raises(ValueError) as caught:
        _image(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith(f"Refused: {file_name}: Upload exactly one image file")


def test_image_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    Image.new("RGB", (64, 48), (10, 20, 30)).save(tmp_path / "ok.png")

    def broken(*_args, **_kwargs):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _image(monkeypatch, tmp_path, str(tmp_path / "ok.png"), ns_extra={"validate_inputs": broken})
    assert not getattr(caught.value, "dimer_refusal", False)
    assert _left(tmp_path) == sorted(DATASET_FILES + IMAGE_FILES)  # nothing removed
