"""The isolated install uses the carried hash lock, with the pins' own package indexes (ENV10-ENV16).

torch `+cpu` lives on the PyTorch CPU index and MMCV 2.1.0 on the OpenMMLab find-links page, so a
`--require-hashes --only-binary :all:` install from PyPI alone would fail; the lock-mode install cell
passes the index lines of `tools/pins.txt` after PyPI.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_notebook as build  # noqa: E402


def _install_cell() -> str:
    nb = json.loads((ROOT / "tutorials" / "swin_segmentation_colab.ipynb").read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_lock_pins_every_direct_pin_with_hashes() -> None:
    lock_text = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    declared = build._pins(ROOT, {"pins_file": "tools/pins.txt"})
    pins = [p for p in declared if "==" in p and not p.startswith("--")]
    build.check_lock(pins, lock_text)  # raises on a missing/different pin or an unhashed entry
    assert build.lock_packages(lock_text)["mmcv"] == "2.1.0"


def test_install_cell_uses_the_lock_with_the_pins_indexes() -> None:
    cell = _install_cell()
    lock_text = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    assert f"LOCK_SHA256 = '{hashlib.sha256(lock_text.encode('utf-8')).hexdigest()}'" in cell
    install = next(line for line in cell.splitlines() if '"pip", "install"' in line)
    assert '"--require-hashes", "--only-binary", ":all:"' in install
    pins_text = (ROOT / "tools" / "pins.txt").read_text(encoding="utf-8")
    for url in re.findall(r"^--(?:extra-index-url|find-links) (\S+)", pins_text, re.M):
        assert f'"{url}"' in install
    assert '"--index-strategy", "unsafe-best-match"' in install
