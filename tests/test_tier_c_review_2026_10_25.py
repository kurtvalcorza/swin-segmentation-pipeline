"""Tier C review findings T87-1..T87-5 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "swin_segmentation_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    md = "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    return " ".join(md.split())


def test_t87_1_shortfall_is_not_called_a_tie():
    md = _markdown()
    assert "Roughly match it." not in md and "could only tie it" not in md
    assert "An adapted mIoU near 1.0" not in md and "The mIoU is close to its ceiling" not in md
    assert "Fall short." in md and "mIoU 0.914" in md


def test_t87_2_no_stale_no_hosted_run_claims():
    md = _markdown()
    for stale in ("no Colab run is recorded yet", "No hosted loss values are recorded",
                  "No metric values of a hosted run are recorded"):
        assert stale not in md, stale
    assert "7.4856 → 0.9187 → 0.6023" in md


def test_t87_3_numbers():
    md = _markdown()
    assert "28.3M" not in md and "0.387" not in md and "27,520,698" in md


def test_t87_4_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "verified by construction" in md


def test_t87_5_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] Z. Liu" in md and "`DIMER_ISOLATED_ENV`" in md
