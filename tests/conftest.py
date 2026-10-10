"""Shared pytest configuration.

`test_adaptation.py` imports torch at module level (CI installs the CPU wheel). Without torch, as in a
lightweight offline check, the module is skipped rather than reported as a collection error.
"""

import importlib.util

collect_ignore = [] if importlib.util.find_spec("torch") else ["test_adaptation.py"]
