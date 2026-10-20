"""Sentinel: a read-only, alert-only organism that lives on a machine, learns what is normal, and tells you when it is not."""

import os as _os
import sys as _sys

# The Sentinel reuses the crash-safe store of the living organism in lkm/ (repository root). Make it importable when run from this folder.
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _os.path.isdir(_os.path.join(_REPO, "lkm")) and _REPO not in _sys.path:
    _sys.path.append(_REPO)
