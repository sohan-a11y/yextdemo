import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # guard_organism/
REPO = os.path.dirname(HERE)                                                 # repository root (holds lkm/)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
