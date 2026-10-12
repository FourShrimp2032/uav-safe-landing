import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))          # the safe_landing package
sys.path.insert(0, str(HERE.parents[4] / "tools"))  # make_worlds.py (world definitions)
