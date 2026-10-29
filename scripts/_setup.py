"""
Put the library folder src/ on sys.path, so that the numbered scripts can `import slm`.

Every script in scripts/ starts with `import _setup` (run the scripts from this folder, e.g.
`cd scripts && python3 01_ingest_corpus.py`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # <repository>/src
