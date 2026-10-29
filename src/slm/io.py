"""
Small input/output helpers shared by the numbered scripts.

* save_json / load_json read and write the result files (results/*.json for the data steps,
  results/<run>/*.json for the analyses of one annotation run).  save_json converts numpy and
  pandas values to plain JSON types.
* Timer prints when a long step starts and how long it took, so that the logs in results/logs/
  show where the time went.

Used by every script in scripts/, for example `python3 02_select_cohort.py`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def save_json(obj, path):
    """
    Write `obj` as indented JSON to `path`, creating the parent folder if needed.

    Values the json module cannot serialize are converted by `conv`: numpy integers and floats to
    int and float, numpy arrays to (nested) lists, pandas Timestamps to their string form
    ("YYYY-MM-DD HH:MM:SS"), sets to sorted lists, and anything else (numpy booleans included) to
    str(o).  Float NaN is written as NaN, which Python's json module reads back.

    Args:
        obj: JSON-like structure (dicts, lists, numbers, strings) that may hold numpy/pandas values.
        path: destination file (str or Path).
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        # called by json.dumps only for objects it cannot serialize itself
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (pd.Timestamp,)): return str(o)
        if isinstance(o, set): return sorted(o)
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=conv))


def load_json(path):
    """Read a JSON file (str or Path) and return the decoded object."""
    return json.loads(Path(path).read_text())


class Timer:
    """
    Context manager that prints when a labeled step starts and how long it took.

    Example::

        with Timer("rate functions"):
            ...

    prints "[HH:MM:SS] rate functions ..." on entry and "[HH:MM:SS] rate functions done in 12.3s" on
    exit (local clock time; wall-clock duration in seconds).
    """
    def __init__(self, label): self.label = label
    def __enter__(self):
        self.t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} ...", flush=True)
        return self
    def __exit__(self, *a):
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} done in {time.time()-self.t:.1f}s", flush=True)
