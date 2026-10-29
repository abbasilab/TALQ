#!/usr/bin/env python3
"""
Pipeline test: write a synthetic stand-in for the stored model outputs.

The synthetic export has the schema of the stored outputs of the annotation job (one row per
response: speaker_id, window_start, symptom_title, replicate, response) and covers the real cohort
windows (data/processed/windows.parquet, 9,913 windows), so that steps 5-12 can be run end to end
without the stored outputs (run name `synthetic`).  Ratings come from a simple simulation: six
nonnegative latent factors with random symptom loadings, an autoregressive factor path per account
whose level depends on one of three random regimes, and Gaussian noise; confidences are drawn around
70.  About 1% of requests are dropped (no response), about 0.4% of responses are wrapped in a fenced
code block after introductory text and about 0.2% are truncated, to exercise the parser.  The data
are never used for any reported result.

Inputs
  data/processed/windows.parquet (step 3), config/symptoms.yaml
Outputs
  data/llm_outputs/_synthetic/<P.SYMPTOM_JOB>/part-<i>.parquet   (500,000 rows per part)

Usage (from the repository root):
  python3 tests/make_synthetic_llm_export.py
  cd scripts && python3 05_parse_collate.py --run synthetic      (then steps 6-12 with --run synthetic)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# --help prints this description; the step has no options
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm import params as P  # noqa: E402

# ---- step: simulation parameters (fixed seed) ----
rng = np.random.RandomState(7)                   # one generator for every random draw below
sym = yaml.safe_load(open(P.CONFIG / "symptoms.yaml"))
win = pd.read_parquet(P.PROCESSED / "windows.parquet", columns=["speaker_id", "window_index", "window_start"])
K, S = 6, len(sym)                               # latent factors, symptoms
H = rng.gamma(0.5, 1.0, size=(K, S)) * (rng.rand(K, S) < 0.35)   # sparse nonnegative loadings
H[:, :] += 0.02
regimes = rng.randint(0, 3, size=win["speaker_id"].nunique())    # regime 0, 1 or 2 per account
reg_of = dict(zip(sorted(win["speaker_id"].unique()), regimes))

# ---- step: one response per (window, symptom, sample) ----
rows = []
for s, g in win.groupby("speaker_id"):
    r = reg_of[s]
    z = rng.gamma(2, 2, size=K) * (1 + r)        # factor levels scale with the regime
    for _, w in g.sort_values("window_index").iterrows():
        z = 0.7 * z + 0.3 * rng.gamma(2, 2, size=K) * (1 + r)     # autoregressive factor path
        x = np.clip(z @ H * 4 + rng.normal(0, 5, size=S), 0, 100)  # window means on the 0-100 scale
        conf = np.clip(rng.normal(70, 12, size=S), 0, 100)
        for j, sp in enumerate(sym):
            for rep in range(5):
                if rng.rand() < 0.01:   # ~1 % of requests without a stored response
                    continue
                rating = int(np.clip(x[j] + rng.normal(0, 4), 0, 100))
                c = int(np.clip(conf[j] + rng.normal(0, 5), 0, 100))
                resp = json.dumps({"reasoning": "synthetic", "rating": rating, "confidence": c}, indent=2)
                u = rng.rand()
                if u < 0.004:
                    resp = "Here is my answer:\n```json\n" + resp + "\n```"   # embedded JSON
                elif u < 0.006:
                    resp = resp[:-10]          # truncated JSON
                rows.append({"speaker_id": s, "window_start": w["window_start"],
                             "symptom_title": sp["title"], "replicate": rep, "response": resp})

# ---- step: shuffle (random completion order) and write parquet parts ----
df = pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)  # random completion order
out = P.LLM_OUTPUTS / "_synthetic" / P.SYMPTOM_JOB
out.mkdir(parents=True, exist_ok=True)
for i, start in enumerate(range(0, len(df), 500_000)):
    df.iloc[start:start + 500_000].to_parquet(out / f"part-{i:05d}.parquet", index=False)
print(len(df), "synthetic rows written to", out)
