"""Regenerate the submission from the provided data:  python scripts/run_all.py [--team NAME]
Needs cache/dead_signal.npy, which is rebuilt here from the tracking (deterministic, no labels)."""
import sys, os, argparse, hashlib
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, pandas as pd
from mtah import io, pipeline
ap = argparse.ArgumentParser(); ap.add_argument("--team", default="team"); a = ap.parse_args()
sig = os.path.join(ROOT, "cache", "dead_signal.npy")
if not os.path.exists(sig):
    import dead_signals, _common
    d = io.load(); pb, _ = dead_signals.dead_probability(d); np.save(sig, pb.astype(np.float32))
lab = pipeline.run()
out = os.path.join(ROOT, "submission", a.team + ".csv")
pd.DataFrame({"frame": np.arange(len(lab)), "label": lab}).to_csv(out, index=False)
print(out, hashlib.sha256(open(out, "rb").read()).hexdigest())
