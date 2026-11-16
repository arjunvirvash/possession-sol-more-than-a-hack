import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from mtah import pipeline
name = sys.argv[1]; params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
lab = pipeline.run(params)
out = os.path.join(os.path.dirname(__file__), "..", "evaluation", "results", name + ".csv")
pd.DataFrame({"frame": np.arange(len(lab)), "label": lab}).to_csv(out, index=False)
g = np.r_[lab[11185:90652], lab[110808:190198]]
print(name, {k: round((g == k).mean(), 3) for k in "HAD"})
