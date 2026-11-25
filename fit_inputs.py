"""Turn on-site observations into inputs.json.

Expected files in data/ (minutes after 11:45, 1 dp; see data/*_template.csv):
  customers.csv : day, arrive, start, end, type (small|large), outcome (served|abandoned|balked)
  Abandoned rows: `start` empty and `end` = time the customer left the queue.

Usage: python fit_inputs.py [--data data] [--out inputs.json]
Fits: block arrival rates (mean over days), P(small), gamma service per type (MLE),
exponential patience with right-censoring (MLE mean = total waiting time / #abandons).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "inputs.json"))
    a = ap.parse_args()
    df = pd.read_csv(Path(a.data) / "customers.csv")
    inp = json.load(open(HERE / "inputs.json"))
    inp["balk_prob"] = 0.0  # balking is not modelled separately; abandonment covers it
    L = inp["block_len_min"]
    nb = len(inp["arrival_rate_per_min"])
    days = df["day"].nunique()
    blk = (df["arrive"] // L).clip(0, nb - 1).astype(int)
    inp["arrival_rate_per_min"] = [round(float((blk == b).sum() / days / L), 4) for b in range(nb)]
    inp["p_small"] = round(float((df["type"] == "small").mean()), 3)
    served = df[df["outcome"] == "served"].copy()
    served["svc"] = served["end"] - served["start"]
    for kind in ("small", "large"):
        x = served.loc[served["type"] == kind, "svc"].to_numpy()
        shape, _, scale = stats.gamma.fit(x, floc=0)
        inp["service"][kind] = {"mean_min": round(float(x.mean()), 3), "shape": round(float(shape), 3)}
        ks = stats.kstest(x, "gamma", args=(shape, 0, scale))
        print(f"{kind}: n={len(x)} mean={x.mean():.2f} gamma shape={shape:.2f} KS p={ks.pvalue:.3f}")
    ab = df["outcome"] == "abandoned"
    waited = np.where(ab, df["end"] - df["arrive"], df["start"] - df["arrive"])
    waited = waited[df["outcome"] != "balked"]
    n_ab = int((df["outcome"] == "abandoned").sum())
    if n_ab:
        inp["patience_mean_min"] = round(float(waited.sum() / n_ab), 3)
    inp["source"] = f"observed ({days} days, {len(df)} customers)"
    json.dump(inp, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
