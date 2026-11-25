"""Run validation, main experiment, and sensitivity analysis. Writes results/*.json and figures.

Usage: python run_experiments.py [--reps 300]
Seeds: SeedSequence(3410) spawns one child per replication; every policy in a
replication shares that child's stream (common random numbers).
"""
import argparse
import copy
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl3410"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
from scipy import stats

from simulation import BASELINE, generate_stream, load_inputs, policies, simulate

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
MEASURES = ["p_bad", "p_abandon", "mean_wait", "util", "cashier_min"]


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return float(x.mean()), float(h)


def run(inp, pols, reps, seed=3410, **kw):
    """Return {policy: {measure: array(reps)}} with CRN across policies."""
    res = {p["name"]: {m: [] for m in MEASURES} for p in pols}
    for ss in SeedSequence(seed).spawn(reps):
        stream = generate_stream(inp, default_rng(ss), **kw)
        for p in pols:
            r = simulate(stream, p, inp)
            for m in MEASURES:
                res[p["name"]][m].append(r[m])
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in res.items()}


def summarize(res):
    out = {}
    for name, d in res.items():
        out[name] = {m: ci(d[m]) for m in MEASURES}
        if name != BASELINE:
            out[name]["diff_p_bad"] = ci(d["p_bad"] - res[BASELINE]["p_bad"])
            out[name]["diff_wait"] = ci(d["mean_wait"] - res[BASELINE]["mean_wait"])
    return out


def validate():
    """M/M/c, no abandonment/balk, one long stationary run vs Erlang C."""
    inp = {"block_len_min": 100000.0, "arrival_rate_per_min": [1.5], "p_small": 1.0,
           "service": {"small": {"mean_min": 1.0, "shape": 1.0}, "large": {"mean_min": 1.0, "shape": 1.0}},
           "patience_mean_min": 1e12, "balk_queue_threshold": 10**9, "balk_prob": 0.0,
           "window_min": 100000.0, "wait_threshold_min": 5.0, "kiosk_share_of_small": 0.0}
    c, lam, mu = 2, 1.5, 1.0
    a = lam / mu
    rho = a / c
    erl = (a**c / math.factorial(c)) / (1 - rho)
    p0 = 1 / (sum(a**k / math.factorial(k) for k in range(c)) + erl)
    C = erl * p0
    Wq_th = C / (c * mu - lam)
    sims = []
    for ss in SeedSequence(99).spawn(5):
        s = generate_stream(inp, default_rng(ss))
        sims.append(simulate(s, {"name": "v", "servers": [(0, None, "any")] * c, "disc": "fcfs"}, inp)["mean_wait"])
    m, h = ci(sims)
    return {"theory_Wq_min": Wq_th, "sim_Wq_min": m, "sim_halfwidth": h, "C_c_a": C}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()
    inp = load_inputs()
    pols = policies()
    R = args.reps
    report = {"reps": R, "validation": validate()}
    print("validation", report["validation"])

    main_res = run(inp, pols, R)
    report["main"] = summarize(main_res)
    for k, v in report["main"].items():
        print(f"{k:24s} p_bad={v['p_bad'][0]:.3f}±{v['p_bad'][1]:.3f} wait={v['mean_wait'][0]:.2f} util={v['util'][0]:.2f} cm={v['cashier_min'][0]:.0f}")

    # ---- sensitivity: one-at-a-time, compare all policies
    scen = {
        "Base": {},
        "Arrivals +20%": {"rate_scale": 1.2},
        "Arrivals -20%": {"rate_scale": 0.8},
        "Service +20%": {"service_scale": 1.2},
        "Service -20%": {"service_scale": 0.8},
        "Patience mean 3 min": {"inp": {"patience_mean_min": 3.0}},
        "Patience mean 10 min": {"inp": {"patience_mean_min": 10.0}},
        "Small share 40%": {"inp": {"p_small": 0.4}},
        "Small share 80%": {"inp": {"p_small": 0.8}},
    }
    sens = {}
    for sname, s in scen.items():
        i2 = copy.deepcopy(inp)
        i2.update(s.get("inp", {}))
        kw = {k: v for k, v in s.items() if k != "inp"}
        r = run(i2, pols, max(R // 2, 100), seed=777, **kw)
        sm = summarize(r)
        sens[sname] = {k: {"p_bad": v["p_bad"], "cashier_min": v["cashier_min"][0]} for k, v in sm.items()}
        ok = {k: v["p_bad"][0] for k, v in sm.items()}
        feasible = {k: sm[k]["cashier_min"][0] for k in ok if ok[k] <= inp["target_bad_fraction"]}
        sens[sname]["_cheapest_meeting_target"] = min(feasible, key=feasible.get) if feasible else None
        sens[sname]["_best_p_bad"] = min(ok, key=ok.get)
        print(sname, "cheapest meeting target:", sens[sname]["_cheapest_meeting_target"], "| lowest p_bad:", sens[sname]["_best_p_bad"])
    report["sensitivity"] = sens

    with open(OUT / "results.json", "w") as f:
        json.dump(report, f, indent=1)

    # ---- figures
    names = [p["name"] for p in pols]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = [report["main"][n]["cashier_min"][0] for n in names]
    y = [report["main"][n]["p_bad"][0] for n in names]
    e = [report["main"][n]["p_bad"][1] for n in names]
    ax.errorbar(x, y, yerr=e, fmt="o", capsize=3)
    for n, xi, yi in zip(names, x, y):
        ax.annotate(n, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(inp["target_bad_fraction"], ls="--", c="r", lw=0.8)
    ax.set_xlabel("Staffed cashier-minutes per lunch peak")
    ax.set_ylabel("P(wait > 5 min or lost)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tradeoff.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    L = inp["block_len_min"]
    ax.bar(range(len(inp["arrival_rate_per_min"])), [l * 60 for l in inp["arrival_rate_per_min"]], width=0.8)
    ax.set_xticks(range(len(inp["block_labels"])))
    ax.set_xticklabels(inp["block_labels"])
    ax.set_ylabel("Arrival rate (per hour)")
    ax.set_xlabel("15-min block start")
    fig.tight_layout()
    fig.savefig(OUT / "fig_arrivals.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    show = ["FCFS c=2", "FCFS c=3", "Schedule 2-3-2", "Express 2+1", "Priority small c=2"]
    w = 0.16
    for k, n in enumerate(show):
        ax.bar(np.arange(len(sens)) + k * w, [sens[s][n]["p_bad"][0] for s in sens], w, label=n)
    ax.axhline(inp["target_bad_fraction"], ls="--", c="r", lw=0.8)
    ax.set_xticks(np.arange(len(sens)) + 2 * w)
    ax.set_xticklabels(list(sens), rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("P(wait > 5 min or lost)")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sens.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
