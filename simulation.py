"""Discrete-event simulation of the CUHK Cafe (Tolo) ordering counter.

Times are in minutes after 11:45. One replication = one weekday lunch peak
(window [0, 105) min) starting empty; arrivals stop at the end of the window
and the queue is served until it empties, so no customer is left unfinished.

Common random numbers: `generate_stream` draws every random input of a
replication (arrival times, order type, service time, patience, balk and kiosk
uniforms) BEFORE any policy is run. All policies in one replication process the
identical stream, so policy differences are paired.
"""
from heapq import heappop, heappush
from itertools import count
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_inputs(path=None):
    with open(path or HERE / "inputs.json") as f:
        return json.load(f)


# ----------------------------------------------------------------- policies
def fcfs(c):
    return {"name": f"FCFS c={c}", "servers": [(0, None, "any")] * c, "disc": "fcfs"}


def policies(window=105):
    """Candidate policies. Server tuple = (start_min, end_min or None, kind).
    kind 'any' serves every order; 'small' (express) serves small orders only.
    end None = stays on until the queue is empty."""
    peak = (30, 75)  # 12:15-13:00
    return [
        fcfs(1), fcfs(2), fcfs(3), fcfs(4),
        {"name": "Schedule 2-3-2", "disc": "fcfs",
         "servers": [(0, None, "any")] * 2 + [(peak[0], peak[1], "any")]},
        {"name": "Express 1+1", "disc": "fcfs",
         "servers": [(0, None, "any"), (0, None, "small")]},
        {"name": "Express 2+1", "disc": "fcfs",
         "servers": [(0, None, "any")] * 2 + [(0, None, "small")]},
        {"name": "Priority small c=2", "disc": "priority",
         "servers": [(0, None, "any")] * 2},
        {"name": "Sched 2 + peak express", "disc": "fcfs",
         "servers": [(0, None, "any")] * 2 + [(peak[0], peak[1], "small")]},
    ]


BASELINE = "FCFS c=2"


# ------------------------------------------------------------------ inputs
def generate_stream(inp, rng, rate_scale=1.0, service_scale=1.0):
    """All random inputs of one replication (independent of the policy)."""
    L = inp["block_len_min"]
    times = []
    for b, lam in enumerate(inp["arrival_rate_per_min"]):
        n = rng.poisson(lam * rate_scale * L)  # NHPP: Poisson count + uniform times per block
        times.append(b * L + rng.uniform(0, L, n))
    t = np.sort(np.concatenate(times))
    n = len(t)
    small = rng.random(n) < inp["p_small"]
    svc = np.empty(n)
    for kind, mask in (("small", small), ("large", ~small)):
        p = inp["service"][kind]
        svc[mask] = rng.gamma(p["shape"], p["mean_min"] * service_scale / p["shape"], mask.sum())
    patience = rng.exponential(inp["patience_mean_min"], n)
    u_balk = rng.random(n)
    u_kiosk = rng.random(n)
    return {"t": t, "small": small, "svc": svc, "patience": patience,
            "u_balk": u_balk, "u_kiosk": u_kiosk}


# ------------------------------------------------------------- simulation
def simulate(stream, policy, inp):
    """Run one policy on one stream. Returns per-replication measures."""
    W = inp["window_min"]
    t_arr, small, svc = stream["t"], stream["small"], stream["svc"]
    n = len(t_arr)
    kiosk = small & (stream["u_kiosk"] < inp["kiosk_share_of_small"])
    servers = policy["servers"]
    ns = len(servers)
    active = [False] * ns
    busy = [False] * ns
    busy_time = [0.0] * ns
    status = np.zeros(n, dtype=int)  # 0 kiosk/none,1 served,2 abandoned,3 balked
    wait = np.zeros(n)
    waiting = []  # customer ids in arrival order
    seq = count()
    fel = []
    for j, (s, e, _) in enumerate(servers):
        heappush(fel, (s, next(seq), "start", j))
        if e is not None:
            heappush(fel, (e, next(seq), "end", j))
    for i in range(n):
        if not kiosk[i]:
            heappush(fel, (t_arr[i], next(seq), "arr", i))

    def dispatch(t):
        for j in range(ns):
            if not waiting:
                return
            if active[j] and not busy[j]:
                kind = servers[j][2]
                pick = None
                if kind == "small":
                    for k, i in enumerate(waiting):
                        if small[i]:
                            pick = k
                            break
                elif policy["disc"] == "priority":
                    for k, i in enumerate(waiting):
                        if small[i]:
                            pick = k
                            break
                    if pick is None:
                        pick = 0
                else:
                    pick = 0
                if pick is None:
                    continue
                i = waiting.pop(pick)
                wait[i] = t - t_arr[i]
                status[i] = 1
                busy[j] = True
                d = svc[i]
                lo = max(t, servers[j][0])
                hi = min(t + d, servers[j][1] if servers[j][1] is not None else W)
                hi = min(hi, W)
                busy_time[j] += max(0.0, hi - lo)
                heappush(fel, (t + d, next(seq), "dep", j))

    while fel:
        t, _, ev, x = heappop(fel)
        if ev == "arr":
            i = x
            if len(waiting) >= inp["balk_queue_threshold"] and stream["u_balk"][i] < inp["balk_prob"]:
                status[i] = 3
            else:
                waiting.append(i)
                heappush(fel, (t + stream["patience"][i], next(seq), "ab", i))
                dispatch(t)
        elif ev == "ab":
            if x in waiting:
                waiting.remove(x)
                status[x] = 2
                wait[x] = stream["patience"][x]
        elif ev == "dep":
            busy[x] = False
            dispatch(t)
        elif ev == "start":
            active[x] = True
            dispatch(t)
        elif ev == "end":
            active[x] = False

    staffed = sum((min(e, W) if e is not None else W) - s for s, e, _ in servers)
    cashier_min_total = staffed
    cust = ~kiosk
    n_c = int(cust.sum())
    served = status == 1
    bad = (status == 2) | (status == 3) | (served & (wait > inp["wait_threshold_min"]))
    return {
        "n": int(n),
        "p_bad": float(bad.sum() / n) if n else 0.0,  # kiosk users count as good
        "p_abandon": float(((status == 2) | (status == 3)).sum() / n) if n else 0.0,
        "mean_wait": float(wait[cust].sum() / n_c) if n_c else 0.0,  # balk = 0 wait, abandon = time waited
        "mean_wait_served": float(wait[served].mean()) if served.any() else 0.0,
        "util": float(sum(busy_time) / staffed) if staffed else 0.0,
        "cashier_min": float(cashier_min_total),
    }
