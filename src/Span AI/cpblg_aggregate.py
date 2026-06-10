"""
cpblg_aggregate.py
==================
Publication-grade multi-SEED aggregation of the CP-BLG de-risk benchmark.
SYNTHETIC data -- method demonstration, not a clinical claim.

For a given emergence regime we repeat the whole pipeline (simulate -> split ->
calibrate each detector to a matched 10% false-alarm rate -> evaluate) over
several independent seeds, and report MEAN +/- STD of the denominator-fair
metrics. This guards against reporting a single lucky run -- the result must hold
on average, with an honest spread.

Usage:  python cpblg_aggregate.py indolent
        python cpblg_aggregate.py slow
        python cpblg_aggregate.py fast
"""
from __future__ import annotations
import json, os, sys, numpy as np
from paths import RESULTS_DIR
from simulate import simulate, SimConfig
from cpblg_benchmark import split, calibrate, evaluate, SCORERS, TARGET_FAR
from cpblg_regimes import REGIMES

SEEDS = [7, 11, 23, 42, 99]
N_PATIENTS = 500
OUT = os.path.join(RESULTS_DIR, "cpblg_aggregate.json")
KEYS = ["flagged_frac", "early_sens", "pre_lod_sens"]


def run_seed(regime_key, seed):
    cfg = REGIMES[regime_key]
    pts = simulate(SimConfig(n_patients=N_PATIENTS, seed=seed,
                             growth_lo=cfg["growth_lo"], growth_hi=cfg["growth_hi"]))
    cal, test = split(pts, seed=seed)
    out = {}
    for name, scorer in SCORERS:
        thr = calibrate(cal, scorer, TARGET_FAR)
        res = evaluate(test, scorer, thr)
        out[name] = {k: res[k] for k in KEYS}
    return out


def aggregate(regime_key):
    per = {name: {k: [] for k in KEYS} for name, _ in SCORERS}
    for s in SEEDS:
        r = run_seed(regime_key, s)
        for name, _ in SCORERS:
            for k in KEYS:
                per[name][k].append(r[name][k])
    summary = {}
    for name, _ in SCORERS:
        summary[name] = {k: dict(mean=float(np.mean(per[name][k])),
                                 std=float(np.std(per[name][k])))
                         for k in KEYS}
    return summary


def merge_save(regime_key, summary):
    data = {"config": dict(target_far=TARGET_FAR, n_patients=N_PATIENTS,
                           seeds=SEEDS,
                           DISCLAIMER="ALL DATA SYNTHETIC -- method demo only"),
            "regimes": {}}
    if os.path.exists(OUT):
        try:
            data = json.load(open(OUT)); data.setdefault("regimes", {})
        except Exception:
            pass
    data["regimes"][regime_key] = dict(label=REGIMES[regime_key]["label"],
                                       growth=[REGIMES[regime_key]["growth_lo"],
                                               REGIMES[regime_key]["growth_hi"]],
                                       summary=summary)
    json.dump(data, open(OUT, "w"), indent=2)


def main():
    rk = sys.argv[1] if len(sys.argv) > 1 else "slow"
    summary = aggregate(rk)
    print(f"\n=== {REGIMES[rk]['label']} (growth {REGIMES[rk]['growth_lo']}-"
          f"{REGIMES[rk]['growth_hi']}) | {len(SEEDS)} seeds, n={N_PATIENTS} ===")
    print(f"{'detector':<15}{'flagged':>14}{'early>=12wk':>15}{'pre-LoD':>14}")
    for name, _ in SCORERS:
        s = summary[name]
        print(f"{name:<15}"
              f"{s['flagged_frac']['mean']*100:>9.0f}+-{s['flagged_frac']['std']*100:<3.0f}"
              f"{s['early_sens']['mean']*100:>10.0f}+-{s['early_sens']['std']*100:<3.0f}"
              f"{s['pre_lod_sens']['mean']*100:>9.0f}+-{s['pre_lod_sens']['std']*100:<3.0f}")
    merge_save(rk, summary)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
