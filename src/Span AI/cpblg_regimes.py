"""
cpblg_regimes.py
================
Official multi-regime de-risk benchmark for the NOVEL CP-BLG detector.
SYNTHETIC data (simulate.py) -- method demonstration, not a clinical claim.

Runs the CP-BLG vs commodity-snapshot vs slope-CUSUM race across SUBCLONE
EMERGENCE regimes (how fast the resistant clone grows out of the sub-LoD zone),
at a matched 10% false-alarm rate, and reports the denominator-fair EARLY
SENSITIVITY (fraction of all progressions caught >= 12 weeks ahead) plus the
pre-LoD sensitivity (caught before the variant is callable at all).

The scientific point is deliberately falsifiable: CP-BLG's advantage is large in
the INDOLENT / SLOW regime (long sub-LoD dwell -- the real hard early-resistance
setting) and SHRINKS in the FAST regime (where the value rises quickly enough
that even a single snapshot catches it). We report all regimes, including the one
where the method barely helps.

Usage:
    python cpblg_regimes.py indolent      # one regime per process (keeps it fast)
    python cpblg_regimes.py slow
    python cpblg_regimes.py fast
    python cpblg_regimes.py all            # all three in one process (slower)
"""
from __future__ import annotations
import json, os, sys, numpy as np
from paths import RESULTS_DIR
from simulate import simulate, SimConfig
from cpblg_benchmark import split, calibrate, evaluate, SCORERS, TARGET_FAR

REGIMES = {
    "indolent": dict(growth_lo=0.04, growth_hi=0.07,
                     label="indolent (long sub-LoD dwell)"),
    "slow":     dict(growth_lo=0.06, growth_hi=0.11,
                     label="slow"),
    "fast":     dict(growth_lo=0.10, growth_hi=0.20,
                     label="fast (short sub-LoD dwell)"),
}
N_PATIENTS = 800
SEED = 7
OUT = os.path.join(RESULTS_DIR, "cpblg_regimes.json")


def run_regime(key, n=N_PATIENTS, seed=SEED):
    cfg = REGIMES[key]
    pts = simulate(SimConfig(n_patients=n, seed=seed,
                             growth_lo=cfg["growth_lo"], growth_hi=cfg["growth_hi"]))
    cal, test = split(pts, seed=seed)
    rows = []
    for name, scorer in SCORERS:
        thr = calibrate(cal, scorer, TARGET_FAR)
        res = evaluate(test, scorer, thr)
        res.pop("records", None)
        res["model"] = name
        rows.append(res)
    return dict(regime=key, label=cfg["label"],
                growth=[cfg["growth_lo"], cfg["growth_hi"]], rows=rows)


def _print(block):
    print(f"\n=== regime: {block['label']}  (growth {block['growth'][0]}-{block['growth'][1]}) ===")
    print(f"{'detector':<15}{'flagged':>9}{'early>=12wk':>12}{'pre-LoD':>9}{'med-lead-d':>11}")
    for r in block["rows"]:
        print(f"{r['model']:<15}{r['flagged_frac']:>9.0%}{r['early_sens']:>12.0%}"
              f"{r['pre_lod_sens']:>9.0%}{r['median_lead_d']:>11.0f}")


def merge_save(block):
    data = {"config": dict(target_far=TARGET_FAR, n_patients=N_PATIENTS, seed=SEED,
                           DISCLAIMER="ALL DATA SYNTHETIC -- method demo only"),
            "regimes": {}}
    if os.path.exists(OUT):
        try:
            data = json.load(open(OUT))
            data.setdefault("regimes", {})
        except Exception:
            pass
    data["regimes"][block["regime"]] = block
    json.dump(data, open(OUT, "w"), indent=2)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = list(REGIMES) if which == "all" else [which]
    for k in keys:
        block = run_regime(k)
        _print(block)
        merge_save(block)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
