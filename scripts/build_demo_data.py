"""
build_demo_data.py
==================
Emit the playback data behind the interactive trajectory demo on the Span AI
website (span-ai/public/data/span-demo.json).

WHY THIS EXISTS
---------------
The website must never re-implement CP-BLG in JavaScript. A second
implementation drifts from the paper silently, and the first time it disagrees
with results/cpblg_regimes.json the site is quietly publishing a number the
method does not produce. So: the real detector runs here, in Python, and the
browser only replays what it emitted.

Config is pinned to the published benchmark (cpblg_regimes.py): N=800, seed=7,
matched false-alarm rate 10%. The cohort numbers this writes therefore reproduce
results/cpblg_regimes.json exactly -- that is the point, and `--check` asserts it.

ALL DATA IS SYNTHETIC (simulate.py). This is a method demonstration, not a
clinical claim, and every consumer of this file is required to say so.

HONESTY NOTE -- read before changing the headline metric
--------------------------------------------------------
Do NOT surface `median_lead_wk` as the demo's payoff number. In the indolent
regime CP-BLG shows a *lower* median lead than the snapshot rule (17.6 wk vs
37.3 wk) purely because the snapshot rule only ever fires on the small set of
extreme, fast-rising cases -- its median is computed over an easy subset. This
is exactly the failure mode cpblg_benchmark.evaluate() documents in its own
docstring. The denominator-fair metrics are `early_sens` and `pre_lod_sens`,
which is what this script emits as the headline.

Usage:
    /usr/bin/python3 scripts/build_demo_data.py            # write the JSON
    /usr/bin/python3 scripts/build_demo_data.py --check    # verify vs published
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "Span AI"))

import cpblg as C  # noqa: E402
from cpblg_benchmark import (  # noqa: E402
    TARGET_FAR,
    calibrate,
    evaluate,
    first_alarm_week,
    patient_paths,
    split,
)
from simulate import SimConfig, _subclone_vaf, simulate  # noqa: E402

# Pinned to cpblg_regimes.py so the emitted cohort stats reproduce the paper.
REGIMES = {
    "indolent": dict(growth_lo=0.04, growth_hi=0.07, label="indolent (long sub-LoD dwell)"),
    "slow": dict(growth_lo=0.06, growth_hi=0.11, label="slow"),
    "fast": dict(growth_lo=0.10, growth_hi=0.20, label="fast (short sub-LoD dwell)"),
}
N_PATIENTS = 800
SEED = 7
LEAD_MIN_WK = 12.0

# Traces are picked at fixed rank percentiles of CP-BLG lead among test
# progressors -- NOT hand-picked. The spread deliberately reaches down to a case
# where the method barely helps, so the demo cannot be accused of cherry-picking.
# Order matters: the first entry is what the page shows on load. p90 is the
# canonical demonstration (fires pre-LoD, well ahead of imaging); the set then
# steps down THROUGH the median, where the detector fires late and helps
# nobody. Showing the median failure is the point -- only ~23% of progressions
# are caught 12+ weeks early, and the cohort view carries that base rate.
TRACE_PERCENTILES = [90, 80, 70, 50]
COHORT_SAMPLE = 60

DEFAULT_OUT = os.path.normpath(
    os.path.join(_ROOT, "..", "span-ai", "public", "data", "span-demo.json")
)
PUBLISHED = os.path.join(_ROOT, "results", "cpblg_regimes.json")
AGGREGATE = os.path.join(_ROOT, "results", "cpblg_aggregate.json")

MODELS = [("snapshot", C.snapshot_score_path), ("span", C.cpblg_score_path)]
MODEL_LABEL = {"snapshot": "snapshot", "span": "CP-BLG (ours)"}


def _r(x, n):
    """Round for transport; keep infinities JSON-safe as null."""
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, n)


def _is_progressor(p):
    return p["cause"] >= 0 and np.isfinite(p["t_imaging"])


def build_trace(p, thresholds, score_paths):
    """One patient's full playback record."""
    weeks = np.array([w for w, _ in p["draws"]], float)
    feats = np.stack([f for _, f in p["draws"]])
    mech = feats[:, C.MECH_COLS]
    reported = mech.max(axis=1)  # 0.0 == non-detect, and what `snapshot` scores

    # Latent subclone VAF. The simulator stores onset/growth/ceiling explicitly
    # "for plotting/illustration ONLY -- a real assay never observes these".
    latent = []
    if np.isfinite(p["onset"]):
        grid = np.arange(0.0, float(weeks[-1]) + 3.0, 2.0)
        latent = [
            [_r(g, 1), _r(_subclone_vaf(g, p["onset"], p["growth"], p["ceiling"]), 5)]
            for g in grid
        ]

    out = {
        "lod": _r(p["lod"], 5),
        "t_molecular": _r(p["t_molecular"], 1),
        "t_imaging": _r(p["t_imaging"], 1),
        "weeks": [_r(w, 1) for w in weeks],
        "vaf": [_r(v, 5) for v in reported],
        "detect": [int(v > 0) for v in reported],
        "latent": latent,
        "models": {},
    }
    for key, _ in MODELS:
        score = score_paths[key]
        thr = thresholds[key]
        alarm = first_alarm_week(weeks, score, thr)
        out["models"][key] = {
            "score": [_r(s, 3) for s in score],
            "threshold": _r(thr, 4),
            "alarm_wk": _r(alarm, 1),
            "lead_wk": _r(p["t_imaging"] - alarm, 1) if np.isfinite(alarm) else None,
            "pre_lod": bool(np.isfinite(alarm) and alarm < p["t_molecular"]),
        }
    return out


def run_regime(key, verbose=True):
    cfg = REGIMES[key]
    pts = simulate(
        SimConfig(
            n_patients=N_PATIENTS,
            seed=SEED,
            growth_lo=cfg["growth_lo"],
            growth_hi=cfg["growth_hi"],
        )
    )
    cal, test = split(pts, seed=SEED)

    thresholds, cohort = {}, {}
    for name, scorer in MODELS:
        thr = calibrate(cal, scorer, TARGET_FAR)
        res = evaluate(test, scorer, thr, lead_min_wk=LEAD_MIN_WK)
        thresholds[name] = thr
        cohort[name] = {
            "early_sens": _r(res["early_sens"], 4),
            "pre_lod_sens": _r(res["pre_lod_sens"], 4),
            "flagged_frac": _r(res["flagged_frac"], 4),
            "n_early": int(round(res["early_sens"] * res["n_prog"])),
            "n_prog": int(res["n_prog"]),
            # Carried for provenance/auditing only. Deliberately NOT a headline:
            # see the honesty note at the top of this file.
            "median_lead_wk_DO_NOT_HEADLINE": _r(res["median_lead_wk"], 1),
        }
        if verbose:
            print(
                f"  {MODEL_LABEL[name]:<14} thr={thr:<8.4g} "
                f"early_sens={res['early_sens']:.1%} pre_lod={res['pre_lod_sens']:.1%}"
            )

    # Per-patient score paths, computed once and reused for both trace selection
    # and the cohort sample.
    progressors = [p for p in test if _is_progressor(p)]
    per_patient = []
    for p in progressors:
        paths, ok = {}, True
        for name, scorer in MODELS:
            r = patient_paths(p, scorer)
            if r is None:
                ok = False
                break
            paths[name] = r[1]
        if not ok:
            continue  # <3 draws: excluded from evaluate()'s numerator the same way
        weeks = np.array([w for w, _ in p["draws"]], float)
        entry = {"p": p, "paths": paths, "early": {}, "lead": {}}
        for name, _ in MODELS:
            a = first_alarm_week(weeks, paths[name], thresholds[name])
            lead = p["t_imaging"] - a if np.isfinite(a) else -np.inf
            entry["lead"][name] = lead
            entry["early"][name] = bool(np.isfinite(a) and a < p["t_imaging"] and lead >= LEAD_MIN_WK)
        per_patient.append(entry)

    # Cohort sample for the counter view: first N in the deterministic test
    # order, NOT sorted by outcome.
    sample = [
        {"snapshot": int(e["early"]["snapshot"]), "span": int(e["early"]["span"])}
        for e in per_patient[:COHORT_SAMPLE]
    ]

    # Traces at fixed rank percentiles of CP-BLG lead.
    ranked = sorted(per_patient, key=lambda e: e["lead"]["span"])
    traces = []
    seen = set()
    for pct in TRACE_PERCENTILES:
        i = int(round((pct / 100.0) * (len(ranked) - 1)))
        if i in seen:
            continue
        seen.add(i)
        e = ranked[i]
        t = build_trace(e["p"], thresholds, e["paths"])
        t["percentile"] = pct
        traces.append(t)

    snap_e = cohort["snapshot"]["early_sens"] or 0.0
    span_e = cohort["span"]["early_sens"] or 0.0
    return {
        "key": key,
        "label": cfg["label"],
        "growth": [cfg["growth_lo"], cfg["growth_hi"]],
        # `aggregate` (5 seeds) is what the page QUOTES; `run` (seed 7) is what
        # the page DRAWS. Keeping both is what stops the demo contradicting
        # fig-regimes.png, which is captioned from the aggregate.
        "aggregate": load_aggregate(key),
        "cohort": {
            "n_prog": cohort["span"]["n_prog"],
            "sample_n": len(sample),
            "models": cohort,
            "early_sens_ratio": _r(span_e / snap_e, 2) if snap_e > 0 else None,
            "sample": sample,
        },
        "traces": traces,
    }


def load_aggregate(key):
    """The 5-seed aggregate is the number the site should QUOTE.

    A single seed is a single run: across seeds the slow regime swings from
    1.9x to 2.3x. fig-regimes.png (already on research.html) is captioned from
    this 5-seed aggregate, so the demo must quote the same source or the site
    contradicts its own figure. The seed-7 run is still what gets *drawn* -- it
    is one honest run, labelled as such -- but the stated result is the mean.
    """
    if not os.path.exists(AGGREGATE):
        return None
    agg = json.load(open(AGGREGATE))
    block = agg.get("regimes", {}).get(key)
    if not block:
        return None
    out = {"seeds": agg.get("config", {}).get("seeds", []), "models": {}}
    for name in ("snapshot", "span"):
        s = block["summary"][MODEL_LABEL[name]]
        out["models"][name] = {
            metric: {"mean": _r(s[metric]["mean"], 4), "std": _r(s[metric]["std"], 4)}
            for metric in ("early_sens", "pre_lod_sens", "flagged_frac")
        }
    snap = out["models"]["snapshot"]["early_sens"]["mean"]
    span = out["models"]["span"]["early_sens"]["mean"]
    out["early_sens_ratio"] = _r(span / snap, 2) if snap else None
    return out


def check_against_published(data):
    """The emitted cohort stats must equal the committed benchmark exactly."""
    if not os.path.exists(PUBLISHED):
        print(f"! {PUBLISHED} not found; skipping cross-check")
        return True
    pub = json.load(open(PUBLISHED))
    ok = True
    for key, block in data["regimes"].items():
        pubr = pub.get("regimes", {}).get(key)
        if not pubr:
            print(f"! regime {key} absent from published results; skipping")
            continue
        by_model = {r["model"]: r for r in pubr["rows"]}
        for name in ("snapshot", "span"):
            got = block["cohort"]["models"][name]
            want = by_model[MODEL_LABEL[name]]
            for metric in ("early_sens", "pre_lod_sens", "flagged_frac"):
                a, b = got[metric], round(float(want[metric]), 4)
                if abs(a - b) > 1e-4:
                    print(f"  MISMATCH {key}/{name}/{metric}: emitted {a} vs published {b}")
                    ok = False
    print("  cohort stats match results/cpblg_regimes.json" if ok else "  MISMATCHES ABOVE")
    return ok


def main():
    check_only = "--check" in sys.argv
    out_path = DEFAULT_OUT
    for a in sys.argv[1:]:
        if not a.startswith("--"):
            out_path = a

    data = {
        "meta": {
            "generator": "span-detector/scripts/build_demo_data.py",
            "detector": "CP-BLG (censored-Poisson Bayesian latent-growth change-point)",
            "cohort": "synthetic HR+/HER2- mBC on 1L CDK4/6i (simulate.py)",
            "DISCLAIMER": "ALL DATA SYNTHETIC -- method demo only",
            "target_far": TARGET_FAR,
            "n_patients": N_PATIENTS,
            "seed": SEED,
            "lead_min_wk": LEAD_MIN_WK,
            "headline_metric": "early_sens",
            "headline_metric_note": (
                "Early sensitivity = fraction of ALL progressions flagged at least "
                "12 weeks ahead at a matched 10% false-alarm rate. median_lead_wk is "
                "deliberately not a headline: it is computed only over cases a "
                "detector caught, so a detector that fires on fewer, easier patients "
                "scores misleadingly well on it."
            ),
            "quote_vs_draw": (
                "Quote regimes[k].aggregate (mean of 5 seeds) -- it is the source "
                "fig-regimes.png is captioned from. Draw regimes[k].cohort.sample, "
                "which is one run at seed 7 and must be labelled as such. Across "
                "seeds the slow regime ranges 1.9x-2.3x, so a single run must never "
                "be presented as the result."
            ),
            "trace_selection": (
                f"Fixed rank percentiles {TRACE_PERCENTILES} of CP-BLG lead among test "
                "progressors. Not hand-picked; the range deliberately includes cases "
                "where the method does not help."
            ),
        },
        "regimes": {},
    }

    for key in REGIMES:
        print(f"regime: {key}")
        data["regimes"][key] = run_regime(key)

    print("\ncross-check vs published benchmark:")
    ok = check_against_published(data)

    if check_only:
        return 0 if ok else 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size = os.path.getsize(out_path)
    print(f"\nSaved -> {out_path}  ({size/1024:.1f} KB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
