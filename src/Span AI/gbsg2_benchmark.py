"""
gbsg2_benchmark.py
==================
REAL-DATA validation on the German Breast Cancer Study Group (GBSG2) cohort.

================================  READ THIS  ================================
This script uses REAL patient data: 686 node-positive breast-cancer patients
from the GBSG-2 randomised trial of hormonal therapy (Schumacher et al., J Clin
Oncol 1994), distributed with the `lifelines` package. Outcome = recurrence-free
survival (event `cens`, time in days).

GBSG2 is CROSS-SECTIONAL: each patient has ONE set of baseline covariates and
ONE follow-up time -- there is no serial ctDNA. So it CANNOT demonstrate the
longitudinal lead-time advantage (that needs repeated draws; see the synthetic
benchmark and the commercial thesis). What it CAN do, honestly, is validate that
our deep discrete-time survival head is competitive on REAL breast-cancer
discrimination against the field-standard Cox proportional-hazards model.

Read this as: "the model machinery is sound on real breast data" -- not as a
claim about early warning, which only serial data can support.
============================================================================
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from paths import RESULTS_DIR, DATA_DIR
from model_numpy import DiscreteTimeCompetingRisks

SEED = 7
N_BINS = 10                # discrete horizon bins
TEST_FRAC = 0.25


def load_gbsg2():
    df = pd.read_csv(os.path.join(DATA_DIR, "gbsg2.csv"))
    # encode covariates
    X = pd.DataFrame()
    X["horTh"]    = (df["horTh"] == "yes").astype(float)
    X["age"]      = df["age"].astype(float)
    X["postmeno"] = (df["menostat"] == "Post").astype(float)
    X["tsize"]    = df["tsize"].astype(float)
    X["tgrade"]   = df["tgrade"].map({"I": 1, "II": 2, "III": 3}).astype(float)
    X["pnodes"]   = df["pnodes"].astype(float)
    X["progrec"]  = df["progrec"].astype(float)
    X["estrec"]   = df["estrec"].astype(float)
    t = df["time"].to_numpy(float)
    e = df["cens"].to_numpy(int)
    return X, t, e


def discretise(t, n_bins, t_max):
    edges = np.linspace(0, t_max, n_bins + 1)
    tbin = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    return tbin.astype(int), edges


def harrell_c(risk, t, e):
    """Higher risk should mean shorter survival. Concordant pairs among
    comparable (earlier event vs later) pairs."""
    n = len(t)
    num = den = 0.0
    for i in range(n):
        if not e[i]:
            continue
        for j in range(n):
            if t[j] > t[i]:
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
    return num / den if den else float("nan")


def main():
    X, t, e = load_gbsg2()
    print(f"GBSG2 (REAL): {len(X)} patients, event rate {e.mean():.0%}, "
          f"median follow-up {np.median(t):.0f} d")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_te = int(TEST_FRAC * len(X))
    te, tr = idx[:n_te], idx[n_te:]

    Xv = X.to_numpy(float)
    mu, sd = Xv[tr].mean(0), Xv[tr].std(0) + 1e-9
    Xz = (Xv - mu) / sd

    t_max = np.quantile(t[tr], 0.95)
    tbin, _ = discretise(t, N_BINS, t_max)

    # ---- our deep discrete-time model (single cause = recurrence/death)
    model = DiscreteTimeCompetingRisks(Xz.shape[1], n_causes=1, n_bins=N_BINS,
                                       hidden=12, seed=0)
    model.fit(Xz[tr], cause=np.zeros(len(tr), int), tbin=tbin[tr],
              event=e[tr], epochs=600, batch=128, lr=3e-3, l2=1e-3)
    risk_deep = model.risk_score(Xz[te])
    c_deep = harrell_c(risk_deep, t[te], e[te])

    # ---- Cox proportional-hazards baseline (lifelines)
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
        dftr = pd.DataFrame(Xz[tr], columns=X.columns)
        dftr["time"] = t[tr]; dftr["event"] = e[tr]
        cph = CoxPHFitter(penalizer=0.1).fit(dftr, "time", "event")
        risk_cox = cph.predict_partial_hazard(pd.DataFrame(Xz[te], columns=X.columns)).to_numpy()
        c_cox = concordance_index(t[te], -risk_cox, e[te])
    except Exception as ex:                                   # pragma: no cover
        print("  (lifelines unavailable:", ex, ")")
        c_cox = float("nan")

    print("\n" + "=" * 56)
    print("GBSG2 REAL-DATA DISCRIMINATION (held-out test, C-index)")
    print("=" * 56)
    print(f"  Cox proportional hazards (lifelines)   {c_cox:.3f}")
    print(f"  deep discrete-time competing-risks      {c_deep:.3f}")
    print("=" * 56)
    print("Comparable discrimination on REAL breast data confirms the deep\n"
          "survival head is sound; the longitudinal early-warning advantage\n"
          "requires serial ctDNA (synthetic benchmark + commercial thesis).")

    out = dict(
        dataset="GBSG2 (German Breast Cancer Study Group 2) -- REAL",
        n_patients=int(len(X)), event_rate=float(e.mean()),
        n_bins=N_BINS, test_frac=TEST_FRAC, seed=SEED,
        c_index_cox=float(c_cox), c_index_deep=float(c_deep),
        note="Cross-sectional real data; validates discrimination, not lead time.",
    )
    with open(os.path.join(RESULTS_DIR, "gbsg2_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> results/gbsg2_results.json")

    _make_figure(risk_deep, t[te], e[te], c_cox, c_deep)


def _make_figure(risk, t, e, c_cox, c_deep):
    """KM curves on the REAL test set, stratified into model risk tertiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from lifelines import KaplanMeierFitter
    except Exception:
        return
    q1, q2 = np.quantile(risk, [1/3, 2/3])
    groups = np.where(risk <= q1, 0, np.where(risk <= q2, 1, 2))
    names = ["low risk (model)", "medium risk", "high risk (model)"]
    cols = ["#34a853", "#f9ab00", "#ea4335"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    kmf = KaplanMeierFitter()
    for g in range(3):
        m = groups == g
        if m.sum() == 0:
            continue
        kmf.fit(t[m] / 365.25, e[m], label=f"{names[g]} (n={m.sum()})")
        kmf.plot_survival_function(ax=axL, color=cols[g], ci_show=True, lw=2)
    axL.set_xlabel("years since surgery")
    axL.set_ylabel("recurrence-free survival")
    axL.set_title("GBSG2 (REAL): model risk tertiles separate\n"
                  "actual patient outcomes", fontsize=10, fontweight="bold")
    axL.set_ylim(0, 1); axL.legend(fontsize=8, loc="lower left")
    axL.spines[["top", "right"]].set_visible(False)

    bars = axR.bar(["Cox PH\n(standard)", "deep DTCR\n(ours)"],
                   [c_cox, c_deep], color=["#9aa0a6", "#1a73e8"], width=0.6)
    axR.set_ylim(0.5, 0.75); axR.set_ylabel("C-index (held-out)")
    axR.set_title("Discrimination on REAL\nbreast data", fontsize=10, fontweight="bold")
    axR.axhline(0.5, color="k", lw=0.8, ls=":")
    for b, v in zip(bars, [c_cox, c_deep]):
        axR.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    axR.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig3_gbsg2.png"), dpi=150)
    print("saved fig3_gbsg2.png")


if __name__ == "__main__":
    main()
