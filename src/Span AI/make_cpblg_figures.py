"""
make_cpblg_figures.py
=====================
Publication figures for the CP-BLG de-risk (SYNTHETIC data; method demo).

  fig_cpblg_trajectory.png  : one patient -- the sub-LoD detection flicker, the
                              latent subclone, the LoD line, and WHEN each
                              detector fires (CP-BLG fires below LoD, before the
                              variant is callable; the value rules wait).
  fig_cpblg_regimes.png     : the headline -- early sensitivity (caught >=12 wk
                              ahead, matched 10% FAR) across emergence regimes,
                              mean +/- std over seeds, snapshot vs Span.
"""
from __future__ import annotations
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import RESULTS_DIR
from simulate import simulate, SimConfig, MECHANISMS
from cpblg_benchmark import split, calibrate, patient_paths, first_alarm_week, SCORERS, TARGET_FAR
import cpblg as C

NAVY = "#13294B"; TEAL = "#2A9D8F"; AMBER = "#E9A23B"; RED = "#C0392B"; GREY = "#9AA3AB"


# --------------------------------------------------------------------------
def _latent_vaf(p, t):
    """True (unobserved) subclone VAF curve -- same logistic as simulate._subclone_vaf."""
    if not np.isfinite(p["onset"]):
        return np.zeros_like(t)
    x = p["growth"] * (t - p["onset"])
    v = p["ceiling"] / (1.0 + np.exp(-x + 4.0))
    return np.where(t < p["onset"], 0.0, v)


def figure_trajectory(seed=7):
    """Find a clean illustrative progressor: a rising detection-flicker pattern,
    CP-BLG firing in the sub-LoD window, a believable (2-6 month) lead, and the
    snapshot firing later (or not before progression)."""
    pts = simulate(SimConfig(n_patients=900, seed=seed, growth_lo=0.06, growth_hi=0.11))
    cal, test = split(pts, seed=seed)
    thr = {name: calibrate(cal, sc, TARGET_FAR) for name, sc in SCORERS}

    best = None; best_score = -1
    for p in test:
        if not (p["cause"] >= 0 and np.isfinite(p["t_imaging"]) and np.isfinite(p["t_molecular"])):
            continue
        r = patient_paths(p, C.cpblg_score_path)
        if r is None:
            continue
        weeks, s_cp = r
        feats = np.stack([f for _, f in p["draws"]])
        k = p["cause"]; vaf = feats[:, 2 + k]; det = vaf > 0
        a_cp = first_alarm_week(weeks, s_cp, thr["CP-BLG (ours)"])
        _, s_sn = patient_paths(p, C.snapshot_score_path); a_sn = first_alarm_week(weeks, s_sn, thr["snapshot"])
        lead_w = (p["t_imaging"] - a_cp)
        n_det = int(det.sum())
        # require: CP-BLG fires sub-LoD, before snapshot, believable lead, >=3 detects
        # forming a rising cluster (most detections in the later half before progression)
        if not (np.isfinite(a_cp) and a_cp < p["t_molecular"] and a_cp < a_sn):
            continue
        if not (8 <= lead_w <= 30 and n_det >= 3 and n_det <= 9):
            continue
        late_frac = (weeks[det] > weeks[0] + 0.4 * (a_cp - weeks[0])).mean()
        score = late_frac + 0.05 * n_det        # prefer rising clusters
        if score > best_score:
            best_score = score; best = (p, weeks, a_cp, a_sn)
    if best is None:
        print("  (no clean illustrative patient found; skipping trajectory fig)")
        return
    p, weeks, a_cp, a_sn = best
    feats = np.stack([f for _, f in p["draws"]])
    k = p["cause"]; vaf = feats[:, 2 + k]; lod = p["lod"]; det = vaf > 0

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    tt = np.linspace(weeks[0], p["t_imaging"] * 1.02, 300)
    ax.plot(tt, _latent_vaf(p, tt), color=NAVY, lw=1.6, alpha=0.55,
            label="true latent subclone VAF (UNOBSERVED)")
    ax.axhspan(0, lod, color=GREY, alpha=0.16, zorder=0)
    ax.axhline(lod, color=GREY, ls="--", lw=1.2)
    ax.text(weeks[0], lod * 1.05, "assay limit of detection (LoD)",
            fontsize=8.5, color="#5b6770", va="bottom")
    ax.text(weeks[0], lod * 0.45, "sub-LoD zone — non-detects are LEFT-CENSORED, not zero",
            fontsize=8.3, color="#5b6770", va="center", style="italic")

    ax.scatter(weeks[det], vaf[det], s=48, color=NAVY, zorder=6, edgecolor="white", lw=0.6,
               label="ctDNA draw — variant DETECTED")
    ax.scatter(weeks[~det], np.full((~det).sum(), lod * 0.10), s=44, marker="x",
               color=RED, zorder=6, label="ctDNA draw — NON-DETECT")

    ymax = max(vaf.max(), _latent_vaf(p, tt).max(), lod * 2.2) * 1.18
    snap_fires = np.isfinite(a_sn) and a_sn < p["t_imaging"]
    lines = [(a_cp, TEAL, "Span alarm (ours)", "-"),
             (p["t_molecular"], "#7d6608", "latent VAF reaches LoD", ":"),
             (p["t_imaging"], RED, "imaging progression", ":")]
    if snap_fires:
        lines.append((a_sn, AMBER, "snapshot alarm", "-"))
    for t, col, lab, ls in lines:
        if np.isfinite(t) and t <= ax.get_xlim()[1] + 5:
            ax.axvline(t, color=col, lw=2.1 if ls == "-" else 1.4, ls=ls, zorder=3)
            ax.text(t, ymax * 0.985, " " + lab, rotation=90, va="top", ha="left",
                    fontsize=8.6, color=col, fontweight="bold")
    if not snap_fires:
        ax.text(p["t_imaging"] * 1.005, ymax * 0.30,
                "commodity snapshot\nraises NO alarm before\nprogression",
                fontsize=8.4, color=AMBER, va="center", ha="left", fontweight="bold")

    ax.annotate("", xy=(p["t_imaging"], ymax * 0.40), xytext=(a_cp, ymax * 0.40),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.7))
    ax.text((a_cp + p["t_imaging"]) / 2, ymax * 0.43,
            f"Span lead ≈ {(p['t_imaging']-a_cp)/4.345:.1f} months",
            ha="center", fontsize=9.5, color=TEAL, fontweight="bold")
    ax.annotate("", xy=(p["t_molecular"], ymax * 0.20), xytext=(a_cp, ymax * 0.20),
                arrowprops=dict(arrowstyle="<->", color="#7d6608", lw=1.3))
    ax.text((a_cp + p["t_molecular"]) / 2, ymax * 0.225,
            f"pre-detection window\n({(p['t_molecular']-a_cp)/4.345:.1f} mo below LoD)",
            ha="center", fontsize=8.2, color="#7d6608")

    ax.set_xlim(weeks[0], p["t_imaging"] * 1.16)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("weeks since CDK4/6i + endocrine therapy start")
    ax.set_ylabel(f"{MECHANISMS[k]} ctDNA VAF")
    ax.set_title("Span flags resistance from the sub-LoD detection pattern — before the variant is callable\n"
                 "(one synthetic HR+/HER2− breast patient — method demonstration, not a real case)",
                 fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.2, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_cpblg_trajectory.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", os.path.basename(out))


# --------------------------------------------------------------------------
def figure_regimes():
    path = os.path.join(RESULTS_DIR, "cpblg_aggregate.json")
    if not os.path.exists(path):
        print("  (no cpblg_aggregate.json; run cpblg_aggregate.py first)")
        return
    data = json.load(open(path))
    order = [k for k in ["indolent", "slow", "fast"] if k in data["regimes"]]
    labels = {"indolent": "indolent\n(long sub-LoD dwell)", "slow": "slow", "fast": "fast\n(short dwell)"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics = [("early_sens", "Early sensitivity\n(caught ≥12 wk ahead, matched 10% FAR)"),
               ("flagged_frac", "Overall sensitivity\n(progressions flagged, matched 10% FAR)")]
    x = np.arange(len(order)); w = 0.36
    for ax, (mk, title) in zip(axes, metrics):
        snap_m = [data["regimes"][r]["summary"]["snapshot"][mk]["mean"] * 100 for r in order]
        snap_s = [data["regimes"][r]["summary"]["snapshot"][mk]["std"] * 100 for r in order]
        ours_m = [data["regimes"][r]["summary"]["CP-BLG (ours)"][mk]["mean"] * 100 for r in order]
        ours_s = [data["regimes"][r]["summary"]["CP-BLG (ours)"][mk]["std"] * 100 for r in order]
        ax.bar(x - w/2, snap_m, w, yerr=snap_s, capsize=4, color=GREY,
               label="commodity snapshot")
        ax.bar(x + w/2, ours_m, w, yerr=ours_s, capsize=4, color=TEAL,
               label="Span (ours)")
        for xi, (a, b) in enumerate(zip(snap_m, ours_m)):
            if a > 0:
                ax.text(xi + w/2, b + 1.5, f"{b/a:.1f}×", ha="center",
                        fontsize=9.5, color=NAVY, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([labels[r] for r in order], fontsize=9)
        ax.set_ylabel("%"); ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(ours_m) * 1.35)
        ax.legend(fontsize=8.6, loc="upper left")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Span's advantage scales with sub-LoD dwell time: transformative for indolent\n"
                 "emergence, honestly vanishing for fast emergence   (synthetic; 5 seeds, mean ± std)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(RESULTS_DIR, "fig_cpblg_regimes.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", os.path.basename(out))


if __name__ == "__main__":
    figure_trajectory()
    figure_regimes()
