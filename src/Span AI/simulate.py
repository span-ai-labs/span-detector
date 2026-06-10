"""
simulate.py
===========
SYNTHETIC serial-ctDNA TRAJECTORY GENERATOR
HR+/HER2- METASTATIC BREAST CANCER on first-line CDK4/6 inhibitor + endocrine
therapy, at the point of approaching ENDOCRINE / CDK4/6 RESISTANCE.

================================  READ THIS  ================================
EVERY number this file emits is SIMULATED. No real patient data is used or
implied. The generator exists to demonstrate that a *longitudinal* model can
recover resistance lead-time a *single-snapshot* model cannot -- i.e. to prove
the METHOD, before any real serial-ctDNA breast cohort exists (such a cohort is
not public; that scarcity is the commercial thesis).

It is calibrated to PUBLISHED, citable parameters so the synthetic world is
biologically plausible, NOT to manufacture a clinical result:

  * HR+/HER2- is the largest biomarker-defined solid-tumour population; ESR1
    mutations emerge in ~30-40% of patients on first-line aromatase-inhibitor +
    CDK4/6i and drive acquired endocrine resistance.
        (Brett et al., Breast Cancer Res 2021; Bidard et al., PADA-1, Lancet Oncol 2022)
  * Resistance arises via COMPETING mechanisms; approximate shares among
    progressors used here:
        ESR1 mutation         ~38%   (ligand-independent ER signalling)
        PIK3CA pathway         ~18%
        RB1 loss               ~10%   (loss of CDK4/6-dependence)
        HER2-activating mut    ~6%
        (remainder: unknown/other -> right-censored)
  * PADA-1 showed that acting on a RISING ESR1 ctDNA signal BEFORE radiographic
    progression (switching to fulvestrant) doubled progression-free survival --
    direct evidence that ctDNA leads imaging by a clinically actionable window.
        (Bidard et al., PADA-1, Lancet Oncol 2022)
  * ctDNA detects molecular progression months before imaging across solid
    tumours (e.g., TRACERx NSCLC ~151 days; analogous lead in HR+ breast).
        (Abbosh/Frankell et al., TRACERx, Nature 2023)

These citations describe the SHAPE of the simulation, not a claim about this
code's outputs. See the paper (paper/span_breast.pdf), "Honesty boundary".
============================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# ---- competing resistance mechanisms (the "causes" in competing-risks survival)
MECHANISMS = ["ESR1", "PIK3CA", "RB1_loss", "HER2_mut"]
N_CAUSES = len(MECHANISMS)

# approximate share of *progressing* patients per mechanism (rest -> "other")
MECH_SHARE = np.array([0.38, 0.18, 0.10, 0.06])     # sums 0.72; 0.28 -> censored

# fraction of patients who develop (modelled) resistance during follow-up
P_RESIST = 0.85


@dataclass
class SimConfig:
    n_patients: int = 2000
    draw_interval_weeks: float = 6.0      # serial ctDNA roughly every 6 wks
    draw_jitter_weeks: float = 1.5
    max_followup_weeks: float = 150.0     # ~3 yrs (1L CDK4/6i mPFS ~2 yrs)
    # imaging lag after molecular crossing: months-scale, lognormal (>0).
    imaging_lag_log_mean: float = np.log(22.0)   # ~22 wks median
    imaging_lag_log_sigma: float = 0.5
    # ---- subclone biology / assay regime --------------------------------
    # REAL acquired resistance emerges at LOW VAF, right at the assay's limit
    # of detection, buried in noise -- which is exactly why a single snapshot
    # is unreliable early. We simulate that hard regime explicitly, and report
    # sub-LoD per-variant calls as NON-DETECTS (left-censored), as real panels do.
    growth_lo: float = 0.09
    growth_hi: float = 0.18
    ceiling_lo: float = 0.03              # subclone plateaus LOW (near-LoD early)
    ceiling_hi: float = 0.16
    lod_lo: float = 0.004                 # assay limit of detection (VAF)
    lod_hi: float = 0.010
    noise_lo: float = 0.004               # per-variant assay noise (on a detection)
    noise_hi: float = 0.012
    spike_prob: float = 0.05              # sporadic spurious (false-positive) detection
    spike_max: float = 0.030
    censor_below_lod: bool = True         # report sub-LoD variant calls as NON-DETECT
    # Finite-molecule POISSON SAMPLING: at low VAF a true clone is detected only
    # stochastically, so early calls FLICKER (detect/non-detect/detect). This is
    # the real reason low-VAF ctDNA is hard -- and the regime where reading the
    # DETECTION PATTERN beats reading a single value or a value-slope.
    detect_scale: float = 0.020           # VAF at which detection prob ~ 63%
    seed: int = 7


# ---------------------------------------------------------------------------
def _patient_baseline(rng):
    """Static clinical covariates known at treatment start."""
    age = np.clip(rng.normal(61, 11), 28, 90)
    postmeno = rng.random() < 0.70
    er_high = rng.random() < 0.85                       # HR+ -> mostly ER-high
    visceral = rng.random() < 0.45                      # visceral metastases
    burden = np.clip(rng.lognormal(mean=np.log(0.05), sigma=0.6), 0.002, 0.5)
    return dict(age=age, postmeno=int(postmeno), er_high=int(er_high),
                visceral=int(visceral), baseline_tf=burden)


def _assign_fate(rng):
    if rng.random() > P_RESIST:
        return None, np.inf
    p = np.append(MECH_SHARE, 1 - MECH_SHARE.sum())
    idx = rng.choice(N_CAUSES + 1, p=p)
    if idx == N_CAUSES:
        return None, np.inf                            # 'other' -> censored
    cause = idx
    # ESR1 tends to emerge a bit earlier on AI; RB1 later. (weeks)
    scale = [52.0, 60.0, 70.0, 58.0][cause]
    onset = rng.weibull(1.6) * scale + 8.0
    return cause, onset


def _subclone_vaf(t, onset, growth, ceiling):
    if t < onset:
        return 0.0
    x = growth * (t - onset)
    return ceiling / (1.0 + np.exp(-x + 4.0))


def simulate(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    patients = []

    for _ in range(cfg.n_patients):
        base = _patient_baseline(rng)
        cause, onset = _assign_fate(rng)

        growth = rng.uniform(cfg.growth_lo, cfg.growth_hi)
        ceiling = rng.uniform(cfg.ceiling_lo, cfg.ceiling_hi)
        lod = rng.uniform(cfg.lod_lo, cfg.lod_hi)

        # realistic assay/biology noise (why a single snapshot is weak)
        noise_scale = rng.uniform(cfg.noise_lo, cfg.noise_hi)
        mech_baseline = rng.uniform(0.0, 0.006, size=N_CAUSES)

        # truncal tumour signal (e.g. PIK3CA/TP53 clonal): drops on therapy
        driver0 = base["baseline_tf"] * rng.uniform(0.5, 1.0)
        driver_nadir_wk = rng.uniform(6, 12)

        draw_weeks = []
        t = rng.uniform(0, cfg.draw_interval_weeks)
        while t < cfg.max_followup_weeks:
            draw_weeks.append(t)
            t += max(2.0, rng.normal(cfg.draw_interval_weeks, cfg.draw_jitter_weeks))

        t_molecular = np.inf
        if cause is not None:
            grid = np.arange(0, cfg.max_followup_weeks, 0.5)
            vafs = np.array([_subclone_vaf(g, onset, growth, ceiling) for g in grid])
            crossed = np.where(vafs > lod)[0]
            if len(crossed):
                t_molecular = grid[crossed[0]]

        if np.isfinite(t_molecular):
            lag = rng.lognormal(cfg.imaging_lag_log_mean, cfg.imaging_lag_log_sigma)
            t_imaging = t_molecular + lag
        else:
            t_imaging = np.inf

        t_dropout = rng.exponential(170.0)
        t_censor = min(cfg.max_followup_weeks, t_dropout)

        static = np.array([base["age"] / 100.0, base["postmeno"], base["er_high"],
                           base["visceral"], base["baseline_tf"]], dtype=np.float32)
        draws = []
        for w in draw_weeks:
            if w > t_censor + 1e-6:
                break
            if w <= driver_nadir_wk:
                driver = driver0 * np.exp(-0.25 * w)
            else:
                rise = 0.0
                if cause is not None:
                    rise = 0.4 * _subclone_vaf(w, onset, growth, ceiling)
                driver = driver0 * np.exp(-0.25 * driver_nadir_wk) + rise
            mech_vafs = np.zeros(N_CAUSES, dtype=np.float32)
            for k in range(N_CAUSES):
                f_true = 0.0
                if cause is not None and k == cause:
                    f_true = _subclone_vaf(w, onset, growth, ceiling)
                # ---- finite-molecule POISSON detection: low VAF -> flicker ----
                p_det = 1.0 - np.exp(-f_true / cfg.detect_scale) if f_true > 0 else 0.0
                if rng.random() < p_det:
                    val = max(f_true + abs(rng.normal(0, noise_scale)), lod)
                else:
                    val = 0.0                               # NON-DETECT (censored)
                # sporadic spurious (false-positive) detection -- one-off, no trend
                if rng.random() < cfg.spike_prob:
                    val = max(val, lod + rng.uniform(0, cfg.spike_max))
                mech_vafs[k] = val
            tf = max(0.0, driver + float(mech_vafs.sum()) + rng.normal(0, 0.003))
            feats = np.concatenate([[driver, tf], mech_vafs]).astype(np.float32)
            draws.append((float(w), feats))

        patients.append(dict(
            static=static, draws=draws, lod=float(lod),
            cause=(cause if cause is not None else -1),
            t_molecular=float(t_molecular), t_imaging=float(t_imaging),
            t_censor=float(t_censor),
            # latent subclone parameters (for plotting/illustration ONLY; a real
            # assay never observes these -- it only sees the censored draws)
            onset=(float(onset) if cause is not None else np.inf),
            growth=float(growth), ceiling=float(ceiling),
        ))

    return patients


DRAW_FEATURES = ["truncal_vaf", "ctdna_fraction"] + [f"vaf_{m}" for m in MECHANISMS]
STATIC_FEATURES = ["age_scaled", "postmeno", "er_high", "visceral", "baseline_tf"]


if __name__ == "__main__":
    pts = simulate(SimConfig(n_patients=500))
    n_res = sum(p["cause"] >= 0 for p in pts)
    leads = [p["t_imaging"] - p["t_molecular"]
             for p in pts if np.isfinite(p["t_imaging"])]
    print(f"HR+/HER2- breast (synthetic): {len(pts)} patients, "
          f"{n_res} progress ({n_res/len(pts):.0%})")
    print(f"median draws/patient: {np.median([len(p['draws']) for p in pts]):.0f}")
    print(f"median molecular->imaging lead: {np.median(leads):.1f} wks "
          f"(~{np.median(leads)*7:.0f} days)")
