"""
cpblg.py
========
CP-BLG : Censored-Poisson Bayesian Latent-Growth change-point detector.

================================  WHAT THIS IS  ============================
A NOVEL sequential detector for serial ctDNA, and the technical de-risk for the
Span AI thesis: that resistance can be flagged MONTHS before imaging by reading
the *trajectory*, specifically the part of it that lives BELOW the assay's limit
of detection (LoD).

THE IDEA NO ONE HAS ASSEMBLED FOR ctDNA (the IP):
    A real panel reports a per-variant call only ABOVE LoD; below it the result
    is a NON-DETECT. Under finite-molecule Poisson sampling a true low-VAF clone
    is detected only intermittently -- the calls FLICKER. Every value-based rule
    (a single snapshot threshold, or a slope on the reported VAF) treats a
    non-detect as zero and is therefore BLIND until the clone crosses LoD.

    CP-BLG instead models the DETECTION PROCESS itself. Each draw is a Bernoulli
    detect/non-detect whose probability rises as the latent subclone grows. We
    run an online change-point likelihood: is there a past draw after which the
    detection probability began to rise (H1), versus a flat background (H0)?
    Because the censoring + Poisson detection model is built into the likelihood,
    the detector accumulates evidence of growth from the PATTERN of faint and
    intermittent calls -- BEFORE the variant is ever reliably callable. It then
    fires a sequential generalised-likelihood-ratio (GLR) alarm with calibrated
    false-alarm control, aggregated across the competing resistance mechanisms.

This is a GENERATIVE detector (a proper likelihood), not a feature-fed MLP.
That is the point: the lead time it recovers comes from correctly modelling the
sub-LoD censoring, which is exactly the regime real assays operate in.

ALL PATIENT DATA IS SYNTHETIC (see simulate.py). This file demonstrates the
METHOD; it is not a clinical claim.
===========================================================================
"""
from __future__ import annotations
import numpy as np
from simulate import N_CAUSES

MECH_COLS = list(range(2, 2 + N_CAUSES))     # mechanism VAF channels in a draw
R_GRID = np.array([0.10, 0.20, 0.40, 0.80])  # candidate post-onset rise rates (per week, logit scale)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def channel_glr(t, d, p0):
    """One-sided generalised-likelihood-ratio for an UPWARD change-point in the
    Bernoulli DETECTION rate of a single variant channel.

    H0: detect prob constant = p0  (background: assay false-positive / spurious).
    H1(tau, r): for draws after onset tau, logit(p) rises linearly at rate r;
                before tau it sits at the background p0.
    Returns (glr, onset_tau, rate_r). glr is 2*log-likelihood-ratio (Wilks scale).
    Larger glr  => stronger evidence a subclone has begun growing (possibly still
    BELOW LoD, because d can flicker on while the reported VAF is a non-detect)."""
    t = np.asarray(t, float); d = np.asarray(d, float)
    n = len(d)
    if n < 3:
        return 0.0, np.nan, 0.0
    p0 = min(max(p0, 0.02), 0.5)
    base = _logit(p0)
    ll0 = np.sum(d * np.log(p0) + (1 - d) * np.log(1 - p0))
    best = 0.0; best_tau = np.nan; best_r = 0.0
    for ci in range(n - 1):                       # onset at observed draw t[ci]
        tau = t[ci]
        dt = np.maximum(0.0, t - tau)
        for r in R_GRID:
            p = _sig(base + r * dt)
            p = np.clip(p, 1e-4, 1 - 1e-4)
            ll1 = np.sum(d * np.log(p) + (1 - d) * np.log(1 - p))
            glr = 2.0 * (ll1 - ll0)
            if glr > best:
                best, best_tau, best_r = glr, tau, r
    return float(best), float(best_tau), float(best_r)


def cpblg_score_path(weeks, mech_hist, background=0.06):
    """Sequential CP-BLG alarm score at every landmark (draw index >=2).
    mech_hist: (T, N_CAUSES) reported VAF, 0 == non-detect.
    Returns array `score[j]` = max over mechanisms of the censored-detection GLR
    using draws 0..j  (competing-risks aggregation: ANY subclone growing)."""
    weeks = np.asarray(weeks, float)
    T = mech_hist.shape[0]
    scores = np.full(T, 0.0)
    det = (mech_hist > 0).astype(float)           # detect / non-detect matrix
    for j in range(2, T):
        s = 0.0
        for k in range(N_CAUSES):
            g, _, _ = channel_glr(weeks[:j + 1], det[:j + 1, k], background)
            if g > s:
                s = g
        scores[j] = s
    return scores


# --------------------------------------------------------------------------
# Value-based competitors (what a commodity report / a naive longitudinal rule
# can do), evaluated under the SAME false-alarm budget for a fair race.
# --------------------------------------------------------------------------
def snapshot_score_path(weeks, mech_hist):
    """Commodity single-test rule: the score IS the latest max reported VAF."""
    return mech_hist.max(axis=1).astype(float)


def slope_cusum_path(weeks, mech_hist, lod=0.006):
    """Strong value-based longitudinal heuristic: a one-sided CUSUM (Page) on the
    Tobit-imputed reported VAF (non-detect -> lod/2). Catches a rising VALUE, but
    cannot see growth while the value is still a flat run of non-detects."""
    weeks = np.asarray(weeks, float)
    T = mech_hist.shape[0]
    y = np.where(mech_hist > 0, mech_hist, 0.5 * lod)     # Tobit imputation
    out = np.zeros(T)
    for j in range(2, T):
        best = 0.0
        for k in range(N_CAUSES):
            v = y[:j + 1, k]
            inc = np.diff(v)
            c = 0.0; peak = 0.0
            for x in inc:
                c = max(0.0, c + x)                       # Page CUSUM, positive drift
                peak = max(peak, c)
            best = max(best, peak)
        out[j] = best
    return out
