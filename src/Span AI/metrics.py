"""
metrics.py
==========
Evaluation for the dynamic competing-risks benchmark.

The headline metric is LEAD TIME: how many weeks before imaging progression a
model's risk score first crosses an alarm threshold, with the threshold
calibrated to a FIXED false-alarm rate so the comparison is apples-to-apples.
A snapshot model only sees the current draw; a longitudinal model sees the
trajectory, so it should alarm earlier for the same false-alarm budget.
"""

from __future__ import annotations
import numpy as np


def c_index(risk, tbin, event, max_anchors=800, seed=0):
    """Harrell-style concordance using risk vs discrete event time (any-cause).
    Vectorised over comparators; anchors optionally subsampled for speed."""
    ev = np.where(event)[0]
    if len(ev) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(ev) > max_anchors:
        ev = rng.choice(ev, max_anchors, replace=False)
    conc = tied = total = 0.0
    for i in ev:
        comp = tbin > tbin[i]            # comparator outlives anchor's event bin
        m = comp.sum()
        if m == 0:
            continue
        rj = risk[comp]
        conc += np.sum(risk[i] > rj)
        tied += np.sum(risk[i] == rj)
        total += m
    return (conc + 0.5 * tied) / total if total else float("nan")


def auc_within(risk, tbin, event, window_bin):
    """AUC for 'progresses within window' (any cause). Drops ambiguous censored."""
    positive = event & (tbin <= window_bin)
    neg_known = (~positive) & (((event) & (tbin > window_bin)) |
                               ((~event) & (tbin >= window_bin)))
    keep = positive | neg_known
    y = positive[keep].astype(int)
    s = risk[keep]
    return _auc(y, s)


def _auc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    s_sorted = s[order]; r = ranks[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def calibrate_threshold(risk, tbin, event, window_bin, target_far=0.10):
    """Pick tau so that ~target_far of 'truly-not-soon' landmarks raise an alarm."""
    positive = event & (tbin <= window_bin)
    neg_known = (~positive) & (((event) & (tbin > window_bin)) |
                               ((~event) & (tbin >= window_bin)))
    negs = risk[neg_known]
    if len(negs) == 0:
        return 0.5
    return float(np.quantile(negs, 1 - target_far))


def lead_times(samples, risk, tau, window_bin, bin_weeks):
    """
    Patient-level lead time (weeks) = t_imaging - earliest landmark week whose
    alarm score >= tau (and which is itself within reach of the event).
    Only true progressors (finite imaging time) are scored.
    Returns (median_lead, detection_rate, per_patient_dict).
    """
    pid = samples["pid"]; L = samples["L"]
    # map sample -> patient imaging time via the label we stored
    leads = {}
    detected = {}
    # we need t_imaging per patient: reconstruct from event samples is messy;
    # caller passes it through samples["t_imaging_by_pid"]
    timg = samples["t_imaging_by_pid"]
    for p in np.unique(pid):
        if not np.isfinite(timg.get(p, np.inf)):
            continue
        E = timg[p]
        mask = (pid == p)
        order = np.argsort(L[mask])
        Ls = L[mask][order]; rs = risk[mask][order]
        alarmed = np.where((rs >= tau) & (Ls < E))[0]
        if len(alarmed):
            first_L = Ls[alarmed[0]]
            leads[p] = E - first_L
            detected[p] = 1
        else:
            leads[p] = 0.0
            detected[p] = 0
    vals = np.array(list(leads.values()))
    det = np.array(list(detected.values()))
    med = float(np.median(vals[det == 1])) if det.sum() else 0.0
    return med, float(det.mean()), leads
