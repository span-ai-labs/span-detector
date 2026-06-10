# Span AI — Seeing below the limit of detection

Detecting endocrine / CDK4/6 resistance in HR+/HER2− metastatic breast cancer
from **serial ctDNA**, *months before the variant is even callable on a single
blood test* — by reading the **pattern of detections and non-detects** over time
instead of any one value.

**The polished write-up is the paper: `paper/span_cpblg.pdf`** (share this with
investors / KOLs).

---

## The one-line idea

A non-detect on a liquid biopsy is **not a zero** — it means the resistant clone
is below the assay's limit of detection (LoD). At low VAF, detection *flickers*
(finite-molecule Poisson sampling): detect, non-detect, detect. A single value —
or its slope — is blind during this sub-LoD dwell. **The rising rate of
detections is not.** The **Span detector** models that censored detection process
directly and fires a change-point alarm before the variant is callable at all.

## The Span detector (the novel method)

**Span** = a Censored-Poisson Bayesian Latent-Growth (CP-BLG) change-point
detector. It:

1. models each draw's detect/non-detect as a **Bernoulli detection process**
   under Poisson sampling + LoD left-censoring;
2. runs a sequential **generalised-likelihood-ratio (GLR)** test for an upward
   change-point in the per-variant detection rate;
3. **aggregates across competing resistance mechanisms** (ESR1 / PIK3CA / RB1 /
   HER2);
4. fires at a threshold **calibrated to a matched 10% false-alarm rate**.

It has **no trainable parameters** — a transparent decision rule, nothing to
overfit. The advantage is *structural*, not learned.

## The headline result (synthetic, 5 seeds, mean ± std)

At a **matched 10% false-alarm rate**, in the clinically hard **indolent**
regime (resistance dwells near the LoD for months):

| detector | overall sens. | early sens. (≥12 wk) | pre-LoD sens. |
|---|---|---|---|
| commodity snapshot | 15 ± 1% | 11 ± 2% | 8 ± 2% |
| slope-CUSUM (ablation) | 15 ± 1% | 11 ± 2% | 7 ± 2% |
| **Span (ours)** | **34 ± 4%** | **25 ± 3%** | **14 ± 2%** |

**~2.3× more impending progressions caught, with non-overlapping error bars.**

The effect is **falsifiable**: the advantage shrinks monotonically as clones
emerge faster (long sub-LoD dwell → big gain; fast emergence → the snapshot
catches it too). And **slope-CUSUM ties the snapshot** — a built-in ablation
proving the gain comes from the censored *detection-process* model, not the
value trajectory. We report the regimes where Span barely helps.

## Real-data grounding

`gbsg2_benchmark.py`: on **686 real breast-cancer patients** (GBSG-2), the deep
competing-risks survival backbone matches the standard Cox model (C-index
**0.67 vs 0.68**) and cleanly separates real outcomes into risk groups — proof
the machinery is sound on real biology.

## Honesty boundary

The serial-ctDNA trajectories are **synthetic and clearly labelled as such** at
the top of every generating file. No public serial-ctDNA breast cohort exists yet
— *that scarcity is the commercial thesis*. This is a **method demonstration**
that the lead-time claim is *achievable*, not a clinical result. Parameters are
calibrated to published literature (PADA-1, TRACERx, ESR1 emergence rates), not
tuned to manufacture a win.

---

## What's in here

```
span_benchmark/
├── README.md                ← you are here (plain English)
├── METHODS.md               ← technical notes
├── paper/
│   ├── span_cpblg.pdf       ← THE PAPER (the Span detector) — send this
│   └── span_cpblg.tex       ← LaTeX source (compiles with pdflatex)
├── src/Span AI/
│   ├── simulate.py          ← SYNTHETIC near-LoD ctDNA histories (labelled, not real)
│   ├── cpblg.py             ← ★ the Span detector (CP-BLG): GLR change-point on the detection process
│   ├── cpblg_benchmark.py   ← matched-FAR calibration + denominator-fair metrics
│   ├── cpblg_regimes.py     ← emergence-regime dose-response (indolent / slow / fast)
│   ├── cpblg_aggregate.py   ← multi-seed mean ± std (the locked headline numbers)
│   ├── make_cpblg_figures.py← the paper's figures
│   └── gbsg2_benchmark.py   ← REAL breast data (686 patients) vs Cox baseline
└── results/
    ├── cpblg_aggregate.json / cpblg_regimes.json
    ├── fig_cpblg_regimes.png     ← headline: advantage scales with sub-LoD dwell
    ├── fig_cpblg_trajectory.png  ← one patient: Span fires below LoD, before the scan
    └── fig3_gbsg2.png            ← real-data validation
```

## Reproduce

```bash
cd "src/Span AI"
python cpblg_aggregate.py indolent     # one regime per process (keeps it fast)
python cpblg_aggregate.py slow
python cpblg_aggregate.py fast
python make_cpblg_figures.py           # regenerate the figures
python gbsg2_benchmark.py              # real-data check
cd ../../paper && pdflatex span_cpblg.tex && pdflatex span_cpblg.tex
```
