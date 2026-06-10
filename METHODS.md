# SPAN-EGFR: A Dynamic ctDNA Resistance-Prediction Benchmark

*A reproducible, public methods benchmark showing that a longitudinal model
extracts resistance lead-time that a single-snapshot model cannot.*

**Status:** synthetic proof-of-method. No patient data. See "Honesty boundary".

---

## 0. The one-paragraph version (read this first)

People are asking you to "de-risk the model." You cannot show a real clinical
result yet, because the real data doesn't exist outside hospital walls — that is
literally your thesis and your moat. What you *can* show, today, honestly, is
that **your mathematical approach works**: given the same information at the same
moment, a model that reads the *trajectory* of ctDNA flags impending resistance
**weeks earlier** and **catches more patients** than a model that reads a single
test report — at an identical false-alarm rate. This repo proves exactly that on
a transparent, fully-disclosed synthetic cohort, and tells anyone how to
reproduce and extend it. That is what a "public benchmark" means here.

---

## 1. The honesty boundary (the part that protects the company)

There are two very different things, and conflating them will end the company:

- **Fabricating results** — generating numbers and implying they came from real
  patients. This is fraud. Curigliano, Tutt, and an ex-NCCN panelist signing his
  name to your retrospective will catch it, and a retraction is terminal. **We do
  not do this, anywhere.**
- **A methods demonstration on clearly-labelled synthetic data** — "here is our
  model on simulated trajectories built from published parameters, proving the
  math separates signal from noise before imaging does." This is standard
  practice in every methods paper before a cohort exists. **This is what this
  repo is**, and every file says so at the top.

The synthetic world here is calibrated to *published* numbers (resistance ~90%
on osimertinib; MET amp ~15%, C797S ~7%; TRACERx ~151-day molecular→imaging
window). Those citations describe the *shape of the simulation*, not a claim
about real performance. The deliverable is a **method and a task definition**,
not a clinical validation.

---

## 2. The mathematical reframing (this is the IP claim)

### What a test-maker computes
Guardant / Natera / Foundation produce, from a single blood draw at time *t*, an
estimate of whether a mutation is present **right now**:

```
    test-maker:   f(x_t)  →  P(mutation present at time t)
```

It is a **cross-sectional classifier**. It has no notion of *when* resistance
will become clinically actionable, and no memory of the patient's previous draws.

### What Span AI computes
Given the *sequence* of draws up to a landmark time *L*, predict the **time and
mechanism** of future progression — a continuously-updated forecast:

```
    Span AI:   g( x_{t_1}, x_{t_2}, …, x_L )  →  CIF_k(τ | history),  for each
               resistance mechanism k and every future horizon τ
```

This is a **dynamic competing-risks survival** problem. Two pieces of standard
survival theory make it precise:

1. **Competing risks.** C797S, MET amplification, PIK3CA, and small-cell
   transformation are *mutually exclusive first events* — a patient progresses
   via one of them. The right object is the **cause-specific cumulative incidence
   function** (CIF):

   ```
       CIF_k(τ | H_L) = P( progress by time τ  AND  via mechanism k | history H_L )
   ```

   Not a per-mutation probability — a *distribution over (which mechanism, when)*.

2. **Dynamic prediction.** The forecast is re-issued at every new draw, each time
   conditioning on the *full* history H_L. The clinically meaningful quantity is
   **lead time**:

   ```
       lead(patient) = t_imaging_progression  −  (first L where alarm score ≥ τ)
   ```

   i.e., how far ahead of the scan we raise the flag.

### Why the trajectory wins (the mechanism, in one line)
ctDNA is **noisy**. A single elevated draw is ambiguous — assay noise and
background shedding produce spurious spikes. Averaging several draws cuts that
noise by ~√n and a slope term detects the *rise* while the absolute level is
still low. So at a fixed false-alarm budget the longitudinal model can use a
**lower, earlier** alarm threshold. This is not hand-waving; it is the entire
measured effect in §5, and it is exactly the gap a snapshot test cannot close.

> **The defensible claim, stated mathematically:** test-makers estimate
> `P(mutation | x_t)`; we estimate the cause-specific CIF `CIF_k(τ | H_L)` and
> its induced lead-time. The first is a point on a curve; the second *is* the
> curve. You cannot recover the second from a sequence of the first without
> modelling the trajectory — which requires being in the longitudinal workflow.

---

## 3. The benchmark task (so it can be *public*)

A benchmark is a frozen task other people can score against. Ours:

- **Cohort:** EGFR-mutant NSCLC, first-line osimertinib, serial ctDNA every ~7
  weeks (synthetic generator in `src/simulate.py`, fixed seed).
- **Prediction unit:** one *landmark* per draw, using only history up to that
  draw (no future leakage).
- **Target:** discrete-time competing-risks label — `(mechanism, time-bin)` of
  imaging progression, or right-censoring.
- **Primary metric:** **median lead time** before imaging at a **matched 10%
  false-alarm rate** (so models can't buy lead-time with reckless alarms).
- **Secondary:** time-dependent C-index; AUC for "progression within 24 weeks";
  sensitivity at the fixed false-alarm rate.
- **The rule that makes it fair:** snapshot and longitudinal models see the
  *same landmarks* and predict the *same targets*; they differ only in whether
  they may use history. Any gap is therefore attributable to the trajectory.

This is the contribution you can put on arXiv / a `papers-with-code`-style page:
not "we beat SOTA," but "**here is a clean, reproducible task that isolates the
value of longitudinal modelling in resistance prediction, plus a baseline and a
reference model.**" That is genuinely novel — nobody has framed resistance
prediction as a public dynamic-competing-risks benchmark.

---

## 4. Models

| | input | what it represents | where |
|---|---|---|---|
| **Snapshot baseline** | latest draw only | a test-maker's single report | `model_numpy.py` |
| **Longitudinal (demo)** | engineered trajectory features (denoised level, slope) | Span AI's layer, runs with zero install | `model_numpy.py` |
| **Dynamic-DeepHit (production)** | raw serial draws, learned end-to-end by a GRU | the model for the paper/cohort | `model_torch.py` |

All three share one output head — a softmax over `{(mechanism k, time-bin j)}`
plus a "no-event" cell — and one loss, the **DeepHit** censoring-aware
log-likelihood plus a cause-specific ranking term (Lee et al., AAAI 2018 / IEEE
TBME 2020). The censored-likelihood has a clean closed form used in both
implementations:

```
    L = − log ( Σ_{m ∈ A} q_m ) ,   A = allowed output cells consistent with what
                                         we observed (the event cell, or all
                                         cells implying survival past censoring)
    dL/dz = q − (q ⊙ 1_A) / (q · 1_A)        # exact gradient, see model_numpy.py
```

The NumPy model trains on *hand-built* trajectory features; the GRU *learns* them
from raw draws and generalises to irregular timing and many more biomarkers. We
report the NumPy numbers because they run anywhere; the GRU is the production
path once `torch` is available.

---

## 5. Results (synthetic; seed = 7)

Matched false-alarm rate = 10%. Higher is better for all columns.

| model | C-index | AUC (≤24 wk) | median lead | sensitivity |
|---|---|---|---|---|
| snapshot (test-maker proxy) | 0.741 | 0.864 | **140 days** | 74% |
| **longitudinal (Span AI)** | **0.768** | **0.894** | **168 days** | **81%** |

**Headline: +28 days of lead time and +7 points of sensitivity at the same
false-alarm rate**, purely from reading the trajectory instead of the latest
draw. The 168-day figure sits right in the TRACERx ~151-day molecular→imaging
window — the longitudinal model recovers essentially the whole biological lead,
while the snapshot model gives much of it back to noise.

Figures: `results/fig1_metrics.png` (the table, visually) and
`results/fig2_trajectory.png` (one patient: the longitudinal risk crosses its
alarm before the snapshot's, and well before the scan).

Reproduce: `cd src && python run_benchmark.py`.

---

## 6. Graduating from synthetic to real (the honest data roadmap)

Use real public data in two layers, then your private cohort:

**Layer A — validate the *method* on real longitudinal survival data (public, now).**
The dynamic-competing-risks machinery is domain-agnostic. Show it works on a real
serial-biomarker dataset before claiming anything in oncology:
- **PBC2** (primary biliary cirrhosis, serial labs + survival) — the standard
  public benchmark for joint/dynamic survival models; ships with R `JM`/`JMbayes`
  and Python ports. Real, longitudinal, downloadable today.
- **SUPPORT**, **METABRIC** — public survival datasets used in the original
  DeepHit paper (cross-sectional, good for the competing-risks head).

**Layer B — real oncology genomics (public / application-based).**
- **AACR Project GENIE** and **GENIE BPC** via cBioPortal — real EGFR-mutant NSCLC
  genomic + clinical data; BPC adds treatment and outcome timelines. Public.
- **MSK-CHORD / MSK-MET** (cBioPortal) — large clinical-genomic cohorts with
  outcomes.
- **TRACERx** ctDNA data — controlled access via EGA; requires a data-access
  application, but it is the closest real serial-ctDNA-with-outcomes resource and
  worth applying for.
- Note (matches your deck): **TCGA / GEO / Pan-Cancer Atlas are cross-sectional**
  — fine for diagnostic classification, useless for trajectories. Don't lean on
  them for this claim.

**Layer C — your moat.** Tata + Krishnamurthy's prospective Huntsman arm produce
the paired *serial-ctDNA → resistance-mechanism → outcome* sequences that exist
nowhere public. That's the asset; Layers A–B only de-risk the method around it.

**Recommended de-risking artifact for investors/KOLs:** this synthetic benchmark
(method + task) **plus** a PBC2 run (method on real longitudinal data) **plus**
the EGFR-mutant slice of GENIE for biological face-validity. Three honest pieces,
zero fabrication, and a clear line to where the real signal comes from.

---

## 7. What synthetic data *can* and *cannot* show

**Can:** that the estimator is identifiable and learnable; that longitudinal
representation beats snapshot under realistic noise; that the lead-time metric
behaves; that the competing-risks head calibrates. This de-risks *the method*.

**Cannot:** the real effect size, real mechanism frequencies in *your* cohort,
real assay noise structure, or clinical utility. Those require Tata/Huntsman.
**Never present §5 numbers as anything but synthetic.** They prove the math is
sound, not that the product works on patients — that's what the pilots are for.

---

## 8. How to run

```bash
cd src
python run_benchmark.py     # full benchmark, prints the table, writes results/
python make_figures.py      # regenerates the two figures
python simulate.py          # sanity-check the synthetic generator
# production GRU (needs torch):  pip install torch && python model_torch.py
```

