# T101: Random telegraph noise source

Status: done
Depends on: T100
Repo: nv-spin-manipulation

## Goal
Model a field that jumps between discrete values, as expected from a bubble or wall segment
hopping between two pinned positions (hypothesis H1 in PLAN). Echo signals from such noise
are non-Gaussian, so this also introduces the notion of non-Gaussian sources in `nvsim`.

## Physics
Two-state process B(t) ∈ {b1, b2} with switching rates γ12 (1→2) and γ21 (2→1), stationary
occupations p1 = γ21/(γ12+γ21). Mean b̄ = p1 b1 + p2 b2 acts as a static offset (refocused by an
echo). Fluctuating part: amplitude Δ = (b2 − b1), correlation time τ_c = 1/(γ12+γ21),
autocorrelation p1 p2 Δ² exp(−|t|/τ_c), PSD Lorentzian (same form as OUNoise).
For the symmetric case the Hahn-echo and CPMG coherences have closed forms
(e.g. Bergli, Galperin & Altshuler, New J. Phys. 11, 025002 (2009); Cywiński et al.,
PRB 77, 174509 (2008)); in the strong-coupling limit (2πγΔτ ≫ 1) the echo decays roughly as
exp(−2γ_s τ), not Gaussian.

## Requirements
* `TelegraphNoise(b_low, b_high, rate_up, rate_down)` in `nvsim/noise.py` (or a new module),
  with exact sampling on the time grid (draw switching times, not just Bernoulli per step),
  `autocorrelation`, `psd`, `max_step`, and `gaussian = False`.
* Add a `gaussian` attribute (default True) to the existing sources; make
  `filter_function.phase_variance`/`coherence_decay` raise or warn for non-Gaussian input,
  and make `study.py` skip the Gaussian-theory curve for such sources (with a note in the plot).
* Analytic echo / CPMG coherence for symmetric RTN as a helper (for tests and fits).
* Example: `examples/telegraph_noise_demo.py`: echo vs τ for weak and strong coupling,
  MC vs analytic vs (wrong) Gaussian theory.
* README table entry.

## Acceptance
* Tests: sampled traces reproduce occupations and autocorrelation; Monte Carlo echo agrees
  with the analytic symmetric-RTN result within statistical error in weak and strong coupling;
  in the weak-coupling limit it agrees with the Gaussian theory.

## Report

**Done (2026-10-15).**

* `TelegraphNoise(b_low, b_high, rate_up, rate_down)` in `nvsim/noise.py`, plus
  `TelegraphNoise.symmetric(b, rate)` (levels +-b). Properties `p_low`, `p_high`, `mean`,
  `amplitude`, `correlation_time`, `b_rms`. Exact sampling: the initial state is drawn from the
  stationary occupations, dwell times from exponentials (`switching_times`), and the state is
  read off at the grid times. `autocorrelation` = p_low p_high Delta^2 exp(-|t|/tau_c), `psd`
  Lorentzian, `max_step` = min(tau_c/20, 0.1 rad / (2 pi gamma_e |Delta|)) so the
  trapezoidal integration across a switch stays accurate in strong coupling.
  `gaussian = False` (class attribute).
* `gaussian` / `mean` class defaults on `NoiseSource` (True / 0), helpers `is_gaussian`,
  `non_gaussian`. `NarrowbandNoise(gaussian=False)` already had the flag.
* `filter_function.phase_variance` / `coherence_decay` raise `ValueError` for non-Gaussian
  sources; `allow_non_gaussian=True` warns and returns the Gaussian approximation (used in the
  demo for the "wrong" curve).
* `study.py`: `NoiseStudy.gaussian_theory` / `theory_note`; with non-Gaussian noise the theory
  curve is skipped, the signal and spectroscopy figure titles and `summary.txt` say why.
* New `nvsim/telegraph.py`:
  * `telegraph_coherence(segments, source, transition)`: exact <exp(i phi)> for any telegraph
    source (asymmetric, non-zero mean) and any compiled sequence, by the two-state
    stochastic Liouville / transfer-matrix method (pi pulses with the cos(pi u) toggling
    profile in 16 steps).
  * Closed forms for symmetric RTN, ideal pulses: `rtn_fid_coherence`, `rtn_echo_coherence`
    (Bergli et al. 2009), `rtn_cpmg_coherence` (2x2 matrices in the sum/difference basis;
    CPMG-1 = echo; also valid for XY4/XY8 with ideal pulses). Vectorised over tau, continuous
    at mu = 0.
* `examples/telegraph_noise_demo.py` -> `output/telegraph_noise_echo.png`: traces and echo
  coherence vs 2 tau, weak and strong coupling, Monte Carlo vs exact vs Gaussian.
* `tests/test_telegraph.py` (15 tests); README tables (noise sources, theory, layout, demos).

**Results** (Rabi 200 MHz, 4000 realisations, coherence = Monte Carlo / exact / Gaussian):

| case | 2 pi gamma_e b tau_c | sequence | Monte Carlo | exact | Gaussian |
|---|---|---|---|---|---|
| weak, +-2 uT, 2 /us | 0.088 | echo 2 x 5 us | 0.754 +- 0.005 | 0.749 | 0.751 |
| weak | 0.088 | echo 2 x 10 us | 0.557 +- 0.008 | 0.548 | 0.550 |
| strong, +-20 uT, 0.2 /us | 8.8 | echo 2 x 1 us | 0.703 +- 0.009 | 0.697 | 0.083 |
| strong | 8.8 | echo 2 x 3 us | 0.296 +- 0.012 | 0.316 | 0.000 |
| strong | 8.8 | CPMG-4, tau = 1.5 us | 0.256 +- 0.012 | 0.248 | 0.005 |

Closed forms and the transfer matrix agree to < 1e-4 (finite-pulse difference). Full suite:
67 tests OK.

**Deviations / notes**

* `autocorrelation` now means the autocovariance of the fluctuating part for every source
  (identical to before for the zero-mean sources); the telegraph mean is exposed as `mean`
  and is a static offset (refocused by echoes, a phase for Ramsey). Module docs updated.
* Because the guard covers every non-Gaussian source, a study with
  `NarrowbandNoise(gaussian=False)` (`NOISE_GAUSSIAN = False` in `studies/noise_study.py`)
  now also omits the Gaussian curve; before, the Gaussian approximation was drawn silently.
* `studies/compare_studies.py` computes theory only (no Monte Carlo by default), so it raises
  the `ValueError` if a non-Gaussian source is added there; use `--monte-carlo` scenarios or
  extend it with `nvsim.telegraph` if needed.

**Open questions**

* Studies could show an exact theory curve for mixed noise, since independent sources
  factorise: C = exp(-<phi^2>_Gaussian / 2) x prod telegraph_coherence(...). Not done (out of
  the brief); cheap to add if useful for fitting H1.
* Asymmetric RTN with large mean: `telegraph_coherence` returns a complex C; the studies'
  signal model uses Re C, which is right for readout along the coherence axis (all built-in
  sequences) but not for a 90-degree readout phase.

