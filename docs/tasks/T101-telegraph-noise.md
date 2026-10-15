# T101: Random telegraph noise source

Status: todo
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
