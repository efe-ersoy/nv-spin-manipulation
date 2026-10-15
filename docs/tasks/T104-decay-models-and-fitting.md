# T104: Decay models and fitting for measured echo / DD curves

Status: done
Depends on: T101, T102
Repo: nv-spin-manipulation

## Goal
Generic tools to fit measured echo or dynamical-decoupling decay curves with physical noise
models, in particular "intrinsic NV decay × telegraph fluctuator(s)", so that hotspot data
(nvti T004) and simulated maps can be compared by parameters, not by eye.

## Requirements
* Exact theory for mixed noise (independent sources factorise):
  C = exp(−⟨φ²⟩_Gaussian / 2) × Π_k telegraph_coherence_k. Use it in `study.py` so studies with
  telegraph + Gaussian sources show an exact theory curve again (T101 open question).
* `nvsim.fitting` (generic, no sample-specific code):
  * signal model: S(t) = offset + amplitude × C(t), with C from (a) stretched exponential
    exp(−(t/T2)^n), (b) any Gaussian source set, (c) symmetric or asymmetric telegraph
    source(s), (d) products of these. The time axis is the total free evolution; the user
    states the sequence (echo, CPMG-N, XY8-N) and pulse durations.
  * `fit_decay(t, y, sigma, model, p0, bounds)` → parameters, covariance, χ², AIC/BIC.
    Robust starting values for telegraph fits (grid over rate and amplitude, then
    least squares), since the telegraph likelihood is multimodal in strong coupling.
  * Joint fits of several curves that share parameters (e.g. the same Γ at all heights,
    amplitude per height; or the same intrinsic T2 on and off a hotspot).
* The experimental observable: `se_norm` = (N₂ − N₁)/(N₂ + N₁) via the `Readout` model; amplitude
  and offset free by default.
* Example: synthetic hotspot data (intrinsic T2 = 10 µs, n = 1.5; telegraph Γ = 0.3 MHz,
  Δ = 5 µT; noise similar to the screenshot) fitted with all models; table of AIC.
* Chore: `requirements-lock.txt` (`pip freeze`) next to `pyproject.toml` (PLAN §10).

## Acceptance
* Parameter recovery on synthetic data within 2σ for weak and strong coupling; strong
  coupling shows Δ only as a lower bound (document this degeneracy).
* Tests for the factorised theory against Monte Carlo with mixed sources.

## Report

**Done (2026-10-15).**

* **Mixed-noise theory** (`nvsim/coherence.py`): `exact_coherence(s)` = Gaussian decay (all
  Gaussian sources) x prod of telegraph coherences; raises for sources without exact theory
  (`TraceNoise`, fixed-amplitude `NarrowbandNoise`). `study.py` uses it: studies with
  telegraph + Gaussian noise show an exact theory curve again ("exact theory (Gaussian x
  telegraph)"); the note in the plots is now only for unsupported sources
  (`theory_available`, `theory_label`, `theory_note`). `compare_studies.py` uses it too.
* **Telegraph speed-up**: `telegraph_coherences(segments_list, ...)` propagates sequences
  of the same family together with a closed-form batched 2x2 exponential (overflow-safe,
  matches scipy to 3e-14); `telegraph_coherence` is a wrapper.
* **Readout**: `Readout.se_norm(p0_n1, p0_n2)`, `se_norm_amplitude`, `se_norm_from_coherence`;
  se_norm = -+ contrast C/(2 - contrast), i.e. -0.176 C for contrast 0.3, which matches the
  measured -0.175 at short delay if N2 is the dark-ending readout (`n2_bright=False`
  default; still to be confirmed).
* **`nvsim/fitting.py`**: `DecaySequence(kind, n, t_pi)` (echo / CPMG-N / XY4-N / XY8-N,
  parametrised by total free evolution), components `StretchedExp`, `GaussianNoiseDecay`
  (any Gaussian source set from a factory), `TelegraphDecay` (symmetric or asymmetric, zero
  mean, exact theory incl. finite pulses), `DecayModel` (product, offset + amplitude x C),
  `fit_decay`, `fit_joint` (shared parameters by local name, per-curve names
  `"<curve>:<param>"`), `FitResult` (params, errors, covariance, chi2, dof, AIC, BIC,
  `table()`, `signal()`), `profile_chi2`, `profile_interval`. Offset/amplitude free by
  default and solved linearly at every grid point; the grid (telegraph rate x jump, 13 x 13
  log points, and intrinsic T2) gives robust starts, then bounded least squares from the 3
  best. Component results are memoised, so grid searches are cheap (single fit ~0.4 s,
  joint on/off fit ~0.2 s).
* `examples/decay_fitting_demo.py` -> `output/decay_fitting/` (figures for two noise levels,
  `aic_table.md`); `tests/test_fitting.py` (14 tests); `requirements-lock.txt` (pip freeze of
  the repo venv); README (fitting section, mixed theory, se_norm, layout, demos, lock file).

**Results**

* Factorised theory vs Monte Carlo (4000 realisations), all within statistical error:
  OU + telegraph (echo), 13C bath + telegraph (CPMG-4), asymmetric telegraph with mean
  offset + quasi-static noise (Ramsey; complex C).
* Parameter recovery (echo, 40 points 0.4-32 us, 20 seeds each):
  * weak coupling (rate 1 /us, jump 6 uT, sigma 0.004): pulls of rate and jump within 2 sigma
    in 19/20, mean -0.1 / 0.0, spread 1.1 / 1.05;
  * strong coupling (rate 0.2 /us, jump 100 uT, sigma 0.012): rate pulls 11/12 within 2 sigma
    (mean 0.17, spread 1.0); the jump is only bounded from below (profile Delta chi^2 = 4:
    e.g. > 10.6 uT, no upper bound).
* Demo (Gamma 0.3 /us, jump 5 uT, intrinsic T2 10 us, n 1.5; joint off + on fits):

  | hotspot model | dAIC (sigma 0.012) | dAIC (sigma 0.004) |
  |---|---|---|
  | separate stretched exps | 0.0 | 0.0 |
  | intrinsic x telegraph | 0.5 | 0.3 |
  | intrinsic x OU (Gaussian) | 0.2 | 0.3 |
  | intrinsic x extra stretched | 0.3 | 0.6 |

  Telegraph parameters recovered within 1 sigma (sigma 0.012: rate 0.26 +- 0.28, jump
  5.3 +- 0.8 uT; sigma 0.004: rate 0.26 +- 0.10, jump 5.11 +- 0.23 uT).

**Deviations / notes**

* Strong-coupling degeneracy (documented in `nvsim.fitting`): when 2 pi gamma_e (jump/2) >> rate
  the echo is ~ exp(-rate t) and depends on the jump only through small oscillatory terms;
  the covariance error of the jump is then meaningless (pull spread ~5.7 in a 30 uT test),
  so quote `profile_interval` instead. With multimodal profiles the Delta chi^2 = 4 interval
  **under-covers**: in 2 of 8 strong-coupling seeds a spurious intermediate-coupling mode
  (jump ~7-9 uT, rate shifted) beat the truth by Delta chi^2 = 4.6 and 8. The minimiser is not
  at fault (checked by refits from the truth). For quoting bounds on real data, calibrate
  the threshold with synthetic data at the fitted parameters.
* **Important for H1:** with the brief's parameters (2 pi gamma_e jump/2 / rate ~ 1.5,
  intermediate coupling) the telegraph model is **not distinguishable** from Gaussian OU noise
  with a matched spectrum, nor from a phenomenological extra stretched exponential: dAIC < 1
  at screenshot noise, at 3x lower noise, and also in a joint echo + CPMG-4 + CPMG-16 fit
  (dAIC 0.4; fitted OU 2.9 uT, tau_c 2.9 us). Echo shapes alone identify a fluctuator as
  non-Gaussian only in strong coupling. The fits still recover the telegraph parameters if
  that model is assumed.
* `TelegraphDecay` fixes the mean field to zero (it is refocused by echo / DD); for Ramsey
  fits with an offset field use a deterministic field or extend the component.

**Open questions**

* Which experiment discriminates H1 telegraph from Gaussian noise at realistic noise? Candidates
  to simulate next: strong-coupling regime (closer approach / larger jumps), CPMG with spacing
  < 1/(2 pi gamma_e jump/2) (a switch then dephases only partially), and the height series
  (shared rate, jump per height).
* AIC assumes Gaussian errors with known sigma; for photon-count data with few counts a
  Poisson likelihood would be more accurate.

