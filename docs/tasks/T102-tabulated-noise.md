# T102: Noise sources from tabulated spectra and time traces

Status: done
Depends on: T100
Repo: nv-spin-manipulation

## Goal
Let `nvsim` consume noise computed elsewhere (micromagnetics or reduced models in `magtex`),
either as a power spectral density or as explicit time traces of B_nv(t).

## Requirements
* `SpectrumNoise(f, psd)`: stationary Gaussian noise defined by a tabulated one-sided PSD
  [mT²/MHz] on a frequency grid. `psd` interpolates; `autocorrelation` via numerical
  cosine transform (document accuracy); `sample` via spectral synthesis (random phases /
  Gaussian amplitudes, FFT), with correct variance on the requested grid; `max_step` from the
  highest frequency with significant weight.
* `TraceNoise(t, traces)`: replays given time traces (n_traces × n_t). `sample` draws random
  traces and random start offsets (requires traces longer than the sequence), with
  `gaussian = False` unless declared otherwise; `autocorrelation`/`psd` estimated from the
  traces (Welch) for plotting.
* Helpers to build both from the data contract (`S_nv` or `B_nv(t)` in SI, converted to
  mT / MHz / µs).
* README table entries.

## Acceptance
* `SpectrumNoise` built from `OUNoise.psd` reproduces OUNoise echo decay (theory and MC)
  within tolerance.
* `TraceNoise` fed with OU traces reproduces the OU result; fed with telegraph traces
  reproduces T101 results.

## Report

**Done (2026-10-15).** New module `nvsim/tabulated.py`; `SpectrumNoise`, `TraceNoise` exported
from `nvsim`.

* `SpectrumNoise(f, psd)` [MHz, mT^2/MHz], Gaussian.
  * `psd`: linear between table points, flat from the first point down to 0 Hz, zero above
    the last. `variance`, `b_rms`, exact `cumulative(f)` (piecewise quadratic).
  * `autocorrelation`: cosine transform of the interpolated PSD by the trapezoidal rule on a
    uniform frequency grid with step h = 1/(32 N_lag d), evaluated with one FFT (frequencies
    folded modulo the FFT length, so f_j * lag_k = jk/N exactly). Error at the largest lag
    <~ (2 pi/32)^2/12 ~ 0.3 % of the variance (measured ~1e-6 of the variance for an OU
    table); at uniform lags (the case of the filter-function theory) it is evaluated exactly
    on the lag grid, otherwise interpolated from a grid with step <= 1/(16 f_max).
  * `sample`: FFT spectral synthesis with period 16 x duration. Each bin gets a complex
    Gaussian amplitude whose variance is the exact integral of the interpolated PSD over the
    bin; bins above the FFT length are folded (they alias onto the grid exactly as sampling a
    continuous process would), so the sample variance is correct on any grid (checked on a
    0.5 us grid with most of the table above Nyquist). A random sub-bin frequency offset per
    realisation (16 strata) keeps narrow lines at their true frequency; correlations at lag L
    are smeared by ~sinc(L / (16 x duration)) (< 0.7 %).
  * `max_step` = 1/(20 f_95), f_95 = frequency below which 95 % of the variance lies.
* `TraceNoise(t, traces, gaussian=False)` [us, mT]: random trace and continuous random start
  offset per realisation, linear interpolation onto the requested grid; raises if the
  sequence is longer than the traces. `mean` (static offset), `b_rms`, biased autocovariance
  (FFT, averaged over traces) and Welch PSD (`nperseg = min(n_t, 16384)`) about the common
  mean; `max_step` = trace step. `gaussian` must be declared; default False, so the Gaussian
  theory refuses it (T101 guard).
* Data contract helpers: `spectrum_noise_from_si`, `trace_noise_from_si` (SI -> mT/MHz/us:
  S x 1e12, f x 1e-6, t x 1e6, B x 1e3), `spectrum_noise_from_dataset(ds, x, y, var="S_nv")`
  (dims (f, y, x)) and `trace_noise_from_dataset(ds, x, y, var="B_nv")` (dims (t[, run], y, x)),
  nearest pixel, `units` attributes checked when present.
* `tests/test_tabulated.py` (16 tests), README (noise table, data-contract paragraph, layout).

**Results** (echo coherence, Rabi 200 MHz, 4000 realisations unless noted):

| echo | OU theory | SpectrumNoise(OU psd) theory | SpectrumNoise MC | TraceNoise(OU traces) MC |
|---|---|---|---|---|
| 2 x 1 us | 0.8460 | 0.8460 | 0.8509 +- 0.0031 | 0.8460 +- 0.0033 |
| 2 x 2 us | 0.4693 | 0.4693 | 0.4709 +- 0.0036 (3 seeds x 8000) | 0.4700 +- 0.0086 |

| echo | telegraph exact (T101) | TraceNoise(telegraph traces) MC |
|---|---|---|
| 2 x 1 us | 0.6969 | 0.6828 +- 0.0097 |
| 2 x 3 us | 0.3161 | 0.3035 +- 0.0116 |

OU table: 3001 points to 50 MHz (99.8 % of the OU variance; the 1/f^2 tail beyond is the only
loss). OU traces: 100 x 200 us at 10 ns; telegraph traces: 60 x 400 us at 5 ns. A study with a
`SpectrumNoise` keeps the Gaussian-theory curve, one with `TraceNoise` drops it with the note.
Full suite: 83 tests OK (~20 s).

**Deviations / notes**

* No xarray dependency: the dataset helpers only use `ds[name].values / .dims / .attrs`
  (works with `xarray.Dataset`, tested with a minimal stand-in). Reading `.nc` files is left to
  the caller (nv-texture-imaging has xarray).
* `TraceNoise` realisations are independent only as long as the traces contain many
  non-overlapping sequence windows; Monte Carlo error bars do not account for re-use of the
  same trace data (documented in the class docstring).
* `SpectrumNoise` sampling cost scales with (number of bins up to f_max) once per call and
  16 x sequence length per realisation; a table extending far above the significant band
  (e.g. 50 MHz for a 1 us OU process) is fine but slower than `OUNoise`.

**Open questions**

* `line_frequencies` (auto x axis, spectrum plot range in studies) only knows
  `NarrowbandNoise`/AC fields; peaks inside a tabulated spectrum are not detected, so studies
  with only `SpectrumNoise` use the fallback axis. A peak finder on `SpectrumNoise` would fix
  this if needed.
* Should `TraceNoise` support non-uniform `t` (resampling on load)? Not needed for the data
  contract as written.

