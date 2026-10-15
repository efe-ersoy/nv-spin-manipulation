# T102: Noise sources from tabulated spectra and time traces

Status: todo
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
