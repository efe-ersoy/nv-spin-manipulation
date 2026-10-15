# T103: Echo / DD signal maps (many pixels at once)

Status: done
Depends on: T101, T102
Repo: nv-spin-manipulation

## Goal
Compute the measured quantity of a scanning echo or DD experiment for every pixel of a map
efficiently, so simulated noise maps can be compared with images like run 141.

## Requirements
* A function that takes one sequence (e.g. `spin_echo(tau)` with given pulse parameters) and
  per-pixel noise descriptions (arrays of parameters for a given source type, or per-pixel
  PSDs on a common frequency grid) and returns the coherence per pixel.
  * Gaussian sources: vectorise the filter-function integral (the filter function is the same
    for all pixels: compute |Y(f)|² once, then a matrix–vector product with the PSDs; treat
    spectral lines separately).
  * Non-Gaussian sources: Monte Carlo per pixel, with an option to reuse noise histories and
    a progress bar; document cost.
* Experimental observable: the user's scans record `se_norm` = (N₂ − N₁)/(N₂ + N₁), with N₁,
  N₂ the photon counts of the two sequences ending with +π/2 and −π/2. Implement it from the
  two final populations via the readout model (`Readout`), so its maximum magnitude is set by
  the PL contrast (measured ≈ 0.175 at short delay off the hotspot). Which phase is N₂ is
  still to be confirmed: keep the sign convention a parameter.
* Deterministic per-pixel fields (e.g. a 32 kHz oscillation amplitude map for T003) supported
  through the same interface.

* Chore (if not done in T104): `requirements-lock.txt` (`pip freeze`) next to
  `pyproject.toml`.

## Acceptance
* For a 100 × 100 map with Gaussian OU noise of random parameters, the vectorised result
  matches a per-pixel loop to 1e-6 and runs in < 1 s.
* MC path matches Gaussian path for Gaussian sources within statistical error.

## Report

**Done (2026-10-15, after T104).** New module `nvsim/maps.py`.

* Per-pixel descriptions: `PixelSource(factory, **params)` (array parameters per pixel,
  broadcast to the map shape; works with any factory, e.g. `OUNoise`,
  `TelegraphNoise.symmetric`, `CarbonBath`), `PixelSpectrum(f, psd, lines)` (per-pixel PSDs
  on a common grid, psd shape (n_f, *map), zero-width lines as variance maps),
  `PixelField(StaticField | ACField, random_phase=False, **params)`; plain sources / fields
  apply to every pixel.
* `coherence_map(sequence, drive, sources, fields, dt, shape)` -> complex coherence per
  pixel, factorised as in `nvsim.coherence`:
  * Gaussian: the toggling-function correlation (new shared helper
    `filter_function.toggling_correlation`, also used by `phase_variance` - refactor checked
    bit-for-bit to < 2e-15) is computed once; known source types (QuasiStatic, OU, Gaussian
    Narrowband) give a vectorised autocorrelation matrix, so the phase variance is one
    matrix-vector product (blocks of 2048 pixels); other Gaussian types fall back to their own
    `autocorrelation`. PSD maps: the variance is linear in the PSD node values, so node
    weights W_j (= integral hat_j |Y|^2, computed through the same time-domain sum with
    `SpectrumNoise`) are computed once and the map is `S . W`; lines add
    variance x |Y(f0)|^2 from the new complex `filter_function.toggling_transform`.
  * Telegraph: exact transfer matrix batched over pixels
    (`telegraph.telegraph_coherence_array`).
  * Fields: exp(i phi) with phi = 2 pi gamma_e integral y B dt from Y(f) (static: Y(0)); random-phase
    AC fields and fixed-amplitude zero-width `NarrowbandNoise` give J0(2 pi gamma_e A |Y(f)|) exactly.
  * Sources without exact theory (TraceNoise, other non-Gaussian) raise -> Monte Carlo.
* `signal_map(...)` -> `MapResult` with coherence, the two readout populations and
  `se_norm` = (N2 - N1)/(N2 + N1) via `Readout` (T104), from the noise-free populations of the
  sequence and of `with_flipped_readout(sequence)` (readout phase + pi); `n2_bright` selects
  which is N2 (default: N2 = dark-ending readout, as in the hotspot screenshot; note that
  run 141 has positive se_norm, so its convention may be the other one).
* `monte_carlo_map(...)`: `simulate_ensemble` per pixel and readout (the two readouts share
  noise histories), `reuse_noise=True` = same random numbers in every pixel, `progress`
  (stderr line or callable), errors per pixel (se_norm error: linear sum of the two readouts,
  an upper bound).
* Deterministic per-pixel fields go through the same interface (`PixelField`), e.g. the
  32 kHz tip oscillation for T003: `PixelField(ACField, random_phase=True,
  amplitude=A_osc * |grad B|, frequency=0.032)`.
* `examples/pixel_map_demo.py` -> `output/pixel_maps/se_norm_maps.png`; `tests/test_maps.py`
  (10 tests); README section. Lock file: done in T104.

**Results**

* Acceptance: 100 x 100 map, OU noise with random b (0.5-5 uT) and tau_c (0.2-5 us), echo
  2 x 2.5 us: vectorised 0.21 s; per-pixel loop (`coherence_decay`, same grid) 2.1 s;
  max |difference| 1.7e-15 (also 6e-16 vs the per-pixel default grid). Test asserts < 1 s
  and < 1e-6.
* Monte Carlo path vs Gaussian path (6 OU pixels, 2000-3000 realisations): coherence pulls
  0.0-0.4; se_norm within the quoted errors.
* Other checks: PSD map vs per-pixel `SpectrumNoise` loop 1e-16 (6e-11 with a spectral line,
  analytic vs time-domain line); telegraph map vs single-pixel exact theory 0 / 2e-16;
  fixed-phase AC / static field map vs full simulation within 3e-4 in P0 (linear-response
  phase; pulse-detuning effect of the field neglected); random-phase AC (J0) vs Monte Carlo
  within ~1.5 sigma.
* Demo (100 x 100 px, echo 2 x 5 us, each scenario 0.16-0.25 s): localised telegraph
  fluctuators (rate 0.08 /us, jump up to 40 uT) give flat-topped spots with a sharp edge
  (strong-coupling saturation; the echo loss does not depend on the jump until the jump is
  small), while a random-phase tip oscillation (3 nm in the gradient of a synthetic lattice)
  gives a lattice-periodic pattern following |dB/dx|. Monte Carlo line cut (17 px x 400
  realisations, 0.9 s) follows the theory.
* Monte Carlo cost: ~0.05 s per pixel at 400 realisations (echo, both readouts), ~0.3 s at
  3000; a 100 x 100 Monte Carlo map is minutes to tens of minutes.

**Deviations / notes**

* Telegraph and random-phase sources are handled exactly in the theory path (not by Monte
  Carlo as the brief anticipated), since T104 made that possible; Monte Carlo remains for
  sources without exact theory.
* Field phases use the toggling-frame (linear-response) form, consistent with the filter
  functions: valid for fields << Rabi frequency.
* `map_shape` comes from the per-pixel descriptions; pass `shape=` when only uniform
  sources are used.

**Open questions**

* Which readout is N2 (`n2_bright`) - the hotspot screenshot (negative se_norm) and run 141
  (positive) suggest different conventions or a different tracked transition.
* T003 will need the tip-oscillation phase model confirmed (random relative to the sequence
  vs locked); both are supported.

