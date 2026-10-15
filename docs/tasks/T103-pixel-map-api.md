# T103: Echo / DD signal maps (many pixels at once)

Status: todo
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

## Acceptance
* For a 100 × 100 map with Gaussian OU noise of random parameters, the vectorised result
  matches a per-pixel loop to 1e-6 and runs in < 1 s.
* MC path matches Gaussian path for Gaussian sources within statistical error.

## Report
