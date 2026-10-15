# nv-spin-manipulation

Simulation and visualisation of NV-centre ground-state spin manipulation with
[QuTiP](https://qutip.org). Units: **MHz, µs, mT** (frequencies are cyclic; no 2π).
Use the helpers `uT` and `G` for other field units: `4 * uT` = 0.004 mT, `50 * G` = 5 mT.

## Install

Requires Python ≥ 3.11. Dependencies (numpy, scipy, matplotlib, qutip, pillow) are
listed in [pyproject.toml](pyproject.toml).

```
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate elsewhere)
pip install -e .                  # editable install of nvsim
```

From another repository's venv (e.g. `nv-texture-imaging`):
`pip install -e ../nv-spin-manipulation`. Inside this repo the scripts also run without
installing, because VS Code puts the repo root on `PYTHONPATH` (see below).

## Noise studies (start here)

[studies/noise_study.py](studies/noise_study.py) is the main entry point for
asking "how do different pulse sequences react to my magnetic noise?". Edit its
CONFIGURATION block — bias field, MW drive, noise sources, deterministic fields,
which sequences to scan over which τ, readout — and run

```
python studies/noise_study.py            # add --no-show to only write files, --quick for a fast look
```

It writes to `output/<study name>/`:

| file | contents |
|---|---|
| `summary.txt` | parameters and derived numbers (π time, ¹³C Larmor frequency, T₂*) |
| `noise_spectrum.png` | noise PSD and the filter function of every sequence |
| `noise_traces.png` | example realisations of the noise field |
| `signal_vs_tau.png` | PL (or P(\|0⟩)) vs τ per sequence: Monte Carlo, Gaussian theory, no-noise |
| `spectroscopy.png` | CPMG/XY coherence vs probed frequency 1/(2(τ + t_π)), above the PSD |
| `<sequence>.csv` | all numbers per sequence |

All sequences share one x axis (`X_AXIS`):

* `"spacing"` (default): the pulse spacing τ in the CPMG convention
  τ/2 – π – τ – … – π – τ/2. The spin echo is plotted as CPMG-1 (τ = total free
  evolution) and Ramsey has a single window τ. CPMG/XY dips of a noise line at
  f all sit at τ = 1/(2f) − t_π, and every sequence refocuses the line again at
  τ = 2k/f (every window, including the τ/2 edges, then holds whole periods).
* `"total"`: total free-evolution time, to compare how long each sequence keeps
  coherence.

`X_RANGE = "auto"` runs up to the first revival of the slowest noise line (2.2/f_min),
or 3 T₂* if that is longer. Copy the script to keep several studies side by side.

### Noise sources (`nvsim/noise.py`)

All are classical fields along the NV axis (amplitudes in mT). Every source has a
`gaussian` flag; only Gaussian sources may be fed to the Gaussian filter-function theory.

| source | spectrum | typical use |
|---|---|---|
| `QuasiStaticNoise(b_rms)` | constant per shot | slow drifts, inhomogeneous broadening (T₂*) |
| `OUNoise(b_rms, correlation_time)` | Lorentzian at 0 | fluctuating spin bath |
| `NarrowbandNoise(b_rms, frequency, linewidth)` | Lorentzian at ±f₀ (linewidth 0 = single line) | noise at a chosen frequency; `gaussian=False` for fixed amplitude, random phase |
| `CarbonBath(field, b_static, b_larmor, linewidth, abundance)` | quasi-static + line at f_L = γ_C·B | ¹³C bath: T₂*, echo collapses/revivals, DD dips at τ + t_π = 1/(2f_L) |
| `TelegraphNoise(b_low, b_high, rate_up, rate_down)`, `TelegraphNoise.symmetric(b, rate)` | Lorentzian at 0, τ_c = 1/(rate_up + rate_down); **non-Gaussian** | a field jumping between two values (bubble / wall segment hopping between pinning sites); mean = static offset |

The `CarbonBath` defaults (4 µT + 4 µT at 1.1 %) are phenomenological; tune them to your sample.

### How the signal is computed

* **Monte Carlo** (`nvsim/ensemble.py`): many noise histories propagated at once,
  with exact free evolution and exact pulses per realisation; fresh noise for every τ.
* **Gaussian theory** (`nvsim/filter_function.py`): coherence decays as
  exp(−⟨φ²⟩/2), with ⟨φ²⟩ = (2πγ)² ∫ S(f) |Y(f)|² df evaluated exactly in the time
  domain from the noise autocorrelation. Exact for the Gaussian sources; finite
  pulses are approximated (y(t) ∝ cos during π pulses). `phase_variance` /
  `coherence_decay` raise for non-Gaussian sources (`allow_non_gaussian=True` warns
  and returns the Gaussian approximation); studies then omit the theory curve and
  say so in the plot.
* **Telegraph theory** (`nvsim/telegraph.py`): exact coherence for telegraph noise.
  `telegraph_coherence(segments, source)` works for any levels, rates and sequence
  (two-state transfer matrix); closed forms for symmetric noise with ideal pulses:
  `rtn_fid_coherence`, `rtn_echo_coherence`, `rtn_cpmg_coherence` (also XY4/XY8).
  Weak coupling (2πγ_e b τ_c ≪ 1) reproduces the Gaussian result; strong coupling
  gives ≈ exp(−rate · total time) for the echo.
* **Readout** (`nvsim/readout.py`): `Readout(contrast=0.3)` maps populations to
  normalised PL (1 for |0⟩, 1 − contrast for |±1⟩); `photons_per_shot` + `shots`
  add photon shot noise.

The Monte Carlo and theory agree within statistical error (tested in `tests/test_noise.py`
and `tests/test_telegraph.py`).

## Building blocks

```python
import numpy as np
import nvsim
from nvsim import ACField, StaticField, uT
from nvsim.plotting import plot_result
from nvsim.bloch import animate_bloch, save_animation

nv = nvsim.NVCenter.from_field(Bz=5.0)            # mT; or NVCenter(f_minus=..., f_plus=...)
mw = nvsim.MWChannel("-1", rabi_frequency=20, detuning=0)   # or frequency=2729.9

result = nvsim.simulate(nvsim.spin_echo(tau=1.0), nv, mw, fields=[StaticField(10 * uT)])
plot_result(result)                    # sequence, field, populations, accumulated phase
save_animation(animate_bloch(result), "output/echo.gif")

# final populations vs a parameter; any argument may be a function of the value
fringes = nvsim.sweep(np.linspace(0, 2, 101), lambda tau: nvsim.ramsey(tau), nv, mw)
noisy = nvsim.sweep_ensemble(np.linspace(0.1, 2, 50), lambda tau: nvsim.xy8(tau, 2), nv, mw,
                             noise=[nvsim.CarbonBath(field=5.0)], n_realizations=300)
```

### Sequence library (`nvsim/sequences.py`)

| function | sequence |
|---|---|
| `rabi(duration)` | single pulse |
| `ramsey(tau)` | π/2 – τ – π/2 |
| `spin_echo(tau)` | π/2 – τ – π – τ – π/2 |
| `cpmg(tau, n)` | π/2ₓ – [τ/2 – π_y – τ/2]ⁿ – π/2 |
| `xy4(tau, n)`, `xy8(tau, n)` | same frame with π phases XYXY / XYXYYXYX |
| `dynamical_decoupling(tau, phases)` | any list of π-pulse phases |

`tau` is free-evolution time (pulse lengths are not subtracted), so the π-pulse
period of CPMG/XY is `tau + t_π`. Default readout phases bring a field-free spin
back to |0⟩. Custom sequences are built from `Pulse` (rotation `angle` or
`duration`, `phase` in rad or `"x"`, `"y"`, `"-x"`, `"-y"`, optional per-pulse
Rabi frequency) and `Wait`.

### Fields (`nvsim/fields.py`)

`StaticField(Bz)` and `ACField(amplitude, frequency, phase)` along the NV axis (mT),
on top of the bias field already in `NVCenter`. `ACField.sine_centered_at(b, f, t)`
builds a field that changes sign at time `t` (e.g. at the echo pulse).

### Accumulated phase (`nvsim/phase.py`)

`result.accumulated_phase()` is the angle, about +z, between the spin and a
*reference spin* that sees the same pulses but no fields and no detuning,
during free evolution, wrapped to (−π, π]. A π pulse about any equatorial axis
mirrors the equatorial plane, so φ → −φ across every refocusing pulse. Plots and
animations show this with dashed connectors and, on the Bloch sphere, the dashed
reference spin plus an arc labelled φ / −φ.

## Model

Spin-1 in the {|+1⟩, |0⟩, |−1⟩} basis. Each transition |0⟩↔|m⟩ is treated in the
rotating frame of its MW carrier (RWA). Pulses with phase 0 rotate about +x; on
the Bloch sphere of a transition |0⟩ is the north pole. `crosstalk=True` lets
each carrier also drive the other transition (this is where the absolute
transition frequencies matter). `c_ops` takes Lindblad operators (hook for T1/T2).

## Layout

```
nvsim/          library
  system.py  drive.py  pulses.py  sequences.py  fields.py      spin, MW, sequences, fields
  simulate.py  results.py  phase.py                            deterministic simulation
  noise.py  ensemble.py  filter_function.py  readout.py        noise, Monte Carlo, theory, PL
  telegraph.py                                                exact telegraph-noise coherence
  plotting.py  bloch.py  noise_plots.py                        figures and animations
  study.py                                                     NoiseStudy + run_study
studies/        configurable studies (noise_study.py)
examples/       demos of single features
tests/          unittest suite:  python -m unittest discover tests
output/         generated figures, GIFs and CSVs
```

| demo | shows |
|---|---|
| `examples/rabi_demo.py` | resonant / detuned Rabi, Bloch trajectories |
| `examples/ramsey_demo.py` | Ramsey trace and fringes |
| `examples/spin_echo_demo.py` | echo under no / static / AC fields, φ → −φ, echo vs Ramsey, AC response vs theory |
| `examples/dynamical_decoupling_demo.py` | XY8 under a matched AC field, filter narrowing for echo / CPMG / XY8 |
| `examples/telegraph_noise_demo.py` | echo under telegraph noise, weak vs strong coupling: Monte Carlo vs exact vs (wrong) Gaussian theory |

### VS Code

`.vscode/settings.json` selects `.venv` and runs scripts from the project root
with the root on `PYTHONPATH`, so the Run ▶ button works on any file in
`examples/` or `studies/`. `.vscode/launch.json` has debug configurations for the
current file (with or without `--no-show`) and for the test suite. Open a new
terminal after first loading the workspace so the terminal settings take effect.
