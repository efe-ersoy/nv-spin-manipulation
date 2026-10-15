"""Monte Carlo simulation of a pulse sequence under classical magnetic noise.

Each realisation of the noise (see :mod:`nvsim.noise`) is a different field
history B(t) along the NV axis; the measured signal is the average over many
of them, as in an experiment that averages many shots. All realisations are
propagated at once:

* free evolution is exact: the phase is ``2 pi gamma_e`` times the integral of
  B(t), taken from the noise samples with the trapezoidal rule;
* pulses are split into sub-steps no longer than the noise grid step, each
  with the mean field of that sub-step (exact matrix exponential per
  realisation).

Deterministic :mod:`~nvsim.fields` and MW detunings are included the same way
as in :func:`nvsim.simulate` (crosstalk and extra Hamiltonian terms are not).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.linalg import expm

from .constants import GAMMA_E, TWO_PI
from .drive import MWChannel
from .fields import total_field
from .noise import flatten, max_step, sample_total
from .pulses import PulseSequence
from .results import SweepResult
from .simulate import _Model, channel_map
from .system import LEVELS, NVCenter, as_state

#: Grid points per sequence at minimum, and the hard upper limit.
MIN_GRID_POINTS = 400
MAX_GRID_POINTS = 200_000

_SZ = np.array([1.0, 0.0, -1.0])


@dataclass
class EnsembleResult:
    """Final populations averaged over the noise realisations."""

    populations: dict[str, float]
    uncertainty: dict[str, float]
    n_realizations: int
    final_states: np.ndarray | None = None  # (n, 3) kets, if requested

    def population(self, level) -> float:
        return self.populations[level]


def field_time_scale(sources, fields) -> float:
    """Longest step [us] over which the noise and the deterministic fields are
    effectively constant (inf if everything is static)."""
    candidates = [max_step(sources)]
    candidates += [1.0 / (20 * f.max_rate()) for f in fields if f.max_rate() > 0]
    return min(candidates)


def noise_grid_step(sources, fields, duration: float) -> float:
    """Grid step [us] for sampling and integrating the noise: resolves the
    noise and fields, with at least ``MIN_GRID_POINTS`` points per sequence."""
    step = min(field_time_scale(sources, fields), duration / MIN_GRID_POINTS)
    return max(step, duration / MAX_GRID_POINTS)


def simulate_ensemble(
    sequence: PulseSequence,
    nv: NVCenter,
    drives: MWChannel | Iterable[MWChannel],
    noise=(),
    fields=(),
    n_realizations: int = 300,
    initial_state="0",
    rng: np.random.Generator | int | None = None,
    dt: float | None = None,
    keep_states: bool = False,
) -> EnsembleResult:
    """Average final populations of ``sequence`` over noise realisations.

    Parameters
    ----------
    noise : iterable of noise sources
        See :mod:`nvsim.noise`, e.g. ``[CarbonBath(field=50), NarrowbandNoise(...)]``.
    fields : iterable of StaticField / ACField
        Deterministic fields, identical in every realisation.
    n_realizations : int
        Number of noise histories to average.
    rng : Generator or int, optional
        Random generator or seed.
    dt : float, optional
        Noise grid step [us]; chosen from the noise spectra by default.
    """
    rng = np.random.default_rng(rng)
    channels = channel_map(drives)
    segments = sequence.compile(channels)
    model = _Model(nv, channels, False, [], [])
    sources = flatten(noise)
    fields = tuple(fields)
    duration = segments[-1].end if segments else 0.0

    psi0 = as_state(initial_state)
    if not psi0.isket:
        raise ValueError("simulate_ensemble needs a pure initial state (ket or level label).")
    psi = np.repeat(psi0.full().reshape(1, 3), n_realizations, axis=0).astype(complex)

    if duration > 0:
        if dt is None:
            dt = noise_grid_step(sources, fields, duration)
        pulse_step = field_time_scale(sources, fields)
        n_cells = max(1, ceil(duration / dt))
        grid = np.linspace(0.0, duration, n_cells + 1)
        dt = grid[1] - grid[0]
        b = sample_total(sources, grid, n_realizations, rng) + total_field(fields, grid)[None, :]
        cumulative = np.zeros_like(b)
        cumulative[:, 1:] = np.cumsum(0.5 * (b[:, 1:] + b[:, :-1]) * dt, axis=1)

        def integral(t0: float, t1: float) -> np.ndarray:
            """Integral of B over [t0, t1] for every realisation [mT us]."""

            def at(t):
                x = min(max(t / dt, 0.0), n_cells)
                i = min(int(x), n_cells - 1)
                w = x - i
                return cumulative[:, i] * (1 - w) + cumulative[:, i + 1] * w

            return at(t1) - at(t0)

        coupling = TWO_PI * GAMMA_E
        for segment in segments:
            H = model.segment_hamiltonian(segment)[0].full()
            if not segment.is_pulse:
                phase = np.real(np.diag(H))[None, :] * segment.duration
                phase = phase + coupling * integral(segment.start, segment.end)[:, None] * _SZ[None, :]
                psi *= np.exp(-1j * phase)
                continue
            # Sub-steps only where the field changes during the pulse (not on the
            # possibly much finer integration grid).
            n_sub = max(1, ceil(segment.duration / max(dt, pulse_step) - 1e-9))
            edges = np.linspace(segment.start, segment.end, n_sub + 1)
            for t0, t1 in zip(edges[:-1], edges[1:]):
                h = t1 - t0
                b_mean = integral(t0, t1) / h
                Hs = H[None, :, :] + coupling * b_mean[:, None, None] * np.diag(_SZ)[None, :, :]
                psi = np.einsum("rij,rj->ri", expm(-1j * h * Hs), psi)

    probs = np.abs(psi) ** 2
    mean = probs.mean(axis=0)
    sem = probs.std(axis=0, ddof=1) / np.sqrt(n_realizations) if n_realizations > 1 else np.zeros(3)
    return EnsembleResult(
        populations={lv: float(mean[i]) for i, lv in enumerate(LEVELS)},
        uncertainty={lv: float(sem[i]) for i, lv in enumerate(LEVELS)},
        n_realizations=n_realizations,
        final_states=psi if keep_states else None,
    )


def sweep_ensemble(
    values: Iterable[float],
    sequence: PulseSequence | Callable[[float], PulseSequence],
    nv: NVCenter | Callable[[float], NVCenter],
    drives,
    noise=(),
    fields=(),
    n_realizations: int = 300,
    initial_state="0",
    rng: np.random.Generator | int | None = None,
    parameter: str = "parameter",
    dt: float | None = None,
) -> SweepResult:
    """:func:`simulate_ensemble` for each value; fresh noise for every point.

    ``sequence``, ``nv``, ``drives``, ``noise`` and ``fields`` may be functions
    of the swept value, as in :func:`nvsim.sweep`.
    """
    rng = np.random.default_rng(rng)
    values = np.asarray(list(values), dtype=float)

    def resolve(arg, v):
        return arg(v) if callable(arg) else arg

    results = [
        simulate_ensemble(resolve(sequence, v), resolve(nv, v), resolve(drives, v), resolve(noise, v),
                          resolve(fields, v), n_realizations, initial_state, rng, dt)
        for v in values
    ]
    populations = {lv: np.array([r.populations[lv] for r in results]) for lv in LEVELS}
    uncertainty = {lv: np.array([r.uncertainty[lv] for r in results]) for lv in LEVELS}
    return SweepResult(values, populations, parameter, uncertainty=uncertainty)
