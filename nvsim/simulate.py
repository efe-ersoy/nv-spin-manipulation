"""Propagate the NV spin through a pulse sequence.

Model
-----
Each transition |0> <-> |m> is described in the rotating frame of its
microwave carrier (or of the transition itself if no channel addresses it).
Within the rotating-wave approximation the Hamiltonian is (h = 1, angular units)::

    H = sum_m  -2 pi Delta_m |m><m|
      + sum_{active drives k}  pi Omega_k (e^{-i phi_k} |0><k| + h.c.)
      + 2 pi gamma B(t) Sz                       (fields along the NV axis)

with Delta_m = f_carrier,m - f_m. Phase phi = 0 rotates about +x, pi/2 about +y.

With ``crosstalk=True`` each carrier also drives the *other* transition (equal
coupling, as for a linearly polarised MW field), producing a term rotating at
the frequency difference between the carrier and that transition's frame. This
is where the absolute transition frequencies enter.

The sequence is propagated segment by segment:

* constant Hamiltonian -> exact matrix exponentials;
* diagonal but time-dependent Hamiltonian (free evolution in an AC field) ->
  exact, from the time integral of the coefficients;
* anything else (pulses during an AC field, crosstalk) -> :func:`qutip.mesolve`.
"""

from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from math import ceil

import numpy as np
import qutip as qt

from .constants import TWO_PI
from .drive import MWChannel
from .fields import Field
from .pulses import PulseSequence, Segment
from .results import SimulationResult, SweepResult
from .system import (
    LEVELS,
    TRANSITIONS,
    NVCenter,
    as_state,
    projector,
    transition_operator,
)

#: Output samples per period of the fastest Rabi / detuning / field frequency.
SAMPLES_PER_CYCLE = 40
#: Minimum output samples per segment, so short pulses are resolved too.
MIN_SAMPLES_PER_SEGMENT = 20
#: Maximum ODE step, per period of the fastest time-dependent term.
SOLVER_STEPS_PER_CYCLE = 20

#: An extra Hamiltonian term in angular units (rad/us): a constant Qobj, or a
#: (Qobj, f) pair where f(t) is a coefficient of the absolute time t [us].
HamiltonianTerm = qt.Qobj | tuple[qt.Qobj, Callable[[float], complex]]

_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(8)


def channel_map(drives: MWChannel | Iterable[MWChannel]) -> dict[str, MWChannel]:
    """Index MW channels by transition, rejecting duplicates."""
    if isinstance(drives, MWChannel):
        drives = [drives]
    channels: dict[str, MWChannel] = {}
    for channel in drives:
        if channel.transition in channels:
            raise ValueError(
                f"Two MW channels address transition {channel.transition!r}."
            )
        channels[channel.transition] = channel
    return channels


def frame_frequencies(nv: NVCenter, channels: dict[str, MWChannel]) -> dict[str, float]:
    """Rotating-frame frequency [MHz] of each transition."""
    return {
        m: channels[m].carrier_frequency(nv)
        if m in channels
        else nv.transition_frequency(m)
        for m in TRANSITIONS
    }


class _Model:
    """Hamiltonian builder for one simulation configuration."""

    def __init__(self, nv, channels, crosstalk, extra_terms, field_rates):
        self.nv = nv
        self.channels = channels
        self.crosstalk = crosstalk
        self.field_rates = list(field_rates)
        self.frames = frame_frequencies(nv, channels)
        self.detunings = {
            m: self.frames[m] - nv.transition_frequency(m) for m in TRANSITIONS
        }

        self.static = qt.qzero(3)
        for m in TRANSITIONS:
            self.static += -TWO_PI * self.detunings[m] * projector(m)
        self.static_td: list[tuple[qt.Qobj, Callable]] = []
        for term in extra_terms:
            if isinstance(term, qt.Qobj):
                self.static += term
            else:
                op, coeff = term
                self.static_td.append((op, coeff))

    def segment_hamiltonian(self, segment: Segment):
        """Constant part and time-dependent (op, f(t)) terms during ``segment``."""
        H = self.static.copy()
        td = list(self.static_td)
        for drive in segment.drives:
            amp = 0.5 * TWO_PI * drive.rabi_frequency
            sig = transition_operator(drive.transition)
            H += amp * (
                np.exp(-1j * drive.phase) * sig + np.exp(1j * drive.phase) * sig.dag()
            )
            if not self.crosstalk:
                continue
            carrier = self.channels[drive.transition].carrier_frequency(self.nv)
            for other in TRANSITIONS:
                if other == drive.transition:
                    continue
                w = TWO_PI * (carrier - self.frames[other])
                sig_o = transition_operator(other)
                td.append(
                    (
                        amp * sig_o,
                        lambda t, w=w, p=drive.phase: np.exp(1j * (w * t - p)),
                    )
                )
                td.append(
                    (
                        amp * sig_o.dag(),
                        lambda t, w=w, p=drive.phase: np.exp(-1j * (w * t - p)),
                    )
                )
        return H, td

    def sample_rate(self, segments: Sequence[Segment]) -> float:
        """Fastest Rabi / detuning / field frequency [MHz] the output should resolve."""
        rates = [abs(d) for d in self.detunings.values()] + self.field_rates
        for segment in segments:
            for drive in segment.drives:
                rates.append(
                    np.hypot(drive.rabi_frequency, self.detunings[drive.transition])
                )
        return max(rates, default=0.0)

    def crosstalk_rate(self, segments: Sequence[Segment]) -> float:
        """Fastest crosstalk oscillation frequency [MHz] (0 if crosstalk is off)."""
        if not self.crosstalk:
            return 0.0
        rates = [0.0]
        for segment in segments:
            for drive in segment.drives:
                carrier = self.channels[drive.transition].carrier_frequency(self.nv)
                rates += [
                    abs(carrier - self.frames[m])
                    for m in TRANSITIONS
                    if m != drive.transition
                ]
        return max(rates)


def _propagate_constant(state, H, c_ops, offsets):
    """Exact propagation under a constant H on a uniform time grid."""
    out = [state]
    if len(offsets) < 2:
        return out
    step = offsets[1] - offsets[0]
    if c_ops:
        P = (qt.liouvillian(H, c_ops) * step).expm()
        vec = qt.operator_to_vector(state)
        for _ in offsets[1:]:
            vec = P * vec
            out.append(qt.vector_to_operator(vec))
    else:
        U = (-1j * step * H).expm()
        Ud = U.dag()
        for _ in offsets[1:]:
            state = U * state if state.isket else U * state * Ud
            out.append(state)
    return out


def _is_diagonal(op: qt.Qobj) -> bool:
    m = op.full()
    return np.allclose(m, np.diag(np.diag(m)))


def _integrate(f, a: float, b: float, max_step: float) -> complex:
    """Integral of f(t) over [a, b] with 8-point Gauss-Legendre on sub-intervals."""
    n = max(1, ceil((b - a) / max_step))
    edges = np.linspace(a, b, n + 1)
    half = 0.5 * np.diff(edges)[:, None]
    x = half * _GAUSS_NODES[None, :] + 0.5 * (edges[:-1] + edges[1:])[:, None]
    try:
        y = np.broadcast_to(np.asarray(f(x)), x.shape)
    except (TypeError, ValueError):  # coefficient not vectorised
        y = np.vectorize(f)(x)
    return complex(np.sum(half * _GAUSS_WEIGHTS[None, :] * y))


def _propagate_diagonal(state, H, td, t0, offsets, max_step):
    """Exact propagation when H and all time-dependent terms are diagonal."""
    h0 = np.real(np.diag(H.full()))
    ops = [np.diag(op.full()) for op, _ in td]
    phase = np.zeros(3, dtype=complex)
    out = [state]
    for a, b in pairwise(offsets):
        phase += h0 * (b - a)
        for d, (_, f) in zip(ops, td):
            phase += d * _integrate(f, t0 + a, t0 + b, max_step)
        U = qt.Qobj(np.diag(np.exp(-1j * phase)), dims=H.dims)
        out.append(U * state if state.isket else U * state * U.dag())
    return out


def simulate(
    sequence: PulseSequence,
    nv: NVCenter,
    drives: MWChannel | Iterable[MWChannel],
    initial_state="0",
    *,
    fields: Iterable[Field] = (),
    dt: float | None = None,
    store_trajectory: bool = True,
    crosstalk: bool = False,
    extra_terms: Iterable[HamiltonianTerm] = (),
    c_ops: Iterable[qt.Qobj] = (),
    solver_options: dict | None = None,
) -> SimulationResult:
    """Simulate ``sequence`` on ``nv`` driven by the MW channels ``drives``.

    Parameters
    ----------
    sequence : PulseSequence
    nv : NVCenter
        Transition frequencies.
    drives : MWChannel or iterable of MWChannel
        At most one channel per transition; sets Rabi frequencies, detunings
        and the rotating frames.
    initial_state : str, ket or density matrix
        Defaults to |0> (optically polarised NV).
    fields : iterable of StaticField / ACField
        Magnetic fields along the NV axis (see :mod:`nvsim.fields`).
    dt : float, optional
        Output sampling interval [us]. Defaults to 1/40 of the period of the
        fastest Rabi / detuning / field frequency. Each segment gets at least
        ``MIN_SAMPLES_PER_SEGMENT`` samples regardless.
    store_trajectory : bool
        If False only the segment boundaries are stored - much faster for
        parameter sweeps where only the final state matters.
    crosstalk : bool
        Let each carrier drive the other transition too (see module docs).
    extra_terms : iterable
        Additional Hamiltonian terms in rad/us: a Qobj, or ``(Qobj, f)`` with
        ``f(t)`` a function of absolute time.
    c_ops : iterable of Qobj
        Lindblad collapse operators (hook for T1 / T2).
    solver_options : dict, optional
        Extra options for :func:`qutip.mesolve` (time-dependent segments only).
    """
    channels = channel_map(drives)
    segments = sequence.compile(channels)
    fields = tuple(fields)
    terms = [t for f in fields for t in f.hamiltonian_terms()] + list(extra_terms)
    model = _Model(nv, channels, crosstalk, terms, [f.max_rate() for f in fields])
    c_ops = list(c_ops)

    if dt is None:
        rate = model.sample_rate(segments)
        total = segments[-1].end if segments else 0.0
        dt = 1.0 / (SAMPLES_PER_CYCLE * rate) if rate > 0 else max(total, 1e-3) / 200
    xrate = model.crosstalk_rate(segments)
    max_step = min(dt, 1.0 / (SOLVER_STEPS_PER_CYCLE * xrate)) if xrate > 0 else dt
    options = {"atol": 1e-10, "rtol": 1e-8, "nsteps": 10**6, "max_step": max_step}
    options.update(solver_options or {})

    initial = as_state(initial_state)
    state = qt.ket2dm(initial) if c_ops and initial.isket else initial

    times, states = [0.0], [state]
    for segment in segments:
        H, td = model.segment_hamiltonian(segment)
        n_steps = (
            max(MIN_SAMPLES_PER_SEGMENT, ceil(segment.duration / dt - 1e-9))
            if store_trajectory
            else 1
        )
        offsets = np.linspace(0.0, segment.duration, n_steps + 1)
        if not td:
            seg_states = _propagate_constant(state, H, c_ops, offsets)
        elif not c_ops and _is_diagonal(H) and all(_is_diagonal(op) for op, _ in td):
            seg_states = _propagate_diagonal(
                state, H, td, segment.start, offsets, max_step
            )
        else:
            evo = qt.QobjEvo([H] + [[op, f] for op, f in td])
            seg_states = qt.mesolve(
                evo, state, segment.start + offsets, c_ops=c_ops, options=options
            ).states
        times.extend(segment.start + offsets[1:])
        states.extend(seg_states[1:])
        state = seg_states[-1]

    return SimulationResult(
        np.asarray(times),
        states,
        segments,
        sequence,
        nv,
        channels,
        fields=fields,
        initial_state=initial,
        dt=dt,
        store_trajectory=store_trajectory,
    )


def sweep(
    values: Iterable[float],
    sequence: PulseSequence | Callable[[float], PulseSequence],
    nv: NVCenter | Callable[[float], NVCenter],
    drives: MWChannel | Iterable[MWChannel] | Callable,
    initial_state="0",
    *,
    parameter: str = "parameter",
    **simulate_kwargs,
) -> SweepResult:
    """Run one simulation per value and collect the final populations.

    ``sequence``, ``nv``, ``drives`` and any keyword argument of
    :func:`simulate` may be a fixed object or a function of the swept value,
    e.g. ``sequence=lambda tau: ramsey(tau)``,
    ``drives=lambda d: MWChannel('-1', 10, detuning=d)`` or
    ``fields=lambda b: [StaticField(b)]``.
    """
    values = np.asarray(list(values), dtype=float)

    def resolve(arg, v):
        return arg(v) if callable(arg) else arg

    simulate_kwargs.setdefault("store_trajectory", False)
    finals = []
    for v in values:
        kwargs = {key: resolve(arg, v) for key, arg in simulate_kwargs.items()}
        result = simulate(
            resolve(sequence, v),
            resolve(nv, v),
            resolve(drives, v),
            initial_state,
            **kwargs,
        )
        finals.append(result.final_state)
    populations = {
        level: np.real(np.array([qt.expect(projector(level), s) for s in finals]))
        for level in LEVELS
    }
    return SweepResult(values, populations, parameter, finals)
