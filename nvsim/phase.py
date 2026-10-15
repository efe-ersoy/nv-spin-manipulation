"""Accumulated phase of the spin during free evolution.

The phase is measured against a *reference spin*: the same pulse sequence
simulated with resonant carriers and no fields. During free evolution both
spins lie in the equatorial plane; the accumulated phase phi is the signed
angle about +z from the reference to the actual spin, wrapped to (-pi, pi].
Any detuning or field along the NV axis makes phi grow (for the -1 transition
d(phi)/dt = 2*pi*(Delta + gamma*B)).

A pi pulse about an equatorial axis mirrors the equatorial plane:
R(pi) R_z(phi) = R_z(-phi) R(pi). The reference is flipped along with the spin,
so the phase directly after an echo pulse is -phi. Under a static field the
second free evolution then winds it back to 0 (refocusing); under an AC field
whose sign flips at the pulse it keeps growing (AC sensing).
"""

from dataclasses import dataclass

import numpy as np

from .drive import MWChannel
from .pulses import Segment
from .results import SimulationResult
from .system import normalize_transition


def reference_result(result: SimulationResult) -> SimulationResult:
    """The field-free, resonant counterpart of ``result`` on the same time grid."""
    if "reference" not in result._cache:
        from .simulate import simulate

        ideal = [MWChannel(m, ch.rabi_frequency) for m, ch in result.channels.items()]
        ref = simulate(
            result.sequence,
            result.nv,
            ideal,
            result.initial_state,
            dt=result.dt,
            store_trajectory=result.store_trajectory,
        )
        if len(ref.times) != len(result.times) or not np.allclose(
            ref.times, result.times
        ):
            raise RuntimeError("Reference simulation is not on the same time grid.")
        result._cache["reference"] = ref
    return result._cache["reference"]


def free_evolution_mask(result: SimulationResult) -> np.ndarray:
    """True for samples inside (or on the edges of) a free-evolution segment."""
    times = result.times
    eps = 1e-9 * max(result.duration, 1.0)
    mask = np.zeros(len(times), dtype=bool)
    for segment in result.segments:
        if not segment.is_pulse:
            mask |= (times >= segment.start - eps) & (times <= segment.end + eps)
    return mask


def accumulated_phase(
    result: SimulationResult, transition="-1", min_coherence: float = 0.5
) -> np.ndarray:
    """Accumulated phase [rad] in (-pi, pi] at each sample; NaN during pulses,
    and wherever the reference spin is not in the equatorial plane (e.g. before
    the first pi/2 pulse)."""
    transition = normalize_transition(transition)
    r = reference_result(result).bloch_vector(transition)
    v = result.bloch_vector(transition)
    phi = np.arctan2(
        r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0], r[:, 0] * v[:, 0] + r[:, 1] * v[:, 1]
    )
    valid = free_evolution_mask(result) & (np.hypot(r[:, 0], r[:, 1]) > min_coherence)
    return np.where(valid, phi, np.nan)


@dataclass(frozen=True)
class PhaseJump:
    """Phase directly before and after a pulse that sits between two free evolutions."""

    segment: Segment
    t_before: float
    phase_before: float
    t_after: float
    phase_after: float


def phase_jumps(result: SimulationResult, transition="-1") -> list[PhaseJump]:
    """The phase jumps across refocusing pulses (phi -> -phi for a pi pulse)."""
    phi = result.accumulated_phase(transition)
    jumps = []
    for segment in result.segments:
        if not segment.is_pulse:
            continue
        i = int(np.argmin(np.abs(result.times - segment.start)))
        j = int(np.argmin(np.abs(result.times - segment.end)))
        if np.isfinite(phi[i]) and np.isfinite(phi[j]):
            jumps.append(
                PhaseJump(segment, result.times[i], phi[i], result.times[j], phi[j])
            )
    return jumps
