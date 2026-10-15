"""Containers for simulation output."""

from dataclasses import dataclass, field

import numpy as np
import qutip as qt

from .drive import MWChannel
from .pulses import PulseSequence, Segment
from .system import (
    LEVELS,
    NVCenter,
    normalize_level,
    normalize_transition,
    projector,
    qubit_operators,
)


@dataclass
class SimulationResult:
    """Time evolution of the NV spin through a pulse sequence.

    Attributes
    ----------
    times : ndarray
        Sample times [us], starting at 0.
    states : list of Qobj
        State (ket, or density matrix if collapse operators were used) at each
        time, in the rotating frame of the microwave carriers.
    segments : list of Segment
        The compiled sequence: when each pulse / wait starts and ends.
    fields : tuple
        The magnetic fields the spin was exposed to.
    """

    times: np.ndarray
    states: list[qt.Qobj]
    segments: list[Segment]
    sequence: PulseSequence
    nv: NVCenter
    channels: dict[str, MWChannel]
    fields: tuple = ()
    initial_state: qt.Qobj | None = None
    dt: float | None = None
    store_trajectory: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def final_state(self) -> qt.Qobj:
        return self.states[-1]

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def population(self, level) -> np.ndarray:
        """Population of the m_s level ``level`` at every sample time."""
        level = normalize_level(level)
        key = ("pop", level)
        if key not in self._cache:
            self._cache[key] = np.real(qt.expect(projector(level), self.states))
        return self._cache[key]

    @property
    def populations(self) -> dict[str, np.ndarray]:
        """Populations of all three levels, keyed '+1', '0', '-1'."""
        return {level: self.population(level) for level in LEVELS}

    @property
    def final_populations(self) -> dict[str, float]:
        return {level: float(p[-1]) for level, p in self.populations.items()}

    def bloch_vector(self, transition="-1") -> np.ndarray:
        """Bloch vector (x, y, z) of the {|0>, |m>} subspace, shape (N, 3).

        |0> is the north pole. The vector length drops below 1 when population
        leaves the subspace or the state is mixed.
        """
        transition = normalize_transition(transition)
        key = ("bloch", transition)
        if key not in self._cache:
            ops = qubit_operators(transition)
            self._cache[key] = np.real(
                np.column_stack([qt.expect(op, self.states) for op in ops])
            )
        return self._cache[key]

    def accumulated_phase(self, transition="-1") -> np.ndarray:
        """Phase picked up relative to a field-free, resonant reference spin.

        See :func:`nvsim.phase.accumulated_phase`; NaN outside free evolution.
        """
        from .phase import accumulated_phase

        transition = normalize_transition(transition)
        key = ("phase", transition)
        if key not in self._cache:
            self._cache[key] = accumulated_phase(self, transition)
        return self._cache[key]

    def segment_at(self, t: float) -> Segment | None:
        """The segment that produced the state at time ``t``.

        A sample at a boundary belongs to the segment that just ended; t = 0
        maps to the first segment. None outside the sequence.
        """
        if self.segments and t <= self.segments[0].start:
            return self.segments[0]
        for segment in self.segments:
            if segment.start < t <= segment.end:
                return segment
        return None


@dataclass
class SweepResult:
    """Final-state populations as a function of one swept parameter.

    For Monte Carlo sweeps ``uncertainty`` holds the standard error of each
    population (same keys); it is None for deterministic sweeps.
    """

    values: np.ndarray
    populations: dict[str, np.ndarray]
    parameter: str = "parameter"
    final_states: list[qt.Qobj] = field(default_factory=list, repr=False)
    uncertainty: dict[str, np.ndarray] | None = None

    def population(self, level) -> np.ndarray:
        return self.populations[normalize_level(level)]

    def population_error(self, level) -> np.ndarray:
        """Standard error of the population (zeros for deterministic sweeps)."""
        if self.uncertainty is None:
            return np.zeros_like(self.values)
        return self.uncertainty[normalize_level(level)]
