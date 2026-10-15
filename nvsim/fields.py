"""Magnetic fields acting on the NV spin during a sequence.

Only the component along the NV axis is modelled. It couples as
``2*pi*gamma*B(t)*Sz``, which commutes with the rotating frames, so it simply
shifts the |0> <-> |-1> transition by -gamma*B and |0> <-> |+1> by +gamma*B.
On the Bloch sphere of the -1 transition a positive field makes the spin
precess by +2*pi*gamma*B per microsecond about +z (opposite sign for +1).
Transverse components only cause small off-resonant shifts and are neglected.

These fields come *on top of* the bias field already contained in the
transition frequencies of :class:`~nvsim.system.NVCenter`.
"""

from dataclasses import dataclass

import numpy as np

from .constants import GAMMA_E, TWO_PI
from .system import spin_operators


@dataclass(frozen=True)
class StaticField:
    """A constant field ``Bz`` [mT] along the NV axis (e.g. a DC field to sense)."""

    Bz: float

    def value(self, t):
        """Field [mT] at time(s) ``t`` [us]."""
        return np.full_like(np.asarray(t, dtype=float), self.Bz)

    def hamiltonian_terms(self) -> list:
        return [TWO_PI * GAMMA_E * self.Bz * spin_operators()[2]]

    def max_rate(self) -> float:
        """Fastest frequency [MHz] this field introduces."""
        return GAMMA_E * abs(self.Bz)


@dataclass(frozen=True)
class ACField:
    """An oscillating field ``B(t) = amplitude * cos(2*pi*frequency*t + phase)``.

    ``t`` is measured from the start of the sequence, so ``phase`` fixes the
    field's timing relative to the pulses. ``amplitude`` in mT, ``frequency``
    in MHz, ``phase`` in rad.
    """

    amplitude: float
    frequency: float
    phase: float = 0.0

    def value(self, t):
        return self.amplitude * np.cos(
            TWO_PI * self.frequency * np.asarray(t, dtype=float) + self.phase
        )

    def hamiltonian_terms(self) -> list:
        w, p = TWO_PI * self.frequency, self.phase
        op = TWO_PI * GAMMA_E * self.amplitude * spin_operators()[2]
        return [(op, lambda t: np.cos(w * t + p))]

    def max_rate(self) -> float:
        return max(abs(self.frequency), GAMMA_E * abs(self.amplitude))

    @classmethod
    def sine_centered_at(
        cls, amplitude: float, frequency: float, t_center: float
    ) -> "ACField":
        """Field ``-amplitude * sin(2*pi*f*(t - t_center))``: positive just before
        ``t_center`` and changing sign there. With ``t_center`` at the echo pulse
        and ``frequency = 1 / (2 * echo half-period)`` this is the field a spin
        echo is most sensitive to."""
        return cls(
            amplitude, frequency, phase=np.pi / 2 - TWO_PI * frequency * t_center
        )


Field = StaticField | ACField


def total_field(fields, t):
    """Sum of the axial fields [mT] at time(s) ``t`` [us]."""
    t = np.asarray(t, dtype=float)
    return sum((f.value(t) for f in fields), np.zeros_like(t))
