"""Microwave channels: one phase-coherent carrier per NV transition."""

from dataclasses import dataclass

from .system import NVCenter, normalize_transition


@dataclass(frozen=True)
class MWChannel:
    """A microwave source addressing one transition |0> <-> |m>.

    The channel's carrier sets the rotating frame of its transition, so every
    pulse played on it is phase-coherent with the others, and during free
    evolution the spin precesses at the detuning.

    Parameters
    ----------
    transition : str
        '-1' or '+1': the transition this source is tuned to.
    rabi_frequency : float
        Resonant Rabi frequency Omega [MHz]. A resonant pulse of duration t
        rotates the spin by 2*pi*Omega*t, so a pi pulse lasts 1 / (2*Omega).
    detuning : float, optional
        Carrier frequency minus transition frequency [MHz].
    frequency : float, optional
        Absolute carrier frequency [MHz]. Give either ``detuning`` or
        ``frequency``; with neither, the channel is resonant.
    """

    transition: str = "-1"
    rabi_frequency: float = 10.0
    detuning: float | None = None
    frequency: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "transition", normalize_transition(self.transition))
        if self.detuning is not None and self.frequency is not None:
            raise ValueError("Specify either `detuning` or `frequency`, not both.")
        if self.rabi_frequency < 0:
            raise ValueError("`rabi_frequency` must be non-negative.")

    def detuning_for(self, nv: NVCenter) -> float:
        """Detuning [MHz] of the carrier from this channel's transition."""
        if self.frequency is not None:
            return self.frequency - nv.transition_frequency(self.transition)
        return 0.0 if self.detuning is None else self.detuning

    def carrier_frequency(self, nv: NVCenter) -> float:
        """Absolute carrier frequency [MHz]."""
        return nv.transition_frequency(self.transition) + self.detuning_for(nv)
