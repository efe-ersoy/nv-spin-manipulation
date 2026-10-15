"""Pulse-sequence primitives.

A :class:`PulseSequence` is an ordered list of elements (:class:`Pulse`,
:class:`Wait`). Compiling it against the microwave channels turns it into a
list of :class:`Segment` objects - time intervals during which the drive
configuration is constant - which is what the simulator propagates.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .constants import TWO_PI
from .drive import MWChannel
from .system import normalize_transition

_PHASE_ALIASES = {
    "x": 0.0,
    "+x": 0.0,
    "y": 0.5 * np.pi,
    "+y": 0.5 * np.pi,
    "-x": np.pi,
    "-y": 1.5 * np.pi,
}


def parse_phase(phase) -> float:
    """Pulse phase in radians; accepts a number or one of 'x', 'y', '-x', '-y'."""
    if isinstance(phase, str):
        try:
            return _PHASE_ALIASES[phase.strip().lower()]
        except KeyError:
            raise ValueError(
                f"Unknown phase {phase!r}; use a number or x, y, -x, -y."
            ) from None
    return float(phase)


def phase_label(phase: float) -> str:
    for name in ("x", "y", "-x", "-y"):
        if np.isclose(phase, _PHASE_ALIASES[name]):
            return name
    return f"{np.degrees(phase):.0f}°"


def _angle_label(angle: float) -> str:
    frac = angle / np.pi
    for value, text in (
        (1.0, "π"),
        (0.5, "π/2"),
        (2.0, "2π"),
        (0.25, "π/4"),
        (1.5, "3π/2"),
    ):
        if np.isclose(frac, value):
            return text
    return f"{frac:.3g}π"


@dataclass(frozen=True)
class Pulse:
    """A square microwave pulse on one transition.

    Give the pulse either a rotation ``angle`` or a ``duration``.

    Parameters
    ----------
    angle : float, optional
        Nominal rotation angle [rad] on resonance, e.g. ``np.pi`` for a pi
        pulse. The duration is ``angle / (2*pi*Omega)``. With detuning the
        actual rotation is about a tilted axis, as in an experiment.
    transition : str
        '-1' or '+1'.
    phase : float or str
        Rotation-axis phase in the rotating frame [rad] or 'x', 'y', '-x', '-y'.
        Phase 0 rotates about +x, pi/2 about +y.
    duration : float, optional
        Pulse length [us].
    rabi_frequency : float, optional
        Override the channel's Rabi frequency [MHz] for this pulse only.
    label : str, optional
        Name shown in plots and animations.
    """

    angle: float | None = None
    transition: str = "-1"
    phase: float | str = 0.0
    duration: float | None = None
    rabi_frequency: float | None = None
    label: str | None = None

    def __post_init__(self):
        if (self.angle is None) == (self.duration is None):
            raise ValueError("Give a Pulse exactly one of `angle` or `duration`.")
        if self.duration is not None and self.duration < 0:
            raise ValueError("Pulse duration must be non-negative.")
        object.__setattr__(self, "transition", normalize_transition(self.transition))
        phase = parse_phase(self.phase)
        if self.angle is not None and self.angle < 0:
            # A negative rotation is a positive rotation about the opposite axis.
            object.__setattr__(self, "angle", -self.angle)
            phase += np.pi
        object.__setattr__(self, "phase", phase % (2 * np.pi))

    def rabi_for(self, channels: Mapping[str, MWChannel]) -> float:
        """Rabi frequency [MHz] this pulse is played with."""
        if self.transition not in channels:
            raise KeyError(f"No MWChannel defined for transition {self.transition!r}.")
        if self.rabi_frequency is not None:
            return self.rabi_frequency
        return channels[self.transition].rabi_frequency

    def duration_for(self, channels: Mapping[str, MWChannel]) -> float:
        """Pulse length [us] given the channel's Rabi frequency."""
        rabi = self.rabi_for(channels)
        if self.duration is not None:
            return self.duration
        if rabi == 0:
            raise ValueError(
                "Cannot convert a rotation angle to a duration at zero Rabi frequency."
            )
        return self.angle / (TWO_PI * rabi)

    def describe(self) -> str:
        if self.label:
            return self.label
        size = (
            _angle_label(self.angle)
            if self.angle is not None
            else f"{self.duration:.3g} µs"
        )
        return f"{size}_{phase_label(self.phase)} ({self.transition})"


@dataclass(frozen=True)
class Wait:
    """Free evolution (no microwave) for ``duration`` microseconds."""

    duration: float
    label: str = "free evolution"

    def __post_init__(self):
        if self.duration < 0:
            raise ValueError("Wait duration must be non-negative.")

    def describe(self) -> str:
        return self.label


@dataclass(frozen=True)
class DriveTerm:
    """A drive that is on during a segment."""

    transition: str
    rabi_frequency: float
    phase: float


@dataclass(frozen=True)
class Segment:
    """Time interval with a constant drive configuration."""

    start: float
    duration: float
    drives: tuple[DriveTerm, ...]
    label: str

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def is_pulse(self) -> bool:
        return bool(self.drives)


class PulseSequence:
    """An ordered list of pulses and waits.

    Sequences can be concatenated with ``+`` and extended with :meth:`append`.
    """

    def __init__(self, elements: Iterable[Pulse | Wait] = (), name: str = "sequence"):
        self.elements: list[Pulse | Wait] = list(elements)
        self.name = name

    def append(self, element: Pulse | Wait) -> "PulseSequence":
        self.elements.append(element)
        return self

    def __add__(self, other: "PulseSequence") -> "PulseSequence":
        return PulseSequence(
            self.elements + list(other.elements), name=f"{self.name} + {other.name}"
        )

    def __iter__(self):
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __repr__(self) -> str:
        body = ", ".join(e.describe() for e in self.elements)
        return f"PulseSequence({self.name!r}: {body})"

    def compile(self, channels: Mapping[str, MWChannel]) -> list[Segment]:
        """Lay the elements out in time; zero-length elements are dropped."""
        segments, t = [], 0.0
        for element in self.elements:
            if isinstance(element, Pulse):
                duration = element.duration_for(channels)
                drives = (
                    DriveTerm(
                        element.transition, element.rabi_for(channels), element.phase
                    ),
                )
            elif isinstance(element, Wait):
                duration, drives = element.duration, ()
            else:
                raise TypeError(f"Unsupported sequence element {element!r}.")
            if duration > 0:
                segments.append(Segment(t, duration, drives, element.describe()))
                t += duration
        return segments

    def total_duration(self, channels: Mapping[str, MWChannel]) -> float:
        segments = self.compile(channels)
        return segments[-1].end if segments else 0.0
