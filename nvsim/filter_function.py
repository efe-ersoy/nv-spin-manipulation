"""Filter functions and Gaussian-noise decay of pulse sequences.

During a sequence the spin picks up the phase ``phi = 2 pi gamma_e
integral y(t) B(t) dt``, where the toggling function y(t) is +-1 during free
evolution and changes sign at every pi pulse (the echo maps phi -> -phi). For
Gaussian noise with autocorrelation C(t) the coherence decays as
``exp(-<phi^2>/2)`` with

    <phi^2> = (2 pi gamma_e)^2  int int y(t) y(t') C(t - t') dt dt'
            = (2 pi gamma_e)^2  int_0^inf S(f) |Y(f)|^2 df

(S one-sided PSD, Y the Fourier transform of y). |Y(f)|^2 is the filter
function: it shows which noise frequencies a sequence picks up.

Conventions for y(t): the first pulse (a pi/2) starts the coherence, every
following pi pulse flips the sign (smoothly as cos(pi u) while the pulse is
on, u the fraction of the pulse), and the last pulse (readout) ends it. Pulses
with other rotation angles in the middle of a sequence are not supported.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .constants import GAMMA_E, TWO_PI
from .noise import max_step, total_autocorrelation
from .pulses import Segment

#: Relative tolerance for recognising a pi pulse from Rabi frequency x duration.
PI_TOLERANCE = 0.05
#: Upper limit on the time-grid size used for the phase variance.
MAX_GRID_POINTS = 400_000


@dataclass(frozen=True)
class TogglingPiece:
    """A piece of the toggling function: ``sign`` during free evolution
    (``kind='free'``) or ``sign * cos(pi u)`` during a pi pulse (``kind='pi'``)."""

    start: float
    end: float
    kind: str
    sign: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def _rotation_angle(segment: Segment) -> float:
    return sum(TWO_PI * d.rabi_frequency * segment.duration for d in segment.drives)


def toggling_pieces(segments: list[Segment]) -> list[TogglingPiece]:
    """The toggling function of a compiled sequence as a list of pieces."""
    pulses = [i for i, s in enumerate(segments) if s.is_pulse]
    if len(pulses) < 2:
        return []
    first, last = pulses[0], pulses[-1]
    pieces, sign = [], 1
    for segment in segments[first + 1 : last]:
        if not segment.is_pulse:
            pieces.append(TogglingPiece(segment.start, segment.end, "free", sign))
            continue
        angle = _rotation_angle(segment)
        if abs(angle / np.pi - 1) > PI_TOLERANCE:
            raise ValueError(
                f"Filter function: only pi pulses are supported between the first and the "
                f"last pulse, got {angle / np.pi:.2f} pi ({segment.label})."
            )
        pieces.append(TogglingPiece(segment.start, segment.end, "pi", sign))
        sign = -sign
    return pieces


def toggling_function(segments: list[Segment], t) -> np.ndarray:
    """y(t) evaluated at times ``t``."""
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    for p in toggling_pieces(segments):
        inside = (t >= p.start) & (t < p.end)
        if p.kind == "free":
            y[inside] = p.sign
        else:
            y[inside] = p.sign * np.cos(np.pi * (t[inside] - p.start) / p.duration)
    return y


def _piece_integral(p: TogglingPiece, a, b):
    """Integral of the piece over [a, b] intersected with the piece (vectorised)."""
    lo, hi = np.maximum(a, p.start), np.minimum(b, p.end)
    hi = np.maximum(hi, lo)
    if p.kind == "free":
        return p.sign * (hi - lo)
    k = np.pi / p.duration
    return p.sign * (np.sin(k * (hi - p.start)) - np.sin(k * (lo - p.start))) / k


def cell_averages(segments: list[Segment], edges: np.ndarray) -> np.ndarray:
    """Average of y(t) over each cell ``[edges[i], edges[i+1]]``."""
    a, b = edges[:-1], edges[1:]
    total = np.zeros(len(a))
    for p in toggling_pieces(segments):
        total += _piece_integral(p, a, b)
    return total / (b - a)


def filter_function(segments: list[Segment], f) -> np.ndarray:
    """|Y(f)|^2 [us^2] at frequencies ``f`` [MHz] (analytic)."""
    w = TWO_PI * np.asarray(f, dtype=float)
    Y = np.zeros_like(w, dtype=complex)
    small = np.abs(w) < 1e-12
    ws = np.where(small, 1.0, w)
    for p in toggling_pieces(segments):
        if p.kind == "free":
            term = (np.exp(-1j * ws * p.start) - np.exp(-1j * ws * p.end)) / (1j * ws)
            term = np.where(small, p.duration, term)
        else:
            k, T = np.pi / p.duration, p.duration
            denom = k**2 - w**2
            near = np.abs(denom) < 1e-9 * k**2
            safe = np.where(near, 1.0, denom)
            term = np.exp(-1j * w * p.start) * 1j * w * (1 + np.exp(-1j * w * T)) / safe
            term = np.where(near, np.exp(-1j * w * p.start) * T / 2, term)
        Y += p.sign * term
    return np.abs(Y) ** 2


def phase_variance(segments: list[Segment], sources, dt: float | None = None) -> float:
    """<phi^2> [rad^2] picked up from the Gaussian noise ``sources``."""
    pieces = toggling_pieces(segments)
    if not pieces or not sources:
        return 0.0
    t0, t1 = pieces[0].start, pieces[-1].end
    if dt is None:
        # Cell averages integrate the toggling pieces (incl. pi pulses) exactly,
        # so the grid only has to resolve the noise correlation function.
        dt = min(max_step(sources), (t1 - t0) / 2000)
    dt = max(dt, (t1 - t0) / MAX_GRID_POINTS)
    n = max(2, int(np.ceil((t1 - t0) / dt)))
    edges = np.linspace(t0, t1, n + 1)
    dt = edges[1] - edges[0]
    y = cell_averages(segments, edges)
    corr = fftconvolve(y, y[::-1])  # lags -(n-1) .. n-1
    lags = (np.arange(len(corr)) - (n - 1)) * dt
    return float((TWO_PI * GAMMA_E) ** 2 * dt**2 * np.sum(total_autocorrelation(sources, lags) * corr))


def coherence_decay(segments: list[Segment], sources, dt: float | None = None) -> float:
    """Gaussian decay factor exp(-<phi^2>/2) of the spin coherence."""
    return float(np.exp(-0.5 * phase_variance(segments, sources, dt)))
