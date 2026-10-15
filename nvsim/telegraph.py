"""Exact spin coherence under random telegraph noise (RTN).

Telegraph noise is not Gaussian, so the filter-function result
exp(-<phi^2>/2) does not apply (except in the weak-coupling, motional-
narrowing limit). The coherence C = <exp(i phi)> can instead be computed
exactly from the two-state Markov process (stochastic Liouville equation):

    dw/dt = (G + i y(t) V) w,   w(0) = (p_low, p_high),   C = w_low(T) + w_high(T)

with G the switching generator, V = 2 pi gamma_e diag(b_low, b_high) and y(t)
the toggling function of the sequence (:mod:`nvsim.filter_function`).
:func:`telegraph_coherence` does this for any :class:`~nvsim.noise.TelegraphNoise`
and any compiled sequence.

For symmetric, zero-mean RTN (levels +-b, switching rate gamma both ways,
coupling v = 2 pi gamma_e b, mu = sqrt(gamma^2 - v^2)) and instantaneous
pulses there are closed forms (Bergli, Galperin & Altshuler, New J. Phys. 11,
025002 (2009)):

    FID (Ramsey), time t:  C = e^{-gamma t} [cosh(mu t) + (gamma/mu) sinh(mu t)]
    Hahn echo, 2 x tau:    C = e^{-2 gamma tau} [1 + (gamma/mu) sinh(2 mu tau)
                                                 + (gamma/mu)^2 (cosh(2 mu tau) - 1)]

Weak coupling (v << gamma) gives the Gaussian (motional-narrowing) result;
strong coupling (v >> gamma) gives C ~ e^{-gamma t_total} for the echo -
every switch during the sequence fully dephases that shot.
:func:`rtn_cpmg_coherence` extends this to CPMG-N by 2 x 2 transfer matrices.
"""

import numpy as np
from scipy.linalg import expm

from .constants import GAMMA_E, TWO_PI
from .filter_function import toggling_pieces
from .noise import TelegraphNoise
from .pulses import Segment
from .system import normalize_transition


def _coupling(transition) -> float:
    """Phase rate per field [rad/(us mT)]; the +1 transition precesses the other way."""
    return TWO_PI * GAMMA_E * (1.0 if normalize_transition(transition) == "-1" else -1.0)


def telegraph_coherence(segments: list[Segment], source: TelegraphNoise, transition="-1",
                        pulse_steps: int = 16) -> complex:
    """Exact <exp(i phi)> from a telegraph source over a compiled sequence.

    Works for any levels and rates (asymmetric, non-zero mean). Free evolution
    is exact; during pi pulses the toggling function ``sign * cos(pi u)`` is
    applied in ``pulse_steps`` piecewise-constant steps (the convention of
    :mod:`nvsim.filter_function`). The real part is the coherence factor that
    multiplies the signal contrast; a non-zero imaginary part comes from the
    mean field (a static offset).
    """
    if not isinstance(source, TelegraphNoise):
        raise TypeError("telegraph_coherence needs a TelegraphNoise source.")
    up, down = source.rate_up, source.rate_down
    generator = np.array([[-up, down], [up, -down]], dtype=complex)
    v = _coupling(transition) * np.diag([source.b_low, source.b_high]).astype(complex)
    w = np.array([source.p_low, source.p_high], dtype=complex)
    for piece in toggling_pieces(segments):
        if piece.kind == "free":
            w = expm((generator + 1j * piece.sign * v) * piece.duration) @ w
            continue
        h = piece.duration / pulse_steps
        for k in range(pulse_steps):
            y = piece.sign * np.cos(np.pi * (k + 0.5) / pulse_steps)
            w = expm((generator + 1j * y * v) * h) @ w
    return complex(w.sum())


# --------------------------------------------------------------------------- symmetric closed forms


def _mu(b, rate):
    v = TWO_PI * GAMMA_E * np.asarray(b, dtype=float)
    return v, np.sqrt(np.asarray(rate, dtype=float) ** 2 - v**2 + 0j)


def _shc(mu, t):
    """sinh(mu t) / mu, continuous at mu = 0."""
    t = np.asarray(t, dtype=float)
    small = np.abs(mu) < 1e-12
    safe = np.where(small, 1.0, mu)
    return np.where(small, t + 0j, np.sinh(safe * t) / safe)


def rtn_fid_coherence(t, b: float, rate: float) -> np.ndarray:
    """Ramsey (free-induction) coherence after free evolution ``t`` [us] under
    symmetric RTN between -b and +b [mT] with switching rate ``rate`` [1/us]."""
    _, mu = _mu(b, rate)
    t = np.asarray(t, dtype=float)
    return np.real(np.exp(-rate * t) * (np.cosh(mu * t) + rate * _shc(mu, t)))


def rtn_echo_coherence(tau, b: float, rate: float) -> np.ndarray:
    """Hahn-echo coherence, each half ``tau`` [us] (as :func:`nvsim.spin_echo`),
    symmetric RTN between -b and +b [mT], switching rate ``rate`` [1/us]."""
    _, mu = _mu(b, rate)
    tau = np.asarray(tau, dtype=float)
    shc1, shc2 = _shc(mu, tau), _shc(mu, 2 * tau)
    return np.real(np.exp(-2 * rate * tau) * (1 + rate * shc2 + 2 * rate**2 * shc1**2))


def _propagator(v, rate, t, sign):
    """2 x 2 propagator in the (sum, difference) basis for toggling sign +-1, shape (..., 2, 2)."""
    mu = np.sqrt(rate**2 - v**2 + 0j)
    t = np.asarray(t, dtype=float)
    shc = _shc(mu, t)
    decay = np.exp(-rate * t)
    out = np.empty(t.shape + (2, 2), dtype=complex)
    out[..., 0, 0] = decay * (np.cosh(mu * t) + rate * shc)
    out[..., 1, 1] = decay * (np.cosh(mu * t) - rate * shc)
    out[..., 0, 1] = out[..., 1, 0] = decay * 1j * sign * v * shc
    return out


def rtn_cpmg_coherence(tau, n_pulses: int, b: float, rate: float) -> np.ndarray:
    """CPMG-N coherence with instantaneous pi pulses, pulse spacing ``tau`` [us]
    (tau/2 - pi - tau - ... - pi - tau/2, as :func:`nvsim.cpmg`), symmetric RTN
    between -b and +b [mT] with switching rate ``rate`` [1/us].

    The same result holds for XY4 / XY8 with ideal pulses (they share the
    toggling function). CPMG-1 with spacing 2 tau is the Hahn echo.
    """
    v = TWO_PI * GAMMA_E * float(b)
    tau = np.asarray(tau, dtype=float)
    w = np.zeros(tau.shape + (2,), dtype=complex)
    w[..., 0] = 1.0  # stationary start: sum = 1, difference = 0
    windows = [(tau / 2, 1)] + [(tau, (-1) ** k) for k in range(1, n_pulses)] + [(tau / 2, (-1) ** n_pulses)]
    for length, sign in windows:
        w = np.einsum("...ij,...j->...i", _propagator(v, rate, length, sign), w)
    return np.real(w[..., 0])
