"""Classical magnetic noise along the NV axis.

Every source is a stationary random field B(t) [mT] that shifts the spin like
a fluctuating :class:`~nvsim.fields.StaticField`. A source provides

* ``sample(t, n, rng)`` - ``n`` realisations on a uniform time grid ``t`` [us],
  shape ``(n, len(t))``; used by the Monte Carlo engine (:mod:`nvsim.ensemble`);
* ``autocorrelation(lag)`` - autocovariance ``<dB(t) dB(t + lag)>`` [mT^2] of the
  fluctuating part ``dB = B - mean``; used by the Gaussian filter-function
  theory (:mod:`nvsim.filter_function`). ``mean`` is 0 for all sources except
  :class:`TelegraphNoise`, whose mean acts as a static offset;
* ``psd(f)`` - one-sided power spectral density [mT^2/MHz] of the continuous
  part (``integral psd df = b_rms^2``), and ``lines()`` - spectral lines
  ``(frequency, variance)`` for components with zero linewidth;
* ``max_step()`` - the largest time step [us] that resolves the process;
* ``gaussian`` - True if the process is Gaussian. The filter-function theory is
  exact only for Gaussian sources and refuses the others; for telegraph noise
  use the exact results in :mod:`nvsim.telegraph`.

Composite sources (e.g. :class:`CarbonBath`) expose their parts via
``components()``; use :func:`flatten` to get the primitive sources.

Spectra are Lorentzian: an Ornstein-Uhlenbeck process with correlation time
tau_c has ``psd(f) = 4 b^2 tau_c / (1 + (2 pi f tau_c)^2)``; a narrowband
source at f0 with FWHM linewidth Gamma has the same shape shifted to +-f0, with
tau_c = 1 / (pi Gamma). Telegraph noise has the OU spectrum too, but it is not
Gaussian.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .constants import GAMMA_C13, GAMMA_E, TWO_PI, uT

#: Points per period used to resolve oscillating noise on the time grid.
STEPS_PER_PERIOD = 20


def _ar1(n: int, num: int, rho: float, sigma: float, rng, complex_: bool = False) -> np.ndarray:
    """``n`` stationary AR(1) series of length ``num`` with variance ``sigma^2``
    (per quadrature if complex) and lag-1 correlation ``rho``."""

    def normal(shape):
        if complex_:
            return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        return rng.standard_normal(shape)

    x0 = sigma * normal(n)
    if num == 1:
        return x0[:, None]
    innovations = sigma * np.sqrt(1.0 - rho**2) * normal((n, num - 1))
    rest, _ = lfilter([1.0], [1.0, -rho], innovations, axis=1, zi=(rho * x0)[:, None])
    return np.concatenate([x0[:, None], rest], axis=1)


def _grid_step(t: np.ndarray) -> float:
    return float(t[1] - t[0]) if len(t) > 1 else 0.0


class NoiseSource:
    """Base class: a primitive source is its own only component.

    ``gaussian`` and ``mean`` are class-level defaults (Gaussian, zero mean);
    sources that differ override them.
    """

    label: str = "noise"
    gaussian = True
    mean = 0.0

    def components(self) -> list["NoiseSource"]:
        return [self]

    def lines(self) -> list[tuple[float, float]]:
        return []

    def max_step(self) -> float:
        return np.inf


@dataclass(frozen=True)
class QuasiStaticNoise(NoiseSource):
    """A field that is constant during each sequence but random from shot to
    shot (Gaussian, rms ``b_rms`` [mT]). Causes Ramsey decay with
    T2* = sqrt(2) / (2 pi gamma_e b_rms); fully refocused by an echo."""

    b_rms: float
    label: str = "quasi-static"

    def sample(self, t, n, rng):
        return np.repeat(self.b_rms * rng.standard_normal((n, 1)), len(t), axis=1)

    def autocorrelation(self, lag):
        return np.full_like(np.asarray(lag, dtype=float), self.b_rms**2)

    def psd(self, f):
        return np.zeros_like(np.asarray(f, dtype=float))

    def lines(self):
        return [(0.0, self.b_rms**2)]


@dataclass(frozen=True)
class OUNoise(NoiseSource):
    """Ornstein-Uhlenbeck noise: rms ``b_rms`` [mT], correlation time
    ``correlation_time`` [us]; Lorentzian spectrum centred at zero frequency.
    A common model for a fluctuating electron/nuclear spin bath."""

    b_rms: float
    correlation_time: float
    label: str = "OU noise"

    def sample(self, t, n, rng):
        rho = np.exp(-_grid_step(t) / self.correlation_time)
        return _ar1(n, len(t), rho, self.b_rms, rng)

    def autocorrelation(self, lag):
        return self.b_rms**2 * np.exp(-np.abs(np.asarray(lag, dtype=float)) / self.correlation_time)

    def psd(self, f):
        tc = self.correlation_time
        return 4 * self.b_rms**2 * tc / (1 + (TWO_PI * np.asarray(f, dtype=float) * tc) ** 2)

    def max_step(self):
        return self.correlation_time / 5


@dataclass(frozen=True)
class NarrowbandNoise(NoiseSource):
    """Noise concentrated around ``frequency`` [MHz]: rms ``b_rms`` [mT] and a
    Lorentzian line of FWHM ``linewidth`` [MHz] (0 = a single frequency with
    random phase in every shot).

    With ``gaussian=True`` (default) the two quadratures are independent
    Ornstein-Uhlenbeck processes, so the field is Gaussian and the filter-
    function theory is exact. With ``gaussian=False`` the amplitude is fixed
    (sqrt(2) * b_rms) and only the phase is random / diffusing - a coherent
    signal of unknown phase.
    """

    b_rms: float
    frequency: float
    linewidth: float = 0.0
    gaussian: bool = True
    label: str = "narrowband noise"

    @property
    def correlation_time(self) -> float:
        return 1.0 / (np.pi * self.linewidth) if self.linewidth > 0 else np.inf

    def sample(self, t, n, rng):
        dt, tc = _grid_step(t), self.correlation_time
        if self.gaussian:
            rho = np.exp(-dt / tc) if np.isfinite(tc) else 1.0
            z = _ar1(n, len(t), rho, self.b_rms, rng, complex_=True)
        else:
            steps = np.sqrt(TWO_PI * self.linewidth * dt) * rng.standard_normal((n, len(t)))
            steps[:, 0] = TWO_PI * rng.random(n)
            z = np.sqrt(2) * self.b_rms * np.exp(1j * np.cumsum(steps, axis=1))
        return np.real(z * np.exp(1j * TWO_PI * self.frequency * np.asarray(t))[None, :])

    def autocorrelation(self, lag):
        lag = np.abs(np.asarray(lag, dtype=float))
        envelope = np.exp(-lag / self.correlation_time) if self.linewidth > 0 else 1.0
        return self.b_rms**2 * envelope * np.cos(TWO_PI * self.frequency * lag)

    def psd(self, f):
        f = np.asarray(f, dtype=float)
        if self.linewidth == 0:
            return np.zeros_like(f)
        tc = self.correlation_time

        def lorentz(x):
            return 1 / (1 + (TWO_PI * x * tc) ** 2)

        return 2 * self.b_rms**2 * tc * (lorentz(f - self.frequency) + lorentz(f + self.frequency))

    def lines(self):
        return [(self.frequency, self.b_rms**2)] if self.linewidth == 0 else []

    def max_step(self):
        step = 1.0 / (STEPS_PER_PERIOD * (abs(self.frequency) + 3 * self.linewidth))
        return min(step, self.correlation_time / 5) if np.isfinite(step) else self.correlation_time / 5


@dataclass(frozen=True)
class TelegraphNoise(NoiseSource):
    """Random telegraph noise: B(t) jumps between ``b_low`` and ``b_high`` [mT].

    A two-state Markov process, e.g. a bubble or domain-wall segment hopping
    between two pinned positions. ``rate_up`` (low -> high) and ``rate_down``
    (high -> low) are switching rates [1/us]: the mean dwell time in the low
    state is 1/rate_up.

    * occupations: p_high = rate_up / (rate_up + rate_down), p_low = 1 - p_high;
    * the mean ``p_low b_low + p_high b_high`` is a static offset (refocused by
      an echo);
    * fluctuations: amplitude Delta = b_high - b_low, correlation time
      tau_c = 1 / (rate_up + rate_down), autocovariance
      p_low p_high Delta^2 exp(-|t| / tau_c) - the same Lorentzian spectrum as
      :class:`OUNoise`, but the process is **not Gaussian**: in strong coupling
      (2 pi gamma_e Delta tau_c >> 1) the echo decays roughly as
      exp(-rate * total time) instead of the Gaussian prediction. Use
      :mod:`nvsim.telegraph` for the exact coherence.

    Sampling is exact: dwell times are drawn from exponential distributions
    and the state is read off at the grid times.
    """

    b_low: float
    b_high: float
    rate_up: float
    rate_down: float
    label: str = "telegraph noise"
    gaussian = False  # class attribute, not a field

    def __post_init__(self):
        if self.rate_up < 0 or self.rate_down < 0 or self.rate_up + self.rate_down <= 0:
            raise ValueError("TelegraphNoise needs non-negative rates with a positive sum.")

    @classmethod
    def symmetric(cls, b: float, rate: float, label: str = "telegraph noise") -> "TelegraphNoise":
        """Zero-mean telegraph noise between -b and +b, switching rate ``rate`` both ways."""
        return cls(-b, b, rate, rate, label=label)

    @property
    def p_high(self) -> float:
        return self.rate_up / (self.rate_up + self.rate_down)

    @property
    def p_low(self) -> float:
        return 1.0 - self.p_high

    @property
    def mean(self) -> float:
        return self.p_low * self.b_low + self.p_high * self.b_high

    @property
    def amplitude(self) -> float:
        """Jump size Delta = b_high - b_low [mT]."""
        return self.b_high - self.b_low

    @property
    def correlation_time(self) -> float:
        return 1.0 / (self.rate_up + self.rate_down)

    @property
    def b_rms(self) -> float:
        """rms of the fluctuating part, sqrt(p_low p_high) |Delta| [mT]."""
        return float(np.sqrt(self.p_low * self.p_high) * abs(self.amplitude))

    def switching_times(self, t0: float, t1: float, n: int, rng):
        """Initial states (True = high) at ``t0`` and the switching times in
        [t0, t1], shape (n, K) padded with inf."""
        high = rng.random(n) < self.p_high
        dwell_high = 1 / self.rate_down if self.rate_down > 0 else np.inf
        dwell_low = 1 / self.rate_up if self.rate_up > 0 else np.inf
        now, state, times = np.full(n, float(t0)), high.copy(), []
        while True:
            now = now + np.where(state, dwell_high, dwell_low) * rng.standard_exponential(n)
            if np.all(now > t1):
                break
            times.append(np.where(now <= t1, now, np.inf))
            state = ~state
        return high, (np.stack(times, axis=1) if times else np.full((n, 0), np.inf))

    def sample(self, t, n, rng):
        t = np.asarray(t, dtype=float)
        high, times = self.switching_times(t[0], t[-1], n, rng)
        counts = np.empty((n, len(t)), dtype=int)
        for r in range(n):
            counts[r] = np.searchsorted(times[r], t, side="right")
        state = high[:, None] ^ (counts % 2 == 1)
        return np.where(state, self.b_high, self.b_low)

    def autocorrelation(self, lag):
        lag = np.abs(np.asarray(lag, dtype=float))
        return self.b_rms**2 * np.exp(-lag / self.correlation_time)

    def psd(self, f):
        tc = self.correlation_time
        return 4 * self.b_rms**2 * tc / (1 + (TWO_PI * np.asarray(f, dtype=float) * tc) ** 2)

    def max_step(self):
        # Resolve the dwell times, and keep the phase error of a switch inside a
        # grid cell (trapezoidal rule) below ~0.1 rad.
        coupling = TWO_PI * GAMMA_E * abs(self.amplitude)
        return min(self.correlation_time / 20, 0.1 / coupling if coupling > 0 else np.inf)


@dataclass(frozen=True)
class CarbonBath(NoiseSource):
    """Phenomenological classical model of the 13C nuclear spin bath.

    Each 13C nucleus contributes a field ``A_zz I_z + A_zx I_x`` along the NV
    axis. The ``I_z`` part is static during a sequence (random from shot to
    shot), the ``I_x`` part precesses at the 13C Larmor frequency
    ``f_L = gamma_C13 * field``. The bath is therefore modelled as

    * quasi-static Gaussian noise with rms ``b_static`` (sets T2*), plus
    * narrowband noise at ``f_L`` with rms ``b_larmor`` and FWHM ``linewidth``,
      which produces the spin-echo collapses / revivals at multiples of 1/f_L
      and the CPMG/XY8 dips at tau + t_pi = 1/(2 f_L) (and odd harmonics).

    The defaults are typical for natural abundance (1.1 %), giving
    T2* ~ 2 us; both amplitudes scale as sqrt(abundance / 1.1 %). In reality
    the Larmor component weakens at high field (it scales with the transverse
    hyperfine coupling over the Larmor frequency) - adjust ``b_larmor`` to match
    your sample.

    Parameters
    ----------
    field : float
        Bias field along the NV axis [mT]; sets the Larmor frequency.
    """

    field: float
    b_static: float = 4 * uT
    b_larmor: float = 4 * uT
    linewidth: float = 0.002
    abundance: float = 0.011
    label: str = "13C bath"

    @classmethod
    def for_nv(cls, nv, **kwargs) -> "CarbonBath":
        """Bath at the bias field implied by ``nv``'s transition frequencies."""
        return cls(field=abs(nv.Bz), **kwargs)

    @property
    def larmor_frequency(self) -> float:
        return GAMMA_C13 * self.field

    def components(self):
        scale = np.sqrt(self.abundance / 0.011)
        return [
            QuasiStaticNoise(scale * self.b_static, label=f"{self.label}: quasi-static"),
            NarrowbandNoise(scale * self.b_larmor, self.larmor_frequency, self.linewidth,
                            label=f"{self.label}: Larmor {1e3 * self.larmor_frequency:.0f} kHz"),
        ]


def flatten(sources) -> list[NoiseSource]:
    """Primitive sources contained in ``sources`` (composites expanded)."""
    return [c for s in sources for c in s.components()]


def non_gaussian(sources) -> list[NoiseSource]:
    """The primitive sources that are not Gaussian."""
    return [s for s in flatten(sources) if not s.gaussian]


def is_gaussian(sources) -> bool:
    """True if every primitive source is Gaussian (filter-function theory exact)."""
    return not non_gaussian(sources)


def total_autocorrelation(sources, lag) -> np.ndarray:
    lag = np.asarray(lag, dtype=float)
    return sum((s.autocorrelation(lag) for s in flatten(sources)), np.zeros_like(lag))


def total_psd(sources, f) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    return sum((s.psd(f) for s in flatten(sources)), np.zeros_like(f))


def sample_total(sources, t, n: int, rng) -> np.ndarray:
    """Sum of independent realisations of all sources, shape ``(n, len(t))`` [mT]."""
    t = np.asarray(t, dtype=float)
    out = np.zeros((n, len(t)))
    for source in flatten(sources):
        out += source.sample(t, n, rng)
    return out


def max_step(sources) -> float:
    return min((s.max_step() for s in flatten(sources)), default=np.inf)
