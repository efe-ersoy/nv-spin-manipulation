"""Noise sources from data computed elsewhere: tabulated spectra and time traces.

* :class:`SpectrumNoise` - stationary Gaussian noise defined by a tabulated
  one-sided PSD (e.g. from micromagnetics or reduced models in ``magtex``).
* :class:`TraceNoise` - replays recorded / simulated time traces of B_nv(t).

Both follow the :mod:`nvsim.noise` interface and can be mixed with the other
sources. Data in the project data contract (NetCDF/xarray, SI units) is
converted with :func:`spectrum_noise_from_si`, :func:`trace_noise_from_si`,
:func:`spectrum_noise_from_dataset` and :func:`trace_noise_from_dataset`.
The dataset helpers only use ``ds[name].values``, ``.dims`` and ``.attrs``, so
they accept an ``xarray.Dataset`` without ``nvsim`` depending on xarray.

Units inside nvsim: mT, MHz, us. Conversions from SI: B [T] x 1e3,
f [Hz] x 1e-6, t [s] x 1e6, S [T^2/Hz] x 1e12 (-> mT^2/MHz).
"""

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import welch

from .noise import NoiseSource

#: Significant-weight quantile used for ``SpectrumNoise.max_step``.
SIGNIFICANT_QUANTILE = 0.95
#: Frequency grid points per period of the largest lag in the autocorrelation
#: transform (error <~ (2 pi / FREQUENCY_OVERSAMPLING)^2 / 12 ~ 0.3 % of the variance).
FREQUENCY_OVERSAMPLING = 32
_MAX_FFT = 2**23
#: Strata of the random sub-bin frequency offset used by SpectrumNoise.sample.
SUB_BIN_OFFSETS = 16


class SpectrumNoise(NoiseSource):
    """Stationary Gaussian noise with a tabulated one-sided PSD.

    Parameters
    ----------
    f : array
        Frequencies [MHz], >= 0, increasing.
    psd : array
        One-sided PSD [mT^2/MHz] at ``f`` (``integral psd df`` = variance).
    oversample : int
        Synthesis period / sequence duration for :meth:`sample`.

    The PSD between table points is linear; below the first tabulated
    frequency it is extended flat to 0 Hz, above the last it is zero.

    * ``sample``: spectral synthesis by FFT. The frequency axis is cut into bins
      of width df = 1 / (oversample x duration), each bin gets a Gaussian
      complex amplitude whose variance is the *exact* integral of the
      interpolated PSD over the bin, and the whole bin grid is shifted by a
      random sub-bin offset per realisation, stratified over 16 values (so
      narrow lines are not pulled to bin centres). The variance of the samples is exact on any grid: content
      above the grid's Nyquist frequency aliases onto the grid as it would for
      a sampled continuous process. Correlations at lag L are smeared by
      ~ sinc(L / (oversample x duration)), i.e. < 0.7 % for oversample = 16.
    * ``autocorrelation``: cosine transform of the interpolated PSD, by the
      trapezoidal rule on a uniform frequency grid (one FFT); the grid step is
      chosen so that the error at the largest requested lag is at most about
      (2 pi / 32)^2 / 12 ~ 0.3 % of the variance (much less at shorter lags;
      ~1e-6 of the variance for a smooth Lorentzian). Features
      narrower than the table spacing are of course not represented.
    * ``max_step``: 1 / (20 f_95), with f_95 the frequency below which 95 % of
      the variance lies.
    """

    gaussian = True

    def __init__(self, f, psd, label: str = "tabulated spectrum", oversample: int = 16):
        f = np.asarray(f, dtype=float).ravel()
        s = np.asarray(psd, dtype=float).ravel()
        if f.shape != s.shape or len(f) < 2:
            raise ValueError("f and psd must be 1-D arrays of the same length (>= 2).")
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(s))) or np.any(f < 0) or np.any(s < 0):
            raise ValueError("f and psd must be finite and non-negative.")
        order = np.argsort(f)
        f, s = f[order], s[order]
        if np.any(np.diff(f) <= 0):
            raise ValueError("f must not contain duplicates.")
        if f[0] > 0:  # flat extension down to 0 Hz
            f, s = np.concatenate([[0.0], f]), np.concatenate([[s[0]], s])
        self.f, self.s = f, s
        self.label = label
        self.oversample = int(oversample)
        widths = np.diff(f)
        self._slope = np.diff(s) / widths
        self._cum = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * widths)])

    def __repr__(self):
        return (f"SpectrumNoise({self.label!r}: {len(self.f)} points, f up to {self.f_max:g} MHz, "
                f"b_rms = {1e3 * self.b_rms:.3g} uT)")

    @property
    def f_max(self) -> float:
        return float(self.f[-1])

    @property
    def variance(self) -> float:
        return float(self._cum[-1])

    @property
    def b_rms(self) -> float:
        return float(np.sqrt(self.variance))

    def psd(self, f):
        return np.interp(np.asarray(f, dtype=float), self.f, self.s, right=0.0)

    def cumulative(self, f):
        """Exact integral of the interpolated PSD from 0 to ``f`` [mT^2]."""
        q = np.clip(np.asarray(f, dtype=float), 0.0, self.f_max)
        i = np.clip(np.searchsorted(self.f, q, side="right") - 1, 0, len(self.f) - 2)
        x = q - self.f[i]
        return self._cum[i] + self.s[i] * x + 0.5 * self._slope[i] * x**2

    def max_step(self):
        target = SIGNIFICANT_QUANTILE * self.variance
        if target <= 0:
            return np.inf
        f_sig = float(np.interp(target, self._cum, self.f))
        return 1.0 / (20.0 * f_sig) if f_sig > 0 else np.inf

    def autocorrelation(self, lag):
        lag = np.abs(np.asarray(lag, dtype=float))
        if lag.size == 0:
            return lag.copy()
        l_max = float(lag.max())
        if l_max == 0:
            return np.full_like(lag, self.variance)
        # Uniform lag grid: the requested spacing if the lags are uniform (exact
        # at the nodes), else fine enough to interpolate linearly.
        unique = np.unique(lag)
        steps = np.diff(unique)
        if len(steps) and np.allclose(steps, steps[0], rtol=1e-6, atol=0):
            d = float(steps[0])
        else:
            d = min(l_max / 8192, 1.0 / (16 * self.f_max))
        k_max = int(np.ceil(l_max / d - 1e-9))
        n_fft = next_fast_len(max(FREQUENCY_OVERSAMPLING * k_max, 1024))
        if n_fft > _MAX_FFT:
            n_fft = _MAX_FFT
        h = 1.0 / (n_fft * d)  # frequency step; f_j * lag_k = j k / n_fft exactly
        n_freq = int(np.ceil(self.f_max / h)) + 1
        freqs = np.arange(n_freq) * h
        weights = self.psd(freqs) * h
        weights[0] *= 0.5  # trapezoid at f = 0 (the PSD is zero beyond f_max)
        folded = np.bincount(np.arange(n_freq) % n_fft, weights=weights, minlength=n_fft)
        corr = np.real(np.fft.fft(folded))[: k_max + 1]
        return np.interp(lag, np.arange(k_max + 1) * d, corr)

    def sample(self, t, n, rng):
        t = np.asarray(t, dtype=float)
        num = len(t)
        if num == 1:
            return self.b_rms * rng.standard_normal((n, 1))
        dt = t[1] - t[0]
        n_fft = next_fast_len(max(self.oversample * num, 64))
        df = 1.0 / (n_fft * dt)
        n_bins = int(np.ceil(self.f_max / df)) + 2
        # Stratified random sub-bin offsets (one per realisation, drawn from
        # SUB_BIN_OFFSETS strata), and the exact power of every bin for each.
        offsets = (np.arange(SUB_BIN_OFFSETS) + rng.random()) / SUB_BIN_OFFSETS * df
        centres = np.arange(n_bins)[None, :] * df + offsets[:, None]
        power = self.cumulative(centres + 0.5 * df) - self.cumulative(np.clip(centres - 0.5 * df, 0.0, None))
        # Bins beyond n_fft alias onto the grid at an existing bin frequency: fold
        # their powers (independent Gaussians add up to one Gaussian).
        index = np.arange(n_bins) % n_fft
        amplitude = np.sqrt(np.stack([np.bincount(index, weights=row, minlength=n_fft) for row in power]))
        stratum = rng.permutation(np.arange(n) % SUB_BIN_OFFSETS)
        rel = t - t[0]
        out = np.empty((n, num))
        chunk = max(1, int(4e6 // n_fft))
        for begin in range(0, n, chunk):
            q = stratum[begin: begin + chunk]
            z = amplitude[q] * (rng.standard_normal((len(q), n_fft)) + 1j * rng.standard_normal((len(q), n_fft)))
            series = np.fft.ifft(z, axis=1)[:, :num] * n_fft
            out[begin: begin + len(q)] = np.real(series * np.exp(2j * np.pi * offsets[q][:, None] * rel[None, :]))
        return out


class TraceNoise(NoiseSource):
    """Replays given time traces B(t) as noise realisations.

    Parameters
    ----------
    t : array
        Uniform sample times [us] of the traces.
    traces : array
        Field [mT], shape (n_traces, len(t)) (or 1-D for a single trace).
    gaussian : bool
        Declare the traces Gaussian (then the filter-function theory may use the
        estimated autocorrelation). Default False.

    ``sample`` picks a random trace and a random (continuous) start offset for
    every realisation and interpolates linearly onto the requested grid; the
    traces must be longer than the sequence. Realisations are only independent
    as long as the traces hold many non-overlapping sequence windows.
    ``autocorrelation`` (biased estimate, averaged over traces) and ``psd``
    (Welch, one-sided, mT^2/MHz, resolution 1 / (min(n_t, 16384) dt)) are
    estimated from the traces about their
    common mean, for plots and diagnostics. The mean is exposed as ``mean``
    (a static offset).
    """

    def __init__(self, t, traces, gaussian: bool = False, label: str = "recorded traces"):
        t = np.asarray(t, dtype=float).ravel()
        traces = np.atleast_2d(np.asarray(traces, dtype=float))
        if traces.shape[1] != len(t) or len(t) < 2:
            raise ValueError("traces must have shape (n_traces, len(t)) with len(t) >= 2.")
        steps = np.diff(t)
        if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-6):
            raise ValueError("t must be uniform and increasing.")
        self.t = t - t[0]
        self.dt = float(steps[0])
        self.traces = traces
        self.gaussian = bool(gaussian)
        self.label = label
        self.mean = float(traces.mean())
        fluct = traces - self.mean
        self.b_rms = float(np.sqrt(np.mean(fluct**2)))
        nperseg = min(len(t), 16384)  # Welch resolution 1 / (nperseg dt)
        self._welch_f, psd = welch(fluct, fs=1.0 / self.dt, nperseg=nperseg, detrend=False, axis=-1)
        self._welch_psd = psd.mean(axis=0)
        n_fft = next_fast_len(2 * len(t))
        spec = np.fft.rfft(fluct, n=n_fft, axis=-1)
        self._acov = np.fft.irfft(np.abs(spec) ** 2, n=n_fft, axis=-1)[:, : len(t)].mean(axis=0) / len(t)

    def __repr__(self):
        return (f"TraceNoise({self.label!r}: {self.traces.shape[0]} traces x {self.duration:g} us, "
                f"dt = {self.dt:g} us, mean = {1e3 * self.mean:.3g} uT, rms = {1e3 * self.b_rms:.3g} uT, "
                f"gaussian={self.gaussian})")

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def sample(self, t, n, rng):
        t = np.asarray(t, dtype=float)
        rel = t - t[0]
        need = float(rel[-1])
        if need > self.duration:
            raise ValueError(f"TraceNoise: sequence ({need:g} us) is longer than the traces ({self.duration:g} us).")
        which = rng.integers(self.traces.shape[0], size=n)
        offsets = rng.random(n) * (self.duration - need)
        pos = (offsets[:, None] + rel[None, :]) / self.dt
        i0 = np.clip(np.floor(pos).astype(int), 0, len(self.t) - 2)
        w = pos - i0
        rows = which[:, None]
        return self.traces[rows, i0] * (1 - w) + self.traces[rows, i0 + 1] * w

    def autocorrelation(self, lag):
        return np.interp(np.abs(np.asarray(lag, dtype=float)), self.t, self._acov, right=0.0)

    def psd(self, f):
        return np.interp(np.asarray(f, dtype=float), self._welch_f, self._welch_psd, right=0.0)

    def max_step(self):
        return self.dt


# --------------------------------------------------------------------------- data contract (SI)

_PSD_UNITS = {"T^2/Hz", "T**2/Hz", "T2/Hz", "T²/Hz", "T^2 Hz^-1", "T2 Hz-1"}
_FIELD_UNITS = {"T", "tesla", "Tesla"}
_FREQ_UNITS = {"Hz", "hertz"}
_TIME_UNITS = {"s", "second", "seconds"}


def spectrum_noise_from_si(f_hz, psd_t2_per_hz, **kwargs) -> SpectrumNoise:
    """:class:`SpectrumNoise` from f [Hz] and a one-sided PSD [T^2/Hz]."""
    return SpectrumNoise(np.asarray(f_hz, dtype=float) * 1e-6, np.asarray(psd_t2_per_hz, dtype=float) * 1e12,
                         **kwargs)


def trace_noise_from_si(t_s, b_tesla, **kwargs) -> TraceNoise:
    """:class:`TraceNoise` from t [s] and traces B_nv [T] (n_traces x n_t)."""
    return TraceNoise(np.asarray(t_s, dtype=float) * 1e6, np.asarray(b_tesla, dtype=float) * 1e3, **kwargs)


def _check_units(obj, allowed: set, name: str):
    units = getattr(obj, "attrs", {}).get("units")
    if units is not None and str(units).strip() not in allowed:
        raise ValueError(f"{name}: expected units {sorted(allowed)[0]!r}, got {units!r}.")


def _pixel(ds, var: str, x, y):
    """Values of ``ds[var]`` at the pixel nearest to (x, y) [m], and the remaining dims."""
    da = ds[var]
    values = np.asarray(da.values, dtype=float)
    dims = list(da.dims)
    for name, target in (("x", x), ("y", y)):
        if name not in dims:
            continue
        axis = dims.index(name)
        coord = np.asarray(ds[name].values, dtype=float)
        if target is None:
            if values.shape[axis] != 1:
                raise ValueError(f"{var} has {values.shape[axis]} {name} values: pass {name}= [m].")
            index = 0
        else:
            index = int(np.argmin(np.abs(coord - target)))
        values = np.take(values, index, axis=axis)
        dims.pop(axis)
    return values, dims, da


def spectrum_noise_from_dataset(ds, x=None, y=None, var: str = "S_nv", label: str | None = None,
                                **kwargs) -> SpectrumNoise:
    """:class:`SpectrumNoise` from a field-map dataset (data contract section 2):
    ``S_nv`` [T^2/Hz] with dims (f, y, x), at the pixel nearest to (x, y) [m]."""
    values, dims, da = _pixel(ds, var, x, y)
    if dims != ["f"]:
        raise ValueError(f"{var} must have dims (f[, y, x]), got remaining dims {dims}.")
    _check_units(da, _PSD_UNITS, var)
    _check_units(ds["f"], _FREQ_UNITS, "f")
    label = label or (f"{var} at ({x:g}, {y:g}) m" if x is not None and y is not None else var)
    return spectrum_noise_from_si(ds["f"].values, values, label=label, **kwargs)


def trace_noise_from_dataset(ds, x=None, y=None, var: str = "B_nv", label: str | None = None,
                             **kwargs) -> TraceNoise:
    """:class:`TraceNoise` from a field-map dataset with a time series ``B_nv`` [T]
    (dims (t, y, x), optionally one more dim enumerating independent runs), at the
    pixel nearest to (x, y) [m]."""
    values, dims, da = _pixel(ds, var, x, y)
    if "t" not in dims or len(dims) > 2:
        raise ValueError(f"{var} must have dims (t[, run][, y, x]), got remaining dims {dims}.")
    _check_units(da, _FIELD_UNITS, var)
    _check_units(ds["t"], _TIME_UNITS, "t")
    traces = np.moveaxis(values, dims.index("t"), -1)
    label = label or (f"{var} traces at ({x:g}, {y:g}) m" if x is not None and y is not None else f"{var} traces")
    return trace_noise_from_si(ds["t"].values, np.atleast_2d(traces), label=label, **kwargs)
