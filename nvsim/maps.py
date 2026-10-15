"""Echo / DD signal maps: the measured quantity for every pixel of a scan at once.

One pulse sequence (e.g. ``spin_echo(tau)`` with the experiment's pulses) is
evaluated for many pixels that differ only in their noise and fields:

* :class:`PixelSource` - one noise-source type with per-pixel parameters, e.g.
  ``PixelSource(OUNoise, b_rms=b_map, correlation_time=tc_map)`` or
  ``PixelSource(TelegraphNoise.symmetric, b=b_map, rate=0.3)``. Array
  parameters are per pixel (all arrays broadcast to the map shape), anything
  else is passed unchanged.
* :class:`PixelSpectrum` - Gaussian noise with per-pixel one-sided PSDs on a
  common frequency grid, ``psd`` of shape (n_f, *map) [mT^2/MHz] (e.g. ``S_nv``
  of the data contract after SI conversion), plus optional spectral lines.
* :class:`PixelField` - deterministic per-pixel fields (``StaticField``,
  ``ACField``). ``random_phase=True`` makes the AC phase uniformly random from
  shot to shot (e.g. a 32 kHz tip oscillation not locked to the sequence).
* plain noise sources / fields: the same in every pixel.

Theory path (:func:`coherence_map`, :func:`signal_map`)
--------------------------------------------------------
Independent contributions multiply (:mod:`nvsim.coherence`):

* Gaussian noise: exp(-<phi^2>/2) with
  ``<phi^2> = (2 pi gamma_e dt)^2 sum_k w_k C_pixel(k dt) R_y(k)``. The
  toggling-function correlation R_y is the same for all pixels, so the map is
  one matrix-vector product: per-pixel autocorrelations x shared weights. For
  per-pixel PSDs the phase variance is linear in the PSD values, so
  ``<phi^2> = S_pixel . W`` with ``W_j = integral hat_j(f) |Y(f)|^2 df`` computed
  once (through the same time-domain sum); zero-width lines add
  ``variance |Y(f0)|^2``. Same numerics as :func:`nvsim.filter_function.phase_variance`.
* telegraph sources: the exact two-state transfer matrix, batched over pixels;
* deterministic fields: a phase factor exp(i phi), phi = 2 pi gamma_e
  integral y(t) B(t) dt from the complex toggling transform Y(f); with a
  random AC phase the average is J0(2 pi gamma_e A |Y(f)|) (exact). Fixed-
  amplitude, zero-linewidth :class:`~nvsim.noise.NarrowbandNoise` is the same.

Deterministic fields enter in this linear-response (toggling-frame) form; their
effect on the pulses themselves (detuning during a pulse) is neglected - fine
for fields << Rabi frequency. Sources without exact theory (traces, other
non-Gaussian processes) need :func:`monte_carlo_map`.

Monte Carlo path (:func:`monte_carlo_map`)
------------------------------------------
:func:`nvsim.simulate_ensemble` for every pixel and both readouts. Cost per
pixel ~ 2 x (n_realizations x sequence grid points); typically 10-100 ms, so a
100 x 100 map takes minutes to tens of minutes. ``reuse_noise=True`` uses the
same random numbers in every pixel (smooth maps, errors correlated between
pixels); the two readouts of a pixel always share their noise histories.

Observable
----------
:func:`signal_map` / :func:`monte_carlo_map` return ``se_norm`` =
(N2 - N1)/(N2 + N1) from the final |0> populations of the sequence and of the
same sequence with the readout pulse phase flipped by pi, via
:class:`~nvsim.readout.Readout`. ``n2_bright`` selects which readout is N2
(default False: N2 is the one that ends dark without noise).
"""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import j0

from .constants import GAMMA_E, TWO_PI
from .drive import MWChannel
from .ensemble import simulate_ensemble
from .fields import ACField, StaticField
from .filter_function import toggling_correlation, toggling_pieces, toggling_transform
from .noise import NarrowbandNoise, NoiseSource, OUNoise, QuasiStaticNoise, TelegraphNoise, flatten
from .pulses import Pulse, PulseSequence
from .readout import Readout
from .simulate import channel_map, simulate
from .system import NVCenter, normalize_transition
from .tabulated import SpectrumNoise
from .telegraph import telegraph_coherence_array

#: Pixels per block when building autocorrelation matrices (memory ~ block x lags).
PIXEL_BLOCK = 2048


# --------------------------------------------------------------------------- per-pixel descriptions


def _broadcast_shape(params: dict) -> tuple:
    arrays = [np.shape(v) for v in params.values() if isinstance(v, np.ndarray)]
    return np.broadcast_shapes(*arrays) if arrays else ()


class _PerPixel:
    """Common machinery: build one object per pixel from per-pixel parameters."""

    def __init__(self, factory: Callable, params: dict):
        self.factory = factory
        self.params = params

    @property
    def shape(self) -> tuple:
        return _broadcast_shape(self.params)

    def at(self, index: tuple, shape: tuple):
        """The object for pixel ``index`` of a map of ``shape``."""
        kwargs = {}
        for key, value in self.params.items():
            kwargs[key] = float(np.broadcast_to(value, shape)[index]) if isinstance(value, np.ndarray) else value
        return self.factory(**kwargs)

    def objects(self, shape: tuple) -> list:
        """Objects for every pixel, in C order."""
        flat = {k: (np.broadcast_to(v, shape).ravel() if isinstance(v, np.ndarray) else None)
                for k, v in self.params.items()}
        out = []
        for i in range(int(np.prod(shape, dtype=int))):
            kwargs = {k: (float(flat[k][i]) if flat[k] is not None else v) for k, v in self.params.items()}
            out.append(self.factory(**kwargs))
        return out


class PixelSource(_PerPixel):
    """A noise source with per-pixel parameters: ``PixelSource(factory, **params)``."""

    def __init__(self, factory: Callable[..., NoiseSource], **params):
        super().__init__(factory, params)


class PixelField(_PerPixel):
    """A deterministic field with per-pixel parameters (``StaticField`` or ``ACField``).

    ``random_phase=True`` (AC fields only): the phase is uniformly random from
    shot to shot instead of fixed relative to the sequence start.
    """

    def __init__(self, factory: Callable, random_phase: bool = False, **params):
        super().__init__(factory, params)
        self.random_phase = random_phase


@dataclass
class PixelSpectrum:
    """Gaussian noise with per-pixel one-sided PSDs on a common grid.

    ``f`` [MHz] (n_f), ``psd`` [mT^2/MHz] with shape (n_f, *map); interpolated
    linearly in f as :class:`~nvsim.tabulated.SpectrumNoise`. ``lines``:
    [(frequency [MHz], variance map [mT^2]), ...] for zero-width lines.
    """

    f: np.ndarray
    psd: np.ndarray
    lines: list = field(default_factory=list)
    label: str = "per-pixel spectrum"

    def __post_init__(self):
        self.f = np.asarray(self.f, dtype=float)
        self.psd = np.asarray(self.psd, dtype=float)
        if self.psd.shape[0] != len(self.f):
            raise ValueError("psd must have shape (len(f), *map).")

    @property
    def shape(self) -> tuple:
        shapes = [self.psd.shape[1:]] + [np.shape(v) for _, v in self.lines]
        return np.broadcast_shapes(*shapes)

    def at(self, index: tuple, shape: tuple) -> list:
        psd = np.broadcast_to(self.psd, (len(self.f),) + shape)[(slice(None),) + index]
        out = [SpectrumNoise(self.f, psd, label=self.label)]
        for f0, variance in self.lines:
            out.append(NarrowbandNoise(float(np.sqrt(np.broadcast_to(variance, shape)[index])), f0,
                                       label=f"{self.label} line {f0:g} MHz"))
        return out


def map_shape(sources=(), fields=()) -> tuple:
    """Common map shape of all per-pixel descriptions."""
    shapes = [s.shape for s in list(sources) + list(fields) if isinstance(s, (_PerPixel, PixelSpectrum))]
    return np.broadcast_shapes(*shapes) if shapes else ()


# --------------------------------------------------------------------------- theory


def _coupling(transition) -> float:
    return TWO_PI * GAMMA_E * (1.0 if normalize_transition(transition) == "-1" else -1.0)


def _acf_matrix(kind: str, arrays: dict, lags: np.ndarray) -> np.ndarray:
    """Autocovariance of n pixels of one known Gaussian source type at ``lags``, shape (n, len(lags))."""
    if kind == "quasi":
        return np.repeat((arrays["b"] ** 2)[:, None], len(lags), axis=1)
    if kind == "ou":
        return arrays["b"][:, None] ** 2 * np.exp(-lags[None, :] / arrays["tc"][:, None])
    # narrowband (Gaussian): b^2 exp(-pi linewidth |lag|) cos(2 pi f0 lag)
    return (arrays["b"][:, None] ** 2 * np.exp(-np.pi * arrays["lw"][:, None] * lags[None, :])
            * np.cos(TWO_PI * arrays["f0"][:, None] * lags[None, :]))


def _classify(primitives: list):
    """Group per-pixel primitive sources (all of the same class) for vectorised evaluation."""
    first = primitives[0]
    if any(type(p) is not type(first) for p in primitives):
        raise ValueError(f"A PixelSource must give the same source type in every pixel ({first.label}).")
    if isinstance(first, TelegraphNoise):
        return "telegraph", {k: np.array([getattr(p, k) for p in primitives])
                             for k in ("b_low", "b_high", "rate_up", "rate_down")}
    if isinstance(first, QuasiStaticNoise):
        return "quasi", {"b": np.array([p.b_rms for p in primitives])}
    if isinstance(first, OUNoise):
        return "ou", {"b": np.array([p.b_rms for p in primitives]),
                      "tc": np.array([p.correlation_time for p in primitives])}
    if isinstance(first, NarrowbandNoise):
        arrays = {"b": np.array([p.b_rms for p in primitives]), "f0": np.array([p.frequency for p in primitives]),
                  "lw": np.array([p.linewidth for p in primitives])}
        if all(p.gaussian for p in primitives):
            return "narrowband", arrays
        if all((not p.gaussian) and p.linewidth == 0 for p in primitives):
            return "random_line", arrays  # fixed amplitude sqrt(2) b, random phase: exact J0
        return "unsupported", primitives
    if all(p.gaussian for p in primitives):
        return "gaussian_objects", primitives
    return "unsupported", primitives


def _expand_sources(sources, shape):
    """Yield (kind, data) groups for every primitive component of every source."""
    n = int(np.prod(shape, dtype=int))
    for source in sources:
        if isinstance(source, PixelSpectrum):
            yield "spectrum", source
            continue
        if isinstance(source, PixelSource):
            per_pixel = [flatten([obj]) for obj in source.objects(shape)]
        else:  # the same in every pixel
            per_pixel = [flatten([source])] * n
        counts = {len(p) for p in per_pixel}
        if len(counts) != 1:
            raise ValueError("A PixelSource must have the same number of components in every pixel.")
        for j in range(counts.pop()):
            yield _classify([p[j] for p in per_pixel])


def _max_step(kind: str, data) -> float:
    if kind == "ou":
        return float(np.min(data["tc"]) / 5)
    if kind == "narrowband":
        lw = data["lw"]
        tc = np.where(lw > 0, 1 / (np.pi * np.where(lw > 0, lw, 1.0)), np.inf)
        step = 1 / (20 * (np.abs(data["f0"]) + 3 * lw))
        return float(np.min(np.minimum(step, tc / 5)))
    if kind == "gaussian_objects":
        return float(min(p.max_step() for p in data))
    if kind == "spectrum":
        return SpectrumNoise(data.f, data.psd.reshape(len(data.f), -1).mean(axis=1)).max_step()
    return np.inf


def coherence_map(sequence: PulseSequence, drive: MWChannel, sources=(), fields=(), dt: float | None = None,
                  shape: tuple | None = None) -> tuple[np.ndarray, float]:
    """Complex coherence of ``sequence`` for every pixel, and the time step used.

    See the module docs for the per-pixel descriptions. Raises ValueError for
    sources without an exact theory (use :func:`monte_carlo_map`).
    """
    sources, fields = list(sources), list(fields)
    shape = map_shape(sources, fields) if shape is None else tuple(shape)
    n = int(np.prod(shape, dtype=int))
    segments = sequence.compile(channel_map(drive))
    kappa = _coupling(drive.transition)
    result = np.ones(n, dtype=complex)
    pieces = toggling_pieces(segments)
    if not pieces:
        return result.reshape(shape), 0.0

    groups = list(_expand_sources(sources, shape))
    bad = [p.label for kind, data in groups if kind == "unsupported" for p in data[:1]]
    if bad:
        raise ValueError(f"No exact theory for {', '.join(bad)}; use monte_carlo_map.")

    # Gaussian part on a common lag grid.
    gaussian_kinds = ("quasi", "ou", "narrowband", "gaussian_objects", "spectrum")
    gaussian = [(k, d) for k, d in groups if k in gaussian_kinds]
    step = 0.0
    if gaussian:
        if dt is None:
            span = pieces[-1].end - pieces[0].start
            dt = min([_max_step(k, d) for k, d in gaussian] + [span / 2000])
        step, corr = toggling_correlation(segments, dt)
        lags = np.arange(len(corr)) * step
        weights = (kappa * step) ** 2 * np.where(np.arange(len(corr)) == 0, 1.0, 2.0) * corr
        variance = np.zeros(n)
        for kind, data in gaussian:
            if kind in ("quasi", "ou", "narrowband"):
                for lo in range(0, n, PIXEL_BLOCK):
                    block = {k: v[lo: lo + PIXEL_BLOCK] for k, v in data.items()}
                    variance[lo: lo + PIXEL_BLOCK] += _acf_matrix(kind, block, lags) @ weights
            elif kind == "gaussian_objects":
                variance += np.array([p.autocorrelation(lags) @ weights for p in data])
            else:  # PixelSpectrum: linear in the PSD values
                unit = np.eye(len(data.f))
                w_nodes = np.array([SpectrumNoise(data.f, unit[j]).autocorrelation(lags) @ weights
                                    for j in range(len(data.f))])
                psd = np.broadcast_to(data.psd, (len(data.f),) + shape).reshape(len(data.f), n)
                variance += w_nodes @ psd
                for f0, var_map in data.lines:
                    y2 = float(np.abs(toggling_transform(segments, [f0])[0]) ** 2)
                    variance += kappa**2 * y2 * np.broadcast_to(var_map, shape).ravel()
        result *= np.exp(-0.5 * variance)

    for kind, data in groups:
        if kind == "telegraph":
            result *= telegraph_coherence_array(segments, data["b_low"], data["b_high"], data["rate_up"],
                                                data["rate_down"], drive.transition)
        elif kind == "random_line":
            result *= j0(kappa * np.sqrt(2) * data["b"] * np.abs(toggling_transform(segments, data["f0"])))

    for f in fields:
        result *= _field_factor(f, segments, kappa, shape)
    return result.reshape(shape), step


def _field_factor(f, segments, kappa: float, shape: tuple) -> np.ndarray:
    """exp(i phi) (or its random-phase average) of a deterministic field, per pixel (flat)."""
    n = int(np.prod(shape, dtype=int))
    if isinstance(f, PixelField):
        objs, random_phase = f.objects(shape), f.random_phase
    else:
        objs, random_phase = [f] * n, False
    first = objs[0]
    if isinstance(first, StaticField):
        y0 = float(np.real(toggling_transform(segments, [0.0])[0]))
        return np.exp(1j * kappa * y0 * np.array([o.Bz for o in objs]))
    if isinstance(first, ACField):
        amplitude = np.array([o.amplitude for o in objs])
        y = toggling_transform(segments, np.array([o.frequency for o in objs]))
        if random_phase:
            return j0(kappa * amplitude * np.abs(y)).astype(complex)
        phase = np.array([o.phase for o in objs])
        return np.exp(1j * kappa * amplitude * np.real(np.exp(1j * phase) * np.conj(y)))
    raise TypeError(f"Unsupported field {first!r}; use StaticField or ACField.")


# --------------------------------------------------------------------------- observable


def with_flipped_readout(sequence: PulseSequence) -> PulseSequence:
    """The same sequence with the phase of its last (readout) pulse shifted by pi."""
    elements = list(sequence.elements)
    for i in range(len(elements) - 1, -1, -1):
        if isinstance(elements[i], Pulse):
            last = elements[i]
            elements[i] = replace(last, phase=last.phase + np.pi,
                                  label=(last.label or "readout") + " (flipped)")
            return PulseSequence(elements, name=sequence.name)
    raise ValueError("Sequence has no pulse to flip.")


@dataclass
class MapResult:
    """Per-pixel results; ``*_error`` are filled by the Monte Carlo path only."""

    coherence: np.ndarray
    p0_n1: np.ndarray
    p0_n2: np.ndarray
    se_norm: np.ndarray
    method: str
    dt: float = 0.0
    coherence_error: np.ndarray | None = None
    p0_n1_error: np.ndarray | None = None
    p0_n2_error: np.ndarray | None = None
    se_norm_error: np.ndarray | None = None


def _noise_free_populations(sequence, drive, nv):
    flipped = with_flipped_readout(sequence)
    p_a = simulate(sequence, nv, drive, store_trajectory=False).final_populations["0"]
    p_b = simulate(flipped, nv, drive, store_trajectory=False).final_populations["0"]
    return flipped, p_a, p_b


def signal_map(sequence: PulseSequence, drive: MWChannel, sources=(), fields=(), readout: Readout | None = None,
               n2_bright: bool = False, nv: NVCenter | None = None, dt: float | None = None,
               shape: tuple | None = None) -> MapResult:
    """Theory map of the coherence, both readout populations and ``se_norm``.

    ``shape`` is needed only if no per-pixel description fixes the map shape."""
    readout = readout or Readout()
    nv = nv or NVCenter()
    _, p_a, p_b = _noise_free_populations(sequence, drive, nv)
    coherence, step = coherence_map(sequence, drive, sources, fields, dt, shape)
    c = np.real(coherence)
    pop_a, pop_b = 0.5 + (p_a - 0.5) * c, 0.5 + (p_b - 0.5) * c
    bright, dark = (pop_a, pop_b) if p_a >= p_b else (pop_b, pop_a)
    n1, n2 = (dark, bright) if n2_bright else (bright, dark)
    return MapResult(coherence, n1, n2, readout.se_norm(n1, n2), "theory", step)


def _pixel_noise_and_fields(sources, fields, index, shape):
    noise, det = [], []
    for s in sources:
        if isinstance(s, PixelSpectrum):
            noise += s.at(index, shape)
        elif isinstance(s, PixelSource):
            noise.append(s.at(index, shape))
        else:
            noise.append(s)
    for f in fields:
        obj = f.at(index, shape) if isinstance(f, PixelField) else f
        if isinstance(f, PixelField) and f.random_phase:
            # random-phase AC field = fixed-amplitude narrowband noise with zero linewidth
            noise.append(NarrowbandNoise(obj.amplitude / np.sqrt(2), obj.frequency, 0.0, gaussian=False,
                                         label="random-phase AC field"))
        else:
            det.append(obj)
    return noise, det


def monte_carlo_map(sequence: PulseSequence, drive: MWChannel, sources=(), fields=(), n_realizations: int = 200,
                    readout: Readout | None = None, n2_bright: bool = False, nv: NVCenter | None = None,
                    seed: int = 0, reuse_noise: bool = True, progress=True,
                    shape: tuple | None = None) -> MapResult:
    """Monte Carlo map (any noise source). See the module docs for cost and options.

    ``progress``: True prints a progress line to stderr, a callable is called as
    ``progress(done, total)``, False is silent. ``se_norm_error`` adds the two
    readouts' errors linearly (they share noise histories, so their errors are
    correlated); it is an upper bound.
    """
    readout = readout or Readout()
    nv = nv or NVCenter()
    sources, fields = list(sources), list(fields)
    shape = map_shape(sources, fields) if shape is None else tuple(shape)
    flipped, p_a_clean, p_b_clean = _noise_free_populations(sequence, drive, nv)
    a_is_bright = p_a_clean >= p_b_clean
    n_pix = int(np.prod(shape, dtype=int))
    pa, pb, ea, eb = (np.empty(shape) for _ in range(4))
    seeds = np.random.SeedSequence(seed).spawn(n_pix)
    start = time.perf_counter()
    for k, index in enumerate(np.ndindex(shape)):
        noise, det = _pixel_noise_and_fields(sources, fields, index, shape)
        pixel_seed = seed if reuse_noise else seeds[k]
        ra = simulate_ensemble(sequence, nv, drive, noise, det, n_realizations, rng=np.random.default_rng(pixel_seed))
        rb = simulate_ensemble(flipped, nv, drive, noise, det, n_realizations, rng=np.random.default_rng(pixel_seed))
        pa[index], pb[index] = ra.populations["0"], rb.populations["0"]
        ea[index], eb[index] = ra.uncertainty["0"], rb.uncertainty["0"]
        if callable(progress):
            progress(k + 1, n_pix)
        elif progress and ((k + 1) % max(1, n_pix // 100) == 0 or k + 1 == n_pix):
            elapsed = time.perf_counter() - start
            eta = elapsed / (k + 1) * (n_pix - k - 1)
            print(f"\rMonte Carlo map: {k + 1}/{n_pix} pixels, {elapsed:.0f} s, ~{eta:.0f} s left",
                  end="" if k + 1 < n_pix else "\n", file=sys.stderr, flush=True)
    bright, dark = (pa, pb) if a_is_bright else (pb, pa)
    e_bright, e_dark = (ea, eb) if a_is_bright else (eb, ea)
    n1, n2 = (dark, bright) if n2_bright else (bright, dark)
    e1, e2 = (e_dark, e_bright) if n2_bright else (e_bright, e_dark)
    se = readout.se_norm(n1, n2)
    pl1, pl2 = readout.pl({"0": n1}), readout.pl({"0": n2})
    total = pl1 + pl2
    # d se / d p0 = contrast * (+-2 pl_other) / total^2
    se_err = readout.contrast * 2 * (pl1 * e2 + pl2 * e1) / total**2
    contrast_clean = (p_a_clean if a_is_bright else p_b_clean) - 0.5
    coherence = (bright - 0.5) / contrast_clean
    return MapResult(coherence, n1, n2, se, "monte carlo", 0.0, e_bright / abs(contrast_clean), e1, e2, se_err)
