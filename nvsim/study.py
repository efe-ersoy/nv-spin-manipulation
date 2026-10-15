"""Noise studies: how pulse sequences respond to magnetic noise.

A :class:`NoiseStudy` collects everything an experiment would fix - the NV,
the MW drive, the noise sources and deterministic fields, the readout model,
and a list of sequence scans (:class:`SequenceScan`, built with
:func:`ramsey_scan`, :func:`echo_scan`, :func:`cpmg_scan`, :func:`xy8_scan`,
...). :func:`run_study` then, for every scan,

* runs the Monte Carlo ensemble at each tau (:mod:`nvsim.ensemble`),
* computes the noise-free signal and the Gaussian filter-function prediction,
* writes figures and CSV data to the output directory.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .drive import MWChannel
from .ensemble import sweep_ensemble
from .filter_function import coherence_decay, filter_function
from .fields import ACField
from .noise import CarbonBath, NarrowbandNoise, OUNoise, QuasiStaticNoise, flatten
from .noise_plots import (
    SEQUENCE_COLORS, plot_filter_functions, plot_noise_spectrum, plot_noise_traces, plot_signal,
)
from .plotting import TEXT_COLOR, style_axes
from .pulses import PulseSequence
from .readout import Readout
from .results import SweepResult
from .sequences import cpmg, ramsey, spin_echo, xy4, xy8
from .simulate import channel_map, sweep
from .constants import GAMMA_E, TWO_PI
from .system import NVCenter


#: Shared x axes for a study: the pulse spacing tau, or the total free-evolution time.
X_AXES = {
    "spacing": "pulse spacing τ (µs)   [Ramsey, echo: total free evolution]",
    "total": "total free-evolution time (µs)",
}


@dataclass
class SequenceScan:
    """A pulse sequence swept along the study's shared x axis.

    Every sequence is parametrised by the same *pulse spacing* tau: the free
    evolution between consecutive pi pulses, with tau/2 before the first and
    after the last one (CPMG convention). The spin echo is CPMG-1
    (tau/2 - pi - tau/2, i.e. tau = total free evolution) and Ramsey has a
    single window of length tau. With ``intervals`` windows of length tau the
    total free evolution is ``intervals * tau``.

    Parameters
    ----------
    label : str
        Name used in plots and file names.
    build : callable
        ``build(tau) -> PulseSequence`` with tau the pulse spacing [us].
    x : array
        Points on the shared axis [us].
    intervals : int
        Number of tau-windows (1 for Ramsey and echo, N for CPMG-N, 8 R for XY8-R).
    axis : {"spacing", "total"}
        Meaning of ``x``: the pulse spacing itself, or the total free-evolution
        time (then tau = x / intervals).
    probe : callable, optional
        ``probe(tau, t_pi2, t_pi) -> f`` [MHz]: the noise frequency the sequence
        is most sensitive to. Enables the spectroscopy plot.
    filter_tau : float, optional
        tau at which the filter function is drawn (default: median of the taus).
    """

    label: str
    build: Callable[[float], PulseSequence]
    x: Sequence[float]
    intervals: int = 1
    axis: str = "spacing"
    probe: Callable[[float, float, float], float] | None = None
    filter_tau: float | None = None

    def __post_init__(self):
        if self.axis not in X_AXES:
            raise ValueError(f"axis must be one of {list(X_AXES)}, got {self.axis!r}")
        self.x = np.asarray(self.x, dtype=float)
        if self.filter_tau is None:
            self.filter_tau = float(np.median(self.taus))

    @property
    def taus(self) -> np.ndarray:
        """Pulse spacings [us] actually simulated."""
        return self.x if self.axis == "spacing" else self.x / self.intervals

    @property
    def total_times(self) -> np.ndarray:
        """Total free-evolution time [us] at every point."""
        return self.intervals * self.taus


def _dd_probe(tau, t_pi2, t_pi):
    return 1.0 / (2.0 * (tau + t_pi))


def ramsey_scan(x, axis="spacing", label="Ramsey", **kwargs) -> SequenceScan:
    """Ramsey pi/2 - tau - pi/2 (``kwargs`` go to :func:`nvsim.ramsey`)."""
    return SequenceScan(label, lambda tau: ramsey(tau, **kwargs), x, 1, axis)


def echo_scan(x, axis="spacing", label="Spin echo", **kwargs) -> SequenceScan:
    """Hahn echo as CPMG-1: tau/2 - pi - tau/2, tau = total free evolution.

    The echo filter is broad, so it is left out of the spectroscopy view.
    """
    return SequenceScan(label, lambda tau: spin_echo(tau / 2, **kwargs), x, 1, axis)


def cpmg_scan(x, n_pulses: int, axis="spacing", label=None, **kwargs) -> SequenceScan:
    return SequenceScan(label or f"CPMG-{n_pulses}", lambda tau: cpmg(tau, n_pulses, **kwargs), x,
                        n_pulses, axis, _dd_probe)


def xy4_scan(x, repetitions: int = 1, axis="spacing", label=None, **kwargs) -> SequenceScan:
    return SequenceScan(label or f"XY4-{repetitions}", lambda tau: xy4(tau, repetitions, **kwargs), x,
                        4 * repetitions, axis, _dd_probe)


def xy8_scan(x, repetitions: int = 1, axis="spacing", label=None, **kwargs) -> SequenceScan:
    return SequenceScan(label or f"XY8-{repetitions}", lambda tau: xy8(tau, repetitions, **kwargs), x,
                        8 * repetitions, axis, _dd_probe)


# --------------------------------------------------------------------------- automatic x range


def line_frequencies(noise=(), fields=()) -> list[float]:
    """Frequencies [MHz] of all narrowband noise lines and AC fields (> 0)."""
    freqs = [s.frequency for s in flatten(noise) if isinstance(s, NarrowbandNoise)]
    freqs += [f.frequency for f in fields if isinstance(f, ACField)]
    return sorted(f for f in freqs if f > 0)


def t2_star(noise) -> float:
    """Ramsey T2* [us] from the slow noise (quasi-static + OU rms); inf if none."""
    slow = [s.b_rms for s in flatten(noise) if isinstance(s, (QuasiStaticNoise, OUNoise))]
    b = float(np.sqrt(np.sum(np.square(slow)))) if slow else 0.0
    return np.sqrt(2) / (TWO_PI * GAMMA_E * b) if b > 0 else np.inf


def auto_axis(noise, t_pi: float, n: int, fields=(), x_min: float = 0.02, fallback: float = 10.0,
              x_max_limit: float = 200.0) -> np.ndarray:
    """Automatic points for the shared x axis [us].

    The axis ends just after the first revival of the slowest noise line,
    x = 2.2 / f_min: there every window of CPMG/XY (and of the echo, which is
    CPMG-1) holds a whole number of periods, so that line is refocused. It is
    extended to 3 T2* if the slow noise decays more slowly. Without lines the
    axis ends at ``fallback`` (or 3 T2*).

    Half of the points are linear in x (overall shape, revivals), half are
    spaced evenly in log of the probed frequency 1/(2(x + t_pi)), which keeps
    the narrow CPMG/XY dips resolved at short tau.
    """
    freqs = line_frequencies(noise, fields)
    x_max = 2.2 / freqs[0] if freqs else fallback
    t2 = t2_star(noise)
    if np.isfinite(t2):
        x_max = max(x_max, 3 * t2)
    x_max = min(x_max, x_max_limit)
    linear = np.linspace(x_min, x_max, n - n // 2)
    f = np.geomspace(1 / (2 * (x_max + t_pi)), 1 / (2 * (x_min + t_pi)), n // 2)
    log_f = 1 / (2 * f) - t_pi
    return np.unique(np.round(np.concatenate([linear, log_f]), 9))


def resolve_axis(spec, auto: Callable[[], np.ndarray], n: int) -> np.ndarray:
    """``"auto"`` -> ``auto()``; ``(start, stop)`` -> ``n`` linear points; else the array itself."""
    if isinstance(spec, str):
        if spec != "auto":
            raise ValueError(f"axis range must be 'auto', (start, stop) or an array, got {spec!r}")
        return auto()
    spec = np.asarray(spec, dtype=float)
    return np.linspace(spec[0], spec[1], n) if spec.shape == (2,) else spec


@dataclass
class NoiseStudy:
    """Configuration of a noise study (see module docs).

    ``signal`` selects what is plotted against tau: ``"pl"`` (normalised
    photoluminescence from ``readout``) or ``"population"`` (P(|0>)).
    """

    name: str
    nv: NVCenter
    drive: MWChannel
    scans: list[SequenceScan]
    noise: list = field(default_factory=list)
    fields: list = field(default_factory=list)
    n_realizations: int = 300
    readout: Readout = field(default_factory=Readout)
    signal: str = "pl"
    seed: int | None = 1
    theory: bool = True
    spectrum_max_frequency: float | None = None
    n_noise_traces: int = 3

    @property
    def t_pi(self) -> float:
        return 1.0 / (2.0 * self.drive.rabi_frequency)

    @property
    def t_pi2(self) -> float:
        return 1.0 / (4.0 * self.drive.rabi_frequency)


@dataclass
class ScanOutcome:
    """Everything computed for one :class:`SequenceScan`."""

    scan: SequenceScan
    monte_carlo: SweepResult
    noise_free: SweepResult
    p0_theory: np.ndarray | None
    signal: np.ndarray
    signal_error: np.ndarray
    signal_theory: np.ndarray | None
    signal_noise_free: np.ndarray
    probe_frequency: np.ndarray | None
    coherence: np.ndarray
    coherence_error: np.ndarray
    coherence_theory: np.ndarray | None


def _signal_label(study: NoiseStudy) -> str:
    return "PL (normalised)" if study.signal == "pl" else "P(|0⟩)"


def _to_signal(study: NoiseStudy, p0, p0_err=None, rng=None):
    """Convert P0 (and its error) to the plotted signal; shot noise only with ``rng``."""
    p0 = np.asarray(p0, dtype=float)
    err = np.zeros_like(p0) if p0_err is None else np.asarray(p0_err, dtype=float)
    if study.signal == "population":
        return p0, err
    if rng is None or not study.readout.shot_noise:
        return study.readout.pl({"0": p0}), study.readout.pl_error(err)
    pl, shot = study.readout.measure({"0": p0}, rng)
    return pl, np.hypot(study.readout.pl_error(err), shot)


def run_scan(study: NoiseStudy, scan: SequenceScan, rng) -> ScanOutcome:
    """Monte Carlo, noise-free and theory signals for one scan."""
    channels = channel_map(study.drive)
    mc = sweep_ensemble(scan.taus, scan.build, study.nv, study.drive, study.noise, study.fields,
                        study.n_realizations, rng=rng, parameter="pulse spacing τ (µs)")
    clean = sweep(scan.taus, scan.build, study.nv, study.drive, fields=study.fields,
                  parameter="pulse spacing τ (µs)")
    p0_clean = clean.population("0")

    p0_theory = None
    if study.theory and study.noise:
        decay = np.array([coherence_decay(scan.build(t).compile(channels), study.noise) for t in scan.taus])
        p0_theory = 0.5 + (p0_clean - 0.5) * decay

    signal, signal_err = _to_signal(study, mc.population("0"), mc.population_error("0"), rng)
    signal_theory = None if p0_theory is None else _to_signal(study, p0_theory)[0]

    # Coherence relative to the noise-free signal (1 = no decay, 0 = fully dephased).
    contrast = p0_clean - 0.5
    usable = np.abs(contrast) > 0.05
    safe = np.where(usable, contrast, 1.0)
    coherence = np.where(usable, (mc.population("0") - 0.5) / safe, np.nan)
    coherence_err = np.where(usable, mc.population_error("0") / np.abs(safe), np.nan)
    coherence_theory = None if p0_theory is None else np.where(usable, (p0_theory - 0.5) / safe, np.nan)

    probe = None
    if scan.probe is not None:
        probe = np.array([scan.probe(t, study.t_pi2, study.t_pi) for t in scan.taus])
    return ScanOutcome(scan, mc, clean, p0_theory, signal, signal_err, signal_theory,
                       _to_signal(study, p0_clean)[0], probe, coherence, coherence_err, coherence_theory)


# --------------------------------------------------------------------------- plots


def _spectrum_range(study: NoiseStudy) -> np.ndarray:
    if study.spectrum_max_frequency is not None:
        f_max = study.spectrum_max_frequency
    else:
        marks = [f0 for s in flatten(study.noise) for f0, _ in s.lines()]
        marks += [getattr(s, "frequency", 0.0) for s in flatten(study.noise)]
        marks += [scan.probe(scan.filter_tau, study.t_pi2, study.t_pi) for scan in study.scans if scan.probe]
        f_max = 2.0 * max(marks + [0.5])
    return np.linspace(0, f_max, 2000)


def figure_spectrum(study: NoiseStudy):
    """Noise PSD above the (normalised) filter functions of every scan."""
    f = _spectrum_range(study)
    channels = channel_map(study.drive)
    fig, (ax_psd, ax_filter) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    plot_noise_spectrum(study.noise, f, ax_psd)
    ax_psd.set_xlabel("")
    curves = [(f"{scan.label}, τ = {scan.filter_tau:.3g} µs", filter_function(scan.build(scan.filter_tau).compile(channels), f))
              for scan in study.scans]
    plot_filter_functions(curves, f, ax_filter)
    fig.suptitle(f"{study.name}: noise spectrum and sequence filter functions", color=TEXT_COLOR)
    return fig


def figure_noise_traces(study: NoiseStudy, rng):
    channels = channel_map(study.drive)
    longest = max(scan.build(scan.taus.max()).total_duration(channels) for scan in study.scans)
    fig, ax = plt.subplots(figsize=(10, 2.8), layout="constrained")
    plot_noise_traces(study.noise, min(longest, 50.0), study.n_noise_traces, rng, ax)
    ax.set_title("Example noise realisations (total field along the NV axis)", color=TEXT_COLOR)
    return fig


def _window_note(scan: SequenceScan) -> str:
    """How the plotted x relates to this sequence's own timing."""
    if scan.axis == "spacing":
        return f"total free evolution = {scan.intervals} τ" if scan.intervals > 1 else "τ = total free evolution"
    return f"pulse spacing τ = T / {scan.intervals}" if scan.intervals > 1 else "τ = T"


def figure_signals(study: NoiseStudy, outcomes: list[ScanOutcome]):
    """Signal vs the shared x axis, one panel per scan (same x and y for all)."""
    n = len(outcomes)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n + 0.8), layout="constrained", squeeze=False,
                             sharex=True, sharey=True)
    for i, (ax, out) in enumerate(zip(axes[:, 0], outcomes)):
        plot_signal(ax, out.scan.x, out.signal, out.signal_error, out.signal_theory, out.signal_noise_free,
                    color=SEQUENCE_COLORS[i % len(SEQUENCE_COLORS)], label=f"{out.scan.label} (Monte Carlo)")
        ax.set_ylabel(_signal_label(study))
        ax.set_title(f"{out.scan.label}   ({_window_note(out.scan)})", color=TEXT_COLOR, loc="left", fontsize=10)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    axes[-1, 0].set_xlabel(X_AXES[outcomes[0].scan.axis])
    axes[-1, 0].set_xlim(0, max(out.scan.x.max() for out in outcomes) * 1.01)
    fig.suptitle(f"{study.name}: {_signal_label(study)}  ({study.n_realizations} noise realisations per point)",
                 color=TEXT_COLOR)
    return fig


def figure_spectroscopy(study: NoiseStudy, outcomes: list[ScanOutcome]):
    """Coherence vs probed frequency for the scans that define one, above the PSD."""
    probing = [(i, o) for i, o in enumerate(outcomes) if o.probe_frequency is not None]
    if not probing:
        return None
    f_lo = min(np.nanmin(o.probe_frequency) for _, o in probing)
    f_hi = max(np.nanmax(o.probe_frequency) for _, o in probing)
    lines = line_frequencies(study.noise, study.fields)
    if lines:  # up to the 3rd harmonic of the highest line; beyond it nothing happens
        f_hi = min(f_hi, 3.5 * lines[-1])
    fig, (ax_c, ax_psd) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, layout="constrained",
                                       gridspec_kw={"height_ratios": [2, 1.2]})
    for i, out in probing:
        color = SEQUENCE_COLORS[i % len(SEQUENCE_COLORS)]
        order = np.argsort(out.probe_frequency)
        x = out.probe_frequency[order]
        ax_c.errorbar(x, out.coherence[order], yerr=out.coherence_error[order], fmt="o", ms=3.5,
                      color=color, elinewidth=1, label=out.scan.label, zorder=3)
        if out.coherence_theory is not None:
            ax_c.plot(x, out.coherence_theory[order], color=color, lw=1.2, ls=(0, (4, 2)))
    ax_c.axhline(0, color="#8a8984", lw=0.8)
    ax_c.set_ylabel("coherence\n(relative to no noise)")
    ax_c.set_ylim(-0.1, 1.1)
    ax_c.set_title("Noise spectroscopy view: dips where the filter overlaps the noise "
                   "(points: Monte Carlo, dashed: Gaussian theory)", color=TEXT_COLOR, fontsize=10, loc="left")
    ax_c.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    style_axes(ax_c)
    f = np.geomspace(0.9 * f_lo, 1.1 * f_hi, 3000)
    plot_noise_spectrum(study.noise, f, ax_psd)
    ax_psd.set_xscale("log")  # probed frequencies span decades; dips have similar relative widths
    ax_psd.set_xlim(f[0], f[-1])
    ax_psd.set_xlabel("probed frequency 1/(2(τ + t_π)) (MHz, log scale)")
    fig.suptitle(f"{study.name}: noise spectroscopy", color=TEXT_COLOR)
    return fig


# --------------------------------------------------------------------------- output


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


def _write_csv(path: Path, study: NoiseStudy, out: ScanOutcome):
    columns = {
        "x_us": out.scan.x,
        "tau_spacing_us": out.scan.taus,
        "total_free_evolution_us": out.scan.total_times,
        "probe_frequency_MHz": out.probe_frequency if out.probe_frequency is not None else np.full_like(out.scan.taus, np.nan),
        "P0_mc": out.monte_carlo.population("0"),
        "P0_mc_err": out.monte_carlo.population_error("0"),
        "Pm1_mc": out.monte_carlo.population("-1"),
        "Pp1_mc": out.monte_carlo.population("+1"),
        "P0_noise_free": out.noise_free.population("0"),
        "P0_theory": out.p0_theory if out.p0_theory is not None else np.full_like(out.scan.taus, np.nan),
        "signal": out.signal,
        "signal_err": out.signal_error,
        "coherence": out.coherence,
        "coherence_err": out.coherence_error,
    }
    data = np.column_stack(list(columns.values()))
    np.savetxt(path, data, delimiter=",", header=",".join(columns), comments="", fmt="%.8g")


def _summary(study: NoiseStudy) -> str:
    lines = [
        f"study: {study.name}",
        f"NV: f_- = {study.nv.f_minus:.3f} MHz, f_+ = {study.nv.f_plus:.3f} MHz, Bz = {study.nv.Bz:.4g} mT",
        f"drive: {study.drive}",
        f"pi/2 = {1e3 * study.t_pi2:.2f} ns, pi = {1e3 * study.t_pi:.2f} ns",
        f"realisations per point: {study.n_realizations}, seed: {study.seed}",
        f"readout: {study.readout}, signal: {study.signal}",
        "noise sources:",
    ]
    for source in study.noise:
        lines.append(f"  {source}")
        if isinstance(source, CarbonBath):
            lines.append(f"    13C Larmor frequency: {1e3 * source.larmor_frequency:.2f} kHz "
                         f"(echo revivals at tau = k * {1 / source.larmor_frequency:.3f} us)")
    if np.isfinite(t2_star(study.noise)):
        lines.append(f"  slow noise (quasi-static + OU) -> T2* ~ {t2_star(study.noise):.3g} us")
    freqs = line_frequencies(study.noise, study.fields)
    if freqs:
        lines.append("  lines: " + ", ".join(f"{1e3 * f:.4g} kHz (DD dip at tau = {1 / (2 * f) - study.t_pi:.4g} us)"
                                            for f in freqs))
    lines.append("fields:")
    lines += [f"  {f}" for f in study.fields] or ["  none"]
    lines.append("scans:")
    if study.scans:
        lines.append(f"  shared x axis: {X_AXES[study.scans[0].axis]}")
    lines += [f"  {s.label}: tau from {s.taus.min():.4g} to {s.taus.max():.4g} us, total free evolution up to "
              f"{s.total_times.max():.4g} us ({len(s.taus)} points)" for s in study.scans]
    return "\n".join(lines) + "\n"


def run_study(study: NoiseStudy, output_dir, show: bool = False) -> list[ScanOutcome]:
    """Run every scan of ``study`` and write figures + data to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(study.seed)
    (output_dir / "summary.txt").write_text(_summary(study), encoding="utf-8")
    print(_summary(study))

    outcomes = []
    for scan in study.scans:
        start = time.perf_counter()
        out = run_scan(study, scan, rng)
        outcomes.append(out)
        _write_csv(output_dir / f"{_slug(scan.label)}.csv", study, out)
        print(f"{scan.label}: {len(scan.taus)} points in {time.perf_counter() - start:.1f} s")

    figures = {
        "noise_spectrum.png": figure_spectrum(study),
        "noise_traces.png": figure_noise_traces(study, rng),
        "signal_vs_tau.png": figure_signals(study, outcomes),
        "spectroscopy.png": figure_spectroscopy(study, outcomes),
    }
    for name, fig in figures.items():
        if fig is not None:
            fig.savefig(output_dir / name, dpi=150)
            print(f"wrote {output_dir / name}")
    if show:
        plt.show()
    else:
        plt.close("all")
    return outcomes
