"""Plots for noise studies: spectra, filter functions, noise traces, signals."""

import matplotlib.pyplot as plt
import numpy as np

from .noise import flatten, sample_total
from .plotting import MUTED_COLOR, TEXT_COLOR, field_scale, style_axes

#: Categorical slots: sequences use 1-4, noise sources 5-8 (colour = entity).
SEQUENCE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"]
SOURCE_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948", "#2a78d6", "#eb6834"]
NOISE_FREE_COLOR = "#b4b3ad"
THEORY_COLOR = TEXT_COLOR
PSD_UNIT = 1e6  # mT^2/MHz -> uT^2/MHz


def plot_noise_spectrum(sources, f, ax=None):
    """One-sided PSD of every primitive source (log scale, µT²/MHz); spectral
    lines (zero linewidth, quasi-static) are drawn as labelled vertical lines."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    f = np.asarray(f, dtype=float)
    peak = 0.0
    for i, source in enumerate(flatten(sources)):
        color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        psd = PSD_UNIT * source.psd(f)
        if np.any(psd > 0):
            ax.plot(f, psd, color=color, lw=2, label=source.label)
            peak = max(peak, psd.max())
        for f0, variance in source.lines():
            ax.axvline(f0, color=color, lw=2, ls=(0, (4, 2)),
                       label=f"{source.label} (line, {1e3 * np.sqrt(variance):.2g} µT rms)")
    ax.set_yscale("log")
    if peak > 0:
        ax.set_ylim(peak * 1e-4, peak * 3)
    ax.set_xlim(f[0], f[-1])
    ax.set_ylabel("PSD (µT²/MHz)")
    ax.set_xlabel("frequency (MHz)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    return style_axes(ax)


def plot_filter_functions(curves, f, ax=None):
    """Normalised filter functions |Y(f)|²; ``curves`` = [(label, values), ...]."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    for i, (label, values) in enumerate(curves):
        peak = np.max(values) if np.max(values) > 0 else 1.0
        ax.plot(f, values / peak, color=SEQUENCE_COLORS[i % len(SEQUENCE_COLORS)], lw=2, label=label)
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("filter |Y(f)|²\n(normalised)")
    ax.set_xlabel("frequency (MHz)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    return style_axes(ax)


def plot_noise_traces(sources, duration: float, n_traces: int = 3, rng=None, ax=None, n_points: int = 4000):
    """A few realisations of the total noise field B(t)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.6))
    t = np.linspace(0, duration, n_points)
    b = sample_total(sources, t, n_traces, np.random.default_rng(rng))
    scale, unit = field_scale(float(np.max(np.abs(b), initial=0.0)))
    for k in range(n_traces):
        ax.plot(t, scale * b[k], lw=1.2, color=SEQUENCE_COLORS[k % len(SEQUENCE_COLORS)], alpha=0.9,
                label=f"realisation {k + 1}")
    ax.axhline(0, color=MUTED_COLOR, lw=0.8, zorder=1)
    ax.set_xlim(0, duration)
    ax.set_xlabel("time (µs)")
    ax.set_ylabel(f"$B_z$ noise ({unit})")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    return style_axes(ax)


def plot_signal(ax, x, y, y_err, theory=None, noise_free=None, color=SEQUENCE_COLORS[0], label="simulation"):
    """Monte Carlo points with error bars, plus optional theory and noise-free curves."""
    if noise_free is not None:
        ax.plot(x, noise_free, color=NOISE_FREE_COLOR, lw=1.5, label="no noise")
    if theory is not None:
        ax.plot(x, theory, color=THEORY_COLOR, lw=1.2, ls=(0, (4, 2)), label="Gaussian theory")
    ax.errorbar(x, y, yerr=y_err, fmt="o", ms=4, color=color, ecolor=color, elinewidth=1,
                capsize=0, label=label, zorder=3)
    return style_axes(ax)
