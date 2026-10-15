"""Static plots: populations, accumulated phase, fields, pulse-sequence diagrams, sweeps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .fields import total_field
from .phase import phase_jumps
from .pulses import Segment, phase_label
from .results import SimulationResult, SweepResult
from .system import TRANSITIONS, normalize_level

# Categorical colours follow the level, never the plot order.
LEVEL_COLORS = {"0": "#2a78d6", "-1": "#eb6834", "+1": "#1baf7a"}
LEVEL_LABELS = {"0": r"$|0\rangle$", "-1": r"$|{-}1\rangle$", "+1": r"$|{+}1\rangle$"}
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def style_axes(ax):
    """Recessive axes: no top/right spines, light grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def _default_levels(populations: dict[str, np.ndarray]) -> list[str]:
    """|0> plus every level that picks up noticeable population."""
    return [
        lv for lv in ("0", "-1", "+1") if lv == "0" or np.max(populations[lv]) > 1e-3
    ]


def shade_pulses(ax, segments: list[Segment], alpha: float = 0.12):
    """Shade the time windows in which a pulse is on, coloured by transition."""
    for segment in segments:
        for drive in segment.drives:
            ax.axvspan(
                segment.start,
                segment.end,
                color=LEVEL_COLORS[drive.transition],
                alpha=alpha,
                linewidth=0,
                zorder=0,
            )


def plot_populations(
    result: SimulationResult, ax=None, levels=None, shade=True, legend=True
):
    """Level populations vs. time; pulse windows shaded."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    pops = result.populations
    levels = (
        _default_levels(pops)
        if levels is None
        else [normalize_level(lv) for lv in levels]
    )
    if shade:
        shade_pulses(ax, result.segments)
    for level in levels:
        ax.plot(
            result.times,
            pops[level],
            color=LEVEL_COLORS[level],
            lw=2,
            label=LEVEL_LABELS[level],
        )
    ax.set_xlim(result.times[0], result.times[-1])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("population")
    if legend:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return style_axes(ax)


def plot_sequence(result: SimulationResult, ax=None):
    """Timing diagram: one row per driven transition, bar height = Rabi frequency."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 1.6))
    rows = driven_transitions(result)
    rabi_max = (
        max((d.rabi_frequency for s in result.segments for d in s.drives), default=1.0)
        or 1.0
    )
    total = result.duration or 1.0
    for row, m in enumerate(rows):
        ax.axhline(row, color=MUTED_COLOR, lw=0.8, zorder=1)
        for segment in result.segments:
            for drive in segment.drives:
                if drive.transition != m:
                    continue
                height = 0.8 * drive.rabi_frequency / rabi_max
                ax.add_patch(
                    Rectangle(
                        (segment.start, row),
                        segment.duration,
                        height,
                        color=LEVEL_COLORS[m],
                        lw=0,
                        zorder=2,
                    )
                )
                if segment.duration > 0.04 * total:
                    ax.text(
                        segment.start + segment.duration / 2,
                        row + height + 0.05,
                        phase_label(drive.phase),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=TEXT_COLOR,
                    )
    ax.set_yticks(range(len(rows)), [f"MW 0↔{m}" for m in rows])
    ax.set_ylim(-0.2, len(rows) - 0.05)
    ax.set_xlim(0, total)
    ax.set_xlabel("time (µs)")
    style_axes(ax)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return ax


def driven_transitions(result: SimulationResult) -> list[str]:
    """Transitions that at least one pulse addresses (default: ['-1'])."""
    driven = [
        m
        for m in TRANSITIONS
        if any(d.transition == m for s in result.segments for d in s.drives)
    ]
    return driven or ["-1"]


def phase_transitions(result: SimulationResult) -> list[str]:
    """Driven transitions for which an accumulated phase is defined somewhere."""
    return [
        m
        for m in driven_transitions(result)
        if np.isfinite(result.accumulated_phase(m)).any()
    ]


def _break_at_wraps(t: np.ndarray, y: np.ndarray):
    """Insert NaNs where y wraps around +-pi so the line is not drawn across."""
    wraps = np.flatnonzero(np.abs(np.diff(y)) > np.pi)
    return np.insert(t, wraps + 1, np.nan), np.insert(y, wraps + 1, np.nan)


def style_phase_axis(ax):
    """Phase axis from -pi to pi with labelled ticks and a zero line."""
    ax.set_ylim(-1.12 * np.pi, 1.12 * np.pi)
    ax.set_yticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"]
    )
    ax.axhline(0, color=MUTED_COLOR, lw=0.8, zorder=1)
    ax.set_ylabel("accumulated\nphase φ")
    return ax


def plot_phase(
    result: SimulationResult, ax=None, transitions=None, shade=True, annotate=True
):
    """Accumulated phase during free evolution (see :mod:`nvsim.phase`).

    Each refocusing pulse is bridged by a dashed connector from the phase
    before it (open circle) to the phase after it (filled circle): phi -> -phi.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2.8))
    transitions = phase_transitions(result) if transitions is None else transitions
    if shade:
        shade_pulses(ax, result.segments)
    for m in transitions:
        color = LEVEL_COLORS[m]
        t, y = _break_at_wraps(result.times, result.accumulated_phase(m))
        ax.plot(
            t,
            y,
            color=color,
            lw=2,
            label=f"φ (0↔{m})" if len(transitions) > 1 else None,
        )
        jumps = phase_jumps(result, m)
        for jump in jumps:
            ax.plot(
                [jump.t_before, jump.t_after],
                [jump.phase_before, jump.phase_after],
                color=color,
                lw=1.2,
                ls=(0, (3, 2)),
            )
            ax.plot(
                jump.t_before,
                jump.phase_before,
                "o",
                ms=5,
                mfc="white",
                mec=color,
                mew=1.5,
            )
            ax.plot(jump.t_after, jump.phase_after, "o", ms=5, color=color)
        if annotate and jumps and abs(jumps[0].phase_before) > 0.15:
            first = jumps[0]
            below = first.phase_after < 0
            ax.annotate(
                "π pulse: φ → −φ",
                xy=(first.t_after, first.phase_after),
                xytext=(8, -10 if below else 10),
                textcoords="offset points",
                fontsize=9,
                color=TEXT_COLOR,
                va="top" if below else "bottom",
            )
    ax.set_xlim(result.times[0], result.times[-1])
    ax.set_xlabel("time (µs)")
    style_phase_axis(ax)
    if len(transitions) > 1:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return style_axes(ax)


def field_scale(bmax: float) -> tuple[float, str]:
    """Factor and unit label for displaying fields given in mT (µT below 1 mT)."""
    return (1e3, "µT") if bmax < 1 else (1.0, "mT")


def plot_field(result: SimulationResult, ax=None, shade=True):
    """Total axial field B_z(t) of the simulated fields (in µT when below 1 mT)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2.0))
    t = np.linspace(result.times[0], result.times[-1], 2000)
    b = total_field(result.fields, t)
    bmax = float(np.max(np.abs(b), initial=0.0))
    scale, unit = field_scale(bmax)
    if shade:
        shade_pulses(ax, result.segments)
    ax.axhline(0, color=MUTED_COLOR, lw=0.8, zorder=1)
    ax.plot(t, scale * b, color=TEXT_COLOR, lw=1.6)
    limit = 1.25 * scale * bmax if bmax > 0 else 1.0
    ax.set_ylim(-limit, limit)
    ax.set_xlim(result.times[0], result.times[-1])
    ax.set_xlabel("time (µs)")
    ax.set_ylabel(f"$B_z$ ({unit})")
    return style_axes(ax)


def plot_result(
    result: SimulationResult, title: str | None = None, show_field=None, show_phase=None
):
    """Sequence diagram, field (if any), populations and accumulated phase (if any)."""
    show_field = bool(result.fields) if show_field is None else show_field
    show_phase = bool(phase_transitions(result)) if show_phase is None else show_phase
    panels = (
        ["sequence"] + ["field"] * show_field + ["populations"] + ["phase"] * show_phase
    )
    heights = {"sequence": 1.0, "field": 1.3, "populations": 2.6, "phase": 2.0}
    plotters = {
        "sequence": plot_sequence,
        "field": plot_field,
        "populations": plot_populations,
        "phase": plot_phase,
    }
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(7.5, 1.0 + sum(heights[p] for p in panels)),
        sharex=True,
        gridspec_kw={"height_ratios": [heights[p] for p in panels]},
        layout="constrained",
    )
    for ax, panel in zip(axes, panels):
        plotters[panel](result, ax=ax)
        if ax is not axes[-1]:
            ax.set_xlabel("")
    fig.suptitle(title or result.sequence.name, color=TEXT_COLOR)
    return fig


def plot_sweep(result: SweepResult, ax=None, levels=None, xlabel=None, marker="o"):
    """Final populations vs. the swept parameter."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    levels = (
        _default_levels(result.populations)
        if levels is None
        else [normalize_level(lv) for lv in levels]
    )
    for level in levels:
        ax.plot(
            result.values,
            result.populations[level],
            color=LEVEL_COLORS[level],
            lw=2,
            marker=marker,
            ms=4,
            label=LEVEL_LABELS[level],
        )
    ax.set_xlabel(xlabel or result.parameter)
    ax.set_ylabel("final population")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return style_axes(ax)


__all__ = [
    "LEVEL_COLORS",
    "LEVEL_LABELS",
    "style_axes",
    "shade_pulses",
    "driven_transitions",
    "phase_transitions",
    "style_phase_axis",
    "plot_populations",
    "plot_phase",
    "plot_field",
    "plot_sequence",
    "plot_result",
    "plot_sweep",
]
