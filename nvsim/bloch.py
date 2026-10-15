"""Bloch-sphere plots and animations of the {|0>, |m>} subspaces.

|0> is drawn at the north pole, |m> (m = -1 or +1) at the south pole. Pulses
with phase 0 rotate about +x, phase pi/2 about +y.

During free evolution the animation also shows the field-free *reference spin*
(dashed grey) and the accumulated phase phi as an arc from the reference to the
actual spin (see :mod:`nvsim.phase`). A refocusing pi pulse mirrors the
equatorial plane, so the arc reappears on the other side of the reference: -phi.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from PIL import Image

from .phase import phase_jumps, reference_result
from .plotting import (
    GRID_COLOR,
    LEVEL_COLORS,
    LEVEL_LABELS,
    MUTED_COLOR,
    TEXT_COLOR,
    driven_transitions,
    phase_transitions,
    plot_field,
    plot_phase,
    plot_populations,
)
from .results import SimulationResult
from .system import normalize_transition

REFERENCE_COLOR = "#8a8984"
ARC_RADIUS = 0.5


def draw_bloch_sphere(ax, transition="-1", elev: float = 20, azim: float = -60):
    """Draw an empty Bloch sphere for the {|0>, |m>} subspace on a 3D axis."""
    transition = normalize_transition(transition)
    u = np.linspace(0, 2 * np.pi, 33)
    v = np.linspace(0, np.pi, 17)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="#d9d8d3",
        alpha=0.10,
        linewidth=0,
        shade=False,
    )
    u = np.linspace(0, 2 * np.pi, 120)
    c, s, z = np.cos(u), np.sin(u), np.zeros_like(u)
    for circle in ((c, s, z), (c, z, s), (z, c, s)):
        ax.plot(*circle, color=GRID_COLOR, lw=0.9)
    for axis in np.eye(3):
        ax.plot(*np.column_stack([-axis, axis]), color=MUTED_COLOR, lw=0.7)
    ax.text(
        0,
        0,
        1.22,
        LEVEL_LABELS["0"],
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=12,
    )
    ax.text(
        0,
        0,
        -1.28,
        LEVEL_LABELS[transition],
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=12,
    )
    ax.text(1.3, 0, 0, "x", ha="center", va="center", color=MUTED_COLOR)
    ax.text(0, 1.22, 0, "y", ha="center", va="center", color=MUTED_COLOR)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1), zoom=1.25)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return ax


def plot_bloch_trajectory(result: SimulationResult, transition=None, ax=None):
    """Static Bloch sphere with the full trajectory, start (○) and end (●) marked."""
    transition = normalize_transition(transition or driven_transitions(result)[0])
    if ax is None:
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(projection="3d")
    draw_bloch_sphere(ax, transition)
    b = result.bloch_vector(transition)
    color = LEVEL_COLORS[transition]
    ax.plot(b[:, 0], b[:, 1], b[:, 2], color=color, lw=1.6)
    ax.plot(*b[0], marker="o", ms=8, mfc="white", mec=color, mew=2)
    ax.plot(*b[-1], marker="o", ms=8, color=color)
    ax.plot(*np.column_stack([np.zeros(3), b[-1]]), color=color, lw=2.5)
    return ax


def _frame_indices(
    result: SimulationResult, max_frames: int, min_frames_per_segment: int
) -> np.ndarray:
    """Pick sample indices so short pulses still get enough frames to be seen."""
    times = result.times
    if not result.segments:
        return np.array([0])
    durations = np.array([s.duration for s in result.segments])
    budget = max(max_frames - min_frames_per_segment * len(durations), 0)
    alloc = min_frames_per_segment + np.round(
        budget * durations / durations.sum()
    ).astype(int)
    eps = 1e-12 * max(times[-1], 1.0)
    picks = [np.array([0])]
    for segment, n in zip(result.segments, alloc):
        lo = np.searchsorted(times, segment.start - eps)
        hi = np.searchsorted(times, segment.end + eps, side="right") - 1
        picks.append(np.round(np.linspace(lo, hi, max(n, 2))).astype(int))
    return np.unique(np.concatenate(picks))


def _format_phase(phi: float) -> str:
    return f"φ = {phi / np.pi:+.2f}π"


class _SphereArtists:
    """Everything drawn on one Bloch sphere that changes from frame to frame."""

    def __init__(self, ax, result: SimulationResult, transition: str, show_phase: bool):
        self.ax = ax
        self.b = result.bloch_vector(transition)
        color = LEVEL_COLORS[transition]
        (self.trail,) = ax.plot([], [], [], color=color, lw=1.3, alpha=0.6)
        (self.vector,) = ax.plot(
            [], [], [], color=color, lw=2.8, solid_capstyle="round"
        )
        (self.tip,) = ax.plot(
            [], [], [], marker="o", ms=8, color=color, mec="white", mew=1.5
        )
        self.dynamic = [self.trail, self.vector, self.tip]

        self.show_phase = show_phase
        if not show_phase:
            return
        self.r = reference_result(result).bloch_vector(transition)
        self.phi = result.accumulated_phase(transition)
        self.flip_segments = {id(j.segment) for j in phase_jumps(result, transition)}
        (self.ref_vector,) = ax.plot(
            [], [], [], color=REFERENCE_COLOR, lw=1.8, ls=(0, (3, 2))
        )
        (self.ref_tip,) = ax.plot(
            [], [], [], marker="o", ms=6, mfc="white", mec=REFERENCE_COLOR, mew=1.5
        )
        (self.arc,) = ax.plot([], [], [], color=color, lw=2.4)
        self.arc_label = ax.text(
            0, 0, 0, "", color=TEXT_COLOR, fontsize=10, ha="center", va="center"
        )
        ax.text2D(
            0.0,
            1.0,
            "- - field-free reference spin",
            transform=ax.transAxes,
            va="top",
            color=REFERENCE_COLOR,
            fontsize=8,
        )
        self.banner = ax.text2D(
            0.5,
            0.0,
            "",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=10,
            fontweight="bold",
        )
        # Reference first so the actual spin is drawn on top of it.
        self.dynamic = [
            self.ref_vector,
            self.ref_tip,
            self.arc,
            self.arc_label,
            self.banner,
        ] + self.dynamic

    def update(self, k: int, trail_start: int | None, segment):
        b = self.b
        if trail_start is not None:
            path = b[trail_start : k + 1]
            self.trail.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
        self.vector.set_data_3d([0, b[k, 0]], [0, b[k, 1]], [0, b[k, 2]])
        self.tip.set_data_3d([b[k, 0]], [b[k, 1]], [b[k, 2]])
        if not self.show_phase:
            return

        r = self.r[k]
        self.ref_vector.set_data_3d([0, r[0]], [0, r[1]], [0, r[2]])
        self.ref_tip.set_data_3d([r[0]], [r[1]], [r[2]])

        phi = self.phi[k]
        if np.isfinite(phi) and abs(phi) > 1e-3:
            start = np.arctan2(r[1], r[0])
            angles = start + np.linspace(0, phi, 40)
            self.arc.set_data_3d(
                ARC_RADIUS * np.cos(angles),
                ARC_RADIUS * np.sin(angles),
                np.zeros_like(angles),
            )
            mid = start + phi / 2
            self.arc_label.set_position_3d(
                (0.72 * np.cos(mid), 0.72 * np.sin(mid), 0.0)
            )
            self.arc_label.set_text("φ" if phi > 0 else "−φ")
        else:
            self.arc.set_data_3d([], [], [])
            self.arc_label.set_text("")

        flipping = segment is not None and id(segment) in self.flip_segments
        self.banner.set_text("π pulse mirrors the spin:  φ → −φ" if flipping else "")


def animate_bloch(
    result: SimulationResult,
    transitions=None,
    show_populations: bool = True,
    show_phase: bool | None = None,
    show_field: bool | None = None,
    max_frames: int = 240,
    min_frames_per_segment: int = 20,
    interval: int = 40,
    trail: bool = True,
    trail_window: float | None = None,
    hold_frames: int = 15,
    figsize=None,
) -> FuncAnimation:
    """Animate the Bloch vector(s) through the sequence.

    Parameters
    ----------
    transitions : str or list of str, optional
        Which subspaces to show, one sphere each. Defaults to the driven ones.
    show_populations : bool
        Add a population panel with a moving time cursor.
    show_phase : bool, optional
        Add the accumulated-phase panel, and draw the reference spin and the
        phase arc on the spheres. Defaults to True whenever the sequence has
        free evolution after a pi/2 pulse.
    show_field : bool, optional
        Add a panel with the axial field B_z(t). Defaults to True if the
        simulation included fields.
    max_frames : int
        Approximate number of animation frames (excluding the final hold).
    min_frames_per_segment : int
        Every pulse / wait gets at least this many frames, so a short pulse in a
        long sequence is not skipped over.
    interval : int
        Delay between frames [ms] for on-screen playback.
    trail : bool
        Draw the path the vector has travelled.
    trail_window : float, optional
        Only draw the last ``trail_window`` microseconds of the path (keeps long
        sequences such as XY8 readable). Default: the whole path.
    hold_frames : int
        Extra frames showing the final state.

    Display with ``plt.show()``, embed in Jupyter with ``anim.to_jshtml()``,
    or write a file with :func:`save_animation`.
    """
    if transitions is None:
        transitions = driven_transitions(result)
    elif isinstance(transitions, str):
        transitions = [transitions]
    transitions = [normalize_transition(m) for m in transitions]
    with_phase = set(phase_transitions(result))
    show_phase = (
        bool(with_phase & set(transitions)) if show_phase is None else show_phase
    )
    show_field = bool(result.fields) if show_field is None else show_field

    panels = (
        ["field"] * show_field
        + ["populations"] * show_populations
        + ["phase"] * show_phase
    )
    heights = {"field": 1.0, "populations": 1.6, "phase": 1.6}
    n = len(transitions)

    # One sphere: sphere on the left, panels stacked on the right.
    # Several spheres: spheres side by side on top, panels below.
    side_by_side = n == 1 and panels
    if figsize is None:
        if side_by_side:
            figsize = (12.5, max(5.6, 1.4 + 1.5 * sum(heights[p] for p in panels)))
        else:
            figsize = (max(4.6 * n, 7.5), 4.6 + 1.5 * sum(heights[p] for p in panels))
    fig = plt.figure(figsize=figsize, layout="constrained")
    if side_by_side:
        grid = fig.add_gridspec(
            len(panels),
            2,
            width_ratios=[1, 1.15],
            height_ratios=[heights[p] for p in panels],
        )
        sphere_specs = [grid[:, 0]]
        panel_specs = [grid[i, 1] for i in range(len(panels))]
    else:
        grid = fig.add_gridspec(
            1 + len(panels), n, height_ratios=[3] + [heights[p] for p in panels]
        )
        sphere_specs = [grid[0, i] for i in range(n)]
        panel_specs = [grid[1 + i, :] for i in range(len(panels))]

    spheres = []
    for spec, m in zip(sphere_specs, transitions):
        ax = fig.add_subplot(spec, projection="3d")
        draw_bloch_sphere(ax, m)
        spheres.append(_SphereArtists(ax, result, m, show_phase and m in with_phase))

    plotters = {
        "field": plot_field,
        "populations": plot_populations,
        "phase": plot_phase,
    }
    cursors, phase_dots, first_ax = [], [], None
    for spec, panel in zip(panel_specs, panels):
        ax = fig.add_subplot(spec, sharex=first_ax)
        first_ax = first_ax or ax
        plotters[panel](result, ax=ax)
        if panel != panels[-1]:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        cursors.append(ax.axvline(0, color=TEXT_COLOR, lw=1.2))
        if panel == "phase":
            for m in transitions:
                if m in with_phase:
                    (dot,) = ax.plot(
                        [], [], "o", ms=7, color=LEVEL_COLORS[m], mec="white", mew=1.2
                    )
                    phase_dots.append((dot, result.accumulated_phase(m)))
    title = fig.suptitle("", color=TEXT_COLOR)
    first_phase = (
        result.accumulated_phase(transitions[0])
        if show_phase and transitions[0] in with_phase
        else None
    )

    indices = _frame_indices(result, max_frames, min_frames_per_segment)
    frames = [(int(k), False) for k in indices] + [
        (int(indices[-1]), True)
    ] * hold_frames

    # Solve the layout once with the longest title, then freeze it: re-running
    # constrained layout on every frame dominates the rendering time.
    title.set_text(
        f"{result.sequence.name}   t = 0.000 µs   [π/2 (readout)]   φ = +0.00π"
    )
    fig.canvas.draw()
    fig.set_layout_engine("none")

    def update(frame):
        k, holding = frame
        t = result.times[k]
        segment = None if holding else result.segment_at(t)
        trail_start = None
        if trail:
            trail_start = (
                0
                if trail_window is None
                else int(np.searchsorted(result.times, t - trail_window))
            )
        for sphere in spheres:
            sphere.update(k, trail_start, segment)
        for cursor in cursors:
            cursor.set_xdata([t, t])
        for dot, phi in phase_dots:
            if np.isfinite(phi[k]):
                dot.set_data([t], [phi[k]])
            else:
                dot.set_data([], [])
        stage = segment.label if segment is not None else "end"
        text = f"{result.sequence.name}   t = {t:.3f} µs   [{stage}]"
        if first_phase is not None and np.isfinite(first_phase[k]):
            text += f"   {_format_phase(first_phase[k])}"
        title.set_text(text)
        return []

    dynamic = (
        [a for s in spheres for a in s.dynamic]
        + cursors
        + [d for d, _ in phase_dots]
        + [title]
    )
    return BlochAnimation(fig, update, frames, dynamic, interval=interval)


class BlochAnimation(FuncAnimation):
    """A FuncAnimation that also knows which artists change between frames,
    so :func:`save_animation` can redraw only those (much faster GIF export)."""

    def __init__(self, fig, update, frames, dynamic_artists, interval=40):
        self.figure_ = fig
        self.update_frame = update
        self.frame_list = list(frames)
        self.dynamic_artists = list(dynamic_artists)
        super().__init__(
            fig, update, frames=self.frame_list, interval=interval, blit=False
        )

    def render_frames(self, dpi: int) -> list[Image.Image]:
        """Render every frame: static background once, moving artists per frame."""
        fig, canvas = self.figure_, self.figure_.canvas
        self._draw_was_started = (
            True  # silences matplotlib's "deleted without rendering" warning
        )
        old_dpi = fig.dpi
        fig.set_dpi(dpi)
        for artist in self.dynamic_artists:
            artist.set_animated(True)  # animated artists are left out of canvas.draw()
        try:
            canvas.draw()
            background = canvas.copy_from_bbox(fig.bbox)
            size = canvas.get_width_height()
            images = []
            for frame in self.frame_list:
                self.update_frame(frame)
                canvas.restore_region(background)
                for artist in self.dynamic_artists:
                    fig.draw_artist(artist)
                images.append(
                    Image.frombuffer(
                        "RGBA", size, bytes(canvas.buffer_rgba()), "raw", "RGBA", 0, 1
                    )
                )
            return images
        finally:
            for artist in self.dynamic_artists:
                artist.set_animated(False)
            fig.set_dpi(old_dpi)


def save_animation(anim: FuncAnimation, path, fps: int = 25, dpi: int = 80):
    """Save to .gif (Pillow, always available) or .mp4 (needs ffmpeg on PATH)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if (
        suffix == ".gif"
        and isinstance(anim, BlochAnimation)
        and hasattr(anim.figure_.canvas, "copy_from_bbox")
    ):
        images = anim.render_frames(dpi)
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        return path
    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not FFMpegWriter.isAvailable():
            raise RuntimeError(
                "Saving .mp4 needs ffmpeg on PATH; save as .gif instead."
            )
        writer = FFMpegWriter(fps=fps)
    else:
        raise ValueError(f"Unsupported animation format {suffix!r}; use .gif or .mp4.")
    anim.save(path, writer=writer, dpi=dpi)
    return path
