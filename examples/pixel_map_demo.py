"""Echo signal maps for many pixels at once (synthetic example).

A 2 x 2 um map (100 x 100 px) measured with a Hahn echo (2 x 5 us), showing the
signal se_norm = (N2 - N1)/(N2 + N1) for three synthetic noise scenarios:

* background: the same Gaussian noise in every pixel (intrinsic decay);
* "fluctuators": a few localised telegraph fluctuators (switching rate 0.08 /us)
  whose field jump falls off with distance from each site, so the echo is lost
  in spots (strong coupling near the sites, weak coupling further out);
* "tip oscillation": a 32 kHz oscillation of the scan position (amplitude
  3 nm, random phase relative to the sequence) in the gradient of a
  synthetic triangular bubble-lattice field, i.e. a per-pixel random-phase AC
  field with amplitude = 3 nm x |dB/dx|;
* both together.

The vectorised theory path computes each 100 x 100 map in well under a second.
A line cut through one spot is checked against the Monte Carlo path.

Run from the repository root:

    python examples/pixel_map_demo.py            # show figures
    python examples/pixel_map_demo.py --no-show  # write output/pixel_maps/
"""

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "output" / "pixel_maps"


def lattice_field(x, y, a=0.444, amplitude=0.1, width=0.12):
    """Synthetic triangular lattice of Gaussian field bumps [mT] (positions in um)."""
    b = np.zeros_like(x)
    for i in range(-2, 8):
        for j in range(-2, 8):
            cx, cy = a * (i + 0.5 * j), a * j * np.sqrt(3) / 2
            b -= amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width**2))
    return b


def main(show: bool):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import nvsim
    from nvsim import ACField, uT
    from nvsim.maps import PixelField, PixelSource, monte_carlo_map, signal_map
    from nvsim.noise import OUNoise, TelegraphNoise
    from nvsim.noise_plots import SEQUENCE_COLORS
    from nvsim.plotting import MUTED_COLOR, TEXT_COLOR, style_axes
    from nvsim.readout import Readout

    OUTPUT.mkdir(parents=True, exist_ok=True)
    drive = nvsim.MWChannel("-1", rabi_frequency=20.0)
    sequence = nvsim.spin_echo(5.0)  # 2 x 5 us
    readout = Readout(contrast=0.3)

    x, y = np.meshgrid(np.linspace(0, 2, 100), np.linspace(0, 2, 100))  # um; rows = y
    pixel = 2 / 99

    background = OUNoise(1.0 * uT, correlation_time=30.0, label="background")
    sites = [(0.75, 1.2), (1.3, 0.55), (1.55, 1.45)]
    distance = np.min([np.hypot(x - sx, y - sy) for sx, sy in sites], axis=0)
    jump = 40 * uT * np.exp(-((distance / 0.15) ** 2))  # field jump seen by the NV [mT]
    fluctuators = PixelSource(TelegraphNoise.symmetric, b=jump / 2, rate=0.08, label="fluctuator")

    b_lattice = lattice_field(x, y)
    gradient = np.abs(np.gradient(b_lattice, pixel * 1e3, axis=1))  # mT / nm
    tip_amplitude = 3.0  # nm
    tip = PixelField(ACField, random_phase=True, amplitude=tip_amplitude * gradient, frequency=0.032)

    scenarios = {
        "background only": ([background], []),
        "fluctuators (H1-like)": ([background, fluctuators], []),
        "tip oscillation (H2-like)": ([background], [tip]),
        "fluctuators + tip oscillation": ([background, fluctuators], [tip]),
    }
    maps, timings = {}, {}
    for name, (sources, fields) in scenarios.items():
        start = time.perf_counter()
        maps[name] = signal_map(sequence, drive, sources, fields, readout=readout, shape=x.shape)
        timings[name] = time.perf_counter() - start
        se = maps[name].se_norm
        print(f"{name:32s} {timings[name]:.2f} s   se_norm {se.min():+.4f} .. {se.max():+.4f}")

    # Line cut through the first site: theory vs Monte Carlo.
    row = int(round(sites[0][1] / pixel))
    cols = np.arange(10, 60, 3)
    cut_sources = [background, PixelSource(TelegraphNoise.symmetric, b=jump[row, cols] / 2, rate=0.08)]
    start = time.perf_counter()
    mc = monte_carlo_map(sequence, drive, cut_sources, n_realizations=400, readout=readout, seed=1)
    mc_time = time.perf_counter() - start
    theory_cut = maps["fluctuators (H1-like)"].se_norm[row, cols]
    print(f"Monte Carlo line cut: {len(cols)} pixels in {mc_time:.1f} s; "
          f"max |MC - theory| = {np.max(np.abs(mc.se_norm - theory_cut)):.4f} "
          f"(MC error up to {mc.se_norm_error.max():.4f})")

    extent = (0, 2, 0, 2)
    vmin = min(m.se_norm.min() for m in maps.values())
    vmax = max(m.se_norm.max() for m in maps.values())
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), layout="constrained")
    inputs = [(b_lattice, "lattice field B_nv (mT)", "viridis"),
              (tip_amplitude * gradient / uT, "tip-oscillation AC amplitude (µT)", "magma"),
              (jump / uT, "fluctuator jump Δ (µT)", "magma")]
    for ax, (img, title, cmap) in zip(axes[0, :3], inputs):
        im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap)
        ax.set_title(title, color=TEXT_COLOR, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
    ax = axes[0, 3]
    ax.plot(x[row, cols], theory_cut, color=SEQUENCE_COLORS[0], lw=2, label="theory")
    ax.errorbar(x[row, cols], mc.se_norm, yerr=mc.se_norm_error, fmt="o", ms=4, color=SEQUENCE_COLORS[1],
                label="Monte Carlo (400 / pixel)")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("se_norm")
    ax.set_title(f"line cut y = {y[row, 0]:.2f} µm (fluctuators)", color=TEXT_COLOR, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    for ax, (name, result) in zip(axes[1], maps.items()):
        im = ax.imshow(result.se_norm, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"se_norm: {name}\n({timings[name]:.2f} s for 100 × 100 px)", color=TEXT_COLOR, fontsize=10)
        if name.startswith("fluct"):
            ax.axhline(y[row, 0], color=MUTED_COLOR, lw=0.6, ls=":")
        fig.colorbar(im, ax=ax, shrink=0.85)
    for ax in axes.ravel()[:3].tolist() + axes[1].tolist():
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    fig.suptitle("Hahn echo 2 × 5 µs: se_norm maps for synthetic noise scenarios (contrast 0.3, N2 = dark readout)",
                 color=TEXT_COLOR)
    path = OUTPUT / "se_norm_maps.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="only save files, do not open windows")
    main(show=not parser.parse_args().no_show)
