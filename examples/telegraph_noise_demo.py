"""Spin echo under random telegraph noise: weak vs strong coupling.

A field jumping between -b and +b with switching rate gamma (correlation time
tau_c = 1/(2 gamma)) has the same Lorentzian spectrum as Ornstein-Uhlenbeck
noise, but it is not Gaussian. The figure compares, for the Hahn echo,

* Monte Carlo (exact sampling of the switches),
* the exact telegraph result (nvsim.telegraph.rtn_echo_coherence),
* the Gaussian filter-function prediction with the same spectrum.

Weak coupling (2 pi gamma_e b tau_c << 1, many switches per radian of phase):
the noise averages to Gaussian and all three agree (motional narrowing).
Strong coupling (2 pi gamma_e b tau_c >> 1): a single switch during the echo
already scrambles the phase, so the coherence is the probability of no switch,
~exp(-gamma * total time) - far above the Gaussian prediction.

Run from the repository root:

    python examples/telegraph_noise_demo.py            # show figures
    python examples/telegraph_noise_demo.py --no-show  # only write output/telegraph_noise_echo.png
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "output"


def main(show: bool):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import nvsim
    from nvsim import GAMMA_E, uT
    from nvsim.ensemble import sweep_ensemble
    from nvsim.filter_function import coherence_decay
    from nvsim.noise import TelegraphNoise
    from nvsim.noise_plots import NOISE_FREE_COLOR, SEQUENCE_COLORS, THEORY_COLOR
    from nvsim.plotting import TEXT_COLOR, style_axes
    from nvsim.simulate import channel_map, sweep
    from nvsim.telegraph import rtn_echo_coherence

    OUTPUT.mkdir(exist_ok=True)
    nv = nvsim.NVCenter.from_field(5.0)  # mT
    drive = nvsim.MWChannel("-1", rabi_frequency=50.0)  # short pulses (pi = 10 ns)
    channels = channel_map(drive)
    regimes = [
        ("Weak coupling", TelegraphNoise.symmetric(2 * uT, rate=2.0), 25.0),
        ("Strong coupling", TelegraphNoise.symmetric(20 * uT, rate=0.2), 12.0),
    ]
    rng = np.random.default_rng(3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 2]})
    for col, (title, source, t_max) in enumerate(regimes):
        b, rate = source.b_high, source.rate_up
        tau_c = source.correlation_time
        regime = 2 * np.pi * GAMMA_E * b * tau_c
        color = SEQUENCE_COLORS[col]

        # Top: example traces of the field.
        ax = axes[0, col]
        t = np.linspace(0, t_max, 3000)
        traces = source.sample(t, 3, rng)
        for k in range(3):
            ax.plot(t, traces[k] / uT + 2.5 * (b / uT) * k, lw=1.0, color=SEQUENCE_COLORS[k])
        ax.set_yticks([])
        ax.set_xlim(0, t_max)
        ax.set_xlabel("time (µs)")
        ax.set_ylabel("B(t), 3 realisations\n(offset for clarity)")
        ax.set_title(f"{title}: ±{b / uT:g} µT, switching rate {rate:g} /µs   "
                     f"(2πγb·τ_c = {regime:.2g})", color=TEXT_COLOR, fontsize=10, loc="left")
        style_axes(ax)

        # Bottom: echo coherence vs total free evolution 2 tau.
        ax = axes[1, col]
        total_fine = np.linspace(0, t_max, 400)
        exact = rtn_echo_coherence(total_fine / 2, b, rate)
        total_mc = np.linspace(0.2, t_max, 30)
        mc = sweep_ensemble(total_mc / 2, nvsim.spin_echo, nv, drive, [source], n_realizations=1500, rng=rng)
        clean = sweep(total_mc / 2, nvsim.spin_echo, nv, drive).population("0")
        coherence = (mc.population("0") - 0.5) / (clean - 0.5)
        error = mc.population_error("0") / np.abs(clean - 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # deliberately applying the Gaussian theory to RTN
            gaussian = [coherence_decay(nvsim.spin_echo(T / 2).compile(channels), [source],
                                        allow_non_gaussian=True) for T in total_fine[1:]]
        ax.plot(total_fine, exact, color=THEORY_COLOR, lw=1.6, label="exact telegraph theory")
        ax.plot(total_fine[1:], gaussian, color=NOISE_FREE_COLOR, lw=2.2, ls=(0, (4, 2)),
                label="Gaussian theory, same spectrum")
        if regime > 1:
            ax.plot(total_fine, np.exp(-rate * total_fine), color=THEORY_COLOR, lw=1, ls=":",
                    label="exp(−γ·2τ): no switch")
        ax.errorbar(total_mc, coherence, yerr=error, fmt="o", ms=4, color=color, elinewidth=1,
                    label="Monte Carlo", zorder=3)
        ax.set_xlim(0, t_max)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("total free evolution 2τ (µs)")
        ax.set_ylabel("echo coherence")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        style_axes(ax)

    fig.suptitle("Hahn echo under random telegraph noise: Gaussian theory works only in weak coupling",
                 color=TEXT_COLOR)
    path = OUTPUT / "telegraph_noise_echo.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="only save files, do not open windows")
    main(show=not parser.parse_args().no_show)
