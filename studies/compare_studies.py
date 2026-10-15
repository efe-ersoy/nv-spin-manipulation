"""Compare noise scenarios side by side to see which feature comes from what.

Each comparison changes ONE thing (sample-noise frequency, the 13C bath, the
sample-noise linewidth) and overlays the resulting PL curves of every sequence
on a shared, logarithmic pulse-spacing axis.

The top panel of every figure is the noise spectrum S(f) read at the frequency
a CPMG/XY sequence probes at that pulse spacing, f = 1/(2(tau + t_pi)) (solid),
and at its 3rd harmonic f = 3/(2(tau + t_pi)) (dashed, weighted 1/9 like the
filter function). Every peak there sits directly above the dip it causes.

Run from the repository root:

    python studies/compare_studies.py                 # Gaussian theory (fast, smooth)
    python studies/compare_studies.py --monte-carlo   # also Monte Carlo points (slower)
    python studies/compare_studies.py --no-show

Output: output/comparisons/<comparison>.png and <comparison>_features.txt,
which lists where each feature is expected.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nvsim import MWChannel, NVCenter, mT, uT
from nvsim.filter_function import coherence_decay
from nvsim.noise import CarbonBath, NarrowbandNoise, total_psd
from nvsim.readout import Readout
from nvsim.simulate import channel_map, sweep
from nvsim.study import NoiseStudy, cpmg_scan, echo_scan, line_frequencies, ramsey_scan, run_scan, t2_star, xy8_scan

# =============================================================================
# Common settings
# =============================================================================

RABI_FREQUENCY = 20.0  # MHz -> pi = 25 ns
T_PI = 1 / (2 * RABI_FREQUENCY)
DRIVE = MWChannel("-1", rabi_frequency=RABI_FREQUENCY)
TAU = np.geomspace(0.05, 50.0, 240)  # pulse spacing [us], log-spaced
READOUT = Readout(contrast=0.3)
MC_REALIZATIONS = 150

SEQUENCES = {
    "Ramsey": lambda x: ramsey_scan(x),
    "Spin echo (CPMG-1)": lambda x: echo_scan(x),
    "CPMG-8": lambda x: cpmg_scan(x, 8),
    "XY8-2 (16 π)": lambda x: xy8_scan(x, 2),
}


def carbon(field=5 * mT, abundance=0.011):
    return CarbonBath(field=field, b_static=2 * uT, b_larmor=2 * uT, linewidth=0.002, abundance=abundance)


def sample(freq=0.5, rms=1.5 * uT, linewidth=0.02):
    return NarrowbandNoise(b_rms=rms, frequency=freq, linewidth=linewidth, label=f"sample {freq:g} MHz")


@dataclass
class Scenario:
    label: str
    noise: list
    bias_field: float = 5 * mT


# Each comparison changes one thing; everything else stays as in the first scenario.
COMPARISONS = {
    "sample_frequency": (
        "Sample-noise frequency (13C 1.1 % at 5 mT in all)",
        [
            Scenario("f_s = 0.25 MHz", [carbon(), sample(0.25)]),
            Scenario("f_s = 0.5 MHz", [carbon(), sample(0.5)]),
            Scenario("f_s = 1 MHz", [carbon(), sample(1.0)]),
        ],
    ),
    "carbon_bath": (
        "13C bath (sample noise 0.5 MHz, 1.5 µT in all)",
        [
            Scenario("no 13C", [sample()]),
            Scenario("13C 1.1 %, 5 mT", [carbon(), sample()]),
            Scenario("13C 1.1 %, 20 mT", [carbon(field=20 * mT), sample()], bias_field=20 * mT),
            Scenario("13C 4.4 %, 5 mT", [carbon(abundance=0.044), sample()]),
        ],
    ),
    "sample_linewidth": (
        "Sample-noise linewidth (0.5 MHz, 1.5 µT rms, no 13C)",
        [
            Scenario("linewidth 2 kHz", [sample(linewidth=0.002)]),
            Scenario("linewidth 20 kHz", [sample(linewidth=0.02)]),
            Scenario("linewidth 100 kHz", [sample(linewidth=0.1)]),
        ],
    ),
}

SCENARIO_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4

# =============================================================================


def theory_pl(scan, nv, noise):
    """PL from the noise-free simulation times the Gaussian decay factor."""
    channels = channel_map(DRIVE)
    p0_clean = sweep(scan.taus, scan.build, nv, DRIVE).population("0")
    decay = np.array([coherence_decay(scan.build(t).compile(channels), noise) for t in scan.taus])
    return READOUT.pl({"0": 0.5 + (p0_clean - 0.5) * decay})


def monte_carlo_pl(scan, nv, noise, seed):
    study = NoiseStudy("mc", nv, DRIVE, [scan], noise, n_realizations=MC_REALIZATIONS, readout=READOUT,
                       theory=False, seed=seed)
    out = run_scan(study, scan, np.random.default_rng(seed))
    return out.signal, out.signal_error


def features(scenario: Scenario) -> list[str]:
    """Where each feature should appear, in pulse spacing tau [us]."""
    lines = [f"{scenario.label}:"]
    t2 = t2_star(scenario.noise)
    if np.isfinite(t2):
        lines.append(f"  Ramsey decays with T2* = {t2:.2f} us (quasi-static part of the 13C bath)")
    for f in line_frequencies(scenario.noise):
        name = "13C Larmor" if abs(f - 0.0107084 * scenario.bias_field) < 1e-9 else "sample line"
        lines.append(f"  {name} f = {1e3 * f:.1f} kHz:")
        lines.append(f"    CPMG/XY dip (fundamental)  tau = 1/(2f) - t_pi = {1 / (2 * f) - T_PI:.3f} us")
        lines.append(f"    CPMG/XY dip (3rd harmonic) tau = 3/(2f) - t_pi = {3 / (2 * f) - T_PI:.3f} us")
        lines.append(f"    echo strongest decay near  tau = 1/f = {1 / f:.3f} us; "
                     f"refocused (revival) at tau = 2k/f = {2 / f:.3f}, {4 / f:.3f} ... us")
    return lines


def figure(title, scenarios, monte_carlo: bool):
    import matplotlib.pyplot as plt

    from nvsim.plotting import TEXT_COLOR, style_axes

    n_rows = 1 + len(SEQUENCES)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.3 * n_rows + 0.8), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.3] + [1] * len(SEQUENCES)})
    probe = 1 / (2 * (TAU + T_PI))
    for i, sc in enumerate(scenarios):
        color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]
        nv = NVCenter.from_field(sc.bias_field)
        psd = 1e6 * total_psd(sc.noise, probe)  # uT^2/MHz
        psd3 = 1e6 * total_psd(sc.noise, 3 * probe) / 9
        axes[0].plot(TAU, psd, color=color, lw=1.8, label=sc.label)
        axes[0].plot(TAU, psd3, color=color, lw=1.2, ls=(0, (3, 2)))
        for ax, (name, make) in zip(axes[1:], SEQUENCES.items()):
            scan = make(TAU)
            ax.plot(TAU, theory_pl(scan, nv, sc.noise), color=color, lw=1.8, label=sc.label)
            if monte_carlo:
                y, err = monte_carlo_pl(scan, nv, sc.noise, seed=i)
                ax.errorbar(TAU, y, yerr=err, fmt="o", ms=2.5, color=color, elinewidth=0.8, alpha=0.7)
        print(f"  {sc.label} done")

    axes[0].set_yscale("log")
    peak = axes[0].get_ylim()[1]
    axes[0].set_ylim(peak * 1e-5, peak * 2)
    axes[0].set_ylabel("S(f) at probed f\n(µT²/MHz)")
    axes[0].set_title("Noise spectrum at the CPMG/XY probed frequency f = 1/(2(τ+t_π)) (solid) "
                      "and its 3rd harmonic (dashed, ×1/9)", color=TEXT_COLOR, fontsize=9, loc="left")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    style_axes(axes[0])
    for ax, name in zip(axes[1:], SEQUENCES):
        ax.set_title(name, color=TEXT_COLOR, fontsize=10, loc="left")
        ax.set_ylabel("PL")
        ax.set_ylim(0.68, 1.02)
        style_axes(ax)
    axes[-1].set_xscale("log")
    axes[-1].set_xlim(TAU[0], TAU[-1])
    axes[-1].set_xlabel("pulse spacing τ (µs, log)   [Ramsey, echo: total free evolution]")
    fig.suptitle(title + ("   (lines: Gaussian theory, points: Monte Carlo)" if monte_carlo else
                          "   (Gaussian theory)"), color=TEXT_COLOR)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="only save files, do not open windows")
    parser.add_argument("--monte-carlo", action="store_true", help="also run the Monte Carlo (slower)")
    parser.add_argument("--only", choices=list(COMPARISONS), help="run a single comparison")
    args = parser.parse_args()
    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(__file__).resolve().parents[1] / "output" / "comparisons"
    output.mkdir(parents=True, exist_ok=True)
    for key, (title, scenarios) in COMPARISONS.items():
        if args.only and key != args.only:
            continue
        text = [title, f"pi pulse t_pi = {1e3 * T_PI:.0f} ns", ""]
        for sc in scenarios:
            text += features(sc) + [""]
        (output / f"{key}_features.txt").write_text("\n".join(text), encoding="utf-8")
        print("\n".join(text))
        fig = figure(title, scenarios, args.monte_carlo)
        fig.savefig(output / f"{key}.png", dpi=150)
        print(f"wrote {output / f'{key}.png'}\n")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
