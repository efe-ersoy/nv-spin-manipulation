"""Fitting echo decays on and off a noise hotspot with physical noise models.

Synthetic data shaped like the hotspot height series: se_norm vs total delay
0.4-32 us, readout contrast 0.3 (se_norm = -0.176 C), noise 0.012 per point.

* off the hotspot: intrinsic NV decay exp(-(t/T2)^n), T2 = 10 us, n = 1.5;
* on the hotspot: the same intrinsic decay x a symmetric telegraph fluctuator,
  switching rate 0.3 /us, jump 5 uT.

Both curves are fitted jointly with four models for the hotspot (the off curve
is always intrinsic only), so their AIC values compare like with like:

1. separate stretched exponentials (no shared parameters);
2. shared intrinsic decay x telegraph fluctuator;
3. shared intrinsic decay x Gaussian OU noise;
4. shared intrinsic decay x an extra stretched exponential.

Also shown: the profile chi^2 of the telegraph jump (how well the data fix it).

Run from the repository root:

    python examples/decay_fitting_demo.py            # show figures
    python examples/decay_fitting_demo.py --no-show  # write output/decay_fitting/

The comparison is repeated with 3x lower noise to show when the models become
distinguishable.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "output" / "decay_fitting"


def main(show: bool):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nvsim import uT
    from nvsim.fitting import (
        Curve, DecayModel, DecaySequence, GaussianNoiseDecay, StretchedExp, TelegraphDecay, fit_joint,
        profile_chi2,
    )
    from nvsim.noise import OUNoise
    from nvsim.noise_plots import SEQUENCE_COLORS
    from nvsim.plotting import MUTED_COLOR, TEXT_COLOR, style_axes
    from nvsim.readout import Readout

    OUTPUT.mkdir(parents=True, exist_ok=True)
    seq = DecaySequence("echo", t_pi=0.05)
    t = np.linspace(0.4, 32, 40)
    amplitude = Readout(contrast=0.3).se_norm_amplitude()
    truth = {"offset": 0.0, "amplitude": amplitude, "intrinsic.T2": 10.0, "intrinsic.n": 1.5,
             "rtn.rate": 0.3, "rtn.jump": 5 * uT}
    truth_global = {"intrinsic.T2": 10.0, "intrinsic.n": 1.5, "on:rtn.rate": 0.3, "on:rtn.jump": 5 * uT}
    intrinsic = DecayModel([StretchedExp()], seq)
    hotspot_truth = DecayModel([StretchedExp(), TelegraphDecay()], seq)
    shared = ["intrinsic.T2", "intrinsic.n"]
    ou = GaussianNoiseDecay(lambda b, tc: [OUNoise(b, tc)], p0={"b": 3 * uT, "tc": 1.0},
                            bounds={"b": (1e-4, 0.1), "tc": (0.01, 100.0)}, name="ou",
                            grid={"b": np.geomspace(0.3 * uT, 30 * uT, 7), "tc": np.geomspace(0.05, 20, 7)})
    candidates = {
        "separate stretched exps": (DecayModel([StretchedExp()], seq), []),
        "intrinsic x telegraph": (hotspot_truth, shared),
        "intrinsic x OU (Gaussian)": (DecayModel([StretchedExp(), ou], seq), shared),
        "intrinsic x extra stretched": (DecayModel([StretchedExp(), StretchedExp("extra", T2=5.0, n=1.0)], seq),
                                        shared),
    }

    report = []
    for sigma, tag in [(0.012, "screenshot noise"), (0.004, "3x lower noise (9x more averaging)")]:
        rng = np.random.default_rng(7)
        y_off = intrinsic.signal(t, truth) + sigma * rng.standard_normal(t.size)
        y_on = hotspot_truth.signal(t, truth) + sigma * rng.standard_normal(t.size)
        fits = {}
        for name, (on_model, share) in candidates.items():
            curves = [Curve(t, y_off, sigma, intrinsic, "off"), Curve(t, y_on, sigma, on_model, "on")]
            fits[name] = fit_joint(curves, shared=share)
        best_aic = min(f.aic for f in fits.values())
        rtn = fits["intrinsic x telegraph"]
        report += [f"## sigma = {sigma} ({tag})", "",
                   "| hotspot model | k | chi2 | reduced chi2 | AIC | dAIC | BIC |", "|---|---|---|---|---|---|---|"]
        for name, f in fits.items():
            report.append(f"| {name} | {f.n_free} | {f.chi2:.1f} | {f.reduced_chi2:.2f} | {f.aic:.1f} | "
                          f"{f.aic - best_aic:.1f} | {f.bic:.1f} |")
        report += ["", "Telegraph model, fitted vs true:", "", "| parameter | fit | error | true | pull |",
                   "|---|---|---|---|---|"]
        for p_name, true_value in truth_global.items():
            pull = (rtn.params[p_name] - true_value) / rtn.errors[p_name]
            report.append(f"| {p_name} | {rtn.params[p_name]:.4g} | {rtn.errors[p_name]:.2g} | "
                          f"{true_value:.4g} | {pull:+.2f} |")
        report.append("")

        jumps = np.geomspace(1 * uT, 300 * uT, 28)
        profile = profile_chi2(rtn, "on:rtn.jump", jumps)
        figure(plt, t, y_off, y_on, sigma, fits, best_aic, jumps, profile, tag, uT, SEQUENCE_COLORS,
               MUTED_COLOR, TEXT_COLOR, style_axes, OUTPUT / f"decay_fits_sigma_{sigma:g}.png")

    table = "\n".join(report)
    (OUTPUT / "aic_table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"wrote {OUTPUT}")
    if show:
        plt.show()
    else:
        plt.close("all")


def figure(plt, t, y_off, y_on, sigma, fits, best_aic, jumps, profile, tag, uT, colors, muted, text, style_axes,
           path):
    rtn = fits["intrinsic x telegraph"]
    fig = plt.figure(figsize=(13, 7.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[2, 1])
    ax_data, ax_res, ax_prof = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
    t_fine = np.linspace(0, 32, 400)
    ax_data.errorbar(t, y_off, yerr=sigma, fmt="s", ms=4, color=muted, elinewidth=0.8, label="off hotspot")
    ax_data.errorbar(t, y_on, yerr=sigma, fmt="o", ms=4, color=text, elinewidth=0.8, label="on hotspot")
    ax_data.plot(t_fine, rtn.signal(0, t_fine), color=muted, lw=1.5, label="off: intrinsic fit")
    for k, (name, f) in enumerate(fits.items()):
        ax_data.plot(t_fine, f.signal(1, t_fine), color=colors[k], lw=1.8,
                     label=f"on: {name} (ΔAIC {f.aic - best_aic:.1f})")
        ax_res.plot(t, (y_on - f.signal(1)) / sigma, "o-", ms=3, lw=1, color=colors[k])
    ax_data.axhline(0, color=muted, lw=0.8)
    ax_data.set_ylabel("se_norm")
    ax_data.set_title(f"Joint fits of the off- and on-hotspot echo (synthetic data, σ = {sigma:g}: {tag})",
                      color=text, loc="left", fontsize=10)
    ax_data.legend(frameon=False, fontsize=8, loc="lower right")
    ax_res.axhline(0, color=muted, lw=0.8)
    ax_res.set_xlabel("total free evolution (µs)")
    ax_res.set_ylabel("on-hotspot\nresidual / σ")
    rel = profile - profile.min()
    ax_prof.plot(jumps / uT, rel, "o-", ms=4, color=colors[1])
    ax_prof.axhline(4, color=muted, lw=1, ls=(0, (4, 2)))
    ax_prof.axvline(5, color=text, lw=1, ls=":")
    top = min(60.0, float(np.nanmax(rel)) + 2)
    ax_prof.set_ylim(-0.5, top)
    ax_prof.text(jumps[0] / uT, 4.3, "Δχ² = 4 (≈ 2σ)", fontsize=8, color=text)
    ax_prof.text(5.3, 0.92 * top, "true jump", fontsize=8, color=text)
    ax_prof.set_xscale("log")
    ax_prof.set_xlabel("telegraph jump Δ (µT)")
    ax_prof.set_ylabel("profile Δχ²")
    ax_prof.set_title("How well the data fix the jump", color=text, loc="left", fontsize=10)
    for ax in (ax_data, ax_res, ax_prof):
        style_axes(ax)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="only save files, do not open windows")
    main(show=not parser.parse_args().no_show)
