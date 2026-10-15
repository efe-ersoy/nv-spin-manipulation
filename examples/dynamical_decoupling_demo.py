"""Dynamical decoupling (CPMG, XY8) as AC magnetometry.

Every pi pulse maps the accumulated phase phi -> -phi. If an AC field changes
sign at every pi pulse, the phase picked up in each interval adds up, so the
spin acts as a lock-in detector at f = 1 / (2 (tau + t_pi)). More pulses give a
narrower filter around that frequency.

Run from the repository root:

    python -m examples.dynamical_decoupling_demo            # show figures
    python -m examples.dynamical_decoupling_demo --no-show  # only write files to output/
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "output"


def main(show: bool):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import nvsim
    from nvsim import ACField, uT
    from nvsim.bloch import animate_bloch, save_animation
    from nvsim.plotting import TEXT_COLOR, plot_result, style_axes

    OUTPUT.mkdir(exist_ok=True)
    nv = nvsim.NVCenter.from_field(Bz=5.0)  # mT
    rabi = 20.0
    channel = nvsim.MWChannel("-1", rabi_frequency=rabi)
    tau = 0.5  # us between pi pulses
    t_pi2, t_pi = 1 / (4 * rabi), 1 / (2 * rabi)
    f_match = 1 / (2 * (tau + t_pi))
    t_first_pi = t_pi2 + tau / 2 + t_pi / 2  # centre of the first pi pulse
    print(
        f"pulse spacing tau + t_pi = {tau + t_pi:.3f} us, matched AC frequency {f_match:.4f} MHz"
    )

    # 1) XY8-1 under a matched AC field: the phase flips at every pi pulse and grows
    b_ac = 5 * uT  # -> about 0.75 pi after 8 pi pulses
    ac = ACField.sine_centered_at(b_ac, f_match, t_first_pi)
    result = nvsim.simulate(nvsim.xy8(tau), nv, channel, fields=[ac])
    fig = plot_result(
        result,
        title=f"XY8-1, matched AC field {b_ac / uT:.0f} µT at {f_match:.3f} MHz",
    )
    fig.savefig(OUTPUT / "xy8_ac_trace.png", dpi=150)

    # 2) Frequency response at fixed pulse spacing: more pi pulses -> more signal
    #    and a narrower filter around f_match (and its odd harmonics).
    b_small = 2 * uT
    freqs = np.linspace(0.05, 3.2, 400)
    sequences = {
        "spin echo = CPMG-1": nvsim.spin_echo(tau / 2),
        "CPMG-4": nvsim.cpmg(tau, 4),
        "XY8-1 (8 π)": nvsim.xy8(tau),
        "XY8-2 (16 π)": nvsim.xy8(tau, repetitions=2),
    }
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4
    fig, (ax_raw, ax_norm) = plt.subplots(
        2, 1, figsize=(8.5, 6.2), sharex=True, layout="constrained"
    )
    for (name, seq), color in zip(sequences.items(), colors):
        response = nvsim.sweep(
            freqs,
            seq,
            nv,
            channel,
            fields=lambda f: [ACField.sine_centered_at(b_small, f, t_first_pi)],
        )
        signal = 1 - response.population("0")
        ax_raw.plot(freqs, signal, color=color, lw=2, label=name)
        ax_norm.plot(freqs, signal / signal.max(), color=color, lw=2, label=name)
    for ax in (ax_raw, ax_norm):
        for harmonic in (1, 3):
            ax.axvline(
                harmonic * f_match, color="#8a8984", lw=1, ls=(0, (3, 2)), zorder=1
            )
        ax.set_ylim(-0.02, 1.05)
        style_axes(ax)
    ax_raw.annotate(
        "f = 1/(2(τ+t_π))",
        xy=(f_match, 1.05),
        xytext=(4, -2),
        textcoords="offset points",
        fontsize=8,
        color=TEXT_COLOR,
        va="top",
    )
    ax_raw.annotate(
        "3f",
        xy=(3 * f_match, 1.05),
        xytext=(4, -2),
        textcoords="offset points",
        fontsize=8,
        color=TEXT_COLOR,
        va="top",
    )
    ax_raw.set_ylabel("1 − P(|0⟩)")
    ax_raw.set_title(
        f"AC response at fixed pulse spacing τ = {tau} µs, B_ac = {b_small / uT:.0f} µT",
        color=TEXT_COLOR,
    )
    ax_raw.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax_norm.set_ylabel("normalised to peak")
    ax_norm.set_xlabel("AC frequency (MHz)")
    fig.savefig(OUTPUT / "dd_ac_response.png", dpi=150)

    # 3) Animation of XY8-1 under the matched AC field
    anim = animate_bloch(
        result,
        max_frames=320,
        min_frames_per_segment=12,
        trail_window=1.5 * (tau + t_pi),
    )
    path = save_animation(anim, OUTPUT / "xy8_ac.gif", fps=25)
    print(f"wrote {path}")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show", action="store_true", help="only save files, do not open windows"
    )
    main(show=not parser.parse_args().no_show)
