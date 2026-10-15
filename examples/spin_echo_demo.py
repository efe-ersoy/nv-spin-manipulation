"""Spin echo under static and oscillating magnetic fields.

Shows how the echo pi pulse turns the phase phi accumulated in the first free
evolution into -phi, so that

* a static field is refocused: the second evolution winds -phi back to 0;
* an AC field that changes sign at the echo pulse is *not*: the second
  evolution keeps adding phase, which is how a spin echo senses AC fields;
* an AC field symmetric about the echo pulse cancels again.

Run from the repository root:

    python -m examples.spin_echo_demo            # show figures
    python -m examples.spin_echo_demo --no-show  # only write files to output/
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
    from nvsim import ACField, StaticField, uT
    from nvsim.bloch import animate_bloch, save_animation
    from nvsim.plotting import (
        TEXT_COLOR,
        plot_field,
        plot_phase,
        plot_populations,
        plot_sweep,
        style_axes,
    )

    OUTPUT.mkdir(exist_ok=True)
    nv = nvsim.NVCenter.from_field(Bz=5.0)  # mT
    rabi = 20.0  # MHz -> pi/2 = 12.5 ns, pi = 25 ns
    channel = nvsim.MWChannel("-1", rabi_frequency=rabi)
    tau = 1.0  # us, each free evolution
    t_pi2, t_pi = 1 / (4 * rabi), 1 / (2 * rabi)
    t_echo = t_pi2 + tau + t_pi / 2  # centre of the echo pulse
    f_match = 1 / (2 * (tau + t_pi))  # AC period = pi-pulse period (centre to centre)
    b_dc, b_ac = 10 * uT, 10 * uT
    seq = nvsim.spin_echo(tau)

    print(
        f"pi pulse {1e3 * t_pi:.0f} ns, tau = {tau} us, matched AC frequency {f_match:.4f} MHz"
    )
    print(
        f"static field: phase per evolution 2*pi*gamma*B*tau = {2 * np.pi * nvsim.GAMMA_E * b_dc * tau:.3f} rad"
    )

    cases = {
        "No field": [],
        f"Static field {b_dc / uT:.0f} µT": [StaticField(b_dc)],
        "AC field, sign flips at π\n(sine about the echo pulse)": [
            ACField.sine_centered_at(b_ac, f_match, t_echo)
        ],
        "AC field, symmetric about π\n(cosine about the echo pulse)": [
            ACField(b_ac, f_match, phase=-2 * np.pi * f_match * t_echo)
        ],
    }
    results = {
        name: nvsim.simulate(seq, nv, channel, fields=fields)
        for name, fields in cases.items()
    }

    # 1) Field, populations and accumulated phase side by side
    fig, axes = plt.subplots(
        3,
        len(cases),
        figsize=(4.2 * len(cases), 7.5),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [1, 1.6, 1.6]},
        layout="constrained",
    )
    for col, (name, result) in enumerate(results.items()):
        plot_field(result, ax=axes[0, col])
        plot_populations(result, ax=axes[1, col], legend=(col == len(cases) - 1))
        plot_phase(result, ax=axes[2, col])
        axes[0, col].set_title(name, color=TEXT_COLOR, fontsize=10)
        axes[0, col].set_ylim(-1.3 * b_ac / uT, 1.3 * b_ac / uT)
        p0 = result.final_populations["0"]
        axes[1, col].text(
            0.02,
            0.04,
            f"final P(|0⟩) = {p0:.3f}",
            transform=axes[1, col].transAxes,
            fontsize=9,
            color=TEXT_COLOR,
        )
        for row in range(2):
            axes[row, col].set_xlabel("")
        if col:
            for row in range(3):
                axes[row, col].set_ylabel("")
    fig.suptitle(
        "Spin echo π/2 – τ – π – τ – π/2: the echo pulse maps φ → −φ", color=TEXT_COLOR
    )
    fig.savefig(OUTPUT / "spin_echo_fields.png", dpi=150)

    # 2) Static field: Ramsey (same total free evolution) vs spin echo
    fields_b = np.linspace(-30, 30, 121) * uT
    ramsey = nvsim.sweep(
        fields_b,
        nvsim.ramsey(2 * tau + t_pi),
        nv,
        channel,
        fields=lambda b: [StaticField(b)],
        parameter="static field (mT)",
    )
    echo = nvsim.sweep(
        fields_b,
        seq,
        nv,
        channel,
        fields=lambda b: [StaticField(b)],
        parameter="static field (mT)",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, layout="constrained")
    plot_sweep(ramsey, ax=axes[0], levels=["0"], marker=None)
    plot_sweep(echo, ax=axes[1], levels=["0"], marker=None)
    axes[0].set_title("Ramsey: static field → fringes", color=TEXT_COLOR)
    axes[1].set_title("Spin echo: static field refocused", color=TEXT_COLOR)
    fig.savefig(OUTPUT / "spin_echo_vs_ramsey_static.png", dpi=150)

    # 3) AC field: echo response vs AC frequency. The field is a sine phase-locked
    #    to the echo pulse (sign change at the pi pulse). For instantaneous pulses
    #    the total phase is phi = 2*gamma*b*(1 - cos(2*pi*f*T)) / f, with T the
    #    time between the centres of the pi/2 and pi pulses, and 1 - P0 = sin^2(phi/2).
    #    It peaks at 2*pi*f*T ~ 2.33, slightly below the frequency whose sign flips
    #    land exactly on the pulses, f = 1/(2T).
    b_small = 3 * uT  # small enough that the response is not saturated
    freqs = np.linspace(0.02, 2.5, 250)
    response = nvsim.sweep(
        freqs,
        seq,
        nv,
        channel,
        fields=lambda f: [ACField.sine_centered_at(b_small, f, t_echo)],
        parameter="AC frequency (MHz)",
    )
    T = t_echo - t_pi2 / 2
    phi_ideal = (
        2 * nvsim.GAMMA_E * b_small * (1 - np.cos(2 * np.pi * freqs * T)) / freqs
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    ax.plot(
        freqs, 1 - response.population("0"), color="#2a78d6", lw=2, label="simulation"
    )
    ax.plot(
        freqs,
        np.sin(phi_ideal / 2) ** 2,
        color=TEXT_COLOR,
        lw=1.2,
        ls=(0, (3, 2)),
        label="ideal-pulse theory",
    )
    ax.axvline(f_match, color="#8a8984", lw=1)
    ax.annotate(
        "sign flips exactly at the pulses\nf = 1 / (2(τ + t_π))",
        xy=(f_match, 0.0),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=8,
        color=TEXT_COLOR,
        va="bottom",
    )
    ax.set_xlabel("AC frequency (MHz)")
    ax.set_ylabel("1 − P(|0⟩)")
    ax.set_title(
        f"Spin echo AC response, τ = {tau} µs, B_ac = {b_small / uT:.0f} µT",
        color=TEXT_COLOR,
    )
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    fig.savefig(OUTPUT / "spin_echo_ac_response.png", dpi=150)

    # 4) Animations: static field (refocused) and matched AC field (not refocused)
    animations = []  # keep references so the animations keep running in plt.show()
    for name, filename in [
        (list(cases)[1], "spin_echo_static.gif"),
        (list(cases)[2], "spin_echo_ac.gif"),
    ]:
        anim = animate_bloch(results[name], max_frames=200, min_frames_per_segment=30)
        path = save_animation(anim, OUTPUT / filename, fps=25)
        print(f"wrote {path}")
        animations.append(anim)

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
