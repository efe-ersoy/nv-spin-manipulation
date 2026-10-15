"""Rabi oscillations on the |0> <-> |-1> transition.

Run from the repository root:

    python -m examples.rabi_demo            # show figures
    python -m examples.rabi_demo --no-show  # only write files to output/
"""

import argparse
from pathlib import Path

import matplotlib

OUTPUT = Path(__file__).resolve().parents[1] / "output"


def main(show: bool):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import nvsim
    from nvsim.bloch import animate_bloch, plot_bloch_trajectory, save_animation
    from nvsim.plotting import plot_result, style_axes

    OUTPUT.mkdir(exist_ok=True)
    nv = nvsim.NVCenter.from_field(Bz=5.0)  # mT; f_- = 2729.9 MHz, f_+ = 3010.1 MHz
    rabi_frequency = 5.0  # MHz -> pi pulse = 100 ns
    print(f"f_- = {nv.f_minus:.1f} MHz, f_+ = {nv.f_plus:.1f} MHz")

    # 1) Resonant Rabi: sequence diagram + populations
    resonant = nvsim.MWChannel("-1", rabi_frequency=rabi_frequency, detuning=0.0)
    result = nvsim.simulate(nvsim.rabi(0.6), nv, resonant)
    fig = plot_result(result, title="Resonant Rabi, Ω = 5 MHz")
    fig.savefig(OUTPUT / "rabi_resonant.png", dpi=150)

    # 2) Detuning dependence: P(-1) vs pulse length for several detunings
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    detunings = [0.0, 2.5, 5.0]
    shades = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
    for detuning, color in zip(detunings, shades):
        channel = nvsim.MWChannel(
            "-1", rabi_frequency=rabi_frequency, detuning=detuning
        )
        r = nvsim.simulate(nvsim.rabi(0.6), nv, channel)
        ax.plot(
            r.times,
            r.population("-1"),
            color=color,
            lw=2,
            label=f"Δ = {detuning:g} MHz",
        )
    ax.set_xlabel("pulse length (µs)")
    ax.set_ylabel(r"population $|{-}1\rangle$")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="upper right", ncols=3)
    ax.set_title("Detuned Rabi: faster, shallower oscillations")
    style_axes(ax)
    fig.savefig(OUTPUT / "rabi_detuning.png", dpi=150)

    # 3) Bloch trajectories: resonant (great circle) vs detuned (tilted axis)
    detuned = nvsim.MWChannel("-1", rabi_frequency=rabi_frequency, detuning=3.0)
    one_cycle = 1 / rabi_frequency
    fig = plt.figure(figsize=(8, 4), layout="constrained")
    for i, (channel, name) in enumerate([(resonant, "Δ = 0"), (detuned, "Δ = 3 MHz")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        plot_bloch_trajectory(nvsim.simulate(nvsim.rabi(one_cycle), nv, channel), ax=ax)
        ax.set_title(name, color="#0b0b0b")
    fig.savefig(OUTPUT / "rabi_bloch.png", dpi=150)

    # 4) Animation of the detuned Rabi rotation
    result = nvsim.simulate(nvsim.rabi(one_cycle), nv, detuned)
    anim = animate_bloch(result, max_frames=120)
    path = save_animation(anim, OUTPUT / "rabi_detuned.gif", fps=25)
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
