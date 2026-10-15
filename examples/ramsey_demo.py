"""Ramsey fringes on the |0> <-> |-1> transition.

Run from the repository root:

    python -m examples.ramsey_demo            # show figures
    python -m examples.ramsey_demo --no-show  # only write files to output/
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
    from nvsim.bloch import animate_bloch, save_animation
    from nvsim.plotting import plot_result, plot_sweep

    OUTPUT.mkdir(exist_ok=True)
    nv = nvsim.NVCenter.from_field(Bz=5.0)  # mT
    # Carrier 2 MHz above the transition -> fringes at 2 MHz.
    channel = nvsim.MWChannel("-1", rabi_frequency=10.0, detuning=2.0)

    # 1) One Ramsey run, full time trace (pi/2 = 25 ns)
    result = nvsim.simulate(nvsim.ramsey(tau=0.375), nv, channel)
    fig = plot_result(result, title="Ramsey, τ = 375 ns, Δ = 2 MHz")
    fig.savefig(OUTPUT / "ramsey_trace.png", dpi=150)

    # 2) Ramsey fringes: final population vs tau, readout pulse about x and y
    taus = np.linspace(0, 1.5, 151)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout="constrained")
    for ax, readout in zip(axes, ["x", "y"]):
        fringes = nvsim.sweep(
            taus,
            lambda tau: nvsim.ramsey(tau, readout_phase=readout),
            nv,
            channel,
            parameter="τ (µs)",
        )
        plot_sweep(fringes, ax=ax, marker=None)
        ax.set_title(f"readout π/2 about {readout}")
    fig.savefig(OUTPUT / "ramsey_fringes.png", dpi=150)

    # 3) Animation: pi/2 -> precession about z at the detuning -> pi/2
    anim = animate_bloch(result, max_frames=180, min_frames_per_segment=30)
    path = save_animation(anim, OUTPUT / "ramsey.gif", fps=25)
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
