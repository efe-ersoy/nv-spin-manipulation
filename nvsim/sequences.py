"""Library of standard pulse sequences.

Each function returns a :class:`~nvsim.pulses.PulseSequence`. Pulse lengths
given as rotation angles are converted to durations using the Rabi frequency
of the corresponding :class:`~nvsim.drive.MWChannel` at simulation time.
"""

import numpy as np

from .pulses import Pulse, PulseSequence, Wait, phase_label, parse_phase


def rabi(duration: float, transition: str = "-1", phase="x") -> PulseSequence:
    """A single square pulse of length ``duration`` [us].

    Simulating it with a dense time grid gives the Rabi oscillation directly;
    to mimic the experiment (one point per pulse length), sweep ``duration``
    with :func:`nvsim.sweep`.
    """
    return PulseSequence(
        [
            Pulse(
                duration=duration, transition=transition, phase=phase, label="MW pulse"
            )
        ],
        name="Rabi",
    )


def ramsey(
    tau: float,
    transition: str = "-1",
    phase="x",
    readout_phase="x",
) -> PulseSequence:
    """Ramsey sequence pi/2 - tau - pi/2.

    Parameters
    ----------
    tau : float
        Free-evolution time between the pulses [us].
    transition : str
        '-1' or '+1'.
    phase, readout_phase : float or str
        Phases of the first and the second (readout) pi/2 pulse.
    """
    return PulseSequence(
        [
            Pulse(np.pi / 2, transition, phase, label="π/2"),
            Wait(tau, label="τ"),
            Pulse(np.pi / 2, transition, readout_phase, label="π/2 (readout)"),
        ],
        name="Ramsey",
    )


def spin_echo(
    tau: float,
    transition: str = "-1",
    phase="x",
    pi_phase="x",
    readout_phase="x",
) -> PulseSequence:
    """Hahn echo pi/2 - tau - pi - tau - pi/2.

    ``tau`` is each of the two free-evolution periods [us] (total 2*tau, pulse
    lengths not included). With the default phases a field-free spin returns
    to |0>. A static field is refocused; an AC field of period ~2*tau whose
    sign flips at the pi pulse is not.
    """
    return PulseSequence(
        [
            Pulse(np.pi / 2, transition, phase, label="π/2"),
            Wait(tau, label="τ"),
            Pulse(np.pi, transition, pi_phase, label="π (echo)"),
            Wait(tau, label="τ"),
            Pulse(np.pi / 2, transition, readout_phase, label="π/2 (readout)"),
        ],
        name="Spin echo",
    )


def dynamical_decoupling(
    tau: float,
    pi_phases,
    transition: str = "-1",
    phase="x",
    readout_phase="-x",
    name: str = "DD",
) -> PulseSequence:
    """pi/2 - [tau/2 - pi - tau/2] x N - pi/2 with the given pi-pulse phases.

    ``tau`` is the free evolution between consecutive pi pulses [us] (so the
    pulse period is tau + t_pi), with tau/2 before the first and after the last
    pi pulse. This is the common building block of CPMG and the XY family.
    """
    elements = [Pulse(np.pi / 2, transition, phase, label="π/2")]
    for k, pi_phase in enumerate(pi_phases):
        elements.append(
            Wait(tau / 2 if k == 0 else tau, label="τ/2" if k == 0 else "τ")
        )
        axis = phase_label(parse_phase(pi_phase))
        elements.append(Pulse(np.pi, transition, pi_phase, label=f"π_{axis} #{k + 1}"))
    elements.append(Wait(tau / 2, label="τ/2"))
    elements.append(Pulse(np.pi / 2, transition, readout_phase, label="π/2 (readout)"))
    return PulseSequence(elements, name=name)


def cpmg(
    tau: float, n_pulses: int, transition: str = "-1", phase="x", readout_phase="-x"
) -> PulseSequence:
    """CPMG-N: pi/2_x - [tau/2 - pi_y - tau/2] x N - pi/2.

    The pi pulses are 90 degrees out of phase with the first pi/2, which makes
    CPMG robust to pi-pulse amplitude errors for this initial state. With the
    default readout phase a field-free spin returns to |0>.
    """
    pi_phase = parse_phase(phase) + np.pi / 2
    return dynamical_decoupling(
        tau,
        [pi_phase] * n_pulses,
        transition,
        phase,
        readout_phase,
        name=f"CPMG-{n_pulses}",
    )


def xy4(
    tau: float,
    repetitions: int = 1,
    transition: str = "-1",
    phase="x",
    readout_phase="-x",
) -> PulseSequence:
    """XY4-N: N repetitions of the pi-pulse block X Y X Y (4N pi pulses)."""
    return dynamical_decoupling(
        tau,
        ["x", "y", "x", "y"] * repetitions,
        transition,
        phase,
        readout_phase,
        name=f"XY4-{repetitions}",
    )


def xy8(
    tau: float,
    repetitions: int = 1,
    transition: str = "-1",
    phase="x",
    readout_phase="-x",
) -> PulseSequence:
    """XY8-N: N repetitions of the pi-pulse block X Y X Y Y X Y X (8N pi pulses).

    Robust to both amplitude and detuning errors of the pi pulses for any
    initial state. With the default readout phase a field-free spin returns to |0>.
    """
    block = ["x", "y", "x", "y", "y", "x", "y", "x"]
    return dynamical_decoupling(
        tau,
        block * repetitions,
        transition,
        phase,
        readout_phase,
        name=f"XY8-{repetitions}",
    )
