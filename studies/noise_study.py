"""Noise study: how do different pulse sequences respond to magnetic noise?

Set the knobs in section 1 (INTERFACE); everything in section 2 (study name,
noise sources, fields, tau ranges, readout) is derived from them. Then run from
the repository root:

    python studies/noise_study.py            # run and show the figures
    python studies/noise_study.py --no-show  # only write files
    python studies/noise_study.py --quick    # fewer points / realisations, for a fast look

Everything is written to output/<study name>/:

    summary.txt          parameters and derived numbers (pi time, T2*, where the DD dips are)
    noise_spectrum.png   noise PSD and the filter function of every sequence
    noise_traces.png     a few realisations of the noise field
    signal_vs_tau.png    PL (or population) vs tau: Monte Carlo, exact theory, no-noise
    spectroscopy.png     CPMG / XY coherence vs probed frequency, above the PSD
    <sequence>.csv       all numbers per sequence

Units: MHz, us, mT (``uT`` = microtesla: ``4 * uT`` = 0.004 mT).
"""

import argparse
from pathlib import Path

from nvsim import ACField, MWChannel, NVCenter, StaticField, mT, uT
from nvsim.noise import CarbonBath, NarrowbandNoise, OUNoise, QuasiStaticNoise  # noqa: F401
from nvsim.readout import Readout
from nvsim.study import (
    NoiseStudy,
    auto_axis,
    cpmg_scan,
    echo_scan,
    ramsey_scan,
    resolve_axis,
    run_study,
    xy4_scan,
    xy8_scan,
)

# =============================================================================
# 1. INTERFACE - set these; everything in section 2 follows automatically
# =============================================================================

# --- NV and microwave drive -------------------------------------------------
BIAS_FIELD = 5 * mT  # along the NV axis; sets f_+- and the 13C Larmor frequency
TRANSITION = "-1"  # "-1" or "+1"
RABI_FREQUENCY = 20.0  # MHz -> pi pulse = 1 / (2 * Rabi) = 25 ns
MW_DETUNING = 0.0  # MHz, carrier minus transition frequency

# --- The noise you want to detect: a narrowband line --------------------------
ADD_SAMPLE_NOISE = True
NOISE_FREQ = 1  # MHz
NOISE_RMS = 1.5 * uT
NOISE_LINEWIDTH = 0.02  # MHz (FWHM); 0 = a single frequency
NOISE_GAUSSIAN = True  # False: fixed amplitude, random phase

# --- 13C nuclear spin bath (quasi-static part + line at the Larmor frequency) ---
ADD_CARBON_BATH = True
CARBON_ABUNDANCE = 0.011  # 1.1 % = natural abundance; amplitudes scale as sqrt
CARBON_B_STATIC = 2 * uT  # quasi-static rms at 1.1 %
CARBON_B_LARMOR = 2 * uT  # rms of the Larmor line at 1.1 %
CARBON_LINEWIDTH = 0.002  # MHz

# --- Slow fluctuating spin bath (Ornstein-Uhlenbeck, Lorentzian at f = 0) -----
ADD_SLOW_BATH = False
SLOW_BATH_RMS = 1 * uT
SLOW_BATH_CORRELATION_TIME = 20.0  # us

# --- Coherent signals, identical in every shot --------------------------------
ADD_AC_SIGNAL = False
AC_FREQ = 0.8  # MHz
AC_AMPLITUDE = 1 * uT
ADD_DC_OFFSET = False
DC_OFFSET = 2 * uT

# --- Sequences -----------------------------------------------------------------
RUN_RAMSEY = True
RUN_ECHO = True
RUN_CPMG = True
RUN_XY8 = True
RUN_XY4 = False
CPMG_PULSES = 8
XY8_REPETITIONS = 2  # 8 pi pulses per repetition
XY4_REPETITIONS = 4  # 4 pi pulses per repetition

# Shared x axis of the PL-vs-tau plots (the same for every sequence):
#   "spacing": pulse spacing tau (CPMG convention tau/2 - pi - tau - ... - pi - tau/2).
#              CPMG / XY dips of a line at f all sit at tau = 1/(2f) - t_pi, and
#              revivals at tau = 2k/f. The echo is CPMG-1 (tau = total free
#              evolution); Ramsey has a single window tau.
#   "total":   total free-evolution time T (tau = T / number of windows) -
#              compares how long each sequence keeps coherence.
X_AXIS = "spacing"
# x range [us]: "auto" (up to the first revival of the slowest noise line, 2.2/f_min,
# or 3 T2* if longer), (start, stop), or an explicit array
X_RANGE = "auto"
N_POINTS = 200

# --- Readout and statistics ------------------------------------------------------
SIGNAL = "pl"  # "pl" (normalised photoluminescence) or "population" (P(|0>))
PL_CONTRAST = 0.3
PHOTONS_PER_SHOT = None  # e.g. 0.03 - set both this and SHOTS to add shot noise
SHOTS = None  # e.g. 200_000
N_REALIZATIONS = 300  # noise histories averaged per tau point
SEED = 1
STUDY_NAME = None  # None: built from the settings above

# =============================================================================
# 2. DERIVED - built from the interface above
# =============================================================================

NV = NVCenter.from_field(BIAS_FIELD)
DRIVE = MWChannel(TRANSITION, rabi_frequency=RABI_FREQUENCY, detuning=MW_DETUNING)
T_PI = 1 / (2 * RABI_FREQUENCY)

NOISE, NAME_PARTS = [], []
if ADD_CARBON_BATH:
    NOISE.append(
        CarbonBath(
            field=BIAS_FIELD,
            b_static=CARBON_B_STATIC,
            b_larmor=CARBON_B_LARMOR,
            linewidth=CARBON_LINEWIDTH,
            abundance=CARBON_ABUNDANCE,
        )
    )
    NAME_PARTS.append(f"13C {100 * CARBON_ABUNDANCE:g}% at {BIAS_FIELD / mT:g} mT")
if ADD_SAMPLE_NOISE:
    NOISE.append(
        NarrowbandNoise(
            b_rms=NOISE_RMS,
            frequency=NOISE_FREQ,
            linewidth=NOISE_LINEWIDTH,
            gaussian=NOISE_GAUSSIAN,
            label=f"sample noise {NOISE_FREQ:g} MHz",
        )
    )
    NAME_PARTS.append(f"{NOISE_FREQ:g} MHz noise {NOISE_RMS / uT:g} uT")
if ADD_SLOW_BATH:
    NOISE.append(
        OUNoise(
            b_rms=SLOW_BATH_RMS,
            correlation_time=SLOW_BATH_CORRELATION_TIME,
            label=f"slow bath, {SLOW_BATH_CORRELATION_TIME:g} us",
        )
    )
    NAME_PARTS.append(f"slow bath {SLOW_BATH_RMS / uT:g} uT")
# Any other source can be added here, e.g. NOISE.append(QuasiStaticNoise(2 * uT))

FIELDS = []
if ADD_AC_SIGNAL:
    FIELDS.append(ACField(amplitude=AC_AMPLITUDE, frequency=AC_FREQ))
    NAME_PARTS.append(f"{AC_FREQ:g} MHz AC {AC_AMPLITUDE / uT:g} uT")
if ADD_DC_OFFSET:
    FIELDS.append(StaticField(Bz=DC_OFFSET))
    NAME_PARTS.append(f"DC {DC_OFFSET / uT:g} uT")

NAME = STUDY_NAME or (" + ".join(NAME_PARTS) if NAME_PARTS else "no noise")

SEQ_KWARGS = {"transition": TRANSITION, "axis": X_AXIS}
X = resolve_axis(X_RANGE, lambda: auto_axis(NOISE, T_PI, N_POINTS, FIELDS), N_POINTS)

SCANS = []
if RUN_RAMSEY:
    SCANS.append(ramsey_scan(X, **SEQ_KWARGS))
if RUN_ECHO:
    SCANS.append(echo_scan(X, **SEQ_KWARGS))
if RUN_CPMG:
    SCANS.append(cpmg_scan(X, n_pulses=CPMG_PULSES, **SEQ_KWARGS))
if RUN_XY8:
    SCANS.append(xy8_scan(X, repetitions=XY8_REPETITIONS, **SEQ_KWARGS))
if RUN_XY4:
    SCANS.append(xy4_scan(X, repetitions=XY4_REPETITIONS, **SEQ_KWARGS))

READOUT = Readout(contrast=PL_CONTRAST, photons_per_shot=PHOTONS_PER_SHOT, shots=SHOTS)

# =============================================================================


def build_study(quick: bool = False) -> NoiseStudy:
    if not SCANS:
        raise SystemExit("No sequences selected: set at least one RUN_* to True.")
    n_realizations = N_REALIZATIONS
    if quick:
        for scan in SCANS:
            scan.x = scan.x[::3]
        n_realizations = max(50, N_REALIZATIONS // 4)
    return NoiseStudy(
        name=NAME,
        nv=NV,
        drive=DRIVE,
        scans=SCANS,
        noise=NOISE,
        fields=FIELDS,
        n_realizations=n_realizations,
        readout=READOUT,
        signal=SIGNAL,
        seed=SEED,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show", action="store_true", help="only save files, do not open windows"
    )
    parser.add_argument(
        "--quick", action="store_true", help="fewer points and realisations"
    )
    args = parser.parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    study = build_study(args.quick)
    slug = "".join(c if c.isalnum() else "_" for c in NAME).strip("_").lower()
    output = Path(__file__).resolve().parents[1] / "output" / slug
    run_study(study, output, show=not args.no_show)


if __name__ == "__main__":
    main()
