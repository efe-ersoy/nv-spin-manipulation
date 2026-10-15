"""Noise sources, filter functions, Monte Carlo ensemble and readout.

Run from the repository root:  python -m unittest discover tests
"""

import unittest

import numpy as np

import nvsim
from nvsim import ACField, MWChannel, NVCenter, StaticField, uT
from nvsim.ensemble import simulate_ensemble
from nvsim.filter_function import filter_function, phase_variance, toggling_function
from nvsim.noise import CarbonBath, NarrowbandNoise, OUNoise, QuasiStaticNoise, flatten
from nvsim.readout import Readout
from nvsim.simulate import channel_map

NV = NVCenter.from_field(50.0)
FAST = MWChannel("-1", rabi_frequency=200.0)  # near-ideal pulses
CHANNELS = channel_map(FAST)
COUPLING = 2 * np.pi * nvsim.GAMMA_E


def segments(seq, channels=CHANNELS):
    return seq.compile(channels)


class TestUnits(unittest.TestCase):
    def test_millitesla(self):
        self.assertAlmostEqual(nvsim.GAMMA_E, 28.025)
        nv = NVCenter.from_field(50 * nvsim.mT)
        self.assertAlmostEqual(nv.f_minus, nvsim.D_GS - 50 * 28.025)
        self.assertAlmostEqual(500 * nvsim.G, 50 * nvsim.mT)
        self.assertAlmostEqual(1000 * uT, 1.0)

    def test_carbon_larmor_frequency(self):
        bath = CarbonBath(field=50.0)
        self.assertAlmostEqual(bath.larmor_frequency, 0.535, places=3)  # 10.71 kHz/mT
        self.assertAlmostEqual(CarbonBath.for_nv(NV).field, 50.0)


class TestNoiseStatistics(unittest.TestCase):
    RNG = np.random.default_rng(0)

    def check_autocorrelation(self, source, lags_steps=(0, 5, 20), dt=0.01, n=4000, rtol=0.06):
        t = np.arange(400) * dt
        x = source.sample(t, n, self.RNG)
        for k in lags_steps:
            measured = np.mean(x[:, : len(t) - k] * x[:, k:])
            expected = float(source.autocorrelation(k * dt))
            self.assertAlmostEqual(measured, expected, delta=rtol * source.b_rms**2)

    def test_quasi_static(self):
        self.check_autocorrelation(QuasiStaticNoise(2 * uT))

    def test_ornstein_uhlenbeck(self):
        self.check_autocorrelation(OUNoise(2 * uT, correlation_time=0.3))

    def test_narrowband_gaussian_and_fixed_amplitude(self):
        self.check_autocorrelation(NarrowbandNoise(2 * uT, frequency=1.3, linewidth=0.5))
        self.check_autocorrelation(NarrowbandNoise(2 * uT, frequency=1.3, linewidth=0.5, gaussian=False))
        self.check_autocorrelation(NarrowbandNoise(2 * uT, frequency=1.3))

    def test_psd_integrates_to_variance(self):
        f = np.linspace(0, 2000, 2_000_001)
        for source in [OUNoise(2 * uT, 0.5), NarrowbandNoise(2 * uT, 3.0, linewidth=0.2)]:
            with self.subTest(source.label):
                self.assertAlmostEqual(np.trapezoid(source.psd(f), f) / source.b_rms**2, 1.0, delta=0.01)

    def test_carbon_bath_components_scale_with_abundance(self):
        parts = flatten([CarbonBath(field=50.0, abundance=0.044)])
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(parts[0].b_rms, 2 * 4 * uT)
        self.assertAlmostEqual(parts[1].frequency, nvsim.GAMMA_C13 * 50.0)


class TestFilterFunction(unittest.TestCase):
    def test_ramsey_and_echo_dc_response(self):
        tau = 1.5
        ramsey = segments(nvsim.ramsey(tau))
        echo = segments(nvsim.spin_echo(tau))
        self.assertAlmostEqual(filter_function(ramsey, [0.0])[0], tau**2, places=6)
        self.assertAlmostEqual(filter_function(echo, [0.0])[0], 0.0, places=6)

    def test_toggling_function_flips_at_pi_pulses(self):
        segs = segments(nvsim.cpmg(1.0, 2))
        waits = [s for s in segs if not s.is_pulse]
        mids = [0.5 * (w.start + w.end) for w in waits]
        np.testing.assert_allclose(toggling_function(segs, mids), [1, -1, 1])

    def test_dd_filter_peaks_at_probe_frequency(self):
        tau = 0.6
        segs = segments(nvsim.xy8(tau, 4))
        f = np.linspace(0.2, 1.5, 5001)
        t_pi = 1 / (2 * FAST.rabi_frequency)
        self.assertAlmostEqual(f[np.argmax(filter_function(segs, f))], 1 / (2 * (tau + t_pi)), delta=0.01)

    def test_filter_function_matches_time_domain_variance(self):
        source = NarrowbandNoise(1 * uT, frequency=0.8)  # single line: <phi^2> = (2 pi g)^2 b^2 |Y(f0)|^2
        segs = segments(nvsim.cpmg(0.5, 4))
        expected = COUPLING**2 * source.b_rms**2 * filter_function(segs, [source.frequency])[0]
        self.assertAlmostEqual(phase_variance(segs, [source]), expected, delta=1e-3 * expected)

    def test_phase_variance_closed_forms(self):
        b, T = 3 * uT, 2.0
        segs = segments(nvsim.ramsey(T))
        T_eff = segs[1].duration
        self.assertAlmostEqual(phase_variance(segs, [QuasiStaticNoise(b)]), (COUPLING * b * T_eff) ** 2, places=6)
        tc = 0.7
        exact = COUPLING**2 * b**2 * 2 * (tc * T_eff - tc**2 * (1 - np.exp(-T_eff / tc)))
        self.assertAlmostEqual(phase_variance(segs, [OUNoise(b, tc)]), exact, delta=1e-3 * exact)


class TestEnsemble(unittest.TestCase):
    def test_no_noise_matches_deterministic_simulation(self):
        ch = MWChannel("-1", 15.0, detuning=0.4)
        fields = [ACField(5 * uT, 0.6, phase=0.2), StaticField(2 * uT)]
        for seq in [nvsim.ramsey(1.3), nvsim.spin_echo(0.9), nvsim.xy8(0.4)]:
            with self.subTest(seq.name):
                det = nvsim.simulate(seq, NV, ch, fields=fields).final_populations
                mc = simulate_ensemble(seq, NV, ch, fields=fields, n_realizations=2, rng=0).populations
                for level in ("0", "-1", "+1"):
                    self.assertAlmostEqual(mc[level], det[level], delta=2e-3)

    def assert_matches_theory(self, seq, noise, drive=FAST, n=4000):
        mc = simulate_ensemble(seq, NV, drive, noise, n_realizations=n, rng=3)
        det = nvsim.simulate(seq, NV, drive, store_trajectory=False).final_populations["0"]
        decay = np.exp(-0.5 * phase_variance(segments(seq, channel_map(drive)), noise))
        theory = 0.5 + (det - 0.5) * decay
        self.assertAlmostEqual(mc.populations["0"], theory, delta=4 * mc.uncertainty["0"] + 2e-3)

    def test_ramsey_quasi_static(self):
        self.assert_matches_theory(nvsim.ramsey(2.0), [QuasiStaticNoise(4 * uT)])

    def test_echo_ou(self):
        self.assert_matches_theory(nvsim.spin_echo(2.0), [OUNoise(4 * uT, 1.0)])

    def test_echo_carbon_collapse(self):
        bath = CarbonBath(field=50.0)
        self.assert_matches_theory(nvsim.spin_echo(1 / (2 * bath.larmor_frequency)), [bath])

    def test_xy8_narrowband_dip(self):
        noise = [NarrowbandNoise(1 * uT, frequency=1.0, linewidth=0.05)]
        t_pi = 1 / (2 * FAST.rabi_frequency)
        self.assert_matches_theory(nvsim.xy8(0.5 - t_pi, 2), noise)

    def test_echo_refocuses_quasi_static_noise(self):
        mc = simulate_ensemble(nvsim.spin_echo(3.0), NV, FAST, [QuasiStaticNoise(5 * uT)], n_realizations=200, rng=1)
        self.assertAlmostEqual(mc.populations["0"], 1.0, places=4)


class TestReadout(unittest.TestCase):
    def test_pl_contrast(self):
        r = Readout(contrast=0.3)
        self.assertAlmostEqual(float(r.pl({"0": 1.0})), 1.0)
        self.assertAlmostEqual(float(r.pl({"0": 0.0})), 0.7)
        self.assertAlmostEqual(float(r.pl_error(0.1)), 0.03)

    def test_shot_noise_level(self):
        r = Readout(contrast=0.3, photons_per_shot=0.05, shots=100_000)
        pl, err = r.measure({"0": np.full(2000, 0.5)}, rng=0)
        self.assertAlmostEqual(np.mean(pl), 0.85, delta=0.002)
        self.assertAlmostEqual(np.std(pl), np.sqrt(0.85 / 5000), delta=0.002)
        self.assertAlmostEqual(np.mean(err), np.sqrt(0.85 / 5000), delta=0.002)


if __name__ == "__main__":
    unittest.main()
