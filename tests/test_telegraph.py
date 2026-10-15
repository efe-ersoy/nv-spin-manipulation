"""Random telegraph noise: sampling, exact coherence, Gaussian guard, Monte Carlo.

Run from the repository root:  python -m unittest discover tests
"""

import unittest
import warnings

import numpy as np

import nvsim
from nvsim import MWChannel, NVCenter, uT
from nvsim.ensemble import simulate_ensemble
from nvsim.filter_function import coherence_decay, phase_variance
from nvsim.noise import NarrowbandNoise, OUNoise, TelegraphNoise, is_gaussian
from nvsim.readout import Readout
from nvsim.simulate import channel_map
from nvsim.study import NoiseStudy, echo_scan, run_scan
from nvsim.telegraph import (
    rtn_cpmg_coherence,
    rtn_echo_coherence,
    rtn_fid_coherence,
    telegraph_coherence,
)

NV = NVCenter.from_field(5.0)
FAST = MWChannel("-1", rabi_frequency=200.0)  # near-ideal pulses
INSTANT = MWChannel("-1", rabi_frequency=1e5)  # effectively instantaneous pulses
WEAK = TelegraphNoise.symmetric(2 * uT, rate=2.0)  # 2 pi gamma b tau_c ~ 0.09
STRONG = TelegraphNoise.symmetric(20 * uT, rate=0.2)  # 2 pi gamma b tau_c ~ 8.8


def segments(seq, drive=INSTANT):
    return seq.compile(channel_map(drive))


class TestTelegraphSampling(unittest.TestCase):
    SOURCE = TelegraphNoise(b_low=-1 * uT, b_high=3 * uT, rate_up=0.5, rate_down=1.5)

    def setUp(self):
        self.dt = 0.02
        self.t = np.arange(0, 30, self.dt)
        self.x = self.SOURCE.sample(self.t, 3000, np.random.default_rng(1))

    def test_levels_and_occupations(self):
        self.assertTrue(np.all(np.isin(self.x, [self.SOURCE.b_low, self.SOURCE.b_high])))
        self.assertAlmostEqual(np.mean(self.x == self.SOURCE.b_high), self.SOURCE.p_high, delta=0.01)
        self.assertAlmostEqual(self.SOURCE.p_high, 0.25)
        self.assertAlmostEqual(np.mean(self.x), self.SOURCE.mean, delta=0.04 * uT)

    def test_autocovariance(self):
        d = self.x - self.SOURCE.mean
        tc = self.SOURCE.correlation_time
        for lag in [0.0, 0.5 * tc, tc, 2 * tc]:
            k = int(round(lag / self.dt))
            measured = np.mean(d[:, : len(self.t) - k] * d[:, k:])
            self.assertAlmostEqual(measured, float(self.SOURCE.autocorrelation(k * self.dt)),
                                   delta=0.04 * self.SOURCE.b_rms**2)

    def test_switching_rate(self):
        # Up-switches per unit time = p_low * rate_up for the stationary process.
        ups = np.sum(np.diff(self.x, axis=1) > 0) / (self.x.shape[0] * (self.t[-1] - self.t[0]))
        self.assertAlmostEqual(ups, self.SOURCE.p_low * self.SOURCE.rate_up, delta=0.02)

    def test_psd_integrates_to_variance(self):
        f = np.linspace(0, 500, 2_000_001)
        self.assertAlmostEqual(np.trapezoid(self.SOURCE.psd(f), f) / self.SOURCE.b_rms**2, 1.0, delta=0.01)

    def test_flags(self):
        self.assertFalse(self.SOURCE.gaussian)
        self.assertFalse(is_gaussian([OUNoise(1 * uT, 1.0), self.SOURCE]))
        self.assertTrue(is_gaussian([OUNoise(1 * uT, 1.0)]))
        with self.assertRaises(ValueError):
            TelegraphNoise(0.0, 1.0, 0.0, 0.0)


class TestExactCoherence(unittest.TestCase):
    TAUS = np.array([0.3, 1.0, 2.5, 6.0])

    def test_echo_closed_form_is_cpmg1(self):
        for src in (WEAK, STRONG):
            b, rate = src.b_high, src.rate_up
            np.testing.assert_allclose(rtn_echo_coherence(self.TAUS, b, rate),
                                       rtn_cpmg_coherence(2 * self.TAUS, 1, b, rate), atol=1e-12)

    def test_transfer_matrix_matches_closed_forms(self):
        for src in (WEAK, STRONG):
            b, rate = src.b_high, src.rate_up
            for tau in self.TAUS:
                with self.subTest(b=b, tau=tau):
                    self.assertAlmostEqual(telegraph_coherence(segments(nvsim.ramsey(tau)), src).real,
                                           float(rtn_fid_coherence(tau, b, rate)), places=4)
                    self.assertAlmostEqual(telegraph_coherence(segments(nvsim.spin_echo(tau)), src).real,
                                           float(rtn_echo_coherence(tau, b, rate)), places=4)
                    self.assertAlmostEqual(telegraph_coherence(segments(nvsim.cpmg(tau, 4)), src).real,
                                           float(rtn_cpmg_coherence(tau, 4, b, rate)), places=4)
                    self.assertAlmostEqual(telegraph_coherence(segments(nvsim.xy8(tau)), src).real,
                                           float(rtn_cpmg_coherence(tau, 8, b, rate)), places=4)

    def test_limits(self):
        np.testing.assert_allclose(rtn_echo_coherence(self.TAUS, 0.0, 1.0), 1.0, atol=1e-12)
        # Strong coupling: the echo decays as exp(-rate * total time) up to small oscillations.
        tau = np.linspace(1, 8, 30)
        envelope = np.exp(-2 * STRONG.rate_up * tau)
        np.testing.assert_allclose(rtn_echo_coherence(tau, STRONG.b_high, STRONG.rate_up), envelope,
                                   atol=0.1 * envelope.max())

    def test_weak_coupling_is_gaussian(self):
        for tau in self.TAUS:
            segs = segments(nvsim.spin_echo(tau))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gaussian = coherence_decay(segs, [WEAK], allow_non_gaussian=True)
            self.assertAlmostEqual(float(rtn_echo_coherence(tau, WEAK.b_high, WEAK.rate_up)), gaussian, delta=0.01)

    def test_strong_coupling_is_not_gaussian(self):
        segs = segments(nvsim.spin_echo(1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gaussian = coherence_decay(segs, [STRONG], allow_non_gaussian=True)
        self.assertGreater(float(rtn_echo_coherence(1.0, STRONG.b_high, STRONG.rate_up)) - gaussian, 0.4)


class TestGaussianGuard(unittest.TestCase):
    def test_theory_refuses_non_gaussian(self):
        segs = segments(nvsim.spin_echo(1.0))
        for src in (STRONG, NarrowbandNoise(1 * uT, 1.0, gaussian=False)):
            with self.subTest(src.label):
                with self.assertRaises(ValueError):
                    phase_variance(segs, [src])
                with self.assertWarns(UserWarning):
                    coherence_decay(segs, [src], allow_non_gaussian=True)

    def test_study_theory_availability(self):
        scan = echo_scan([0.5, 1.0, 2.0])
        kwargs = dict(n_realizations=20, readout=Readout(), seed=0)
        # No exact theory for fixed-amplitude narrowband noise: curve skipped, with a note.
        noisy = NoiseStudy("fixed", NV, FAST, [scan], [NarrowbandNoise(1 * uT, 1.0, gaussian=False)], **kwargs)
        self.assertFalse(noisy.theory_available)
        self.assertIn("non-Gaussian", noisy.theory_note)
        self.assertIsNone(run_scan(noisy, scan, np.random.default_rng(0)).p0_theory)
        # Gaussian and telegraph (T104): exact theory available.
        for noise in ([OUNoise(2 * uT, 1.0)], [STRONG, OUNoise(2 * uT, 1.0)]):
            study = NoiseStudy("ok", NV, FAST, [scan], noise, **kwargs)
            self.assertIsNone(study.theory_note)
            self.assertIsNotNone(run_scan(study, scan, np.random.default_rng(0)).p0_theory)


class TestMonteCarlo(unittest.TestCase):
    N = 4000

    def assert_mc_matches(self, seq, source, expected=None, rng=7):
        mc = simulate_ensemble(seq, NV, FAST, [source], n_realizations=self.N, rng=rng)
        p0_clean = nvsim.simulate(seq, NV, FAST, store_trajectory=False).final_populations["0"]
        contrast = p0_clean - 0.5
        coherence = (mc.populations["0"] - 0.5) / contrast
        error = mc.uncertainty["0"] / abs(contrast)
        if expected is None:
            expected = telegraph_coherence(segments(seq, FAST), source).real
        self.assertAlmostEqual(coherence, expected, delta=4 * error + 0.005)
        return coherence, error

    def test_weak_coupling_echo(self):
        for tau in (5.0, 10.0):
            with self.subTest(tau=tau):
                seq = nvsim.spin_echo(tau)
                self.assert_mc_matches(seq, WEAK, float(rtn_echo_coherence(tau, WEAK.b_high, WEAK.rate_up)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gaussian = coherence_decay(segments(seq, FAST), [WEAK], allow_non_gaussian=True)
                self.assert_mc_matches(seq, WEAK, gaussian, rng=8)

    def test_strong_coupling_echo_and_cpmg(self):
        for seq in (nvsim.spin_echo(1.0), nvsim.spin_echo(3.0), nvsim.cpmg(1.5, 4)):
            with self.subTest(seq.name):
                self.assert_mc_matches(seq, STRONG)

    def test_asymmetric_with_mean_offset(self):
        # Non-zero mean: Ramsey sees it as a phase (Re C oscillates), the echo refocuses it.
        source = TelegraphNoise(b_low=0.0, b_high=10 * uT, rate_up=0.3, rate_down=0.6)
        for seq in (nvsim.ramsey(1.5), nvsim.spin_echo(2.0)):
            with self.subTest(seq.name):
                self.assert_mc_matches(seq, source)


if __name__ == "__main__":
    unittest.main()
