"""Factorised mixed-noise theory, se_norm readout and decay fitting (T104).

Run from the repository root:  python -m unittest discover tests
"""

import unittest

import numpy as np

import nvsim
from nvsim import MWChannel, NVCenter, uT
from nvsim.coherence import exact_coherence, exact_coherences, theory_supported
from nvsim.ensemble import simulate_ensemble
from nvsim.fitting import (
    Curve,
    DecayModel,
    DecaySequence,
    GaussianNoiseDecay,
    StretchedExp,
    TelegraphDecay,
    fit_decay,
    fit_joint,
    profile_interval,
)
from nvsim.noise import CarbonBath, NarrowbandNoise, OUNoise, QuasiStaticNoise, TelegraphNoise
from nvsim.readout import Readout
from nvsim.simulate import channel_map
from nvsim.tabulated import TraceNoise
from nvsim.telegraph import telegraph_coherence, telegraph_coherences

NV = NVCenter.from_field(5.0)
FAST = MWChannel("-1", rabi_frequency=200.0)
CHANNELS = channel_map(FAST)
T = np.linspace(0.4, 32, 40)  # total free evolution, as in the hotspot screenshot
SE_AMPLITUDE = Readout(0.3).se_norm_amplitude()


class TestFactorisedTheory(unittest.TestCase):
    def assert_mc_matches(self, seq, noise, n=4000, rng=11):
        mc = simulate_ensemble(seq, NV, FAST, noise, n_realizations=n, rng=rng)
        p0_clean = nvsim.simulate(seq, NV, FAST, store_trajectory=False).final_populations["0"]
        coherence = exact_coherence(seq.compile(CHANNELS), noise)
        expected = 0.5 + (p0_clean - 0.5) * coherence.real
        self.assertAlmostEqual(mc.populations["0"], expected, delta=4 * mc.uncertainty["0"] + 3e-3)

    def test_ou_plus_telegraph_echo(self):
        noise = [OUNoise(3 * uT, 1.0), TelegraphNoise.symmetric(10 * uT, 0.3)]
        for tau in (0.7, 2.0):
            with self.subTest(tau=tau):
                self.assert_mc_matches(nvsim.spin_echo(tau), noise)

    def test_carbon_plus_telegraph_cpmg(self):
        noise = [CarbonBath(field=50.0, b_static=2 * uT, b_larmor=2 * uT), TelegraphNoise.symmetric(5 * uT, 0.5)]
        self.assert_mc_matches(nvsim.cpmg(0.8, 4), noise)

    def test_asymmetric_telegraph_with_quasi_static_ramsey(self):
        # Non-zero telegraph mean: the coherence is complex; Ramsey reads out Re C.
        noise = [QuasiStaticNoise(1 * uT), TelegraphNoise(0.0, 8 * uT, 0.4, 0.8)]
        self.assertGreater(abs(exact_coherence(nvsim.ramsey(1.2).compile(CHANNELS), noise).imag), 0.05)
        self.assert_mc_matches(nvsim.ramsey(1.2), noise)

    def test_supported_sources(self):
        self.assertTrue(theory_supported([OUNoise(1 * uT, 1.0), TelegraphNoise.symmetric(1 * uT, 1.0)]))
        fixed = NarrowbandNoise(1 * uT, 1.0, gaussian=False)
        traces = TraceNoise(np.arange(0, 10, 0.1), np.zeros((1, 100)))
        for bad in (fixed, traces):
            self.assertFalse(theory_supported([bad]))
            with self.assertRaises(ValueError):
                exact_coherence(nvsim.spin_echo(1.0).compile(CHANNELS), [bad])

    def test_batched_telegraph_matches_single(self):
        source = TelegraphNoise(-3 * uT, 9 * uT, 0.3, 0.7)
        segments = [nvsim.spin_echo(tau).compile(CHANNELS) for tau in (0.0, 0.5, 2.0)]
        segments += [nvsim.xy8(0.4).compile(CHANNELS)]  # different structure in the same batch
        batch = telegraph_coherences(segments, source)
        single = [telegraph_coherence(s, source) for s in segments]
        np.testing.assert_allclose(batch, single, atol=1e-12)
        np.testing.assert_allclose(exact_coherences(segments, [source]), single, atol=1e-12)


class TestSeNorm(unittest.TestCase):
    def test_values_and_sign_convention(self):
        r = Readout(contrast=0.3)
        self.assertAlmostEqual(r.se_norm_amplitude(), -0.3 / 1.7)
        np.testing.assert_allclose(r.se_norm_from_coherence([1.0, 0.0, -1.0]), [-0.3 / 1.7, 0.0, 0.3 / 1.7])
        self.assertAlmostEqual(float(r.se_norm_from_coherence(1.0, n2_bright=True)), 0.3 / 1.7)
        # Direct form from two populations: N1 bright (P0 = 1), N2 dark (P0 = 0).
        self.assertAlmostEqual(float(r.se_norm(1.0, 0.0)), -0.3 / 1.7)

    def test_from_simulated_readouts(self):
        r = Readout(contrast=0.25)
        seq_plus, seq_minus = nvsim.spin_echo(1.0, readout_phase="x"), nvsim.spin_echo(1.0, readout_phase="-x")
        noise = [OUNoise(3 * uT, 1.0)]
        p_plus = simulate_ensemble(seq_plus, NV, FAST, noise, n_realizations=2000, rng=1).populations["0"]
        p_minus = simulate_ensemble(seq_minus, NV, FAST, noise, n_realizations=2000, rng=2).populations["0"]
        coherence = exact_coherence(seq_plus.compile(CHANNELS), noise).real
        self.assertAlmostEqual(float(r.se_norm(p_plus, p_minus)), float(r.se_norm_from_coherence(coherence)),
                               delta=0.004)


class TestDecaySequence(unittest.TestCase):
    def test_total_free_evolution(self):
        for kind, n, windows in (("echo", 1, 1), ("cpmg", 4, 4), ("xy8", 2, 16), ("xy4", 3, 12)):
            with self.subTest(kind=kind):
                seq = DecaySequence(kind, n, t_pi=0.04)
                self.assertEqual(seq.windows, windows)
                waits = [s.duration for s in seq.segments([8.0])[0] if not s.is_pulse]
                self.assertAlmostEqual(sum(waits), 8.0)
        with self.assertRaises(ValueError):
            DecaySequence("hahn")


class TestFitting(unittest.TestCase):
    SEQ = DecaySequence("echo", t_pi=0.05)

    def synthetic(self, model, truth, sigma, seed):
        clean = model.signal(T, truth)
        return clean + sigma * np.random.default_rng(seed).standard_normal(T.size)

    def assert_within(self, result, truth, names, k=2.0):
        for name in names:
            with self.subTest(param=name):
                self.assertLess(abs(result.params[name] - truth[name]), k * result.errors[name])

    def test_weak_coupling_recovery(self):
        model = DecayModel([TelegraphDecay()], self.SEQ)
        truth = {"offset": 0.0, "amplitude": SE_AMPLITUDE, "rtn.rate": 1.0, "rtn.jump": 6 * uT}
        result = fit_decay(T, self.synthetic(model, truth, 0.004, seed=0), 0.004, model)
        self.assert_within(result, truth, ["rtn.rate", "rtn.jump", "amplitude", "offset"])
        self.assertTrue(0.5 < result.reduced_chi2 < 1.6)

    def test_strong_coupling_rate_and_jump_lower_bound(self):
        model = DecayModel([TelegraphDecay()], self.SEQ)
        truth = {"offset": 0.0, "amplitude": SE_AMPLITUDE, "rtn.rate": 0.2, "rtn.jump": 100 * uT}
        result = fit_decay(T, self.synthetic(model, truth, 0.012, seed=0), 0.012, model)
        self.assert_within(result, truth, ["rtn.rate"])
        lo, hi = profile_interval(result, "rtn.jump", np.geomspace(1 * uT, 2000 * uT, 24))
        self.assertTrue(np.isinf(hi))  # jump only bounded from below
        self.assertLess(lo, truth["rtn.jump"])
        self.assertGreater(lo, 3 * uT)  # ... but the lower bound is informative

    def test_joint_fit_shared_intrinsic(self):
        off = DecayModel([StretchedExp()], self.SEQ)
        on = DecayModel([StretchedExp(), TelegraphDecay()], self.SEQ)
        truth = {"offset": 0.0, "amplitude": SE_AMPLITUDE, "intrinsic.T2": 10.0, "intrinsic.n": 1.5,
                 "rtn.rate": 0.3, "rtn.jump": 5 * uT}
        curves = [Curve(T, self.synthetic(off, truth, 0.006, seed=1), 0.006, off, "off"),
                  Curve(T, self.synthetic(on, truth, 0.006, seed=2), 0.006, on, "on")]
        result = fit_joint(curves, shared=["intrinsic.T2", "intrinsic.n"])
        self.assertIn("intrinsic.T2", result.params)
        self.assertIn("on:rtn.rate", result.params)
        self.assertNotIn("off:rtn.rate", result.params)
        global_truth = {"intrinsic.T2": 10.0, "intrinsic.n": 1.5, "on:rtn.rate": 0.3, "on:rtn.jump": 5 * uT}
        self.assert_within(result, global_truth, list(global_truth))

    def test_joint_fit_shared_rate_at_several_heights(self):
        model = DecayModel([TelegraphDecay()], self.SEQ)
        jumps = [60 * uT, 30 * uT, 15 * uT]  # e.g. three heights: same fluctuator, weaker field
        curves = []
        for k, jump in enumerate(jumps):
            truth = {"offset": 0.0, "amplitude": SE_AMPLITUDE, "rtn.rate": 0.25, "rtn.jump": jump}
            curves.append(Curve(T, self.synthetic(model, truth, 0.008, seed=10 + k), 0.008, model, f"h{k}"))
        result = fit_joint(curves, shared=["rtn.rate"])
        self.assert_within(result, {"rtn.rate": 0.25}, ["rtn.rate"])
        self.assertEqual(sum(name.endswith("rtn.jump") for name in result.params), 3)

    def test_model_selection_by_aic(self):
        true_model = DecayModel([StretchedExp(), TelegraphDecay()], self.SEQ)
        truth = {"offset": 0.0, "amplitude": SE_AMPLITUDE, "intrinsic.T2": 10.0, "intrinsic.n": 1.5,
                 "rtn.rate": 0.3, "rtn.jump": 5 * uT}
        y = self.synthetic(true_model, truth, 0.003, seed=3)
        # Intrinsic decay known (e.g. from the off-hotspot fit): compare with and without a fluctuator.
        fixed = {"intrinsic.T2": 10.0, "intrinsic.n": 1.5}
        with_rtn = fit_decay(T, y, 0.003, true_model, fixed=fixed)
        without = fit_decay(T, y, 0.003, DecayModel([StretchedExp()], self.SEQ), fixed=fixed)
        self.assertLess(with_rtn.aic, without.aic - 10)

    def test_gaussian_component_and_fixed_parameters(self):
        model = DecayModel([GaussianNoiseDecay(lambda b, tc: [OUNoise(b, tc)], p0={"b": 2 * uT, "tc": 2.0},
                                               bounds={"b": (1e-4, 0.1), "tc": (0.05, 100.0)}, name="ou")],
                           self.SEQ)
        truth = {"offset": 0.0, "amplitude": SE_AMPLITUDE, "ou.b": 3 * uT, "ou.tc": 1.0}
        y = self.synthetic(model, truth, 0.004, seed=4)
        result = fit_decay(T, y, 0.004, model, fixed={"offset": 0.0})
        self.assertEqual(result.params["offset"], 0.0)
        self.assertNotIn("offset", result.free)
        self.assert_within(result, truth, ["ou.b", "ou.tc"])


if __name__ == "__main__":
    unittest.main()
