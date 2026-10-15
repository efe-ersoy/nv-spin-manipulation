"""Checks of the Rabi / Ramsey physics against analytic results.

Run from the repository root:  python -m unittest discover tests
"""

import unittest

import numpy as np
import qutip as qt

import nvsim
from nvsim import MWChannel, NVCenter, Pulse, PulseSequence, Wait, simulate

NV = NVCenter.from_field(5.0)


class TestSystem(unittest.TestCase):
    def test_transition_frequencies_from_field(self):
        nv = NVCenter.from_field(10.0)
        self.assertAlmostEqual(nv.f_minus, nvsim.D_GS - 10 * nvsim.GAMMA_E)
        self.assertAlmostEqual(nv.f_plus, nvsim.D_GS + 10 * nvsim.GAMMA_E)
        self.assertAlmostEqual(nv.Bz, 10.0)

    def test_channel_frequency_and_detuning_agree(self):
        by_freq = MWChannel("-1", 10, frequency=NV.f_minus + 1.5)
        self.assertAlmostEqual(by_freq.detuning_for(NV), 1.5)
        with self.assertRaises(ValueError):
            MWChannel("-1", 10, detuning=1, frequency=2800)

    def test_pulse_needs_angle_or_duration(self):
        with self.assertRaises(ValueError):
            Pulse()
        with self.assertRaises(ValueError):
            Pulse(np.pi, duration=0.1)


class TestRabi(unittest.TestCase):
    def test_resonant_rabi_oscillation(self):
        r = simulate(nvsim.rabi(0.5), NV, MWChannel("-1", 7.0))
        np.testing.assert_allclose(r.population("0"), np.cos(np.pi * 7.0 * r.times) ** 2, atol=1e-10)
        np.testing.assert_allclose(r.population("+1"), 0, atol=1e-12)

    def test_detuned_rabi_oscillation(self):
        omega, delta = 5.0, 3.0
        r = simulate(nvsim.rabi(0.5), NV, MWChannel("-1", omega, detuning=delta))
        w = np.hypot(omega, delta)
        expected = (omega / w) ** 2 * np.sin(np.pi * w * r.times) ** 2
        np.testing.assert_allclose(r.population("-1"), expected, atol=1e-10)

    def test_plus_one_transition(self):
        r = simulate(PulseSequence([Pulse(np.pi, "+1")]), NV, MWChannel("+1", 10.0))
        self.assertAlmostEqual(r.final_populations["+1"], 1.0, places=10)
        self.assertAlmostEqual(r.final_populations["-1"], 0.0, places=12)

    def test_pi_pulse_duration_from_angle(self):
        seq = PulseSequence([Pulse(np.pi)])
        self.assertAlmostEqual(seq.total_duration({"-1": MWChannel("-1", 5.0)}), 0.1)

    def test_pulse_phase_sets_rotation_axis(self):
        ch = MWChannel("-1", 10.0)
        about_x = simulate(PulseSequence([Pulse(np.pi / 2, phase="x")]), NV, ch)
        about_y = simulate(PulseSequence([Pulse(np.pi / 2, phase="y")]), NV, ch)
        np.testing.assert_allclose(about_x.bloch_vector()[-1], [0, -1, 0], atol=1e-10)
        np.testing.assert_allclose(about_y.bloch_vector()[-1], [1, 0, 0], atol=1e-10)

    def test_negative_angle_rotates_backwards(self):
        ch = MWChannel("-1", 10.0)
        r = simulate(PulseSequence([Pulse(-np.pi / 2, phase="x")]), NV, ch)
        np.testing.assert_allclose(r.bloch_vector()[-1], [0, 1, 0], atol=1e-10)

    def test_bloch_vector_stays_on_sphere(self):
        r = simulate(nvsim.rabi(0.3), NV, MWChannel("-1", 5.0, detuning=2.0))
        np.testing.assert_allclose(np.linalg.norm(r.bloch_vector("-1"), axis=1), 1.0, atol=1e-10)


class TestRamsey(unittest.TestCase):
    # Pulses much faster than the detuning approximate ideal instantaneous pulses.
    CHANNEL = MWChannel("-1", rabi_frequency=500.0, detuning=1.0)
    TAUS = np.linspace(0, 2, 41)

    def test_fringes_readout_x(self):
        s = nvsim.sweep(self.TAUS, lambda tau: nvsim.ramsey(tau), NV, self.CHANNEL)
        np.testing.assert_allclose(s.population("0"), np.sin(np.pi * self.TAUS) ** 2, atol=5e-3)

    def test_fringes_readout_y(self):
        s = nvsim.sweep(self.TAUS, lambda tau: nvsim.ramsey(tau, readout_phase="y"), NV, self.CHANNEL)
        expected = (1 - np.sin(2 * np.pi * self.TAUS)) / 2
        np.testing.assert_allclose(s.population("0"), expected, atol=5e-3)

    def test_sweep_matches_full_simulation(self):
        ch = MWChannel("-1", 10.0, detuning=2.0)
        s = nvsim.sweep([0.3], lambda tau: nvsim.ramsey(tau), NV, ch)
        r = simulate(nvsim.ramsey(0.3), NV, ch)
        self.assertAlmostEqual(s.population("0")[0], r.final_populations["0"], places=10)

    def test_resonant_free_evolution_is_static(self):
        seq = PulseSequence([Pulse(np.pi / 2), Wait(1.0)])
        r = simulate(seq, NV, MWChannel("-1", 10.0))
        wait = r.times >= r.segments[1].start
        np.testing.assert_allclose(r.bloch_vector()[wait], [[0, -1, 0]] * wait.sum(), atol=1e-10)


class TestSolverPaths(unittest.TestCase):
    def test_time_dependent_path_matches_exact_path(self):
        op = 2 * np.pi * 0.7 * nvsim.projector("-1")
        seq = nvsim.ramsey(0.4)
        ch = MWChannel("-1", 10.0)
        exact = simulate(seq, NV, ch, extra_terms=[op])
        ode = simulate(seq, NV, ch, extra_terms=[(op, lambda t: 1.0)])
        np.testing.assert_allclose(ode.population("0")[-1], exact.population("0")[-1], atol=1e-6)

    def test_density_matrix_path_matches_ket_path(self):
        seq = nvsim.ramsey(0.4)
        ch = MWChannel("-1", 10.0, detuning=1.0)
        ket = simulate(seq, NV, ch)
        dm = simulate(seq, NV, ch, c_ops=[0 * qt.qeye(3)])
        self.assertTrue(dm.final_state.isoper)
        np.testing.assert_allclose(dm.population("0"), ket.population("0"), atol=1e-10)

    def test_crosstalk_leakage_is_small_and_off_by_default(self):
        ch = MWChannel("-1", 10.0)
        seq = PulseSequence([Pulse(np.pi)])
        plain = simulate(seq, NV, ch)
        xtalk = simulate(seq, NV, ch, crosstalk=True)
        self.assertEqual(plain.population("+1").max(), 0.0)
        splitting = NV.f_plus - NV.f_minus
        self.assertLess(xtalk.population("+1").max(), 3 * (10.0 / splitting) ** 2)
        self.assertGreater(xtalk.population("+1").max(), 0.0)
        self.assertAlmostEqual(xtalk.final_populations["-1"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
