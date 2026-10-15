"""Spin echo / dynamical decoupling, magnetic fields and accumulated phase.

Run from the repository root:  python -m unittest discover tests
"""

import unittest

import numpy as np

import nvsim
from nvsim import ACField, MWChannel, NVCenter, StaticField, simulate
from nvsim.phase import phase_jumps

NV = NVCenter.from_field(5.0)
FAST = MWChannel("-1", rabi_frequency=200.0)  # near-ideal pulses


class TestSequences(unittest.TestCase):
    def test_pulse_counts_and_phases(self):
        def pi_phases(seq):
            return [e.phase for e in seq if isinstance(e, nvsim.Pulse) and np.isclose(e.angle, np.pi)]

        self.assertEqual(len(pi_phases(nvsim.cpmg(1.0, 6))), 6)
        np.testing.assert_allclose(pi_phases(nvsim.cpmg(1.0, 3)), [np.pi / 2] * 3)
        x, y = 0.0, np.pi / 2
        np.testing.assert_allclose(pi_phases(nvsim.xy8(1.0)), [x, y, x, y, y, x, y, x])
        self.assertEqual(len(pi_phases(nvsim.xy8(1.0, repetitions=3))), 24)
        self.assertEqual(len(pi_phases(nvsim.xy4(1.0, repetitions=2))), 8)

    def test_dd_free_evolution_timing(self):
        waits = [e.duration for e in nvsim.cpmg(0.8, 3) if isinstance(e, nvsim.Wait)]
        np.testing.assert_allclose(waits, [0.4, 0.8, 0.8, 0.4])

    def test_field_free_spin_returns_to_zero(self):
        ch = MWChannel("-1", 10.0)
        for seq in [nvsim.spin_echo(0.7), nvsim.cpmg(0.5, 4), nvsim.xy4(0.5), nvsim.xy8(0.5, 2)]:
            with self.subTest(seq.name):
                self.assertAlmostEqual(simulate(seq, NV, ch).final_populations["0"], 1.0, places=10)

    def test_echo_refocuses_static_field(self):
        for seq in [nvsim.spin_echo(1.0), nvsim.cpmg(1.0, 4), nvsim.xy8(0.5)]:
            with self.subTest(seq.name):
                r = simulate(seq, NV, FAST, fields=[StaticField(0.02)])
                self.assertAlmostEqual(r.final_populations["0"], 1.0, places=3)

    def test_ramsey_is_not_refocused(self):
        r = simulate(nvsim.ramsey(1.0), NV, FAST, fields=[StaticField(0.005)])
        expected = np.sin(np.pi * nvsim.GAMMA_E * 0.005 * 1.0) ** 2
        self.assertAlmostEqual(r.final_populations["0"], expected, places=2)


class TestFields(unittest.TestCase):
    def test_static_field_equals_detuning(self):
        b = 0.03
        by_field = simulate(nvsim.ramsey(0.6), NV, MWChannel("-1", 10.0), fields=[StaticField(b)], dt=2e-3)
        by_detuning = simulate(nvsim.ramsey(0.6), NV, MWChannel("-1", 10.0, detuning=nvsim.GAMMA_E * b), dt=2e-3)
        np.testing.assert_allclose(by_field.population("0"), by_detuning.population("0"), atol=1e-10)

    def test_static_field_shifts_plus_one_the_other_way(self):
        b = 0.03
        by_field = simulate(nvsim.ramsey(0.6, "+1"), NV, MWChannel("+1", 10.0), fields=[StaticField(b)], dt=2e-3)
        by_detuning = simulate(nvsim.ramsey(0.6, "+1"), NV, MWChannel("+1", 10.0, detuning=-nvsim.GAMMA_E * b),
                               dt=2e-3)
        np.testing.assert_allclose(by_field.population("0"), by_detuning.population("0"), atol=1e-10)

    def test_diagonal_fast_path_matches_mesolve(self):
        ac = ACField(0.01, 0.7, phase=0.3)
        seq = nvsim.spin_echo(0.8)
        fast = simulate(seq, NV, MWChannel("-1", 10.0), fields=[ac])
        # A tiny off-diagonal term disables the diagonal fast path during waits.
        tiny = 1e-12 * (nvsim.qubit_operators("+1")[0])
        ode = simulate(seq, NV, MWChannel("-1", 10.0), fields=[ac], extra_terms=[(tiny, lambda t: 1.0)])
        self.assertAlmostEqual(fast.final_populations["0"], ode.final_populations["0"], places=6)

    def test_echo_ac_response_matches_ideal_pulse_theory(self):
        b, tau = 0.003, 1.0
        t_echo = 1 / (4 * 200) + tau + 1 / (4 * 200)
        T = t_echo - 1 / (8 * 200)
        for f in [0.2, 0.37, 0.6, 1.4]:
            with self.subTest(f=f):
                r = simulate(nvsim.spin_echo(tau), NV, FAST, fields=[ACField.sine_centered_at(b, f, t_echo)])
                phi = 2 * nvsim.GAMMA_E * b * (1 - np.cos(2 * np.pi * f * T)) / f
                self.assertAlmostEqual(1 - r.final_populations["0"], np.sin(phi / 2) ** 2, places=3)

    def test_sweep_accepts_callable_fields(self):
        s = nvsim.sweep([0.0, 0.01], nvsim.ramsey(1.0), NV, FAST, fields=lambda b: [StaticField(b)])
        self.assertAlmostEqual(s.population("0")[0], 0.0, places=6)
        self.assertGreater(s.population("0")[1], 0.1)


class TestAccumulatedPhase(unittest.TestCase):
    def test_ramsey_phase_grows_linearly(self):
        b = 0.002
        r = simulate(nvsim.ramsey(2.0), NV, FAST, fields=[StaticField(b)])
        phi = r.accumulated_phase()
        wait = r.segments[1]
        inside = (r.times > wait.start) & (r.times < wait.end)
        expected = np.angle(np.exp(1j * 2 * np.pi * nvsim.GAMMA_E * b * (r.times[inside] - r.times[inside][0])))
        np.testing.assert_allclose(phi[inside] - phi[inside][0], expected, atol=1e-3)
        self.assertTrue(np.all(np.isnan(phi[r.times < r.segments[0].end * 0.99])))

    def test_echo_pulse_flips_phase_and_static_field_refocuses(self):
        b, tau = 0.005, 1.0
        r = simulate(nvsim.spin_echo(tau), NV, FAST, fields=[StaticField(b)])
        (jump,) = phase_jumps(r)
        self.assertAlmostEqual(jump.phase_before, 2 * np.pi * nvsim.GAMMA_E * b * tau, delta=0.02)
        self.assertAlmostEqual(jump.phase_after, -jump.phase_before, delta=1e-3)
        phi = r.accumulated_phase()
        self.assertAlmostEqual(phi[np.isfinite(phi)][-1], 0.0, delta=0.02)

    def test_matched_ac_field_keeps_accumulating(self):
        tau = 1.0
        t_echo = 1 / 800 + tau + 1 / 800
        f = 1 / (2 * (tau + 1 / 400))
        r = simulate(nvsim.spin_echo(tau), NV, FAST, fields=[ACField.sine_centered_at(0.003, f, t_echo)])
        (jump,) = phase_jumps(r)
        phi = r.accumulated_phase()
        self.assertGreater(jump.phase_before, 0)
        self.assertAlmostEqual(jump.phase_after, -jump.phase_before, delta=1e-3)
        self.assertAlmostEqual(phi[np.isfinite(phi)][-1], 2 * jump.phase_after, delta=0.02)

    def test_every_pi_pulse_flips_the_phase(self):
        r = simulate(nvsim.xy8(0.3), NV, FAST, fields=[StaticField(0.005)])
        jumps = phase_jumps(r)
        self.assertEqual(len(jumps), 8)
        for jump in jumps:
            self.assertAlmostEqual(jump.phase_after, -jump.phase_before, delta=1e-3)

    def test_no_phase_without_free_evolution(self):
        r = simulate(nvsim.rabi(0.3), NV, MWChannel("-1", 10.0))
        self.assertTrue(np.all(np.isnan(r.accumulated_phase())))


if __name__ == "__main__":
    unittest.main()
