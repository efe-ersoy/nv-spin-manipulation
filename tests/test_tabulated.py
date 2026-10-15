"""Tabulated noise: SpectrumNoise, TraceNoise and the data-contract helpers.

Run from the repository root:  python -m unittest discover tests
"""

import unittest

import numpy as np

import nvsim
from nvsim import MWChannel, NVCenter, uT
from nvsim.ensemble import simulate_ensemble
from nvsim.filter_function import coherence_decay
from nvsim.noise import NarrowbandNoise, OUNoise, TelegraphNoise
from nvsim.simulate import channel_map
from nvsim.tabulated import (
    SpectrumNoise,
    TraceNoise,
    spectrum_noise_from_dataset,
    spectrum_noise_from_si,
    trace_noise_from_dataset,
    trace_noise_from_si,
)
from nvsim.telegraph import rtn_echo_coherence

NV = NVCenter.from_field(5.0)
FAST = MWChannel("-1", rabi_frequency=200.0)
CHANNELS = channel_map(FAST)
OU = OUNoise(4 * uT, correlation_time=1.0)
F_TABLE = np.concatenate([[0.0], np.geomspace(1e-4, 50.0, 3000)])
OU_TABLE = SpectrumNoise(F_TABLE, OU.psd(F_TABLE), label="OU table")


def echo_coherence_mc(source, tau, n=3000, rng=0):
    mc = simulate_ensemble(nvsim.spin_echo(tau), NV, FAST, [source], n_realizations=n, rng=rng)
    return 2 * mc.populations["0"] - 1, 2 * mc.uncertainty["0"]  # echo returns to |0> without noise


class TestSpectrumNoise(unittest.TestCase):
    def test_validation(self):
        for f, s in [([0.0], [1.0]), ([0, 1], [1, -1]), ([0, 1, 1], [1, 1, 1]), ([0, 1], [1, 2, 3])]:
            with self.assertRaises(ValueError):
                SpectrumNoise(f, s)

    def test_psd_interpolation_and_extension(self):
        sp = SpectrumNoise([1.0, 2.0, 4.0], [2.0, 4.0, 0.0])
        np.testing.assert_allclose(sp.psd([0.0, 0.5, 1.5, 3.0, 5.0]), [2.0, 2.0, 3.0, 2.0, 0.0])
        self.assertAlmostEqual(sp.variance, 2.0 * 1 + 3.0 * 1 + 4.0)  # flat to 0 Hz, two trapezoids
        self.assertAlmostEqual(float(sp.cumulative(1.5)), 2.0 + 0.5 * (2.0 + 3.0) / 2)
        self.assertTrue(sp.gaussian)

    def test_autocorrelation_matches_ou(self):
        uniform = np.arange(0, 3000) * 0.004  # as used by the filter-function theory
        # Absolute error relative to the variance (the tail of C is tiny).
        np.testing.assert_allclose(OU_TABLE.autocorrelation(uniform)[1:], OU.autocorrelation(uniform)[1:],
                                   rtol=2e-3, atol=1e-4 * OU.b_rms**2)
        scattered = np.array([0.003, 0.17, 0.9, 2.5, 7.0])
        np.testing.assert_allclose(OU_TABLE.autocorrelation(scattered), OU.autocorrelation(scattered), rtol=3e-3)
        # Lag 0 = variance; the table stops at 50 MHz and misses the 1/f^2 tail beyond it.
        self.assertAlmostEqual(float(OU_TABLE.autocorrelation([0.0])[0]) / OU.b_rms**2, 1.0, delta=3e-3)

    def test_max_step_from_significant_weight(self):
        # 95 % of the *tabulated* variance (the table misses the tail beyond 50 MHz).
        quantile = 0.95 * OU_TABLE.variance / OU.b_rms**2
        f95 = np.tan(quantile * np.pi / 2) / (2 * np.pi * OU.correlation_time)
        self.assertAlmostEqual(OU_TABLE.max_step(), 1 / (20 * f95), delta=0.02 / (20 * f95))

    def test_sample_variance_and_correlation(self):
        rng = np.random.default_rng(1)
        for dt in (0.01, 0.5):  # 0.5 us: most of the table lies above the grid Nyquist frequency
            with self.subTest(dt=dt):
                t = np.arange(0, 20, dt)
                x = OU_TABLE.sample(t, 3000, rng)
                k = int(round(1.0 / dt))
                self.assertAlmostEqual(x.var() / OU.b_rms**2, 1.0, delta=0.03)
                self.assertAlmostEqual(np.mean(x[:, :-k] * x[:, k:]) / OU.b_rms**2, np.exp(-1.0), delta=0.03)

    def test_narrow_line_keeps_its_frequency(self):
        line = NarrowbandNoise(2 * uT, frequency=1.3, linewidth=0.01)
        f = np.linspace(0, 3, 30001)
        sp = SpectrumNoise(f, line.psd(f))
        lag = np.array([0.0, 2.0, 5.0, 9.0])
        np.testing.assert_allclose(sp.autocorrelation(lag), line.autocorrelation(lag), atol=0.02 * line.b_rms**2)
        t = np.arange(0, 10, 0.02)
        x = sp.sample(t, 2000, np.random.default_rng(2))
        for k in (100, 250, 450):
            self.assertAlmostEqual(np.mean(x[:, :-k] * x[:, k:]), float(line.autocorrelation(k * 0.02)),
                                   delta=0.06 * line.b_rms**2)

    def test_theory_matches_ou(self):
        for seq in (nvsim.spin_echo(1.0), nvsim.spin_echo(3.0), nvsim.cpmg(0.8, 8)):
            with self.subTest(seq.name):
                segs = seq.compile(CHANNELS)
                self.assertAlmostEqual(coherence_decay(segs, [OU_TABLE]), coherence_decay(segs, [OU]), delta=2e-3)

    def test_monte_carlo_matches_ou(self):
        for tau in (1.0, 2.0):
            with self.subTest(tau=tau):
                value, error = echo_coherence_mc(OU_TABLE, tau, rng=int(10 * tau))
                expected = coherence_decay(nvsim.spin_echo(tau).compile(CHANNELS), [OU])
                self.assertAlmostEqual(value, expected, delta=4 * error + 0.005)


class TestTraceNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4)
        t = np.arange(0, 200, 0.01)
        cls.ou_traces = TraceNoise(t, OU.sample(t, 100, rng), gaussian=True, label="OU traces")
        cls.rtn = TelegraphNoise.symmetric(20 * uT, rate=0.2)
        t2 = np.arange(0, 400, 0.005)
        cls.rtn_traces = TraceNoise(t2, cls.rtn.sample(t2, 60, rng), label="RTN traces")

    def test_validation(self):
        with self.assertRaises(ValueError):
            TraceNoise([0, 1, 3], np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            TraceNoise([0, 1, 2], np.zeros((2, 4)))
        short = TraceNoise(np.arange(0, 1, 0.01), np.zeros((1, 100)))
        with self.assertRaises(ValueError):
            short.sample(np.arange(0, 2, 0.01), 3, np.random.default_rng(0))

    def test_sample_replays_trace_values(self):
        t = np.arange(0, 10, 0.5)
        traces = np.vstack([np.full_like(t, 1.0), np.full_like(t, 2.0)])
        src = TraceNoise(t, traces)
        x = src.sample(np.arange(0, 3, 0.1), 50, np.random.default_rng(0))
        self.assertTrue(set(np.unique(x)) <= {1.0, 2.0})
        self.assertFalse(src.gaussian)
        self.assertAlmostEqual(src.mean, 1.5)

    def test_estimates_from_ou_traces(self):
        lag = np.array([0.0, 0.5, 1.0, 2.0])
        np.testing.assert_allclose(self.ou_traces.autocorrelation(lag), OU.autocorrelation(lag), rtol=0.03)
        f = np.array([0.05, 0.3, 1.0])
        np.testing.assert_allclose(self.ou_traces.psd(f), OU.psd(f), rtol=0.15)

    def test_ou_traces_reproduce_ou(self):
        for tau in (1.0, 2.0):
            with self.subTest(tau=tau):
                value, error = echo_coherence_mc(self.ou_traces, tau, n=4000, rng=5)
                expected = coherence_decay(nvsim.spin_echo(tau).compile(CHANNELS), [OU])
                self.assertAlmostEqual(value, expected, delta=4 * error + 0.01)

    def test_telegraph_traces_reproduce_t101(self):
        for tau in (1.0, 3.0):
            with self.subTest(tau=tau):
                value, error = echo_coherence_mc(self.rtn_traces, tau, n=4000, rng=6)
                expected = float(rtn_echo_coherence(tau, self.rtn.b_high, self.rtn.rate_up))
                self.assertAlmostEqual(value, expected, delta=4 * error + 0.01)
        with self.assertRaises(ValueError):  # non-Gaussian traces: no Gaussian theory
            coherence_decay(nvsim.spin_echo(1.0).compile(CHANNELS), [self.rtn_traces])


class FakeVar:
    """Minimal stand-in for an xarray variable: values, dims, attrs."""

    def __init__(self, values, dims, units=None):
        self.values = np.asarray(values, dtype=float)
        self.dims = tuple(dims)
        self.attrs = {} if units is None else {"units": units}


class TestDataContract(unittest.TestCase):
    def test_si_conversion(self):
        sp = spectrum_noise_from_si([0.0, 1e6, 2e6], [4e-12, 4e-12, 0.0])
        np.testing.assert_allclose(sp.f, [0.0, 1.0, 2.0])  # MHz
        np.testing.assert_allclose(sp.s, [4.0, 4.0, 0.0])  # mT^2/MHz
        tr = trace_noise_from_si(np.arange(5) * 1e-8, np.full((1, 5), 2e-6))
        self.assertAlmostEqual(tr.dt, 0.01)  # us
        self.assertAlmostEqual(tr.mean, 2e-3)  # mT

    def field_map(self, psd_units="T^2/Hz"):
        f = np.array([0.0, 1e6, 2e6])
        x, y = np.array([0.0, 100e-9]), np.array([0.0, 50e-9, 100e-9])
        s = np.zeros((3, len(y), len(x)))
        s[:, 2, 1] = [1e-12, 1e-12, 0.0]  # pixel (x = 100 nm, y = 100 nm)
        return {"f": FakeVar(f, ["f"], "Hz"), "x": FakeVar(x, ["x"], "m"), "y": FakeVar(y, ["y"], "m"),
                "S_nv": FakeVar(s, ["f", "y", "x"], psd_units)}

    def test_spectrum_from_field_map(self):
        sp = spectrum_noise_from_dataset(self.field_map(), x=90e-9, y=95e-9)
        np.testing.assert_allclose(sp.s, [1.0, 1.0, 0.0])
        self.assertIn("S_nv", sp.label)
        with self.assertRaises(ValueError):
            spectrum_noise_from_dataset(self.field_map())  # pixel not given
        with self.assertRaises(ValueError):
            spectrum_noise_from_dataset(self.field_map(psd_units="mT^2/MHz"), x=0, y=0)

    def test_traces_from_time_series(self):
        t = np.arange(100) * 1e-8
        b = np.zeros((3, 100, 1, 1))  # dims (run, t, y, x)
        b[1] = 5e-6
        ds = {"t": FakeVar(t, ["t"], "s"), "x": FakeVar([0.0], ["x"], "m"), "y": FakeVar([0.0], ["y"], "m"),
              "B_nv": FakeVar(b, ["run", "t", "y", "x"], "T")}
        tr = trace_noise_from_dataset(ds)
        self.assertEqual(tr.traces.shape, (3, 100))
        np.testing.assert_allclose(tr.traces[1], 5e-3)
        self.assertAlmostEqual(tr.dt, 0.01)


if __name__ == "__main__":
    unittest.main()
