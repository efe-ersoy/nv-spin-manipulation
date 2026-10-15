"""Echo / DD signal maps (T103).

Run from the repository root:  python -m unittest discover tests
"""

import time
import unittest

import numpy as np

import nvsim
from nvsim import ACField, MWChannel, NVCenter, StaticField, uT
from nvsim.ensemble import simulate_ensemble
from nvsim.filter_function import coherence_decay
from nvsim.maps import (
    PixelField,
    PixelSource,
    PixelSpectrum,
    coherence_map,
    map_shape,
    monte_carlo_map,
    signal_map,
    with_flipped_readout,
)
from nvsim.noise import CarbonBath, NarrowbandNoise, OUNoise, QuasiStaticNoise, TelegraphNoise
from nvsim.readout import Readout
from nvsim.simulate import channel_map
from nvsim.tabulated import SpectrumNoise, TraceNoise
from nvsim.telegraph import telegraph_coherence

DRIVE = MWChannel("-1", rabi_frequency=20.0)
SEQ = nvsim.spin_echo(2.5)
SEGMENTS = SEQ.compile(channel_map(DRIVE))
NV = NVCenter()


class TestGaussianMaps(unittest.TestCase):
    def test_acceptance_100x100_ou(self):
        rng = np.random.default_rng(0)
        b = rng.uniform(0.5, 5, (100, 100)) * uT
        tc = rng.uniform(0.2, 5, (100, 100))
        start = time.perf_counter()
        coherence, dt = coherence_map(SEQ, DRIVE, [PixelSource(OUNoise, b_rms=b, correlation_time=tc)])
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0)
        loop = np.array([[coherence_decay(SEGMENTS, [OUNoise(b[i, j], tc[i, j])], dt=dt) for j in range(100)]
                         for i in range(100)])
        self.assertLess(np.abs(coherence.real - loop).max(), 1e-6)
        self.assertLess(np.abs(coherence.imag).max(), 1e-12)

    def test_mixed_uniform_and_per_pixel_sources(self):
        b = np.array([[1.0, 2.0], [3.0, 4.0]]) * uT
        sources = [CarbonBath(field=20.0), PixelSource(QuasiStaticNoise, b_rms=b),
                   PixelSource(NarrowbandNoise, b_rms=b, frequency=0.3, linewidth=0.02)]
        coherence, dt = coherence_map(nvsim.cpmg(1.0, 4), DRIVE, sources)
        segs = nvsim.cpmg(1.0, 4).compile(channel_map(DRIVE))
        for i, j in np.ndindex(2, 2):
            pixel = [CarbonBath(field=20.0), QuasiStaticNoise(b[i, j]), NarrowbandNoise(b[i, j], 0.3, 0.02)]
            self.assertAlmostEqual(coherence[i, j].real, coherence_decay(segs, pixel, dt=dt), places=10)

    def test_pixel_spectrum_with_lines(self):
        rng = np.random.default_rng(1)
        f = np.linspace(0, 3, 301)
        amps = rng.uniform(0.5, 3, (3, 4)) * uT
        psd = np.stack([OUNoise(a, 1.0).psd(f) for a in amps.ravel()], axis=1).reshape(len(f), 3, 4)
        line_var = (rng.uniform(0.2, 1.0, (3, 4)) * uT) ** 2
        spectrum = PixelSpectrum(f, psd, lines=[(0.2, line_var)])
        coherence, dt = coherence_map(SEQ, DRIVE, [spectrum])
        for i, j in np.ndindex(3, 4):
            pixel = [SpectrumNoise(f, psd[:, i, j]), NarrowbandNoise(np.sqrt(line_var[i, j]), 0.2)]
            # the line enters analytically (|Y(f0)|^2) in the map, by the time-domain sum in the loop
            self.assertAlmostEqual(coherence[i, j].real, coherence_decay(SEGMENTS, pixel, dt=dt), places=8)


class TestNonGaussianAndFields(unittest.TestCase):
    def test_telegraph_map_is_exact(self):
        rng = np.random.default_rng(2)
        rate, b = rng.uniform(0.05, 1, (3, 3)), rng.uniform(1, 30, (3, 3)) * uT
        coherence, _ = coherence_map(SEQ, DRIVE, [PixelSource(TelegraphNoise.symmetric, b=b, rate=rate),
                                                  OUNoise(1 * uT, 2.0)])
        gaussian = coherence_decay(SEGMENTS, [OUNoise(1 * uT, 2.0)])
        for i, j in np.ndindex(3, 3):
            expected = gaussian * telegraph_coherence(SEGMENTS, TelegraphNoise.symmetric(b[i, j], rate[i, j]))
            self.assertAlmostEqual(abs(coherence[i, j] - expected), 0.0, places=10)

    def test_fixed_phase_fields_match_simulation(self):
        p0_clean = nvsim.simulate(SEQ, NV, DRIVE, store_trajectory=False).final_populations["0"]
        amp = np.array([2.0, 5.0, 10.0]) * uT
        for field, objects in [
            (PixelField(ACField, amplitude=amp, frequency=0.18, phase=0.4), [ACField(a, 0.18, 0.4) for a in amp]),
            (PixelField(StaticField, Bz=amp), [StaticField(a) for a in amp]),
        ]:
            with self.subTest(type(objects[0]).__name__):
                coherence, _ = coherence_map(SEQ, DRIVE, [], [field])
                for c, obj in zip(coherence, objects):
                    full = nvsim.simulate(SEQ, NV, DRIVE, fields=[obj], store_trajectory=False).final_populations["0"]
                    self.assertAlmostEqual(0.5 + (p0_clean - 0.5) * c.real, full, delta=1e-3)

    def test_random_phase_ac_field_matches_monte_carlo(self):
        amp = np.array([2.0, 5.0]) * uT
        coherence, _ = coherence_map(SEQ, DRIVE, [], [PixelField(ACField, random_phase=True, amplitude=amp,
                                                                 frequency=0.18)])
        for c, a in zip(coherence.real, amp):
            mc = simulate_ensemble(SEQ, NV, DRIVE, [NarrowbandNoise(a / np.sqrt(2), 0.18, 0.0, gaussian=False)],
                                   n_realizations=4000, rng=3)
            self.assertAlmostEqual(2 * mc.populations["0"] - 1, c, delta=4 * 2 * mc.uncertainty["0"] + 3e-3)

    def test_unsupported_sources_raise(self):
        traces = TraceNoise(np.arange(0, 20, 0.01), np.zeros((1, 2000)))
        with self.assertRaises(ValueError):
            coherence_map(SEQ, DRIVE, [traces])


class TestObservableAndMonteCarlo(unittest.TestCase):
    def test_flipped_readout(self):
        flipped = with_flipped_readout(SEQ)
        p = nvsim.simulate(SEQ, NV, DRIVE, store_trajectory=False).final_populations["0"]
        q = nvsim.simulate(flipped, NV, DRIVE, store_trajectory=False).final_populations["0"]
        self.assertAlmostEqual(p + q, 1.0, places=6)
        self.assertEqual(len(flipped), len(SEQ))

    def test_signal_map_se_norm(self):
        b = np.array([0.0, 2.0, 4.0]) * uT
        readout = Readout(contrast=0.3)
        result = signal_map(SEQ, DRIVE, [PixelSource(OUNoise, b_rms=b, correlation_time=1.0)], readout=readout)
        np.testing.assert_allclose(result.se_norm, readout.se_norm_from_coherence(result.coherence), atol=2e-3)
        self.assertAlmostEqual(result.se_norm[0], readout.se_norm_amplitude(), places=3)
        flipped = signal_map(SEQ, DRIVE, [PixelSource(OUNoise, b_rms=b, correlation_time=1.0)], readout=readout,
                             n2_bright=True)
        np.testing.assert_allclose(flipped.se_norm, -result.se_norm, atol=1e-12)

    def test_monte_carlo_matches_gaussian_theory(self):
        rng = np.random.default_rng(4)
        b, tc = rng.uniform(1, 5, (2, 3)) * uT, rng.uniform(0.3, 3, (2, 3))
        sources = [PixelSource(OUNoise, b_rms=b, correlation_time=tc)]
        calls = []
        theory = signal_map(SEQ, DRIVE, sources)
        mc = monte_carlo_map(SEQ, DRIVE, sources, n_realizations=2000, reuse_noise=False, seed=5,
                             progress=lambda done, total: calls.append(done))
        self.assertEqual(calls[-1], 6)
        np.testing.assert_array_less(np.abs(mc.coherence - theory.coherence.real), 4 * mc.coherence_error + 0.01)
        np.testing.assert_array_less(np.abs(mc.se_norm - theory.se_norm), 4 * mc.se_norm_error + 0.002)
        self.assertEqual(map_shape(sources), (2, 3))


if __name__ == "__main__":
    unittest.main()
