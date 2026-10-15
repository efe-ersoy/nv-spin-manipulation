"""Optical readout: from spin populations to NV photoluminescence (PL).

The m_s = +-1 states fluoresce less than m_s = 0 during the readout laser
pulse. With a fractional ``contrast`` C the normalised PL is

    PL = P0 + (1 - C) (P+1 + P-1)    (1 for |0>, 1 - C for |+-1>).

Optionally photon shot noise is added: each point is the average of ``shots``
readouts that detect on average ``photons_per_shot`` photons from |0>.

Echo / DD scans record the normalised difference of the two sequences that end
with +pi/2 and -pi/2,

    se_norm = (N2 - N1) / (N2 + N1),

computed by :meth:`Readout.se_norm` from the two final |0> populations. One
readout maps the coherence C onto P0 = (1 + C)/2 ("bright" ending), the other
onto (1 - C)/2 ("dark" ending), so se_norm = -+ contrast C / (2 - contrast):
its magnitude for full coherence is contrast / (2 - contrast) (0.176 for 0.3).
Which of the two is N2 is not yet confirmed for the lab software, so it is a
parameter (``n2_bright``; default False = N2 is the dark-ending readout, which
gives the negative se_norm measured at short delay).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Readout:
    """PL model.

    Parameters
    ----------
    contrast : float
        Relative PL drop of m_s = +-1 compared to m_s = 0 (typically 0.2-0.3).
    photons_per_shot : float, optional
        Mean detected photons per readout of |0> (e.g. 0.03 for a single NV
        in confocal, many more for ensembles). Enables shot noise together
        with ``shots``.
    shots : int, optional
        Number of repetitions averaged per point.
    """

    contrast: float = 0.3
    photons_per_shot: float | None = None
    shots: int | None = None

    @property
    def shot_noise(self) -> bool:
        return self.photons_per_shot is not None and self.shots is not None

    def pl(self, populations) -> np.ndarray:
        """Normalised PL from a populations dict (arrays or floats)."""
        bright = np.asarray(populations["0"], dtype=float)
        return bright + (1 - self.contrast) * (1 - bright)

    def pl_error(self, p0_error) -> np.ndarray:
        """Propagate an uncertainty on P0 to the PL."""
        return self.contrast * np.asarray(p0_error, dtype=float)

    def measure(self, populations, rng=None) -> tuple[np.ndarray, np.ndarray]:
        """PL including photon shot noise (if configured) and its standard error."""
        pl = self.pl(populations)
        if not self.shot_noise:
            return pl, np.zeros_like(pl)
        rng = np.random.default_rng(rng)
        expected = self.shots * self.photons_per_shot * pl
        counts = rng.poisson(expected)
        norm = self.shots * self.photons_per_shot
        return counts / norm, np.sqrt(np.maximum(counts, 1)) / norm

    def se_norm(self, p0_n1, p0_n2) -> np.ndarray:
        """(N2 - N1) / (N2 + N1) from the final |0> populations of the two readouts
        (no shot noise; counts are proportional to the PL)."""
        pl1, pl2 = self.pl({"0": p0_n1}), self.pl({"0": p0_n2})
        return (pl2 - pl1) / (pl2 + pl1)

    def se_norm_amplitude(self, n2_bright: bool = False) -> float:
        """se_norm for full coherence C = 1: -contrast/(2 - contrast), sign flipped if
        N2 is the bright-ending readout."""
        magnitude = self.contrast / (2.0 - self.contrast)
        return magnitude if n2_bright else -magnitude

    def se_norm_from_coherence(self, coherence, n2_bright: bool = False) -> np.ndarray:
        """se_norm for coherence C (real part; readout along the coherence axis)."""
        c = np.real(np.asarray(coherence))
        bright, dark = 0.5 * (1 + c), 0.5 * (1 - c)
        return self.se_norm(dark, bright) if n2_bright else self.se_norm(bright, dark)

