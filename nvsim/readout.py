"""Optical readout: from spin populations to NV photoluminescence (PL).

The m_s = +-1 states fluoresce less than m_s = 0 during the readout laser
pulse. With a fractional ``contrast`` C the normalised PL is

    PL = P0 + (1 - C) (P+1 + P-1)    (1 for |0>, 1 - C for |+-1>).

Optionally photon shot noise is added: each point is the average of ``shots``
readouts that detect on average ``photons_per_shot`` photons from |0>.
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
