"""Exact coherence for mixed noise: Gaussian sources x telegraph fluctuators.

Independent noise sources add their phases, phi = sum_k phi_k, so the
coherence factorises:

    C = <exp(i phi)> = exp(-<phi^2>_Gaussian / 2) x prod_k <exp(i phi_k)>_telegraph

The Gaussian factor comes from the filter-function theory
(:func:`nvsim.filter_function.coherence_decay`, all Gaussian sources together),
each telegraph factor from :func:`nvsim.telegraph.telegraph_coherence`. Sources
that are neither (e.g. :class:`~nvsim.tabulated.TraceNoise`, fixed-amplitude
:class:`~nvsim.noise.NarrowbandNoise`) have no exact theory here; use the Monte
Carlo ensemble for them.

The result is complex; its real part multiplies the signal contrast when the
readout projects onto the coherence axis (all built-in sequences). A non-zero
imaginary part comes from telegraph mean fields (static offsets).
"""

import numpy as np

from .filter_function import coherence_decay
from .noise import TelegraphNoise, flatten
from .pulses import Segment
from .telegraph import telegraph_coherences


def unsupported_sources(sources) -> list:
    """Primitive sources without an exact theory (neither Gaussian nor telegraph)."""
    return [s for s in flatten(sources) if not (s.gaussian or isinstance(s, TelegraphNoise))]


def theory_supported(sources) -> bool:
    """True if :func:`exact_coherence` can handle every source."""
    return not unsupported_sources(sources)


def _split(sources):
    bad = unsupported_sources(sources)
    if bad:
        names = ", ".join(sorted({s.label for s in bad}))
        raise ValueError(f"No exact theory for {names}: only Gaussian and telegraph sources are supported; "
                         "use the Monte Carlo ensemble.")
    primitive = flatten(sources)
    return [s for s in primitive if s.gaussian], [s for s in primitive if isinstance(s, TelegraphNoise)]


def exact_coherences(segments_list, sources, transition="-1") -> np.ndarray:
    """Complex coherence for each compiled sequence in ``segments_list``."""
    gaussian, telegraph = _split(sources)
    result = np.ones(len(segments_list), dtype=complex)
    if gaussian:
        result *= np.array([coherence_decay(segments, gaussian) for segments in segments_list])
    for source in telegraph:
        result *= telegraph_coherences(segments_list, source, transition)
    return result


def exact_coherence(segments: list[Segment], sources, transition="-1") -> complex:
    """Complex coherence of one compiled sequence under mixed Gaussian + telegraph noise."""
    return complex(exact_coherences([segments], sources, transition)[0])
