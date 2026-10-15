"""Physical constants and unit conventions.

Unit convention used throughout ``nvsim``:

* frequencies in MHz (cyclic, i.e. *not* multiplied by 2*pi)
* times in microseconds
* magnetic fields in millitesla

With these units a frequency ``f`` [MHz] times a time ``t`` [us] is a number of
cycles, so phases are ``2*pi*f*t``. Hamiltonians handed to QuTiP are in angular
units (rad/us), i.e. ``2*pi*f``.

The unit helpers make field values readable: ``4 * uT`` is 0.004 mT and
``50 * G`` is 5 mT.
"""

import numpy as np

TWO_PI = 2.0 * np.pi

#: Field units, expressed in the internal unit (mT).
mT = 1.0
uT = 1e-3
G = 0.1

#: Electron spin gyromagnetic ratio of the NV centre [MHz/mT] (28.025 GHz/T).
GAMMA_E = 28.025

#: 13C nuclear gyromagnetic ratio [MHz/mT] (10.7084 MHz/T).
GAMMA_C13 = 0.0107084

#: NV ground-state zero-field splitting at room temperature [MHz].
D_GS = 2870.0
