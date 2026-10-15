"""The NV ground-state spin-1 system: basis, operators and transition frequencies.

The Hilbert space is the m_s = {+1, 0, -1} triplet, ordered as the eigenbasis of
``qutip.jmat(1, 'z')``: index 0 -> |+1>, index 1 -> |0>, index 2 -> |-1>.

Each microwave transition |0> <-> |m> (m = -1 or +1) defines an effective
two-level system ("qubit"). We use the convention that |0> sits on the north
pole of its Bloch sphere and |m> on the south pole::

    sigma_x = |0><m| + |m><0|
    sigma_y = -i|0><m| + i|m><0|
    sigma_z = |0><0| - |m><m|
"""

from dataclasses import dataclass

import qutip as qt

from .constants import D_GS, GAMMA_E

#: Basis labels in Hilbert-space order.
LEVELS = ("+1", "0", "-1")

#: The two microwave transitions, labelled by the m_s level coupled to |0>.
TRANSITIONS = ("-1", "+1")

_LEVEL_ALIASES = {
    "+1": "+1",
    "1": "+1",
    1: "+1",
    "0": "0",
    0: "0",
    "-1": "-1",
    -1: "-1",
}


def normalize_level(level) -> str:
    """Return the canonical label ('+1', '0' or '-1') for an m_s level."""
    try:
        return _LEVEL_ALIASES[level]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown m_s level {level!r}; use '+1', '0' or '-1'."
        ) from None


def normalize_transition(transition) -> str:
    """Return the canonical label ('-1' or '+1') of a |0> <-> |m> transition."""
    label = normalize_level(transition)
    if label == "0":
        raise ValueError(
            "A transition is labelled by its m_s = -1 or +1 level, not '0'."
        )
    return label


def basis(level) -> qt.Qobj:
    """Ket of the m_s level ``level``."""
    return qt.basis(3, LEVELS.index(normalize_level(level)))


def projector(level) -> qt.Qobj:
    """Projector |m><m| onto the m_s level ``level``."""
    ket = basis(level)
    return ket * ket.dag()


def transition_operator(transition) -> qt.Qobj:
    """Lowering-type operator |0><m| of the transition |0> <-> |m>."""
    return basis("0") * basis(normalize_transition(transition)).dag()


def qubit_operators(transition) -> tuple[qt.Qobj, qt.Qobj, qt.Qobj]:
    """Pauli operators (sx, sy, sz) of the two-level subspace {|0>, |m>}."""
    sig = transition_operator(transition)
    sx = sig + sig.dag()
    sy = -1j * sig + 1j * sig.dag()
    sz = projector("0") - projector(transition)
    return sx, sy, sz


def spin_operators() -> tuple[qt.Qobj, qt.Qobj, qt.Qobj]:
    """Spin-1 operators (Sx, Sy, Sz) in the {+1, 0, -1} basis."""
    return qt.jmat(1, "x"), qt.jmat(1, "y"), qt.jmat(1, "z")


def as_state(state) -> qt.Qobj:
    """Turn a level label, ket or density matrix into a QuTiP state."""
    if isinstance(state, qt.Qobj):
        if state.shape not in ((3, 1), (3, 3)):
            raise ValueError(
                f"Expected a 3-level ket or density matrix, got shape {state.shape}."
            )
        return state
    return basis(state)


@dataclass(frozen=True)
class NVCenter:
    """Transition frequencies of the NV ground-state spin.

    Parameters
    ----------
    f_minus : float
        Frequency of the |0> <-> |-1> transition [MHz].
    f_plus : float
        Frequency of the |0> <-> |+1> transition [MHz].

    Use :meth:`from_field` to derive both from a static field along the NV axis.
    """

    f_minus: float = D_GS
    f_plus: float = D_GS

    @classmethod
    def from_field(
        cls, Bz: float, D: float = D_GS, gamma: float = GAMMA_E
    ) -> "NVCenter":
        """NV with an axial static field ``Bz`` [mT]: f_+- = D +- gamma * Bz."""
        return cls(f_minus=D - gamma * Bz, f_plus=D + gamma * Bz)

    def transition_frequency(self, transition) -> float:
        """Frequency [MHz] of the transition |0> <-> |m>."""
        return self.f_minus if normalize_transition(transition) == "-1" else self.f_plus

    @property
    def zero_field_splitting(self) -> float:
        """Mean of the two transition frequencies [MHz]."""
        return 0.5 * (self.f_plus + self.f_minus)

    @property
    def Bz(self) -> float:
        """Axial field [mT] implied by the splitting of the two transitions."""
        return (self.f_plus - self.f_minus) / (2.0 * GAMMA_E)
