"""Fit measured echo / dynamical-decoupling decay curves with physical noise models.

Signal model
------------

    S(t) = offset + amplitude x C(t),    C = product of the model components

with t the **total free-evolution time** [us] of the stated sequence
(:class:`DecaySequence`: echo, CPMG-N, XY4-N or XY8-N, with the pi-pulse
duration). Components:

* :class:`StretchedExp` - exp(-(t/T2)^n), e.g. the intrinsic NV decay;
* :class:`GaussianNoiseDecay` - any set of Gaussian sources built from the fit
  parameters (filter-function theory);
* :class:`TelegraphDecay` - one symmetric or asymmetric telegraph fluctuator
  (exact two-state theory); use several for several fluctuators.

Independent decay channels multiply (:mod:`nvsim.coherence`). For the
experimental observable ``se_norm`` = (N2 - N1)/(N2 + N1) the signal is
offset + amplitude x C with amplitude ~ -+ contrast/(2 - contrast)
(:meth:`nvsim.readout.Readout.se_norm_amplitude`); offset and amplitude are free
by default and enter linearly.

Fitting
-------

:func:`fit_decay` (one curve) and :func:`fit_joint` (several curves sharing
parameters, e.g. the same switching rate at all heights, or the same intrinsic
T2 on and off a hotspot) minimise chi^2 = sum((y - S)/sigma)^2 with bounded
least squares. Starting values come from a grid over the parameters that have
one (telegraph rate and jump, intrinsic T2), with offset/amplitude solved
linearly at every grid point, followed by least squares from the best grid
points: the telegraph likelihood is multimodal in strong coupling.
Results: parameters, covariance (from the Jacobian, sigma taken as absolute),
chi^2, AIC = chi^2 + 2k and BIC = chi^2 + k ln N.

Degeneracy of telegraph fits
----------------------------

In strong coupling (2 pi gamma_e (jump/2) >> rate) a single switch already
dephases the spin, so C ~ exp(-rate t) (echo) depends on the jump only through
small corrections of order rate / (2 pi gamma_e jump/2). The data then fix the
rate but give only a **lower bound** on the jump; the covariance error of the
jump is meaningless there. Use :func:`profile_chi2` / :func:`profile_interval`
to quote the bound. In weak coupling only the combination jump^2 / rate (the
motional-narrowing decay rate) is well determined.
"""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .constants import uT
from .drive import MWChannel
from .filter_function import coherence_decay
from .noise import TelegraphNoise
from .pulses import PulseSequence
from .sequences import cpmg, spin_echo, xy4, xy8
from .simulate import channel_map
from .telegraph import telegraph_coherences

#: Rabi frequency used for "instantaneous" pulses (t_pi = 0).
_INSTANT_RABI = 1e6
#: Largest number of grid points evaluated for starting values.
MAX_GRID_POINTS = 3000
_LINEAR = ("offset", "amplitude")


# --------------------------------------------------------------------------- sequence


@dataclass(frozen=True)
class DecaySequence:
    """The measured sequence, parametrised by the total free-evolution time.

    Parameters
    ----------
    kind : {"echo", "cpmg", "xy4", "xy8"}
    n : int
        Number of pi pulses for CPMG, repetitions for XY4 / XY8 (ignored for echo).
    t_pi : float
        pi-pulse duration [us]; 0 = ideal instantaneous pulses.
    transition : str
        "-1" or "+1".
    """

    kind: str = "echo"
    n: int = 1
    t_pi: float = 0.0
    transition: str = "-1"

    def __post_init__(self):
        if self.kind not in ("echo", "cpmg", "xy4", "xy8"):
            raise ValueError(f"Unknown sequence kind {self.kind!r}; use echo, cpmg, xy4 or xy8.")
        if self.n < 1 or self.t_pi < 0:
            raise ValueError("n must be >= 1 and t_pi >= 0.")

    @property
    def windows(self) -> int:
        """Number of pulse spacings tau in the total free evolution."""
        return {"echo": 1, "cpmg": self.n, "xy4": 4 * self.n, "xy8": 8 * self.n}[self.kind]

    @property
    def label(self) -> str:
        return "echo" if self.kind == "echo" else f"{self.kind.upper()}-{self.n}"

    @property
    def drive(self) -> MWChannel:
        rabi = 1.0 / (2.0 * self.t_pi) if self.t_pi > 0 else _INSTANT_RABI
        return MWChannel(self.transition, rabi_frequency=rabi)

    def build(self, total: float) -> PulseSequence:
        """Sequence with total free evolution ``total`` [us]."""
        tau = total / self.windows
        kw = {"transition": self.transition}
        if self.kind == "echo":
            return spin_echo(tau / 2, **kw)
        if self.kind == "cpmg":
            return cpmg(tau, self.n, **kw)
        return (xy4 if self.kind == "xy4" else xy8)(tau, self.n, **kw)

    def segments(self, totals) -> list:
        channels = channel_map(self.drive)
        return [self.build(float(t)).compile(channels) for t in np.atleast_1d(totals)]


# --------------------------------------------------------------------------- components


class Component:
    """A multiplicative decay factor C_k(t; parameters)."""

    name: str
    params: tuple[str, ...]

    def __init__(self, name: str, p0: dict, bounds: dict, grid: dict | None = None):
        self.name = name
        self.params = tuple(p0)
        self.p0 = dict(p0)
        self.bounds = dict(bounds)
        self.grid = dict(grid or {})
        self._cache: dict = {}

    def coherence(self, t: np.ndarray, values: dict, sequence: DecaySequence, segments: list) -> np.ndarray:
        raise NotImplementedError

    def _memo(self, key, compute):
        if key not in self._cache:
            if len(self._cache) > 8192:
                self._cache.clear()
            self._cache[key] = compute()
        return self._cache[key]

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, params={self.params})"


class StretchedExp(Component):
    """C = exp(-(t / T2)^n): phenomenological (e.g. intrinsic) decay; t = total free evolution."""

    def __init__(self, name: str = "intrinsic", T2: float = 10.0, n: float = 1.5,
                 T2_bounds=(1e-3, 1e4), n_bounds=(0.5, 4.0), grid: dict | None = None):
        default_grid = {"T2": np.geomspace(max(T2_bounds[0], 0.3), min(T2_bounds[1], 300.0), 11)}
        super().__init__(name, {"T2": T2, "n": n}, {"T2": T2_bounds, "n": n_bounds},
                         default_grid if grid is None else grid)

    def coherence(self, t, values, sequence, segments):
        return np.exp(-((np.asarray(t, dtype=float) / values["T2"]) ** values["n"]))


class GaussianNoiseDecay(Component):
    """Decay from Gaussian noise sources built from the fit parameters.

    ``factory(**params) -> list of sources`` (all Gaussian), e.g.
    ``GaussianNoiseDecay(lambda b, tc: [OUNoise(b, tc)], p0={"b": 2 * uT, "tc": 1.0},
    bounds={"b": (0, 1e-1), "tc": (1e-3, 1e3)}, name="ou")``.
    """

    def __init__(self, factory: Callable[..., list], p0: dict, bounds: dict, name: str = "gauss",
                 grid: dict | None = None):
        super().__init__(name, p0, bounds, grid)
        self.factory = factory

    def coherence(self, t, values, sequence, segments):
        key = (tuple(np.round(t, 12)), sequence, tuple(values[p] for p in self.params))

        def compute():
            sources = self.factory(**{p: values[p] for p in self.params})
            return np.array([coherence_decay(s, sources) for s in segments])

        return self._memo(key, compute)


class TelegraphDecay(Component):
    """One telegraph fluctuator with jump ``jump`` [mT] (b_high - b_low).

    Symmetric (default): parameters ``rate`` (switching rate each way, 1/us) and
    ``jump``. Asymmetric: ``rate_up``, ``rate_down`` and ``jump``. The mean field
    is set to zero (it is refocused by echo / DD sequences). Exact two-state
    theory (:func:`nvsim.telegraph.telegraph_coherences`) on the compiled
    sequence, including finite pulses.
    """

    def __init__(self, name: str = "rtn", asymmetric: bool = False, rate: float = 0.3, jump: float = 5 * uT,
                 rate_bounds=(1e-4, 100.0), jump_bounds=(1e-3 * uT, 10.0), grid: dict | None = None):
        rates = ("rate_up", "rate_down") if asymmetric else ("rate",)
        p0 = {r: rate for r in rates} | {"jump": jump}
        bounds = {r: rate_bounds for r in rates} | {"jump": jump_bounds}
        rate_grid = np.geomspace(max(rate_bounds[0], 0.01), min(rate_bounds[1], 10.0), 13)
        jump_grid = np.geomspace(max(jump_bounds[0], 0.1 * uT), min(jump_bounds[1], 200 * uT), 13)
        default_grid = {r: rate_grid for r in rates} | {"jump": jump_grid}
        super().__init__(name, p0, bounds, default_grid if grid is None else grid)
        self.asymmetric = asymmetric

    def source(self, values: dict) -> TelegraphNoise:
        jump = values["jump"]
        if not self.asymmetric:
            return TelegraphNoise.symmetric(jump / 2, values["rate"], label=self.name)
        up, down = values["rate_up"], values["rate_down"]
        p_high = up / (up + down)
        return TelegraphNoise(-p_high * jump, (1 - p_high) * jump, up, down, label=self.name)

    def coherence(self, t, values, sequence, segments):
        key = (tuple(np.round(t, 12)), sequence, tuple(values[p] for p in self.params))
        return self._memo(key, lambda: np.real(
            telegraph_coherences(segments, self.source(values), sequence.transition)))


# --------------------------------------------------------------------------- model


class DecayModel:
    """S(t) = offset + amplitude x prod(component coherences) for one sequence."""

    def __init__(self, components: Sequence[Component], sequence: DecaySequence | None = None):
        self.components = list(components)
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError(f"Component names must be unique, got {names}.")
        self.sequence = sequence or DecaySequence()
        self._segments: dict = {}

    @property
    def names(self) -> list[str]:
        return list(_LINEAR) + [f"{c.name}.{p}" for c in self.components for p in c.params]

    def defaults(self):
        """(p0, bounds, grid) dicts keyed by local parameter names; p0 of offset /
        amplitude is None (solved from the data)."""
        p0 = {"offset": None, "amplitude": None}
        bounds = {"offset": (-np.inf, np.inf), "amplitude": (-np.inf, np.inf)}
        grid = {}
        for c in self.components:
            for p in c.params:
                p0[f"{c.name}.{p}"] = c.p0[p]
                bounds[f"{c.name}.{p}"] = c.bounds[p]
                if p in c.grid:
                    grid[f"{c.name}.{p}"] = np.asarray(c.grid[p], dtype=float)
        return p0, bounds, grid

    def _segments_for(self, t):
        key = tuple(np.round(t, 12))
        if key not in self._segments:
            self._segments[key] = self.sequence.segments(t)
        return self._segments[key]

    def coherence(self, t, values: dict) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        segments = None
        result = np.ones_like(t)
        for c in self.components:
            local = {p: values[f"{c.name}.{p}"] for p in c.params}
            if not isinstance(c, StretchedExp) and segments is None:
                segments = self._segments_for(t)
            result = result * c.coherence(t, local, self.sequence, segments)
        return result

    def signal(self, t, values: dict) -> np.ndarray:
        return values["offset"] + values["amplitude"] * self.coherence(t, values)

    def __repr__(self):
        return f"DecayModel({self.sequence.label}: {' x '.join(c.name for c in self.components) or '1'})"


# --------------------------------------------------------------------------- fitting


@dataclass
class Curve:
    """One measured curve: total free evolution ``t`` [us], values ``y``, errors ``sigma``."""

    t: np.ndarray
    y: np.ndarray
    sigma: np.ndarray | float
    model: DecayModel
    name: str = ""

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.sigma = np.broadcast_to(np.asarray(self.sigma, dtype=float), self.y.shape).copy()
        if self.t.shape != self.y.shape or np.any(self.sigma <= 0):
            raise ValueError("t, y must have the same shape and sigma must be positive.")


@dataclass
class FitResult:
    """Outcome of :func:`fit_decay` / :func:`fit_joint`."""

    params: dict
    errors: dict
    covariance: np.ndarray
    free: list
    chi2: float
    n_points: int
    success: bool
    message: str
    curves: list
    shared: tuple
    fixed: dict
    _spec: dict = field(default_factory=dict, repr=False)

    @property
    def n_free(self) -> int:
        return len(self.free)

    @property
    def dof(self) -> int:
        return self.n_points - self.n_free

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else np.nan

    @property
    def aic(self) -> float:
        return self.chi2 + 2 * self.n_free

    @property
    def bic(self) -> float:
        return self.chi2 + self.n_free * np.log(self.n_points)

    def curve_values(self, index: int) -> dict:
        """Local parameter values of curve ``index``."""
        curve = self.curves[index]
        mapping = _global_names(self.curves, self.shared)[index]
        return {local: self.params[g] for local, g in mapping.items() if local in curve.model.names}

    def signal(self, index: int = 0, t=None) -> np.ndarray:
        curve = self.curves[index]
        return curve.model.signal(curve.t if t is None else t, self.curve_values(index))

    def coherence(self, index: int = 0, t=None) -> np.ndarray:
        curve = self.curves[index]
        return curve.model.coherence(curve.t if t is None else t, self.curve_values(index))

    def table(self) -> str:
        lines = [f"chi2 = {self.chi2:.2f}, dof = {self.dof}, reduced chi2 = {self.reduced_chi2:.3f}, "
                 f"AIC = {self.aic:.2f}, BIC = {self.bic:.2f}"]
        for name, value in self.params.items():
            err = self.errors.get(name)
            tag = "fixed" if name in self.fixed else (f"+- {err:.4g}" if err is not None else "")
            lines.append(f"  {name:28s} {value:.6g} {tag}")
        return "\n".join(lines)


def _global_names(curves, shared) -> list[dict]:
    """Per curve: local name -> global name."""
    single = len(curves) == 1
    out = []
    for k, curve in enumerate(curves):
        label = curve.name or f"curve{k}"
        out.append({q: (q if single or q in shared else f"{label}:{q}") for q in curve.model.names})
    return out


def _lookup(table: dict | None, global_name: str, local: str, default=None):
    table = table or {}
    if global_name in table:
        return table[global_name]
    if local in table:
        return table[local]
    return default


def fit_decay(t, y, sigma, model: DecayModel, p0: dict | None = None, bounds: dict | None = None,
              fixed: dict | None = None, grid=True, n_starts: int = 3) -> FitResult:
    """Fit one curve; see :func:`fit_joint` for the arguments."""
    return fit_joint([Curve(t, y, sigma, model)], (), p0, bounds, fixed, grid, n_starts)


def fit_joint(curves: Sequence[Curve], shared: Sequence[str] = (), p0: dict | None = None,
              bounds: dict | None = None, fixed: dict | None = None, grid=True,
              n_starts: int = 3) -> FitResult:
    """Fit several curves at once; parameters named in ``shared`` (local names,
    e.g. ``"intrinsic.T2"``) are common to all curves that have them, all others
    are per curve and named ``"<curve name>:<parameter>"``.

    ``p0``, ``bounds`` and ``fixed`` are dicts keyed by global or local names
    (local applies to every curve). ``grid``: True = default grids of the
    components, False = none, or a dict name -> values to override.
    """
    curves = list(curves)
    if len(curves) > 1 and len({c.name for c in curves}) != len(curves):
        raise ValueError("Curves in a joint fit need distinct names.")
    shared = tuple(shared)
    fixed = dict(fixed or {})
    maps = _global_names(curves, shared)

    # Collect global parameters with their local name, start value, bounds, grid.
    info: dict[str, dict] = {}
    for curve, mapping in zip(curves, maps):
        d_p0, d_bounds, d_grid = curve.model.defaults()
        for local, g in mapping.items():
            if g in info:
                continue
            info[g] = {"local": local,
                       "p0": _lookup(p0, g, local, d_p0[local]),
                       "bounds": tuple(_lookup(bounds, g, local, d_bounds[local])),
                       "grid": d_grid.get(local)}
    fixed_values = {g: _lookup(fixed, g, v["local"]) for g, v in info.items()}
    fixed_values = {g: float(v) for g, v in fixed_values.items() if v is not None}
    if isinstance(grid, dict):
        for g, v in info.items():
            override = _lookup(grid, g, v["local"])
            if override is not None:
                v["grid"] = np.asarray(override, dtype=float)
    free = [g for g in info if g not in fixed_values]
    linear = [g for g in free if info[g]["local"] in _LINEAR]
    nonlinear = [g for g in free if g not in linear]
    n_points = sum(len(c.y) for c in curves)

    def full(values: dict) -> dict:
        return {**fixed_values, **values}

    def residuals_from(values: dict) -> np.ndarray:
        out = []
        for curve, mapping in zip(curves, maps):
            local = {q: values[g] for q, g in mapping.items()}
            out.append((curve.y - curve.model.signal(curve.t, local)) / curve.sigma)
        return np.concatenate(out)

    def solve_linear(nonlin: dict):
        """Best linear parameters (weighted least squares) for given nonlinear ones."""
        columns = {g: [] for g in linear}
        rhs = []
        for curve, mapping in zip(curves, maps):
            local = {q: g for q, g in mapping.items()}
            values = full(nonlin)
            coh = curve.model.coherence(curve.t, {q: values.get(g, 0.0) for q, g in local.items()})
            w = 1.0 / curve.sigma
            known = np.zeros_like(curve.y)
            if local["offset"] in fixed_values:
                known += fixed_values[local["offset"]]
            if local["amplitude"] in fixed_values:
                known += fixed_values[local["amplitude"]] * coh
            rhs.append((curve.y - known) * w)
            for g in linear:
                if g == local["offset"]:
                    columns[g].append(w)
                elif g == local["amplitude"]:
                    columns[g].append(coh * w)
                else:
                    columns[g].append(np.zeros_like(curve.y))
        b = np.concatenate(rhs)
        if not linear:
            return {}, float(np.sum(b**2))
        a = np.column_stack([np.concatenate(columns[g]) for g in linear])
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        sol = [float(np.clip(x, *info[g]["bounds"])) for x, g in zip(sol, linear)]
        return dict(zip(linear, sol)), float(np.sum((b - a @ np.array(sol)) ** 2))

    def clip_inside(g, value):
        lo, hi = info[g]["bounds"]
        if np.isfinite(lo) and np.isfinite(hi):
            pad = 1e-9 * (hi - lo)
            return float(np.clip(value, lo + pad, hi - pad))
        return float(np.clip(value, lo, hi))

    base = {g: clip_inside(g, info[g]["p0"]) for g in nonlinear}

    # Starting points: the defaults plus the best points of the grid.
    starts = []
    lin0, chi0 = solve_linear(base)
    starts.append((chi0, {**base, **lin0}))
    grid_names = [g for g in nonlinear if grid is not False and info[g]["grid"] is not None]
    if grid_names:
        axes = [info[g]["grid"] for g in grid_names]
        combos = list(itertools.product(*axes))
        if len(combos) > MAX_GRID_POINTS:
            pick = np.random.default_rng(0).choice(len(combos), MAX_GRID_POINTS, replace=False)
            combos = [combos[i] for i in pick]
        for combo in combos:
            trial = dict(base)
            trial.update({g: clip_inside(g, v) for g, v in zip(grid_names, combo)})
            lin, chi = solve_linear(trial)
            starts.append((chi, {**trial, **lin}))
    starts.sort(key=lambda s: s[0])
    for g in linear:  # user-given linear start values override the data-driven ones
        if info[g]["p0"] is not None:
            for _, s in starts:
                s[g] = clip_inside(g, info[g]["p0"])

    lower = np.array([info[g]["bounds"][0] for g in free])
    upper = np.array([info[g]["bounds"][1] for g in free])

    def fun(x):
        return residuals_from(full(dict(zip(free, x))))

    best = None
    for _, start in starts[: max(1, n_starts)]:
        x0 = np.array([clip_inside(g, start[g]) for g in free])
        if not free:
            break
        sol = least_squares(fun, x0, bounds=(lower, upper), x_scale="jac", method="trf",
                            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
        if best is None or sol.cost < best.cost:
            best = sol

    if best is None:  # nothing free
        values = full({})
        chi2 = float(np.sum(residuals_from(values) ** 2))
        return FitResult(values, {}, np.zeros((0, 0)), [], chi2, n_points, True, "no free parameters",
                         curves, shared, fixed_values, {"p0": p0, "bounds": bounds, "grid": grid})

    values = full(dict(zip(free, best.x)))
    jac = best.jac
    covariance = np.linalg.pinv(jac.T @ jac)
    errors = {g: float(np.sqrt(max(covariance[i, i], 0.0))) for i, g in enumerate(free)}
    params = {g: float(values[g]) for g in info}
    return FitResult(params, errors, covariance, free, float(2 * best.cost), n_points, bool(best.success),
                     best.message, curves, shared, fixed_values, {"p0": p0, "bounds": bounds, "grid": grid})


# --------------------------------------------------------------------------- profiles


def profile_chi2(result: FitResult, name: str, values) -> np.ndarray:
    """Minimum chi^2 with parameter ``name`` (global name) fixed at each value,
    all other free parameters refitted. Every point gets its own grid search
    over the remaining grid parameters (the profile can be multimodal), plus a
    start from the best fit."""
    spec = result._spec
    out = []
    start = {g: v for g, v in result.params.items() if g != name}
    for v in np.asarray(values, dtype=float):
        fixed = {**result.fixed, name: float(v)}
        from_best = fit_joint(result.curves, result.shared, p0={**(spec.get("p0") or {}), **start},
                              bounds=spec.get("bounds"), fixed=fixed, grid=False, n_starts=1)
        from_grid = fit_joint(result.curves, result.shared, p0=spec.get("p0"), bounds=spec.get("bounds"),
                              fixed=fixed, grid=spec.get("grid", True), n_starts=2)
        out.append(min(from_best.chi2, from_grid.chi2))
    return np.array(out)


def profile_interval(result: FitResult, name: str, values, delta: float = 4.0):
    """(lower, upper) envelope of the parameter values whose profile chi^2 is
    within ``delta`` of the minimum (delta = 4: ~2 sigma for one parameter). The
    best-fit value is always added to the scan. If the region reaches an end of
    the scan, that side is returned as -inf / +inf: the data give only a bound
    there (e.g. the telegraph jump in strong coupling). Multimodal profiles give
    the outermost accepted values."""
    values = np.union1d(np.asarray(values, dtype=float), [result.params[name]])
    chi2 = profile_chi2(result, name, values)
    inside = chi2 <= min(chi2.min(), result.chi2) + delta
    accepted = values[inside]
    lo = -np.inf if inside[0] else float(accepted.min())
    hi = np.inf if inside[-1] else float(accepted.max())
    return lo, hi
