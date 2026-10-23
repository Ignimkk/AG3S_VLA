"""Differentiable LIBERO EE-delta action -> EE path map (pure JAX), shared by the sampling-guidance methods.

pi05_libero emits robosuite OSC_POSE delta rows ``a = [dp(3), drot(3), g]`` in ``[-1, 1]``; the commanded position
step is ``0.05 * a[:3]`` m in the **world** frame (``AG3S/adapters/panda/action.py:13``). OSC does not reach a row's
goal in one 50 ms step; the realised step follows the first-order lag fitted on our dev rollouts
(``robots/panda/description/osc_response_fit.json``, R^2 0.992)::

    y_k = alpha * y_{k-1} + beta * 0.05 * a_k[:3]          (world frame)
    p_k = p_0 + R_bw @ sum_{i<=k} y_i                      (base frame, p_k = EE position after row k)

This is linear in ``a``, so the whole path is ``p = p_0 + R_bw @ (C @ (0.05 a) + c_prev * y_{-1})`` with a constant
lower-triangular ``C``. The guidance methods differentiate a cost through this map (the "decoder" of NUDGE, the
"actions -> trajectory" step of OmniGuide). Two other models are available:

* ``"integrator"`` -- ``p_k = p_0 + dt * sum_{i<=k} a_i`` on whatever ``a`` is given (constrained flow matching
  ``_predict_trajectory`` uses this with ``dt = 0.01`` directly on the **normalized** action, see that README);
* ``"commanded"``  -- ``p_k = p_0 + 0.05 * sum a`` (the OSC goal, no lag; an upper bound of the motion).

Orientation is held at the current EE orientation along the path (the rotation rows are not decoded).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

OSC_POS_SCALE = 0.05  # robosuite 1.4.1 osc_pose.json output_max / input_max (AG3S/adapters/panda/action.py:45)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
OSC_RESPONSE_FIT = PROJECT_ROOT / "robots/panda/description/osc_response_fit.json"
#: fallback when the fit file is absent (values of the 2026-10-13 fit)
DEFAULT_RESPONSE = (0.7168853306811694, 0.0706193331348551)
PATH_MODELS = ("osc_response", "integrator", "commanded")


def load_response(path: str | pathlib.Path = OSC_RESPONSE_FIT) -> tuple[float, float, str]:
    p = pathlib.Path(path)
    if p.exists():
        fit = json.loads(p.read_text())
        return float(fit["pos"]["alpha"]), float(fit["pos"]["beta"]), str(p)
    return DEFAULT_RESPONSE[0], DEFAULT_RESPONSE[1], "default(2026-10-13 fit)"


def path_matrices(H: int, model: str = "osc_response", *, alpha: float | None = None, beta: float | None = None,
                  dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """``(C (H, H), c_prev (H,))`` with ``disp_world = C @ a_xyz + c_prev[:, None] * y_prev``."""
    if model not in PATH_MODELS:
        raise ValueError(f"model must be one of {PATH_MODELS}, got {model!r}")
    k = np.arange(H)
    if model == "integrator":
        return np.tril(np.ones((H, H))) * float(dt), np.zeros(H)
    if model == "commanded":
        return np.tril(np.ones((H, H))) * OSC_POS_SCALE, np.zeros(H)
    if alpha is None or beta is None:
        alpha, beta, _ = load_response()
    diff = k[:, None] - k[None, :]  # k - j
    C = np.where(diff >= 0, beta * (1.0 - alpha ** (np.maximum(diff, 0) + 1)) / (1.0 - alpha), 0.0) * OSC_POS_SCALE
    c_prev = np.cumsum(alpha ** (k + 1))
    return C, c_prev


@dataclass
class EEPathModel:
    """EE path (base frame) of an action chunk; ``__call__`` is pure JAX and differentiable in ``a_xyz``.

    ``a_xyz`` is the **raw** (unnormalized) position part ``(H, 3)`` for ``osc_response`` / ``commanded``; for
    ``integrator`` it is whatever the caller decides (CFM feeds the normalized action, as the original code does).
    """

    H: int
    p0: np.ndarray  # (3,) EE position, base frame
    R_bw: np.ndarray = field(default_factory=lambda: np.eye(3))  # world -> base rotation (LIBERO: identity)
    model: str = "osc_response"
    y_prev: np.ndarray = field(default_factory=lambda: np.zeros(3))  # last realised world step (osc_response)
    dt: float = 0.01  # integrator only
    alpha: float | None = None
    beta: float | None = None

    def __post_init__(self):
        self.p0 = np.asarray(self.p0, np.float64).reshape(3)
        self.R_bw = np.asarray(self.R_bw, np.float64).reshape(3, 3)
        self.y_prev = np.asarray(self.y_prev, np.float64).reshape(3)
        if self.model == "osc_response" and (self.alpha is None or self.beta is None):
            self.alpha, self.beta, self.fit_source = load_response()
        else:
            self.fit_source = None
        self.C, self.c_prev = path_matrices(self.H, self.model, alpha=self.alpha, beta=self.beta, dt=self.dt)

    def __call__(self, a_xyz):
        import jax.numpy as jnp

        C = jnp.asarray(self.C, a_xyz.dtype)
        disp_w = C @ a_xyz + jnp.asarray(self.c_prev, a_xyz.dtype)[:, None] * jnp.asarray(self.y_prev, a_xyz.dtype)[None]
        return jnp.asarray(self.p0, a_xyz.dtype)[None] + disp_w @ jnp.asarray(self.R_bw, a_xyz.dtype).T

    def numpy(self, a_xyz: np.ndarray) -> np.ndarray:
        a = np.asarray(a_xyz, np.float64)
        disp_w = self.C @ a + self.c_prev[:, None] * self.y_prev[None]
        return self.p0[None] + disp_w @ self.R_bw.T

    def jacobian_base(self) -> np.ndarray:
        """``J (3H, 3H)``: d vec(p) / d vec(a_xyz) (row-major ``(k, xyz)``) -- constant because the map is linear."""
        return np.kron(self.C, self.R_bw)

    def describe(self) -> dict:
        d = {"model": self.model, "H": self.H}
        if self.model == "osc_response":
            d.update(alpha=self.alpha, beta=self.beta, osc_pos_scale=OSC_POS_SCALE, fit=self.fit_source)
        elif self.model == "integrator":
            d.update(dt=self.dt)
        else:
            d.update(osc_pos_scale=OSC_POS_SCALE)
        return d


__all__ = ["EEPathModel", "OSC_POS_SCALE", "PATH_MODELS", "load_response", "path_matrices"]
