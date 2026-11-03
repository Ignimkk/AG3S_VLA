"""T93 — steer the VLA's own flow-matching sampler with the collision cost of the TrajOpt scene (default off).

Why. TrajOpt repairs the pi0.5 chunk only inside the chunk; when it cannot (a hurdle bar on the path) it HOLDs, and the next
chunk of the VLA plans through the bar again. This module asks the VLA again, with the **same observation and the same noise**,
while every flow step is pushed down the gradient of the collision cost of the x0 prediction. The result is a chunk from the
VLA's own distribution that avoids what the TrajOpt scene says is in the way. It then becomes the TrajOpt reference; TrajOpt,
the HOLD verdict and the gripper columns are unchanged.

Conventions (openpi, `Pi0.sample_actions`): t = 1 is noise, t = 0 is data, `x_t = t * noise + (1 - t) * a`, one Euler step is
`x_next = x_t + dt * v_t` with `dt = -1 / N`, so the x0 prediction is `x_hat0 = x_t - t * v_t`.

| piece | what it is |
|---|---|
| `FmGuideConfig` | scale (lambda), steps (the last N flow steps are guided), per-step joint-change cap, flow steps |
| `OutputAffine` | the policy's output transform (unnormalize, delta -> absolute, slice to 16) measured as `offset + scale * x`; the check that it is per-dimension affine is part of building it, and `scale` is the diagonal Jacobian |
| `CollisionOracle` | violation and cost gradient of a physical chunk, computed by the SAME `CollisionLinearizer` / scene / rows the TrajOpt solve of this request uses |
| `guidance_update` | one flow step's correction (numpy): cost gradient -> x space by the diagonal scale -> `-lambda * g`, clipped in joint radians |
| `GuidedSampler` | the Python-loop sampler: one jitted prefix pass, then one jitted velocity call per flow step, the correction applied on the host in between |
| `FlowGuide` / `BoundResample` | what `SafePolicy` holds (per server) and what it hands the refiner (per request: observation + noise bound) |
| `guide_reference` | the decision, called by `TrajOptChunkRefiner.refine`: violation > 0 -> resample -> keep it if its violation is lower |

Cost: `C(x) = 1/2 sum_rows max(0, -clearance_row)^2` over the rows `CollisionLinearizer.linearize` selects (contact allowances,
support band, held body and look-ahead rules are in `clearance_row` because they are in the linearizer), gradient
`-sum_rows max(0, -clearance_row) * d clearance_row / d q` — the Jacobian the QP uses.

Not in this module: the RTC-style inpainting of the already executing prefix (T93 task §2); TrajOpt's continuity term does it.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["DEFAULT_SCALE", "DEFAULT_MAX_STEP_RAD", "DEFAULT_STEPS", "FmGuideConfig", "FmGuideError", "OutputAffine",
           "CollisionOracle", "guidance_update", "GuidedSampler", "FlowGuide", "BoundResample", "GuideOutcome",
           "guide_reference", "peek_noise"]

#: Defaults fixed from the T93 smoke (T93.impl.md, "smoke": scale 300, last 5 of 10 flow steps, 0.05 rad per step); not changed before the closed loop.
DEFAULT_SCALE = 300.0
DEFAULT_MAX_STEP_RAD = 0.05
DEFAULT_STEPS = 5
#: pi0.5 flow steps when the policy gives none (`Pi0.sample_actions(num_steps=10)`).
DEFAULT_NUM_STEPS = 10


class FmGuideError(RuntimeError):
    """Guidance cannot be built for this policy (not an affine output transform, no JAX policy ...). Never silent."""


@dataclasses.dataclass(frozen=True)
class FmGuideConfig:
    """`serve_safe --fm-guide*`. `steps` counts from the end: the last `steps` of the `num_steps` Euler steps are guided."""

    scale: float = DEFAULT_SCALE            #: lambda of `x_next -= lambda * g_x`
    steps: int = DEFAULT_STEPS              #: guided flow steps (the last ones); >= num_steps = all
    max_step_rad: float = DEFAULT_MAX_STEP_RAD   #: cap on one step's change of any joint column (physical radians)
    num_steps: Optional[int] = None         #: Euler steps; None = the policy's own (`sample_kwargs["num_steps"]`, else 10)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.scale) and self.scale >= 0.0):
            raise ValueError(f"fm-guide scale must be a finite number >= 0, got {self.scale}")
        if int(self.steps) < 1:
            raise ValueError(f"fm-guide steps must be >= 1, got {self.steps}")
        if not (np.isfinite(self.max_step_rad) and self.max_step_rad > 0.0):
            raise ValueError(f"fm-guide max step must be a finite number > 0 rad, got {self.max_step_rad}")

    def record(self) -> dict[str, Any]:
        return {"scale": float(self.scale), "steps": int(self.steps), "max_step_rad": float(self.max_step_rad),
                "num_steps": self.num_steps}


# ------------------------------------------------------------------------------------------------ output transform
@dataclasses.dataclass(frozen=True)
class OutputAffine:
    """`physical[t, j] = offset[t, j] + scale[j] * x[t, j]` for the first `n` model columns (the rest of the 32 is padding).

    Built by `measure`: the policy's real output transform is evaluated at x = 0, x = 1 and a random x and the affine form must
    reproduce it, so a transform that is not per-dimension affine (a coupling between columns, a nonlinearity) stops the server
    instead of giving a wrong gradient.
    """

    offset: np.ndarray   #: (T, n)
    scale: np.ndarray    #: (n,)

    @property
    def n(self) -> int:
        return int(self.scale.shape[0])

    def to_physical(self, x: np.ndarray) -> np.ndarray:
        """`(T, >= n)` model-space actions -> `(T, n)` physical (the columns the output transform keeps)."""
        return self.offset + self.scale[None, :] * np.asarray(x, np.float64)[:, :self.n]

    @classmethod
    def measure(cls, output_transform: Callable[[dict], dict], state: np.ndarray, horizon: int, action_dim: int, *,
                tol: float = 1e-4, seed: int = 0) -> "OutputAffine":
        state = np.asarray(state, np.float64).reshape(-1)

        def apply(x: np.ndarray) -> np.ndarray:
            out = output_transform({"state": state.copy(), "actions": np.array(x, np.float64, copy=True)})
            return np.asarray(out["actions"], np.float64)

        f0 = apply(np.zeros((horizon, action_dim)))
        f1 = apply(np.ones((horizon, action_dim)))
        n = f0.shape[1]
        step = f1 - f0
        scale = step[0].copy()
        if not np.allclose(step, scale[None, :], rtol=0.0, atol=tol * max(1.0, float(np.abs(scale).max(initial=0.0)))):
            raise FmGuideError("the output transform does not scale a column the same at every chunk step "
                               f"(max spread {float(np.abs(step - scale[None, :]).max()):.3g})")
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((horizon, action_dim))
        fx = apply(x)
        want = f0 + scale[None, :] * x[:, :n]
        err = float(np.abs(fx - want).max())
        if not err <= tol * max(1.0, float(np.abs(fx).max())):
            raise FmGuideError(f"the output transform is not per-dimension affine (random-x residual {err:.3g}); "
                               "a diagonal-scale gradient would be wrong")
        return cls(offset=f0, scale=scale)


# ------------------------------------------------------------------------------------------------ cost oracle
class CollisionOracle:
    """Violation and cost gradient of a physical chunk by the linearizer / scene / rows of this request's TrajOpt solve.

    Built by `TrajOptChunkRefiner.refine` after `scene_fn` (so the finger parameter path, the held points and the scene are the
    ones the solve is about to use). Use as a context manager: it mirrors the ESDF to the host exactly as `_solve_standard`
    does, and restores the per-call record attributes the linearizer's clearance calls leave behind
    (`linearize._CLEARANCE_SIDE_ATTRS`), so the solve that follows sees what it would have seen without this probe.
    """

    def __init__(self, optimizer, layout, planned: int, q_now: np.ndarray, scene):
        self.optimizer = optimizer
        self.layout = layout
        self.planned = int(planned)
        self.q_now = np.asarray(q_now, np.float64).reshape(-1)
        self.scene = scene
        self.calls = {"violation": 0, "cost_grad": 0}
        self._saved: Optional[dict] = None

    # ---- context: same preparation as TrajectoryOptimizer._solve_standard / _solve --------------------------------
    def __enter__(self) -> "CollisionOracle":
        from AG3S.trajopt.linearize import _CLEARANCE_SIDE_ATTRS, _MISSING

        lin = self.optimizer.linearizer
        self._saved = {a: lin.__dict__.get(a, _MISSING) for a in _CLEARANCE_SIDE_ATTRS}
        self._missing = _MISSING
        if getattr(self.optimizer.config.collision, "esdf_host_mirror", False):
            mirror = getattr(getattr(self.scene, "esdf", None), "mirror_to_host", None)
            if mirror is not None:
                mirror()
        lin.set_attached(getattr(self.scene, "attached_points", None), getattr(self.scene, "attached_parent_link", None))
        return self

    def __exit__(self, *exc) -> None:
        lin = self.optimizer.linearizer
        for a, v in (self._saved or {}).items():
            if v is self._missing:
                lin.__dict__.pop(a, None)
            else:
                lin.__dict__[a] = v
        self._saved = None

    # ---- measurements ------------------------------------------------------------------------------------------------
    def trajectory(self, chunk: np.ndarray) -> np.ndarray:
        return self.layout.chunk_to_trajectory(chunk)[:, :self.planned]

    def violation(self, chunk: np.ndarray) -> float:
        """The solve's `reference_violation` (m, > 0 = penetration) of `chunk`: look-ahead -> the executed window's worst
        clearance, else the worst clearance over every row (`sqp.TrajectoryOptimizer._solve`, same two calls)."""
        self.calls["violation"] += 1
        opt, lin = self.optimizer, self.optimizer.linearizer
        traj = self.trajectory(chunk)
        if opt.lookahead_from is not None:
            return -float(lin.step_clearances(traj, self.q_now, self.scene)[:opt.lookahead_from].min(initial=np.inf))
        return -float(lin.full_violation(traj, self.q_now, self.scene))

    def cost_grad(self, chunk: np.ndarray) -> tuple[float, np.ndarray]:
        """`(C, dC/d chunk)`; the gradient is `(T, n_valid)` physical-chunk shaped: zero outside the joint columns the
        optimizer owns and beyond the planned steps. `C = 1/2 sum max(0, -value)^2` over the rows `linearize` picks."""
        self.calls["cost_grad"] += 1
        opt = self.optimizer
        chunk = np.asarray(chunk, np.float64)
        traj = self.trajectory(chunk)
        rows = opt.linearizer.linearize(traj, self.q_now, self.scene, opt.config)
        pen = np.where(rows.used & np.isfinite(rows.value) & (rows.value < 0.0), -rows.value, 0.0)   # (H, rows)
        cost = 0.5 * float(np.sum(pen * pen))
        g_traj = -np.einsum("hr,hrj->jh", pen, rows.gradient)                                    # (nq_opt, H)
        grad = np.zeros((chunk.shape[0], chunk.shape[1]))
        grad[:g_traj.shape[1], self.layout.action_columns] = g_traj.T
        return cost, grad


# ------------------------------------------------------------------------------------------------ one flow step
def guidance_update(x_hat0: np.ndarray, oracle, affine: OutputAffine, *, scale: float, max_step_rad: float,
                    ) -> tuple[np.ndarray, float, float]:
    """The correction of one flow step: `(dx[T, D], cost, max |physical change| in rad)`.

    `g_phys = dC/d phys` from the oracle at the x0 prediction; `g_x = scale_diag * g_phys` (chain rule of the diagonal
    affine output transform; padding columns and everything the oracle gives zero stay zero); `dx = -lambda * g_x`; the
    physical change `scale_diag * dx` of every column is clipped to +-`max_step_rad` (then mapped back).
    """
    x_hat0 = np.asarray(x_hat0, np.float64)
    phys = affine.to_physical(x_hat0)
    cost, g_phys = oracle.cost_grad(phys)
    g_phys = np.asarray(g_phys, np.float64)[:, :affine.n]
    d = affine.scale[None, :]
    dphys = -float(scale) * (d * d) * g_phys                # physical change of -lambda * g_x
    dphys = np.clip(dphys, -float(max_step_rad), float(max_step_rad))
    safe = np.where(np.abs(d) > 1e-12, d, 1.0)
    dx_n = np.where(np.abs(d) > 1e-12, dphys / safe, 0.0)
    dx = np.zeros_like(x_hat0)
    dx[:, :affine.n] = dx_n
    return dx, float(cost), float(np.abs(dphys).max(initial=0.0))


# ------------------------------------------------------------------------------------------------ sampler (JAX; imported lazily)
def peek_noise(owner) -> np.ndarray:
    """The noise `Policy.infer` is about to draw from its own key: `(1, action_horizon, action_dim)`.

    `Policy.infer` does `self._rng, sample = jax.random.split(self._rng)` and the model draws
    `jax.random.normal(sample, (b, ah, ad))`, a pure function of the key (jitted or not). Reading it BEFORE `infer` therefore
    gives the noise of that request's sample without touching the `infer` call — the first sample runs the same compiled graph as
    on a server without the flag — and the same array is the "same noise" of the guided resample. (Handing it to `infer` as
    `noise=` would give the same values but a different compiled graph, so it is not done.)"""
    import jax

    model = owner._model   # noqa: SLF001
    _, sample_rng = jax.random.split(owner._rng)   # noqa: SLF001
    return np.asarray(jax.random.normal(sample_rng, (1, model.action_horizon, model.action_dim)))


class GuidedSampler:
    """`Pi0.sample_actions` driven from Python: one jitted prefix pass, then one jitted velocity call per Euler step.

    The correction between steps is host-side numpy (the collision oracle), so there is no callback inside a compiled loop:
    only `(x_t, v_t)` of shape (1, H, D) cross to the host and `dx` back, per guided step."""

    def __init__(self, model):
        from openpi.shared import nnx_utils

        self.model = model
        self._prefix = nnx_utils.module_jit(model.sample_prefix)
        self._velocity = nnx_utils.module_jit(model.sample_velocity)

    def sample(self, observation, noise, num_steps: int,
               guide: Optional[Callable[[np.ndarray, np.ndarray, float, int], Optional[np.ndarray]]] = None,
               first_guided: int = 0):
        """Final x0 `(1, H, D)` (jax array). `guide(x_t, v_t, t, i)` -> dx `(H, D)` or None, called for i >= first_guided."""
        import jax.numpy as jnp

        obs, prefix_mask, kv_cache = self._prefix(observation)
        dt = -1.0 / num_steps
        x = jnp.asarray(noise)
        t = jnp.asarray(1.0, jnp.float32)
        for i in range(num_steps):
            v = self._velocity(obs, prefix_mask, kv_cache, x, t)
            x_next = x + dt * v
            if guide is not None and i >= first_guided:
                dx = guide(np.asarray(x), np.asarray(v), float(t), i)
                if dx is not None:
                    x_next = x_next + jnp.asarray(dx, x_next.dtype)[None]
            x, t = x_next, t + dt
        return x


class CallbackSampler:
    """The same guided sampling as ONE compiled loop: `Pi0.sample_actions(guidance_xv_fn=...)` with the host-side correction
    reached through `jax.pure_callback` at every Euler step. Kept to measure against `GuidedSampler` (T93.impl.md); the server
    uses the Python loop. The compiled function is shared by all requests, the per-request guide is passed through `self._guide`.
    """

    def __init__(self, model, num_steps: int):
        import flax.nnx as nnx
        import jax

        self.model = model
        self.num_steps = int(num_steps)
        self._guide: Optional[Callable] = None
        self._first_guided = 0
        graphdef, state = nnx.split(model)

        def hook(x_t, v_t, t, x_next, t_next):
            dx = jax.pure_callback(self._host, jax.ShapeDtypeStruct(x_next.shape, x_next.dtype), x_t, v_t, t)
            return x_next + dx

        def fun(state, observation, noise):
            module = nnx.merge(graphdef, state)
            return module.sample_actions(jax.random.key(0), observation, num_steps=self.num_steps, noise=noise,
                                         guidance_xv_fn=hook)

        self._fn = jax.jit(fun)
        self._state = state

    def _host(self, x_t, v_t, t):
        i = int(round((1.0 - float(t)) * self.num_steps))
        zero = np.zeros(np.shape(x_t), np.asarray(x_t).dtype)
        if self._guide is None or i < self._first_guided:
            return zero
        dx = self._guide(np.asarray(x_t), np.asarray(v_t), float(t), i)
        return zero if dx is None else np.asarray(dx, zero.dtype)[None]

    def sample(self, observation, noise, num_steps: int, guide=None, first_guided: int = 0):
        import jax.numpy as jnp

        assert int(num_steps) == self.num_steps
        self._guide, self._first_guided = guide, int(first_guided)
        try:
            return self._fn(self._state, observation, jnp.asarray(noise))
        finally:
            self._guide = None


@dataclasses.dataclass
class GuideOutcome:
    chunk: np.ndarray               #: (T, n_valid) float64: the original chunk with the joint columns replaced
    info: dict[str, Any]


class BoundResample:
    """One request's resample: the observation and the noise are bound, the oracle is given at call time."""

    def __init__(self, guide: "FlowGuide", policy_obs: dict, noise: np.ndarray, *, skip: Optional[str] = None):
        self.guide = guide
        self.policy_obs = policy_obs
        self.noise = noise
        #: not None -> guidance is not attempted for this request (the record says why)
        self.skip = skip

    def __call__(self, oracle: CollisionOracle, original: np.ndarray, **kw) -> GuideOutcome:
        return self.guide.resample(self.policy_obs, self.noise, oracle, original, **kw)


class FlowGuide:
    """Per server. Holds the jitted sampler and the knobs; `bind` makes the per-request closure."""

    def __init__(self, policy, config: FmGuideConfig, *, sweep: tuple = ()):
        from AG3S.trajopt.policy_seed import PolicySeedError, rng_owner

        try:
            self.owner = rng_owner(policy)   # the openpi Policy (JAX); PyTorch / missing -> PolicySeedError
        except PolicySeedError as exc:
            raise FmGuideError(f"--fm-guide needs the JAX openpi policy: {exc}") from exc
        self.config = config
        self.model = self.owner._model   # noqa: SLF001
        for need in ("sample_prefix", "sample_velocity"):
            if not hasattr(self.model, need):
                raise FmGuideError(f"the model has no `{need}` (T93 pi0.py change missing)")
        kw = getattr(self.owner, "_sample_kwargs", {}) or {}
        self.num_steps = int(config.num_steps or kw.get("num_steps", DEFAULT_NUM_STEPS))
        self.first_guided = max(0, self.num_steps - int(config.steps))
        #: smoke only (`serve_safe --fm-guide-sweep`): extra `FmGuideConfig`s each violating request is also resampled with, for
        #: the record only (`to.metrics.fm_guide.sweep`); they never decide the chunk. Empty (default) = nothing extra runs.
        self.sweep = tuple(sweep)
        self._sampler: Optional[GuidedSampler] = None
        self._callback: Optional[CallbackSampler] = None
        self.last_info: dict[str, Any] = {}

    @property
    def sampler(self) -> GuidedSampler:
        if self._sampler is None:
            self._sampler = GuidedSampler(self.model)
        return self._sampler

    @property
    def callback_sampler(self) -> CallbackSampler:
        if self._callback is None:
            self._callback = CallbackSampler(self.model, self.num_steps)
        return self._callback

    # ---- per request -------------------------------------------------------------------------------------------------
    def peek_noise(self) -> np.ndarray:
        return peek_noise(self.owner)

    def bind(self, policy_obs: dict, noise: np.ndarray, *, skip: Optional[str] = None) -> BoundResample:
        return BoundResample(self, policy_obs, noise, skip=skip)

    def observation(self, obs: dict):
        """`(Observation, normalized padded state)` — the same input transform `Policy.infer` applies."""
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.owner._input_transform(inputs)   # noqa: SLF001
        state = np.asarray(inputs["state"])
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return _model.Observation.from_dict(batched), state

    def resample(self, policy_obs: dict, noise: np.ndarray, oracle: CollisionOracle, original: np.ndarray, *,
                 cfg: Optional[FmGuideConfig] = None, impl: str = "loop") -> GuideOutcome:
        cfg = cfg or self.config
        first_guided = max(0, self.num_steps - int(cfg.steps))
        original = np.asarray(original, np.float64)
        observation, state = self.observation(policy_obs)
        H, D = int(self.model.action_horizon), int(self.model.action_dim)
        affine = OutputAffine.measure(self.owner._output_transform, state, H, D)   # noqa: SLF001
        costs: list[float] = []
        step_change: list[float] = []

        def guide(x_t: np.ndarray, v_t: np.ndarray, t: float, i: int) -> np.ndarray:
            x_hat0 = x_t[0].astype(np.float64) - t * v_t[0].astype(np.float64)
            dx, cost, change = guidance_update(x_hat0, oracle, affine, scale=cfg.scale, max_step_rad=cfg.max_step_rad)
            costs.append(cost)
            step_change.append(change)
            return dx

        sampler = self.sampler if impl == "loop" else self.callback_sampler
        x0 = sampler.sample(observation, noise, self.num_steps, guide, first_guided)
        out = self.owner._output_transform({"state": np.array(state, copy=True),   # noqa: SLF001
                                            "actions": np.array(np.asarray(x0)[0], copy=True)})
        guided_all = np.asarray(out["actions"], np.float64)
        chunk = original.copy()
        cols = oracle.layout.action_columns
        n = min(chunk.shape[0], guided_all.shape[0])
        chunk[:n, cols] = guided_all[:n, cols]     # every other column (grippers, pinned joints, padding) stays the original's
        info = {"steps": int(self.num_steps - first_guided), "cost_per_step": [round(c, 8) for c in costs],
                "max_phys_step_rad": [round(c, 5) for c in step_change]}
        self.last_info = info
        return GuideOutcome(chunk=chunk, info=info)

    # ---- warm-up -----------------------------------------------------------------------------------------------------
    def warm_up(self, model_config) -> float:
        """Compile the prefix and velocity functions once at server start (a fake observation of the model's shapes);
        returns seconds. The first guided request then does not pay for compilation."""
        import jax

        t0 = time.perf_counter()
        fake = model_config.fake_obs(1)
        # the real observation carries no FAST-only fields (`Observation.from_dict` leaves them None)
        fake = dataclasses.replace(fake, token_ar_mask=None, token_loss_mask=None)
        noise = np.zeros((1, self.model.action_horizon, self.model.action_dim), np.float32)
        x0 = self.sampler.sample(fake, noise, self.num_steps)
        jax.block_until_ready(x0)
        return time.perf_counter() - t0


# ------------------------------------------------------------------------------------------------ the decision
def _resample_once(resample: BoundResample, oracle: CollisionOracle, layout, chunk: np.ndarray, planned: int,
                   **kw) -> tuple[np.ndarray, float, float, float, dict[str, Any]]:
    """`(merged chunk, its violation, max joint change over the planned steps (rad), ms, info)` for one resample.

    Only the joint columns the optimizer owns may come from the resample: grippers, pinned joints and padding are the original
    chunk's, whatever the resampler returned."""
    t1 = time.perf_counter()
    outcome = resample(oracle, chunk, **kw)
    merged = np.array(chunk, np.float64, copy=True)
    merged[:, layout.action_columns] = np.asarray(outcome.chunk, np.float64)[:, layout.action_columns]
    v1 = oracle.violation(merged)
    ms = (time.perf_counter() - t1) * 1000.0
    cols = layout.action_columns
    change = float(np.abs(merged[:planned][:, cols] - np.asarray(chunk, np.float64)[:planned][:, cols]).max(initial=0.0))
    return merged, float(v1), change, ms, outcome.info


def guide_reference(resample: BoundResample, optimizer, layout, chunk: np.ndarray, q_now, scene, planned: int
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    """`(chunk to use as the TrajOpt reference, record)` — called by `TrajOptChunkRefiner.refine` after `scene_fn`.

    1. violation of `chunk` by the solve's own definition (`CollisionOracle.violation`); <= 0 -> nothing else is done and the
       returned chunk is `chunk` itself (the request is bit-identical to a server without the flag).
    2. > 0 -> resample with the same observation and noise, steering each flow step by the oracle's cost gradient.
    3. the guided chunk is used only if its violation is lower; otherwise `chunk` is kept.
    Any failure inside the guided sampling keeps `chunk` and says so in the record (`error`)."""
    cfg = resample.guide.config
    rec: dict[str, Any] = {"applied": False, "violation_before_m": None, "violation_after_m": None, "used": False,
                           "max_joint_change_rad": None, "guide_ms": None, "steps": 0, "config": cfg.record()}
    if resample.skip is not None:
        rec["reason"] = resample.skip
        return chunk, rec
    if scene is None or getattr(scene, "is_empty", False):
        rec["reason"] = "no scene geometry"
        return chunk, rec
    t0 = time.perf_counter()
    with CollisionOracle(optimizer, layout, planned, q_now, scene) as oracle:
        v0 = oracle.violation(chunk)
        rec["violation_before_m"] = float(max(0.0, v0))
        rec["check_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        if not v0 > 0.0:
            rec["reason"] = "no violation"
            return chunk, rec
        try:
            merged, v1, change, guide_ms, info = _resample_once(resample, oracle, layout, chunk, planned)
        except Exception as exc:  # noqa: BLE001 — a guidance failure must not stop the policy: the original chunk stays
            import traceback

            log.warning("fm-guide resample failed (the original chunk is used):\n%s", traceback.format_exc())
            rec.update(error=f"{type(exc).__name__}: {exc}", guide_ms=round((time.perf_counter() - t0) * 1000.0, 3))
            return chunk, rec
        calls = dict(oracle.calls)
        if resample.guide.sweep:
            rec["sweep"] = _sweep_record(resample, oracle, layout, chunk, planned, v0, merged, guide_ms)
    used = bool(v1 < v0)
    rec.update(applied=True, violation_after_m=float(max(0.0, v1)), used=used, max_joint_change_rad=change,
               guide_ms=round(guide_ms, 3), steps=int(info["steps"]), oracle_calls=calls,
               cost_first=(info["cost_per_step"] or [None])[0], cost_last=(info["cost_per_step"] or [None])[-1])
    return (merged if used else chunk), rec


def _sweep_record(resample, oracle, layout, chunk, planned, v0, main_chunk, main_ms) -> dict[str, Any]:
    """Smoke only (`FlowGuide.sweep` non-empty): the same request resampled with other settings and with the `pure_callback`
    sampler, for the record. Never changes the chunk that is used."""
    out: dict[str, Any] = {"settings": []}
    for scfg in resample.guide.sweep:
        try:
            _, v, change, ms, info = _resample_once(resample, oracle, layout, chunk, planned, cfg=scfg)
            out["settings"].append({**scfg.record(), "violation_after_m": float(max(0.0, v)), "max_joint_change_rad": change,
                                    "ms": round(ms, 2), "cost_last": (info["cost_per_step"] or [None])[-1]})
        except Exception as exc:  # noqa: BLE001
            out["settings"].append({**scfg.record(), "error": f"{type(exc).__name__}: {exc}"})
    try:   # the compiled-loop variant with the main setting: the first call compiles, the second is timed
        if not getattr(resample.guide, "_callback_warm", False):
            _resample_once(resample, oracle, layout, chunk, planned, impl="callback")
            resample.guide._callback_warm = True
        cb, v, change, ms, _ = _resample_once(resample, oracle, layout, chunk, planned, impl="callback")
        out["callback"] = {"ms": round(ms, 2), "loop_ms": round(main_ms, 2), "violation_after_m": float(max(0.0, v)),
                           "max_abs_diff_vs_loop_rad": float(np.abs(cb - main_chunk).max(initial=0.0))}
    except Exception as exc:  # noqa: BLE001
        out["callback"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
