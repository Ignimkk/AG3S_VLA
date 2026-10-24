"""Sampling guidance inside the closed loop: one compiled pi0.5 sampler, a new ``guidance_fn`` every policy call.

``SamplingGuidance.make_guidance_fn(obs, norm, noise=...)`` returns a fresh closure per observation whose constants
(obstacles, target, EE pose, noise) change every call. Putting that closure straight into ``jax.jit(sample_actions)``
recompiles the whole network each call. Two ways around it, both without touching openpi or the methods:

* ``callback`` (default): the jitted sampler's ``guidance_fn`` is a fixed ``jax.pure_callback`` hook. On the host it runs
  the current call's guidance on the **JAX CPU backend** (:class:`GuidanceSlot`). The network compiles once.
* ``inline``: the closure is traced to a jaxpr once per call; its constants become **traced arguments** and the jaxpr
  text is the cache key (:func:`hoist`), so the sampler recompiles only when the guidance structure changes (point
  counts, literals). Same arithmetic in one XLA program.

In both modes a guidance function is compiled per distinct jaxpr (``hoist`` cache): structure equal -> one compile.

:class:`GuidedPolicy` mirrors ``openpi.policies.policy.Policy.infer`` (input transform, batch, the same
``self._rng`` split, observation, output transform) and draws the sampling noise exactly as ``Pi0.sample_actions``
does when ``noise`` is ``None`` (``jax.random.normal(rng, (b, H, D))``), so the guidance can use it (``x0_mode
noise``). It has no ``_rng`` itself: ``SeededPolicy`` re-keys the wrapped openpi policy (``_policy`` chain).
Backends: :class:`OpenpiBackend` (real model) and :class:`SurrogateBackend` (CPU smoke, D's straight-flow surrogate).
"""
from __future__ import annotations

import dataclasses
import hashlib
import time
from typing import Any, Callable

import numpy as np

GUIDANCE_EXEC = ("callback", "inline")


def cpu_device():
    import jax

    return jax.devices("cpu")[0]


# --- jaxpr hoisting --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Hoisted:
    key: str
    jaxpr: Any  # jax.core.Jaxpr (no consts)
    consts: list  # concrete constants of this call, in jaxpr.constvars order
    n_eqns: int

    @property
    def is_identity(self) -> bool:
        """Zero equations and the output is the input variable (``return x_next``)."""
        j = self.jaxpr
        return self.n_eqns == 0 and len(j.outvars) == 1 and j.outvars[0] is j.invars[0]


def hoist(fn: Callable, x_shape, x_dtype, t_dtype="float32") -> Hoisted:
    """Trace ``fn(x, t)`` -> jaxpr + constants; ``key`` = sha1 of the jaxpr text (structure + literals + shapes)."""
    import jax

    with jax.default_device(cpu_device()):
        closed = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(tuple(x_shape), x_dtype), jax.ShapeDtypeStruct((), t_dtype))
    text = str(closed.jaxpr)
    key = hashlib.sha1(text.encode()).hexdigest()
    return Hoisted(key=key, jaxpr=closed.jaxpr, consts=list(closed.consts), n_eqns=len(closed.jaxpr.eqns))


def eval_hoisted(jaxpr, consts, x, t):
    """Evaluate a hoisted guidance jaxpr (traced with float32 ``x`` / ``t``); output in ``x``'s dtype."""
    import jax
    import jax.numpy as jnp

    y = jax.core.eval_jaxpr(jaxpr, consts, jnp.asarray(x, jnp.float32), jnp.asarray(t, jnp.float32))[0]
    return y.astype(x.dtype)


class GuidanceSlot:
    """Holds the current call's guidance for the host callback; compiled per jaxpr key on the CPU backend."""

    def __init__(self):
        self.cache: dict[str, Callable] = {}
        self.current: Hoisted | None = None
        self.consts_cpu: list | None = None
        self.steps: list[dict] = []
        self.stats = {"compiles": 0, "hits": 0}

    def set(self, h: Hoisted | None) -> None:
        import jax

        self.steps = []
        self.current = h
        self.consts_cpu = None
        if h is None or h.is_identity:
            return
        if h.key not in self.cache:
            jaxpr = h.jaxpr
            self.cache[h.key] = jax.jit(lambda consts, x, t: eval_hoisted(jaxpr, consts, x, t))
            self.stats["compiles"] += 1
        else:
            self.stats["hits"] += 1
        dev = cpu_device()
        self.consts_cpu = [jax.device_put(c, dev) for c in h.consts]

    def clear(self) -> None:
        self.current, self.consts_cpu = None, None

    def host(self, x, t):
        """``pure_callback`` target: numpy in, numpy out (same dtype / shape)."""
        import jax

        x = np.asarray(x)
        h = self.current
        if h is None or h.is_identity:
            return x
        dev = cpu_device()
        t0 = time.perf_counter()
        y = self.cache[h.key](self.consts_cpu, jax.device_put(x.astype(np.float32), dev),
                              jax.device_put(np.asarray(t, np.float32), dev))  # traced as float32 (hoist)
        y = np.asarray(y, dtype=x.dtype).reshape(x.shape)
        self.steps.append({"t_next": float(np.asarray(t)), "dx_l2": float(np.linalg.norm((y - x).astype(np.float64))),
                           "ms": (time.perf_counter() - t0) * 1e3})
        return y

    def hook(self, x_next, t_next):
        """The fixed ``guidance_fn`` traced into the sampler (identity when no guidance is set)."""
        import jax

        return jax.pure_callback(self.host, jax.ShapeDtypeStruct(x_next.shape, x_next.dtype), x_next, t_next)


# --- backends ---------------------------------------------------------------------------------------------------
class OpenpiBackend:
    """The real openpi ``Policy``: its transforms, its model, ``Pi0.sample_actions`` jitted like ``module_jit``."""

    def __init__(self, policy, slot: GuidanceSlot, *, exec_mode: str = "callback"):
        import flax.nnx as nnx
        import jax

        if exec_mode not in GUIDANCE_EXEC:
            raise ValueError(f"exec_mode must be one of {GUIDANCE_EXEC}")
        self.policy, self.slot, self.exec_mode = policy, slot, exec_mode
        model = policy._model
        self.H, self.D = int(model.action_horizon), int(model.action_dim)
        graphdef, state = nnx.split(model)
        self._state = state
        kw = dict(getattr(policy, "_sample_kwargs", {}) or {})
        self._kw = kw

        def base(state, rng, observation, noise):
            m = nnx.merge(graphdef, state)
            return m.sample_actions(rng, observation, noise=noise, **kw)

        def guided(state, rng, observation, noise):
            m = nnx.merge(graphdef, state)
            return m.sample_actions(rng, observation, noise=noise, guidance_fn=slot.hook, **kw)

        self._graphdef = graphdef
        self._base = jax.jit(base)
        self._guided = jax.jit(guided)
        self._inline: dict[str, Callable] = {}
        H, D = self.H, self.D
        self._noise = jax.jit(lambda rng, b: jax.random.normal(rng, (b, H, D)), static_argnums=1)

    def prepare(self, obs: dict):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return inputs, _model.Observation.from_dict(inputs)

    def batch(self, prepared) -> int:
        return int(prepared[1].state.shape[0])

    def noise(self, rng, b: int):
        return self._noise(rng, b)

    def sample(self, rng, prepared, noise, hoisted: Hoisted | None):
        import flax.nnx as nnx
        import jax

        observation = prepared[1]
        if hoisted is None or hoisted.is_identity:
            return self._base(self._state, rng, observation, noise)
        if self.exec_mode == "callback":
            return self._guided(self._state, rng, observation, noise)
        fn = self._inline.get(hoisted.key)
        if fn is None:
            graphdef, kw, jaxpr = self._graphdef, self._kw, hoisted.jaxpr

            def body(state, rng, observation, noise, consts):
                m = nnx.merge(graphdef, state)
                return m.sample_actions(rng, observation, noise=noise,
                                        guidance_fn=lambda x, t: eval_hoisted(jaxpr, consts, x, t), **kw)

            fn = self._inline[hoisted.key] = jax.jit(body)
            self.slot.stats["compiles"] += 1
        else:
            self.slot.stats["hits"] += 1
        return fn(self._state, rng, observation, noise, hoisted.consts)

    def finish(self, prepared, x) -> dict:
        import jax

        outputs = {"state": prepared[0]["state"], "actions": x}
        outputs = jax.tree.map(lambda a: np.asarray(a[0, ...]), outputs)
        return self.policy._output_transform(outputs)


class SurrogateBackend:
    """CPU smoke: the fake policy's target chunk, D's surrogate Euler loop (``guidance_common.core``) as the network."""

    def __init__(self, fake, slot: GuidanceSlot, *, exec_mode: str = "callback"):
        import jax

        self.fake, self.slot, self.exec_mode = fake, slot, exec_mode
        self.H, self.D = fake.H, fake.D
        from Baseline.paper_baselines.methods.guidance_common.core import surrogate_sample

        kind, steps = fake.kind, fake.num_steps
        self._base = jax.jit(lambda xt, noise: surrogate_sample(xt, noise, num_steps=steps, kind=kind))
        self._guided = jax.jit(lambda xt, noise: surrogate_sample(xt, noise, num_steps=steps, kind=kind,
                                                                   guidance_fn=slot.hook))
        self._inline: dict[str, Callable] = {}
        H, D = self.H, self.D
        self._noise = jax.jit(lambda rng, b: jax.random.normal(rng, (b, H, D)), static_argnums=1)

    def prepare(self, obs: dict):
        return obs, self.fake.target_model_space(obs)

    def batch(self, prepared) -> int:
        return 1

    def noise(self, rng, b: int):
        return self._noise(rng, b)

    def sample(self, rng, prepared, noise, hoisted: Hoisted | None):
        import jax

        from Baseline.paper_baselines.methods.guidance_common.core import surrogate_sample

        xt = prepared[1]
        if hoisted is None or hoisted.is_identity:
            return self._base(xt, noise)
        if self.exec_mode == "callback":
            return self._guided(xt, noise)
        fn = self._inline.get(hoisted.key)
        if fn is None:
            jaxpr, kind, steps = hoisted.jaxpr, self.fake.kind, self.fake.num_steps
            fn = self._inline[hoisted.key] = jax.jit(lambda xt, noise, consts: surrogate_sample(
                xt, noise, num_steps=steps, kind=kind, guidance_fn=lambda x, t: eval_hoisted(jaxpr, consts, x, t)))
            self.slot.stats["compiles"] += 1
        else:
            self.slot.stats["hits"] += 1
        return fn(xt, noise, hoisted.consts)

    def finish(self, prepared, x) -> dict:
        return self.fake.finish(prepared[0], np.asarray(x))


# --- the served guidance policy ----------------------------------------------------------------------------------
class GuidedPolicy:
    """``Policy.infer`` with a method's guidance. Call :meth:`set_call` (scene of this request) before ``infer``."""

    def __init__(self, inner, backend, method, norm: dict, *, record_nominal: bool = True):
        self._policy = inner  # SeededPolicy re-keys inner._rng (policy_seed.rng_owner walks ``_policy``)
        self.backend, self.method, self.norm = backend, method, norm
        self.slot = backend.slot
        self.record_nominal = bool(record_nominal)
        self._pending = None
        self.last: dict = {}

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    def reset(self):
        r = getattr(self._policy, "reset", None)
        if callable(r):
            r()

    def set_call(self, sobs, scene) -> None:
        self._pending = (sobs, scene)

    def infer(self, obs: dict, **kwargs) -> dict:
        import jax

        if self._pending is None:
            raise RuntimeError("GuidedPolicy.infer without set_call (scene of this request)")
        sobs, scene = self._pending
        self._pending = None
        inner = self._policy
        t0 = time.perf_counter()
        prepared = self.backend.prepare(obs)
        inner._rng, sample_rng = jax.random.split(inner._rng)  # the same split as Policy.infer
        noise = self.backend.noise(sample_rng, self.backend.batch(prepared))
        t_prep = time.perf_counter()
        err = None
        hoisted = None
        try:
            with jax.default_device(cpu_device()):
                fn = self.method.make_guidance_fn(sobs, self.norm, noise=np.asarray(noise), scene=scene)
            hoisted = hoist(fn, (self.backend.batch(prepared), self.backend.H, self.backend.D), np.float32)
        except Exception as exc:  # noqa: BLE001 -- guidance unavailable this call: unguided sample, logged
            import traceback

            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        t_build = time.perf_counter()
        need_nom = self.record_nominal or hoisted is None or hoisted.is_identity
        x_nom = self.backend.sample(sample_rng, prepared, noise, None) if need_nom else None
        if x_nom is not None:
            x_nom = np.asarray(x_nom)
        t_nom = time.perf_counter()
        if hoisted is None or hoisted.is_identity:
            x = x_nom
            steps = []
        else:
            if self.backend.exec_mode == "callback":
                self.slot.set(hoisted)  # inline mode passes the constants into the sampler instead
            try:
                x = np.asarray(self.backend.sample(sample_rng, prepared, noise, hoisted))
            except Exception as exc:  # noqa: BLE001 -- guided sampling failed: serve the unguided sample, logged
                import traceback

                err = f"guided sample: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                if x_nom is None:
                    x_nom = np.asarray(self.backend.sample(sample_rng, prepared, noise, None))
                x, hoisted = x_nom, None
            finally:
                steps = list(self.slot.steps)
                self.slot.clear()
        t_g = time.perf_counter()
        out = dict(self.backend.finish(prepared, x))
        nominal = None if x_nom is None else np.asarray(self.backend.finish(prepared, x_nom)["actions"])
        ctx = getattr(self.method, "last_ctx", None) or {}
        self.last = {"nominal_actions": nominal, "guided": hoisted is not None and not hoisted.is_identity,
                     "identity_reason": None if hoisted is None or not hoisted.is_identity else "guidance_fn is identity "
                     "(method skip: e.g. OmniGuide freeze zone / nothing to avoid)", "error": err,
                     "jaxpr_key": None if hoisted is None else hoisted.key[:12],
                     "n_consts": None if hoisted is None else len(hoisted.consts),
                     "denoise_steps": steps, "cache": dict(self.slot.stats),
                     "skip": bool(ctx.get("skip")) if isinstance(ctx, dict) else None,
                     "ms": {"prepare": (t_prep - t0) * 1e3, "build_guidance": (t_build - t_prep) * 1e3,
                            "sample_nominal": (t_nom - t_build) * 1e3, "sample_guided": (t_g - t_nom) * 1e3}}
        out["policy_timing"] = {"infer_ms": (t_g - t_build) * 1e3}
        return out


__all__ = ["GUIDANCE_EXEC", "GuidanceSlot", "GuidedPolicy", "Hoisted", "OpenpiBackend", "SurrogateBackend",
           "eval_hoisted", "hoist"]
