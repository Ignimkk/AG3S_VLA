"""Method server for the LIBERO / SafeLIBERO closed loop (``.venv-openpi-live``): same wire as
``AG3S.trajopt.serve_safe --robot panda`` (openpi websocket, ``libero_safe_client.py`` request), one comparison method.

```
libero_safe_client.py (.venv-libero, unchanged)  --request (policy keys + ag3s/* geometry + panda/*)-->
  MethodServer.infer
    wire_obs.scene_observation(request)                         (non-privileged SceneObservation;
                                                                 --gt: + privileged.extras(request), ag3s/privileged/*)
    none     : PandaVLAOnly(SeededPolicy(policy)).infer(request)  = the VLA-only server, byte for byte
    filter   : same call -> nominal chunk -> sidecar(.venv-baselines).filter(obs, chunk) -> chunk
    guidance : PandaVLAOnly(SeededPolicy(GuidedPolicy(policy))) -> pi0.sample_actions(guidance_fn=method)
  <--response: actions (H, 7) EE deltas; the client executes rows 0..4 (replan 5), "panda/mode" = "method:<arm>"
```

Every call is logged to ``<run>/method_calls/calls.jsonl`` (+ per-episode ``ep<k>_seed<s>.npz``: nominal and served
chunks): intervened rows, ``|delta a|`` (``evaluate_offline.delta_stats``, the offline keys), the method's own ``h``
(filters: ``evaluate_offline.filter_step_info``; guidance: EE / hand min signed distance of the decoded path to the
method's obstacle set, nominal vs guided), perception, timings, fallbacks. ``<run>`` = ``$AG3S_RUN_DIR`` (launcher) or
``--record-dir``. A method failure never breaks the episode: the nominal chunk is served and the error logged.

``--filter-rate step`` (F2, 2026-11-10; filter arms; default ``chunk`` = everything above, unchanged): the client
(``libero_safe_client.py --filter-every-step``) asks every env step. A request with ``ag3s/filter_step = 0`` is a policy
step (same env steps, seq and seed as in chunk mode): the policy runs, the nominal chunk is cached and only row 0 is
filtered (measured EE pose, current observation); ``ag3s/filter_step = k > 0`` filters row ``k`` of the cached chunk on
that step's observation without calling the policy (its RNG does not move). One filter step per env step, so the
adapter state (AEGIS ``z``, KNOWS normals / tracks) advances once per env step. ``calls.jsonl`` gets one row per env step
(``call_kind`` policy | filter, ``k``); ``ep*_seed*.npz`` keeps ``seq`` / ``nominal`` / ``served`` per policy call and adds
``step_*`` arrays per env step. The server metadata carries ``ag3s_filter_rate = step`` (the client refuses a mismatch).

KNOWS F4a (``--target attention``, the knows detector arm's default since 2026-11-10): ``attention_tap.AttentionTap``
between ``SeededPolicy`` and the policy hands the method the step-0 attention grid of the call that produced the chunk
(``extras["knows/attention"]``); the chunk itself is untouched.

Policy loading is ``serve_safe``'s Panda path (``AG3S/adapters/panda/server.py:main``): ``create_trained_policy(
get_config(--config), --checkpoint, default_prompt=...)``. ``--fake-policy constant|recorded`` replaces it on CPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "third_party" / "openpi" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Baseline.paper_baselines.closed_loop import arms as A  # noqa: E402

GUIDE_TOL_RAW = 1e-4  # = evaluate_offline.GUIDE_TOL_RAW (D's intervened-row rule)
NORM_REL = "assets/physical-intelligence/libero/norm_stats.json"
DEFAULT_SCENE_KW = {"voxel": 0.02, "max_points": 1024, "workspace_radius": 0.6, "remove_table": False}
#: methods whose obstacle cloud is not padded to ``max_points``. Empty since CLOSED-LOOP-FIX: Lan-o3dp's circle
#: preparation now ignores the FAR padding points (``obstacle_circles``), so every guidance arm sees one shape.
NO_PADDING: set[str] = set()
#: per-call method context recorded in ``row["guidance"]["ctx"]`` (scalars / short lists of ``method.last_ctx``)
CTX_KEYS = ("skip", "frozen", "freeze_min_dist_m", "freeze_cached_dist_m", "target_source", "n_target_points",
            "attract", "collision", "n_clearance_dropped", "gripper_closed", "n_circles", "circles", "x0_mode")
#: per-step filter diagnostics copied from ``info["steps"]`` (KNOWS adapter, CLOSED-LOOP-FIX) as ``<key>_steps``
STEP_KEYS = ("h_star", "h_star_next", "h_star_next_nominal", "cbf_ok", "relax_m", "need_m", "binding_obstacle")
#: ``row["ms"]`` keys. ``total`` = wall time inside ``MethodServer.infer`` = what the client waits for (minus
#: transport); ``diagnostics`` = logging-only work done after the chunk is final (guidance h, path metrics);
#: ``method`` = ``total - diagnostics`` (the latency the method itself adds); ``serve`` = ``total`` (kept for readers
#: of the V2 / V3 records, where it meant ``total - diagnostics``).
LATENCY_KEYS = ("total", "diagnostics", "method", "serve")
#: F2 (2026-11-10): ``--filter-rate``; request key of the client's per-step mode; server metadata key of the mode
FILTER_RATES = ("chunk", "step")
FILTER_STEP_KEY = "ag3s/filter_step"
FILTER_RATE_META = "ag3s_filter_rate"


_SD_JIT = None


def _sd_jit():
    """Jitted ``geometry.signed_distance`` on the JAX CPU backend (one compile per padded shape)."""
    global _SD_JIT
    if _SD_JIT is None:
        import jax

        from Baseline.paper_baselines.methods.guidance_common.geometry import signed_distance

        _SD_JIT = jax.jit(signed_distance)
    return _SD_JIT


def path_metrics_fast(raw: np.ndarray, scene, H: int) -> dict:
    """``guidance_common.evaluation.path_metrics`` with a jitted, fixed-shape distance (diagnostics only)."""
    from Baseline.paper_baselines.methods.guidance_common.kinematics import EEPathModel

    pm = EEPathModel(H=H, p0=scene.ee_pos, R_bw=scene.R_bw, y_prev=scene.y_prev)
    p = pm.numpy(np.clip(raw[:, :3], -1, 1))
    o = scene.obstacles
    if o.empty:
        return {"ee_min_sdf_m": None, "hand_min_sdf_m": None}
    import jax

    hs = p[:, None, :] + (scene.hand_c - scene.ee_pos[None])[None]
    q = np.concatenate([p, hs.reshape(-1, 3)]).astype(np.float32)
    with jax.default_device(jax.devices("cpu")[0]):  # same backend as the guidance callback, off the GPU
        oj = o.padded(max(1, len(o.points))).as_jax()
        d = np.asarray(_sd_jit()(jax.numpy.asarray(q), oj), np.float64)
    sd, hsd = d[:H], d[H:].reshape(H, -1) - scene.hand_r[None]
    return {"ee_min_sdf_m": float(sd.min()), "hand_min_sdf_m": float(hsd.min())}


def storage_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", PROJECT_ROOT.parent / "AG3S_VLA-storage"))


def default_checkpoint() -> pathlib.Path:
    return storage_root() / "checkpoints/panda/pi05_libero"


# =============================================================================================================
class CallRecorder:
    def __init__(self, root: pathlib.Path | None):
        self.root = None if root is None else pathlib.Path(root)
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self._jsonl = None if self.root is None else open(self.root / "calls.jsonl", "a")
        self._arrays: dict[str, list] = {}
        self._episode = None

    def start_episode(self, episode: int, policy_seed) -> None:
        self.flush()
        self._episode = (int(episode), policy_seed)
        self._arrays = {"seq": [], "nominal": [], "served": []}

    def add_step(self, row: dict, chunk: tuple | None, step_nominal: np.ndarray, step_served: np.ndarray) -> None:
        """``--filter-rate step``: one jsonl row per env step; ``seq`` / ``nominal`` / ``served`` per policy call (``chunk``
        = (nominal chunk, served chunk) on a policy step), ``step_*`` per env step."""
        from Baseline.paper_baselines.methods.interface import jsonable

        if self._jsonl is not None:
            self._jsonl.write(json.dumps(jsonable(row), allow_nan=True) + "\n")
            self._jsonl.flush()
        if self._episode is not None:
            if chunk is not None:
                self._arrays["seq"].append(int(row.get("seq") or 0))
                self._arrays["nominal"].append(np.asarray(chunk[0], np.float64))
                self._arrays["served"].append(np.asarray(chunk[1], np.float64))
            for key, val in (("step_seq", int(row.get("seq") or 0)), ("step_k", int(row.get("k") or 0)),
                             ("step_nominal", np.asarray(step_nominal, np.float64).reshape(-1)),
                             ("step_served", np.asarray(step_served, np.float64).reshape(-1))):
                self._arrays.setdefault(key, []).append(val)

    def add(self, row: dict, nominal: np.ndarray | None, served: np.ndarray) -> None:
        from Baseline.paper_baselines.methods.interface import jsonable

        if self._jsonl is not None:
            self._jsonl.write(json.dumps(jsonable(row), allow_nan=True) + "\n")
            self._jsonl.flush()
        if self._episode is not None:
            self._arrays["seq"].append(int(row.get("seq") or 0))
            self._arrays["nominal"].append(np.asarray(served if nominal is None else nominal, np.float64))
            self._arrays["served"].append(np.asarray(served, np.float64))

    def flush(self) -> None:
        if self.root is None or self._episode is None or not self._arrays.get("seq"):
            return
        k, ps = self._episode
        try:
            arrays = {key: np.stack(v) for key, v in self._arrays.items()}
        except ValueError:
            arrays = {"seq": np.asarray(self._arrays["seq"])}
            for key in ("step_seq", "step_k", "step_nominal", "step_served"):  # --filter-rate step only
                if self._arrays.get(key):
                    arrays[key] = np.stack(self._arrays[key])
        path = self.root / f"ep{k:03d}_seed{ps}.npz"
        np.savez_compressed(path, **arrays)
        self._arrays = {"seq": [], "nominal": [], "served": []}

    def close(self) -> None:
        self.flush()
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None


# =============================================================================================================
class MethodServer:
    """The served object (``WebsocketPolicyServer(policy=...)``)."""

    def __init__(self, inner, arm: dict, *, method=None, sidecar=None, norm: dict | None = None, backend=None,
                 recorder: CallRecorder | None = None, scene_kwargs: dict | None = None,
                 perception_kwargs: dict | None = None, record_nominal: bool = True, target_cameras=None,
                 suite: str | None = None, filter_rate: str = "chunk", attention_tap=None):
        from AG3S.adapters.panda.server import PandaVLAOnly
        from AG3S.trajopt.policy_seed import SeededPolicy

        self.arm, self.family = arm, arm["family"]
        self.inner, self.method, self.sidecar, self.norm = inner, method, sidecar, norm
        self.recorder = recorder or CallRecorder(None)
        self.scene_kwargs = {**DEFAULT_SCENE_KW, **(scene_kwargs or {})}
        self.perception_kwargs = dict(perception_kwargs or {})
        self.target_cameras = list(target_cameras) if target_cameras else None
        self.suite = suite  # extras["suite"] (AEGIS's official workspace box per suite; non-privileged)
        self.guided = None
        if filter_rate not in FILTER_RATES:
            raise ValueError(f"filter_rate must be one of {FILTER_RATES}")
        if filter_rate != "chunk" and self.family != "filter":
            raise ValueError("--filter-rate step applies to filter arms only")
        self.filter_rate = filter_rate
        self.chunk = None  # step mode: {"seq", "A0"} of the last policy step
        self.tap = attention_tap
        if attention_tap is not None and self.family != "filter":
            raise ValueError("the attention tap (KNOWS F4a) applies to filter arms only")
        if self.family == "guidance":
            from Baseline.paper_baselines.closed_loop.guided import GuidedPolicy

            self.guided = GuidedPolicy(inner, backend, method, norm, record_nominal=record_nominal)
            self.vla = PandaVLAOnly(SeededPolicy(self.guided))
        elif attention_tap is not None:  # the tap wraps the policy; the call itself is unchanged (attention_tap.py)
            self.vla = PandaVLAOnly(SeededPolicy(attention_tap))
        else:
            self.vla = PandaVLAOnly(SeededPolicy(inner))
        self.episode = -1
        self.policy_seed = None
        self.last_gripper = None
        self.n_calls = 0
        self.n_fallback = 0

    @property
    def metadata(self):
        if self.filter_rate == "step":  # the client's --filter-every-step checks this (chunk mode: unchanged)
            return {**dict(self.vla.metadata or {}), FILTER_RATE_META: "step"}
        return self.vla.metadata

    def reset(self):
        self.vla.reset()

    def close(self):
        self.recorder.close()
        if self.sidecar is not None:
            self.sidecar.close()

    # --- one call -----------------------------------------------------------------------------------------------
    def _new_episode(self, request: dict) -> None:
        from Baseline.paper_baselines.closed_loop.fake_policy import decode_policy_seed

        self.episode += 1
        self.policy_seed = request.get("ag3s/policy_seed")
        self.last_gripper = None
        if self.family == "filter" and self.sidecar is not None:
            self.sidecar.call("reset")
        if self.family == "guidance" and hasattr(self.method, "reset"):
            self.method.reset()
        self.recorder.start_episode(self.episode, self.policy_seed)
        self.episode_info = decode_policy_seed(self.policy_seed)

    def _target_masks(self, sobs) -> tuple[dict | None, dict]:
        from Baseline.paper_baselines.closed_loop.perception import box_masks
        from Baseline.paper_baselines.closed_loop.wire_obs import obs_to_dict

        if self.arm.get("target") != "detector" or self.sidecar is None:
            return None, {"target": self.arm.get("target")}
        d = obs_to_dict(sobs)
        d["depth"], d["extras"] = {}, {}
        r = self.sidecar.call("target", obs=d, cameras=self.target_cameras or list(sobs.rgb))
        shapes = {c: np.asarray(sobs.depth[c]).shape for c in sobs.depth}
        boxes = {c: r["boxes"].get(c) for c in shapes}
        return box_masks(boxes, shapes), {"target": "detector", "phrase": r.get("phrase"), "scores": r.get("scores"),
                                          "boxes": {c: None if b is None else np.asarray(b).round(1).tolist()
                                                    for c, b in boxes.items()}, "ms": r.get("ms")}

    def infer(self, obs: dict, **kwargs) -> dict:
        from Baseline.paper_baselines.closed_loop.wire_obs import scene_observation

        t0 = time.perf_counter()
        obs = dict(obs)
        seq = obs.get("ag3s/seq")
        if self.filter_rate == "step":
            return self._infer_step(obs, seq, t0, **kwargs)
        if FILTER_STEP_KEY in obs:
            raise ValueError(f"{FILTER_STEP_KEY} in a request but the server runs --filter-rate chunk")
        if obs.get("ag3s/reset") or self.episode < 0:
            self._new_episode(obs)
        if hasattr(self.inner, "set_context"):  # fake policies only (recorded chunk lookup)
            self.inner.set_context(policy_seed=self.policy_seed, seq=seq)
        row = {"episode": self.episode, **self.episode_info, "seq": seq, "arm": self.arm["arm"],
               "variant": self.arm["variant"], "family": self.family}
        nominal = None
        if self.family == "none":
            result = self.vla.infer(obs, **kwargs)
            t_pol = time.perf_counter()
            row["ms"] = {"policy": (t_pol - t0) * 1e3}
        elif self.family == "filter":
            result, nominal = self._filter_call(obs, row, seq, t0, **kwargs)
        else:
            result, nominal = self._guidance_call(obs, row, seq, t0, **kwargs)
        served = np.asarray(result["actions"])
        if served.ndim == 2 and served.shape[0]:
            self.last_gripper = float(served[min(4, served.shape[0] - 1), 6])  # last executed row (replan 5)
        row["ms"]["total"] = (time.perf_counter() - t0) * 1e3  # = client-visible server time (LATENCY_KEYS)
        row["ms"].setdefault("diagnostics", 0.0)
        row["ms"]["method"] = row["ms"]["total"] - row["ms"]["diagnostics"]
        row["ms"]["serve"] = row["ms"]["total"]
        sx = served.astype(np.float64)[:, :3] if served.ndim == 2 and served.shape[0] else np.zeros((0, 3))
        row["served_max_abs_xyz"] = float(np.abs(sx).max()) if sx.size else None
        row["served_rows_out_of_range"] = int(np.count_nonzero(np.abs(sx).max(1) > 1.0)) if sx.size else 0
        self.recorder.add(row, nominal, served)
        self.n_calls += 1
        result["panda/mode"] = f"method:{self.arm['arm']}"
        result["timing_ms"] = {k: round(float(v), 3) for k, v in row["ms"].items()}
        return result

    def _scene_obs(self, obs, seq, row=None):
        from Baseline.paper_baselines.closed_loop.wire_obs import scene_observation

        sobs = scene_observation(obs, last_gripper_cmd=self.last_gripper, seq=seq)
        if self.suite:
            sobs.extras["suite"] = self.suite
        if self.arm.get("privileged"):  # --gt: simulator GT from the client's ag3s/privileged/* keys
            from Baseline.paper_baselines.closed_loop import privileged as P

            sobs.extras.update(P.extras(obs, sobs.extras["T_world_base"], shape=self.arm.get("gt_geom_shape") or "collision"))
            if self.arm.get("gt_source") == "exact":  # multi-link CBF privileged "exact geometry" (doc §4.12)
                sobs.extras["obstacle_ellipsoids"] = P.exact_ellipsoids(sobs.extras["privileged/obstacles"])
            if row is not None:
                row["privileged"] = P.summary(sobs.extras)
        return sobs

    def _infer_step(self, obs, seq, t0, **kwargs) -> dict:
        """``--filter-rate step``: one env step (policy step ``k = 0`` or filter step ``k > 0``, module docstring)."""
        if FILTER_STEP_KEY not in obs:
            raise ValueError(f"--filter-rate step needs {FILTER_STEP_KEY} in every request (client --filter-every-step)")
        k = int(obs[FILTER_STEP_KEY])
        policy_step = k == 0
        if policy_step and (obs.get("ag3s/reset") or self.episode < 0):
            self._new_episode(obs)
        if policy_step and hasattr(self.inner, "set_context"):  # fake policies only (recorded chunk lookup)
            self.inner.set_context(policy_seed=self.policy_seed, seq=seq)
        row = {"episode": self.episode, **self.episode_info, "seq": seq, "arm": self.arm["arm"],
               "variant": self.arm["variant"], "family": self.family, "filter_rate": "step",
               "call_kind": "policy" if policy_step else "filter", "k": k}
        if policy_step:
            result, nominal = self._filter_call(obs, row, seq, t0, rows=slice(0, 1), **kwargs)
            A0 = np.asarray(nominal)
            self.chunk = {"seq": seq, "A0": A0}
            served = np.asarray(result["actions"])  # the chunk, row 0 filtered (rows 1.. are filtered at their steps)
            step_nominal, step_served = A0[0], served[0]
            chunk = (A0, served)
        else:
            if self.chunk is None or self.chunk["seq"] != seq or k >= len(self.chunk["A0"]):
                raise ValueError(f"filter step k={k} of seq {seq} without its policy step (cached: "
                                 f"{None if self.chunk is None else self.chunk['seq']})")
            A0 = self.chunk["A0"]
            row["ms"] = {"policy": 0.0}
            result = {"actions": A0[k:k + 1].copy()}
            result, _ = self._filter_rows(obs, row, seq, A0[k:k + 1], result, 0.0, t0)
            served = np.asarray(result["actions"])
            step_nominal, step_served = A0[k], served[0]
            chunk = None
        self.last_gripper = float(step_served[6])  # the action executed at this step
        row["ms"]["total"] = (time.perf_counter() - t0) * 1e3
        row["ms"].setdefault("diagnostics", 0.0)
        row["ms"]["method"] = row["ms"]["total"] - row["ms"]["diagnostics"]
        row["ms"]["serve"] = row["ms"]["total"]
        sx = np.asarray(step_served, np.float64)[:3]
        row["served_max_abs_xyz"] = float(np.abs(sx).max())
        row["served_rows_out_of_range"] = int(np.abs(sx).max() > 1.0)
        self.recorder.add_step(row, chunk, step_nominal, step_served)
        self.n_calls += 1
        result["panda/mode"] = f"method:{self.arm['arm']}"
        result["panda/filter_step"] = k
        result["timing_ms"] = {kk: round(float(v), 3) for kk, v in row["ms"].items()}
        return result

    def _filter_call(self, obs, row, seq, t0, rows=None, **kwargs):
        result = self.vla.infer(obs, **kwargs)
        t_pol = time.perf_counter()
        A0 = np.asarray(result["actions"])
        row["ms"] = {"policy": (t_pol - t0) * 1e3}
        if rows is None:  # chunk mode: the whole chunk
            return self._filter_rows(obs, row, seq, A0, result, t_pol, t0)[0], A0
        out = {**result, "actions": A0.copy()}  # step mode policy step: filter row 0, keep the nominal rows after it
        res, _ = self._filter_rows(obs, row, seq, A0[rows], {"actions": A0[rows].copy()}, t_pol, t0)
        out["actions"][rows] = np.asarray(res["actions"])
        return out, A0

    def _filter_rows(self, obs, row, seq, A0, result, t_pol, t0):
        """Filter ``A0`` (the chunk, or one row in step mode) on this request's observation; ``result["actions"]`` <- B."""
        from Baseline.paper_baselines.closed_loop.perception import clusters, scene_points
        from Baseline.paper_baselines.closed_loop.wire_obs import obs_to_dict
        from Baseline.paper_baselines.methods.evaluate_offline import delta_stats, filter_step_info

        t_pol = t_pol or time.perf_counter()
        try:
            sobs = self._scene_obs(obs, seq, row)
            self._attach_attention(sobs, row)
            if self.arm["obstacles"] == "depth":
                masks, tinfo = self._target_masks(sobs)
                sp = scene_points(sobs, target_masks=masks, **self.perception_kwargs)
                sobs.extras["obstacle_points"] = clusters(sp["obstacle_points"])
                row["perception"] = {**sp["info"], **tinfo, "clusters": {k: int(len(v)) for k, v in
                                                                         sobs.extras["obstacle_points"].items()}}
            t_perc = time.perf_counter()
            d = obs_to_dict(sobs)
            r = self.sidecar.call("filter", obs=d, actions=np.asarray(A0, np.float64))
            t_f = time.perf_counter()
            B = np.asarray(r["actions"], np.float64)
            iv = np.asarray(r["intervened"], bool)
            served = B.astype(A0.dtype)
            row.update(delta_stats(A0, B, iv))
            row["intervened"] = iv.tolist()
            row.update({k: v for k, v in filter_step_info(self.arm["method"], r["info"]).items()
                        if k in ("h_steps", "h_min", "h_neg_steps", "qp_status", "qp_fail", "emergency_stops")})
            steps = r["info"].get("steps") or []
            for key in STEP_KEYS:
                if any(key in st for st in steps):
                    row[f"{key}_steps"] = [st.get(key) for st in steps]
            if "h_star_steps" in row:  # KNOWS: the true separation max_n h(n) next to the QP's own h(n_stored)
                hs = [v for v in row["h_star_steps"] if v is not None]
                row["h_star_min"] = min(hs) if hs else None
                row["h_star_neg_steps"] = sum(v < 0 for v in hs)
                row["cbf_violated_steps"] = sum(v is False for v in row.get("cbf_ok_steps", []))
                rel, need = row.get("relax_m_steps", []), row.get("need_m_steps", [])
                row["relaxation_covers_need_steps"] = sum(1 for a, b in zip(rel, need)
                                                          if a is not None and b is not None and b > 0 and a >= b)
            obst = r["info"].get("obstacles")
            if isinstance(obst, dict):
                row["filter_obstacles"] = {k: {"c": v.get("c"), "semi_axes": v.get("semi_axes")} for k, v in obst.items()}
            row["filter_perception"] = r["info"].get("perception")
            if r["info"].get("target") is not None:  # KNOWS F4a: Eq. 4 target of this step (excluded from the set)
                row["filter_target"] = r["info"]["target"]
            row["ms"].update({"perception": (t_perc - t_pol) * 1e3, "sidecar_roundtrip": (t_f - t_perc) * 1e3,
                              "filter": r.get("ms")})
            row["fallback"] = None
            result["actions"] = served
        except Exception:  # noqa: BLE001 -- serve the nominal chunk, log why
            self.n_fallback += 1
            row["fallback"] = "nominal (method error)"
            row["error"] = traceback.format_exc()
            logging.error("method %s failed at seq %s:\n%s", self.arm["arm"], seq, row["error"])
        return result, A0

    def _attach_attention(self, sobs, row) -> None:
        """KNOWS F4a: the attention grid of the policy call that produced the current chunk (tap.last)."""
        if self.tap is None:
            return
        last = self.tap.last or {}
        row["attention"] = {k: last.get(k) for k in ("ms", "call", "reason") if last.get(k) is not None}
        if last.get("error"):
            row["attention"]["error"] = last["error"][-500:]
        if last.get("grid") is not None:
            sobs.extras["knows/attention"] = {"grid": np.asarray(last["grid"], np.float32), "seq": last.get("call")}

    def _guidance_call(self, obs, row, seq, t0, **kwargs):
        from Baseline.paper_baselines.closed_loop.perception import scene_points
        from Baseline.paper_baselines.methods.evaluate_offline import delta_stats
        from Baseline.paper_baselines.methods.guidance_common.scene import build_scene

        scene, sobs, err = None, None, None
        try:
            sobs = self._scene_obs(obs, seq, row)
            kw = {k: v for k, v in self.scene_kwargs.items()}
            if self.arm.get("privileged"):  # GT primitives; the depth lift only for a target the method reads
                if self.arm.get("target") == "detector":
                    masks, tinfo = self._target_masks(sobs)
                    sp = scene_points(sobs, target_masks=masks, **self.perception_kwargs)
                    sobs.extras["target_points"] = sp["target_points"]
                    sp_info = sp["info"]
                else:
                    tinfo, sp_info = {"target": "none"}, {}
                    sobs.extras["target_points"] = np.zeros((0, 3))  # no depth lift in build_scene
                scene = build_scene(sobs, obstacle_source="privileged", **kw)
                sp = {"info": sp_info}
            else:
                masks, tinfo = self._target_masks(sobs)
                sp = scene_points(sobs, target_masks=masks, **self.perception_kwargs)
                sobs.extras["obstacle_points"] = sp["obstacle_points"]
                sobs.extras["target_points"] = sp["target_points"]
                scene = build_scene(sobs, obstacle_source="points", **kw)
            if self.arm["method"] not in NO_PADDING:  # fixed shapes -> one guidance compile per structure
                scene.obstacles = scene.obstacles.padded(int(kw.get("max_points", 1024)), 4)
            row["perception"] = {**sp["info"], **tinfo, "scene": scene.info}
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
        t_perc = time.perf_counter()
        if scene is None:  # no scene: unguided sample through the same path
            self.guided.set_call(sobs, None)
            saved, self.guided.method = self.guided.method, _NoGuidance()
            try:
                result = self.vla.infer(obs, **kwargs)
            finally:
                self.guided.method = saved
        else:
            self.guided.set_call(sobs, scene)
            result = self.vla.infer(obs, **kwargs)
        t_s = time.perf_counter()
        g = dict(self.guided.last)
        nominal = g.pop("nominal_actions", None)
        served = np.asarray(result["actions"])
        err = err or g.get("error")
        row["ms"] = {"perception": (t_perc - t0) * 1e3, "policy_and_guidance": (t_s - t_perc) * 1e3,
                     **{f"guided_{k}": v for k, v in (g.get("ms") or {}).items()}}
        row["guidance"] = {k: g.get(k) for k in ("guided", "identity_reason", "jaxpr_key", "n_consts", "cache", "skip")}
        ctx = getattr(self.method, "last_ctx", None) if scene is not None else None
        if isinstance(ctx, dict):
            row["guidance"]["ctx"] = {k: ctx[k] for k in CTX_KEYS if k in ctx}
        row["denoise_steps"] = g.get("denoise_steps")
        t_d = time.perf_counter()
        if nominal is not None:
            nom = np.asarray(nominal, np.float64)
            iv = np.max(np.abs(served.astype(np.float64)[:, :3] - nom[:, :3]), axis=1) > GUIDE_TOL_RAW
            row.update(delta_stats(nom, served, iv))
            row["intervened"] = iv.tolist()
            if scene is not None:
                H = served.shape[0]
                mb, mg = path_metrics_fast(nom, scene, H), path_metrics_fast(served.astype(np.float64), scene, H)
                row["h_nominal"] = {"ee_min_sdf_m": mb["ee_min_sdf_m"], "hand_min_sdf_m": mb["hand_min_sdf_m"]}
                row["h_served"] = {"ee_min_sdf_m": mg["ee_min_sdf_m"], "hand_min_sdf_m": mg["hand_min_sdf_m"]}
                row["h_min"] = mg["hand_min_sdf_m"]
        row["ms"]["diagnostics"] = (time.perf_counter() - t_d) * 1e3  # logging only, after the chunk is final
        row["fallback"] = None if err is None else "unguided sample (guidance error)"
        if err is not None:
            self.n_fallback += 1
            row["error"] = err
            logging.error("guidance %s failed at seq %s:\n%s", self.arm["arm"], seq, err)
        return result, nominal


class _NoGuidance:
    def make_guidance_fn(self, obs, norm, *, noise=None, scene=None):
        return lambda x, t: x


# =============================================================================================================
def load_norm(checkpoint: pathlib.Path) -> dict:
    from Baseline.paper_baselines.methods.guidance_common.core import load_norm_stats

    return load_norm_stats(str(pathlib.Path(checkpoint) / NORM_REL))


def build_server(args, *, policy=None, record_root=None) -> tuple[MethodServer, dict]:
    """Everything below the openpi policy (tests pass a fake ``policy``)."""
    from Baseline.paper_baselines.methods.interface import get_method, jsonable

    filter_rate = getattr(args, "filter_rate", "chunk")
    arm = A.resolve(args.method, args.obstacles, args.target, getattr(args, "gt", None), filter_rate=filter_rate)
    ckpt = pathlib.Path(args.checkpoint or default_checkpoint())
    norm = load_norm(ckpt) if arm["family"] == "guidance" or args.fake_policy != "none" else None
    if policy is None:
        if args.fake_policy != "none":
            from Baseline.paper_baselines.closed_loop.fake_policy import FakePolicy

            policy = FakePolicy(norm, source=args.fake_policy, run_dir=args.fake_source_run, kind=args.fake_kind,
                                source_key=args.fake_source_key)
        else:
            from openpi.policies import policy_config as _policy_config
            from openpi.training import config as _config

            policy = _policy_config.create_trained_policy(_config.get_config(args.config), str(ckpt),
                                                          default_prompt=args.default_prompt)
    method_kwargs = json.loads(args.method_kwargs) if args.method_kwargs else {}
    sidecar, method, backend, info = None, None, None, {"arm": arm}
    run = record_root
    need_sidecar = arm["family"] == "filter" or arm.get("target") == "detector"
    if need_sidecar:
        from Baseline.paper_baselines.closed_loop.sidecar import SidecarClient

        log = None if run is None else pathlib.Path(run) / "sidecar.log"
        env = dict(os.environ)
        if args.sidecar_device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("OMP_NUM_THREADS", "1")
        sidecar = SidecarClient(args.sidecar_python, log_path=log, env=env)
        info["sidecar"] = {"python": sidecar.python, "ping": sidecar.call("ping")["versions"], "log": str(log)}
        det = {"weight": args.detector_weight, "device": args.sidecar_device}
        if arm["family"] == "filter":
            kw = dict(method_kwargs)
            if arm.get("gt_source") == "names":  # AEGIS's VLM names -> SafeLIBERO GT obstacle names
                kw.setdefault("obstacle_source", "detector")
                kw.setdefault("detector_prompt", "obstacle_names")
                kw.setdefault("detector_device", args.sidecar_device)
                kw.setdefault("detector_weight", args.detector_weight)
            elif arm.get("gt_source") == "exact":
                kw.setdefault("obstacle_source", "ellipsoids")
            elif arm.get("gt_source") == "seg":
                kw.setdefault("obstacle_source", "gt_seg")
                kw.setdefault("exclude", "task_objects")  # target / destination are None: nothing excluded
            elif arm["obstacles"] == "detector":
                kw.setdefault("obstacle_source", "detector")
                # AEGIS / multi-link CBF: the papers' VLM names ONE hazard (AEGIS: GLM-4.5V API, multi-link: Qwen3-VL)
                # -> the official AEGIS prompt answered by a local open VLM (vlm_name, 2026-11-10 interim-check fix,
                # doc §4.10; weights from $AG3S_BASELINE_VLM_WEIGHT, a local copy of Qwen/Qwen2.5-VL-7B-Instruct);
                # KNOWS keeps all_objects (its paper has no VLM naming step)
                if arm["method"] in ("aegis", "multilink_cbf"):
                    from Baseline.paper_baselines.providers.vlm_namer import DEFAULT_VLM

                    kw.setdefault("detector_prompt", "vlm_name")
                    kw.setdefault("vlm_weight", os.environ.get("AG3S_BASELINE_VLM_WEIGHT") or DEFAULT_VLM)
                    kw.setdefault("vlm_device", args.sidecar_device)
                else:
                    kw.setdefault("detector_prompt", "all_objects")
                kw.setdefault("detector_device", args.sidecar_device)
                kw.setdefault("detector_weight", args.detector_weight)
                if arm.get("target") == "attention":  # KNOWS F4a (arms.resolve: knows detector default)
                    kw.setdefault("target_exclusion", "attention")
            else:
                kw.setdefault("obstacle_source", "given")
            r = sidecar.call("init", method=arm["method"], kwargs=kw, detector=det)
            info["method_kwargs"], info["provenance"] = kw, r["provenance"]
        else:
            sidecar.call("init", method=None, detector=det)  # detector only (target box)
            info["target_detector"] = det
    if arm["family"] == "guidance":
        from Baseline.paper_baselines.closed_loop.guided import GuidanceSlot, OpenpiBackend, SurrogateBackend

        method = get_method(arm["method"], **method_kwargs)
        slot = GuidanceSlot()
        if args.fake_policy != "none":
            backend = SurrogateBackend(policy, slot, exec_mode=args.guidance_exec)
        else:
            backend = OpenpiBackend(policy, slot, exec_mode=args.guidance_exec)
        info["method_kwargs"], info["provenance"] = method_kwargs, jsonable(method.provenance())
        info["guidance_exec"] = args.guidance_exec
    tap = None
    if arm.get("target") == "attention":  # KNOWS F4a: attention of the call that produced the chunk
        from Baseline.paper_baselines.closed_loop.attention_tap import AttentionTap, OpenpiAttentionBackend

        att_backend = None
        if args.fake_policy == "none" and hasattr(policy, "_model"):
            from openpi.training import config as _config

            att_backend = OpenpiAttentionBackend(policy, _config.get_config(args.config).model)
        tap = AttentionTap(policy, att_backend)
        info["attention"] = (att_backend.describe() if att_backend is not None
                             else {"backend": None, "reason": "fake policy: no attention, KNOWS target = none"})
    rec_dir = None if run is None else pathlib.Path(run) / "method_calls"
    server = MethodServer(policy, arm, method=method, sidecar=sidecar, norm=norm, backend=backend,
                          recorder=CallRecorder(rec_dir), record_nominal=not args.no_record_nominal,
                          scene_kwargs=json.loads(args.scene_kwargs) if args.scene_kwargs else None,
                          suite=args.suite, filter_rate=filter_rate, attention_tap=tap)
    info.update({"config": args.config, "checkpoint": str(ckpt), "fake_policy": args.fake_policy,
                 "fake_source_run": args.fake_source_run, "fake_source_key": args.fake_source_key,
                 "fake_kind": args.fake_kind, "record_dir": None if rec_dir is None else str(rec_dir),
                 "scene_kwargs": server.scene_kwargs, "suite": args.suite, "record_nominal": not args.no_record_nominal,
                 "action_convention": "pi05_libero EE delta chunk (H=10); client executes rows 0-4 (replan 5), "
                                      "same as serve_safe --robot panda"})
    if filter_rate != "chunk":  # key only off the default (chunk-mode config unchanged)
        info["filter_rate"] = {"rate": filter_rate, "rule": "policy every 5 env steps (rows 0-4 as chunk mode); every "
                               "env step one row filtered on the measured EE pose and the current observation "
                               "(client --filter-every-step)"}
    return server, info


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=sorted(A.ARMS))
    ap.add_argument("--obstacles", choices=A.OBSTACLE_SOURCES, default=None, help="filters: detector (default) | depth")
    ap.add_argument("--target", choices=A.TARGET_SOURCES, default=None, help="target exclusion: detector (default) | none")
    ap.add_argument("--gt", nargs="?", const="default", default=None,
                    help="privileged variant (arms.GT_SOURCES; no value = arms.GT_DEFAULT); needs the client's "
                         "--emit-privileged-obstacles")
    ap.add_argument("--suite", default=None, help="task suite -> SceneObservation.extras['suite'] (launcher sets it)")
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--default-prompt", default=None)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--method-kwargs", default=None, help="JSON kwargs for get_method")
    ap.add_argument("--scene-kwargs", default=None, help="JSON kwargs for guidance_common.scene.build_scene")
    ap.add_argument("--guidance-exec", choices=("callback", "inline"), default="callback")
    ap.add_argument("--filter-rate", choices=FILTER_RATES, default="chunk",
                    help="filter arms: chunk (default, whole chunk at the policy call) | step (F2: one row per env step; "
                         "needs the client's --filter-every-step)")
    ap.add_argument("--no-record-nominal", action="store_true", help="guidance: skip the unguided sample per call")
    ap.add_argument("--sidecar-python", default=str(PROJECT_ROOT / ".venv-baselines/bin/python"))
    ap.add_argument("--sidecar-device", default="cpu", help="cpu | cuda (Grounding DINO in the sidecar)")
    ap.add_argument("--detector-weight", default="grounding-dino-tiny")
    ap.add_argument("--fake-policy", choices=("none", "constant", "recorded"), default="none",
                    help="CPU smoke: no checkpoint; constant descent or a recorded run's chunks")
    ap.add_argument("--fake-source-run", default=None, help="run dir of --fake-policy recorded")
    ap.add_argument("--fake-source-key", choices=("actions_response", "nominal"), default="actions_response",
                    help="recorded chunks: the served chunks (episodes/*_calls.npz) or a method run's nominal "
                         "(method_calls/ep*_seed*.npz)")
    ap.add_argument("--fake-kind", choices=("tracking", "constant"), default="tracking",
                    help="CPU surrogate velocity field of the guided sampler (guidance_common.core.surrogate_sample)")
    ap.add_argument("--record-dir", default=None, help="default: $AG3S_RUN_DIR")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    if args.fake_policy == "recorded" and not args.fake_source_run:
        raise SystemExit("--fake-policy recorded needs --fake-source-run")
    run = args.record_dir or os.environ.get("AG3S_RUN_DIR")
    server, info = build_server(args, record_root=run)
    info["port"] = args.port
    if run:
        p = pathlib.Path(run) / "logs" / "method_server_config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x") as f:
            json.dump(info, f, indent=1, default=str)
    logging.info("method server: %s", json.dumps({"arm": info["arm"]["variant"], "port": args.port}))
    from openpi.serving import websocket_policy_server

    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.policy_seed import describe_rng_owner, describe_xla_determinism

    logging.info("%s", describe_rng_owner(server.vla._policy))
    logging.info("%s", describe_xla_determinism(os.environ.get("XLA_FLAGS")))
    ss.exit_cleanly_on_sigterm()
    try:
        websocket_policy_server.WebsocketPolicyServer(policy=server, host="0.0.0.0", port=args.port,
                                                      metadata=server.metadata).serve_forever()
    finally:
        logging.info("method server stopping: calls=%d fallbacks=%d", server.n_calls, server.n_fallback)
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
