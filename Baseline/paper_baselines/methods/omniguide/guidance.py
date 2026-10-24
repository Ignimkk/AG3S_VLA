"""OmniGuide [19] collision (repeller) + target (attractor) guidance as an openpi ``guidance_fn`` (pure JAX).

Ported from ``OmniGuide-Code/omniguide-openpi@663f1cb2`` (Apache-2.0), ``src/openpi/models_pytorch/pi0_pytorch.py``:

* collision cost  (``_compute_collision_cost``, :1008-1028)::

      cost_col = sum_k w_ee * 1[sdf(p_k) <= thr_ee] * log(max(sdf(p_k), 1e-5))
               + sum_k sum_w w_wr * 1[sdf(q_kw) <= thr_wr] * log(max(sdf(q_kw), 1e-5))

  ``p_k`` = EE tip along the decoded chunk, ``q_kw`` = 4 wrist probe points (``_wrist_probe``, :193-196);
* attraction cost (``_compute_vlm_pointer_cost``, :1090-1104): ``-sum_{k < num_guided} ||p_k - centroid(target)||^2``;
* update (:1557-1583)::

      guide = d(cost)/d(x_t) ; guide <- guide / ||guide||_row * min(||guide||_row, max_guide_length)
      x_t   <- x_t + dt * v_t + factor * guide             (cost is *maximised*)

* freeze zone (``_update_freeze_zone``, :956-1002): no guidance while the EE tip is within ``threshold`` of the target
  points (enter) until it leaves ``exit_threshold`` (hysteresis kept across calls; :meth:`reset` clears it);
* EE clearance (``_filter_point_cloud_for_collision``, :346-400): points within 0.05 m (gripper open) / 0.10 m
  (closed) of the tip are dropped from the collision cloud (absorbs the held object).

Parameters are ``config/guidance/vlm_pointer_collision_sim.yaml`` / ``collision.yaml`` (factor 0.4, max_guide_length
1.0, num_guided 10, thresholds 0.2 / 0.1 m, weights 0.1 / 0.01, freeze 0.1 m) -- also the values [9] (NUDGE) reports
for its OmniGuide-on-LIBERO baseline.

Target memory (CLOSED-LOOP-FIX, ``target_cache="episode"``, default): the original queries the VLM pointer + SAM2
**once per subtask** and keeps the segmented points (``_apply_vlm_sam2_filter``, :462-495, ``cache_stale`` only when
the subtask index changes; with ``subtask_monitor.enabled: false`` -- the sim config -- that is once per episode, re-
queried only while the cache is empty). The freeze zone (:956-1002) and the attractor (:1097-1104) read those cached
points, and the exit test uses the cloud cached **at entry** (``cached_task_relevant_pcd``). A per-call target that
moves with the gripper (the held bowl) or is a wrong detection keeps the freeze zone on for the rest of the episode
(V3 smoke: 38 / 56 calls skipped); ``target_cache="call"`` reproduces that older behaviour.

Fixed structure (CLOSED-LOOP-FIX): points inside the EE clearance are moved to ``geometry.FAR`` instead of dropped and
the collision / attraction terms are weighted by 0 / 1 constants instead of being traced conditionally, so the jaxpr
(and its compile) is the same on every call of a scene padded to a fixed size.
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines.methods.guidance_common.core import GuidanceBase
from Baseline.paper_baselines.methods.guidance_common.geometry import FAR, signed_distance
from Baseline.paper_baselines.methods.guidance_common.scene import T_HAND_GRIP, GuidanceScene

#: ``_wrist_probe`` (pi0_pytorch.py:193-196), homogeneous points in the wrist frame; we use the panda_hand frame.
WRIST_PROBE = np.array([[0, 0, -0.05], [0.04, 0, 0.0], [0, 0.04, 0.0], [0, -0.04, 0.0]], np.float64)


class OmniGuideGuidance(GuidanceBase):
    name = "omniguide"
    paper = {"cite": "[19]", "title": "OmniGuide", "arxiv": "2603.10052v1"}
    source = {"repo": "https://github.com/OmniGuide-Code/omniguide-openpi", "sha": "663f1cb2021282cf673fdfd737800b04ba80c893",
              "umbrella": "https://github.com/daniilidis-group/OmniGuide@d649d69849bfc0c56bc822cdbe1f5f8a9fad2706",
              "license": "Apache-2.0 (omniguide-openpi); umbrella repo has no LICENSE file",
              "files": ["src/openpi/models_pytorch/pi0_pytorch.py", "config/guidance/vlm_pointer_collision_sim.yaml",
                        "config/guidance/collision.yaml", "collision/esdf.py"]}

    @classmethod
    def default_params(cls) -> dict:
        return {"guidance_types": ("collision", "attract"), "factor": 0.4, "max_guide_length": 1.0, "num_guided": 10,
                "ee_sdf_threshold": 0.2, "ee_cost_weight": 0.1, "wrist_sdf_threshold": 0.1, "wrist_cost_weight": 0.01,
                "ee_clearance_open": 0.05, "ee_clearance_closed": 0.1, "gripper_closed_opening_m": 0.02,
                "freeze_zone": True, "freeze_threshold": 0.1, "freeze_exit_threshold": 0.1, "sdf_cap": 0.2,
                "target_cache": "episode"}

    TARGET_CACHE = ("episode", "call")

    def __init__(self, **kw):
        super().__init__(**kw)
        if self.params["target_cache"] not in self.TARGET_CACHE:
            raise ValueError(f"target_cache must be one of {self.TARGET_CACHE}")
        self._frozen = False
        self._target = None  # cached_sam2_relevant_points
        self._freeze_pts = None  # cached_task_relevant_pcd (cloud at freeze-zone entry)

    def reset(self) -> None:
        super().reset()
        self._frozen = False
        self._target, self._freeze_pts = None, None

    def _target_points(self, scene: GuidanceScene):
        """The attractor / freeze-zone points of this call: episode cache (original) or this call's detection."""
        cur = scene.target_points if scene.target_points is not None and len(scene.target_points) else None
        if self.params["target_cache"] == "call":
            return cur, "call"
        if self._target is None and cur is not None:  # query while the cache is empty (_apply_vlm_sam2_filter)
            self._target = np.asarray(cur, np.float64).copy()
            return self._target, "cached_now"
        return self._target, ("cached" if self._target is not None else "none")

    def _update_freeze_zone(self, tip: np.ndarray, pts) -> dict:
        """``_update_freeze_zone`` (pi0_pytorch.py:956-1002): enter below ``threshold`` of the relevant points,
        exit when the distance to the cloud cached at entry exceeds ``exit_threshold``."""
        P = self.params
        if pts is None or not len(pts):
            return {"freeze_min_dist_m": None}
        dmin = float(np.min(np.linalg.norm(pts - tip[None], axis=1)))
        if not self._frozen and dmin < P["freeze_threshold"]:
            self._frozen, self._freeze_pts = True, np.asarray(pts, np.float64).copy()
        elif self._frozen and self._freeze_pts is not None:
            d_c = float(np.min(np.linalg.norm(self._freeze_pts - tip[None], axis=1)))
            if d_c > P["freeze_exit_threshold"]:
                self._frozen, self._freeze_pts = False, None
            return {"freeze_min_dist_m": dmin, "freeze_cached_dist_m": d_c}
        return {"freeze_min_dist_m": dmin}

    def _prepare(self, scene: GuidanceScene, obs, norm) -> dict:
        P = self.params
        tip = scene.ee_pos
        closed = float((obs.extras or {}).get("finger_opening_m", 0.04)) < P["gripper_closed_opening_m"]
        clear = P["ee_clearance_closed"] if closed else P["ee_clearance_open"]
        o = scene.obstacles
        n_clear = 0
        if len(o.points):  # EE clearance exclusion only applies to the point cloud (as in OmniGuide)
            o = type(o)(**{**o.__dict__})
            near = np.linalg.norm(o.points - tip[None], axis=1) <= clear
            n_clear = int(np.count_nonzero(near & np.all(np.abs(o.points) < 0.5 * FAR, axis=1)))
            o.points = np.where(near[:, None], FAR, o.points)  # moved to FAR, not dropped: fixed shape
        real = (not o.empty) and (len(o.sphere_c) + len(o.box_c) + len(o.cyl_c) > 0
                                  or bool(np.any(np.all(np.abs(o.points) < 0.5 * FAR, axis=1))))
        ctx = {"obstacles": o.padded().as_jax(), "ee_clearance_m": clear, "gripper_closed": closed,
               "n_clearance_dropped": n_clear}
        # wrist probe offsets relative to the tip (orientation held along the chunk)
        T_hand = scene.ee_pose @ np.linalg.inv(T_HAND_GRIP)
        ctx["wrist_offsets"] = (WRIST_PROBE @ T_hand[:3, :3].T + T_hand[:3, 3]) - tip[None]
        types = set(P["guidance_types"])
        tpts, tsrc = self._target_points(scene)
        c = None if tpts is None else tpts.mean(0)
        ctx["attract"] = ("attract" in types) and c is not None
        ctx["collision"] = "collision" in types and real
        ctx["centroid"] = np.zeros(3) if c is None else c
        # 0 / 1 weights as (1,) arrays: jaxpr constants, not literals -> the same structure whichever term is on
        ctx["w_collision"] = np.array([1.0 if ctx["collision"] else 0.0], np.float32)
        ctx["w_attract"] = np.array([1.0 if ctx["attract"] else 0.0], np.float32)
        ctx["target_source"] = tsrc
        ctx["n_target_points"] = 0 if tpts is None else int(len(tpts))
        if P["freeze_zone"]:
            ctx.update(self._update_freeze_zone(tip, tpts))
        ctx["frozen"] = self._frozen
        ctx["skip"] = self._frozen or not (ctx["attract"] or ctx["collision"])
        return ctx

    def cost(self, x0, ctx: dict, path):
        """OmniGuide cost (to maximise) of one normalized chunk ``x0 (H, D)``."""
        import jax.numpy as jnp

        P = self.params
        p = path(self.raw_xyz(x0, ctx["norm"]))  # (H, 3) tip positions (base)
        total = jnp.zeros((), x0.dtype)
        w_col = jnp.asarray(ctx["w_collision"], x0.dtype)[0]
        w_att = jnp.asarray(ctx["w_attract"], x0.dtype)[0]
        sdf = jnp.minimum(signed_distance(p, ctx["obstacles"]), P["sdf_cap"] + 1.0)
        wr = p[:, None, :] + jnp.asarray(ctx["wrist_offsets"], p.dtype)[None]
        sdf_w = signed_distance(wr, ctx["obstacles"])
        total = total + w_col * (P["ee_cost_weight"] * (sdf <= P["ee_sdf_threshold"])
                                 * jnp.log(jnp.maximum(sdf, 1e-5))).sum()
        total = total + w_col * (P["wrist_cost_weight"] * (sdf_w <= P["wrist_sdf_threshold"])
                                 * jnp.log(jnp.maximum(sdf_w, 1e-5))).sum()
        n = int(P["num_guided"])
        total = total - w_att * jnp.sum((p[:n] - jnp.asarray(ctx["centroid"], p.dtype)[None]) ** 2)
        return total

    def _guide_one(self, x_next, x0, t_next, ctx, path):
        import jax
        import jax.numpy as jnp

        P = self.params
        g = jax.grad(lambda x: self.cost(x, ctx, path))(x0)
        n = jnp.sqrt(jnp.sum(g * g, axis=-1, keepdims=True))
        g = g / (n + 1e-8) * jnp.minimum(n, P["max_guide_length"])
        return x_next + P["factor"] * g


__all__ = ["OmniGuideGuidance", "WRIST_PROBE"]
