"""AG3S-GEO-GRASP — one request of the geometric grasp: accumulate, (re)generate when needed, keep the choice.

`GeoGraspPlanner.step` is what `SafePolicy` calls once per request (and the offline study per recorded request):

1. merge the target's observed points into `TargetPointMemory` (before the grasp, same ATS identity);
2. in the pre-grasp phases (``latched`` / ``closing``): full candidate generation when there is no valid tracked
   grasp or every `refresh_every` requests; otherwise only a re-check of the tracked grasp against this request's
   observation (collision with the newest scene and completed target, IK) — the cheap path;
3. update the task state; if the re-check invalidated the tracked grasp, generate once more in the same request.

The result is the selected grasp (or None and the reason). Nothing here touches the trajectory optimizer.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

from AG3S.ag3s.grasp.candidates import CandidateSet, GeoGraspConfig, generate_candidates
from AG3S.ag3s.grasp.gripper import ParallelGripperSpec
from AG3S.ag3s.grasp.memory import ObstacleMemory, TargetPointMemory
from AG3S.ag3s.grasp.task_state import ACTIVE_PHASES, GraspTaskState, SelectedGrasp


class GeoGraspPlanner:
    def __init__(self, gripper: ParallelGripperSpec, cfg: GeoGraspConfig = GeoGraspConfig(), *,
                 ik_fn: Optional[Callable] = None, memory: Optional[TargetPointMemory] = None,
                 arm_fn: Optional[Callable] = None):
        self.gripper = gripper
        self.cfg = cfg
        self.ik_fn = ik_fn
        #: obstacle points inside the arm links at a joint configuration (`arm_shape.ArmChecker`; `cfg.arm_check`).
        self.arm_fn = arm_fn
        self.obstacles = ObstacleMemory(cfg.obstacle_voxel, cfg.obstacle_max_points) if cfg.obstacle_memory else None
        self.memory = memory if memory is not None else TargetPointMemory()
        self.state = GraspTaskState(gripper, cfg)
        self._since_gen = 10 ** 9
        self.last_set: Optional[CandidateSet] = None

    def reset(self) -> None:
        self.memory.reset()
        if self.obstacles is not None:
            self.obstacles.reset()
        self.state.reset()
        self._since_gen = 10 ** 9
        self.last_set = None

    @property
    def selected(self) -> Optional[SelectedGrasp]:
        return self.state.selected

    def step(self, *, phase: str, target_id: Optional[int], target_state: Optional[str],
             target_points: Optional[np.ndarray], target_centroid: Optional[np.ndarray],
             scene_points: Optional[np.ndarray], vla_pose: Optional[np.ndarray], ik_seed: Optional[np.ndarray],
             observed_time: Optional[float] = None, allowed_links: tuple[str, ...] = (),
             update_memory: bool = True, field_fn: Optional[Callable] = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {}
        if update_memory:
            pre = phase in ("searching",) + ACTIVE_PHASES and target_id is not None
            rec["memory"] = (self.memory.update(target_id, target_points, target_centroid,
                                                observed=target_state == "visible")
                             if pre else self.memory.update(None, None, None))
        self._field_fn = field_fn
        if self.obstacles is not None and phase in ("searching",) + ACTIVE_PHASES:
            rec["obstacles"] = self.obstacles.update(scene_points)
            scene_points = self.obstacles.without(self.memory.points)
            rec["obstacles"]["n_used"] = int(len(scene_points))
        cands: Optional[CandidateSet] = None
        gen = "none"
        if phase in ACTIVE_PHASES and target_id is not None and len(self.memory.points):
            tracked = self.state.selected is not None and self.state.target_id == target_id
            full = (not tracked) or self._since_gen + 1 >= self.cfg.refresh_every
            cands = self._generate(scene_points, vla_pose, ik_seed, recheck_only=not full)
            gen = "full" if full else "recheck"
        st = self.state.update(phase=phase, target_id=target_id, target_state=target_state, candidates=cands,
                               observed_time=observed_time, allowed_links=allowed_links, ik_fn=self.ik_fn,
                               ik_seed=ik_seed, vla_pose=vla_pose, arm_fn=self.arm_fn)
        if gen == "recheck" and self.state.selected is None and phase in ACTIVE_PHASES:
            cands = self._generate(scene_points, vla_pose, ik_seed, recheck_only=False)
            gen = "recheck+full"
            st = self.state.update(phase=phase, target_id=target_id, target_state=target_state, candidates=cands,
                                   observed_time=observed_time, allowed_links=allowed_links, ik_fn=self.ik_fn,
                                   ik_seed=ik_seed, same_request=True, vla_pose=vla_pose, arm_fn=self.arm_fn)
        if gen in ("full", "recheck+full"):
            self._since_gen = 0
        elif gen == "recheck":
            self._since_gen += 1
        self.last_set = cands
        rec.update(generation=gen, state=st, candidate_stats=None if cands is None else cands.stats,
                   n_candidates=0 if cands is None else len(cands.candidates),
                   compute_ms=round((time.perf_counter() - t0) * 1000.0, 2))
        return rec

    def _generate(self, scene_points, vla_pose, ik_seed, *, recheck_only: bool) -> CandidateSet:
        return generate_candidates(self.memory.points, scene_points, self.gripper, self.cfg, vla_pose=vla_pose,
                                   ik_fn=self.ik_fn, ik_seed=ik_seed, recheck_only=recheck_only, arm_fn=self.arm_fn,
                                   field_fn=getattr(self, "_field_fn", None))


__all__ = ["GeoGraspPlanner"]
