"""T85 — the MuJoCo ground-truth task stage a simulator client sends with every request (analysis only).

`pi05_infer --safe-gt-stage` -> `SafeRemoteClient(gt_stage=GtStageSource(...))` -> `ag3s/gt_stage` in the request ->
`serve_safe --hsr-gt-stage` feeds it to `SafePolicy._hsr5_start` in place of the classifier's `result["hsr5"]`.
This is not a system component: the server is told the simulator's truth. It exists to take the stage classifier's
error out of an ATS / rule experiment (`handoff/T85.task.md`).

The definition is T77's (`t77_labels`: `gt_held`, `phase5_sequence`, `SceneProbe`) and is not copied here:

    held     = both fingers of the grasping hand touch the target (MJCF finger bodies) ∧ target centre >= 20 mm above
               request 0 (base frame)
    place    : held ∧ target centre inside the destination crate's floor xy range
    carry    : held ∧ not place
    retreat  : held at an earlier request, not held now
    grasp    : never held yet ∧ grasping-hand palm site - target centre <= 60 mm
    approach : never held yet ∧ > 60 mm

The measurement is the control loop's `qpos` (and mocap) at the moment of the request, run through a private `MjData`
(`SceneProbe.live`), so the loop's own `MjData` is never forwarded or modified.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["GtStageSource"]


class GtStageSource:
    """One run's causal GT stage. `next_stage()` is called once per request, `reset()` at the first request of an
    episode (`pack_request(reset=True)`)."""

    def __init__(self, scene, *, side: str, target_body: Optional[str] = None):
        from AG3S.ag3s.experiments.studies import t77_labels as tl

        if side not in ("left", "right"):
            raise ValueError(f"--safe-gt-stage: the grasping hand must be left or right, got {side!r}")
        model, data = getattr(scene, "model", None), getattr(scene, "data", None)
        if model is None or data is None:
            raise ValueError("--safe-gt-stage needs a scene attached to the running MuJoCo model/data")
        self.side = side
        self.target_body = target_body or tl.TARGET_BODY
        self._data = data
        self._probe = tl.SceneProbe.live(model, data, target_body=self.target_body)
        self._phase = tl.CausalPhase5()
        self.last: dict = {}

    def reset(self) -> None:
        self._phase.reset()
        self.last = {}

    def next_stage(self) -> Optional[str]:
        """This request's stage (one of `t77_labels.PHASES5`), None when the scene could not be measured."""
        mes = self._probe.measure_live(self._data, self.side)
        stage = self._phase.update(mes)
        self.last = {"stage": stage, "side": self.side, "target": self.target_body,
                     **{k: mes[k] for k in ("fingers_touch", "palm_dist_m", "in_basket") if k in mes}}
        return stage
