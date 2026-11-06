"""SL-GTTO2 — offline replay of recorded GT-arm frames through the run's own AG3S (CPU, grounding-only).

Used by `tests/benchmark/test_sl_gtto2_versions.py` (skipped when the storage records are absent) and by the diagnosis in
`AG3S/ag3s/docs/handoff/SL-GTTO2.impl.md`. Per call of an episode (`<stem>_calls.jsonl` + `<stem>_geom.npz` of a run under
`results/benchmark/`):

1. the server is built as the run's manifest started it (`server_argv`, + extra flags) with `AdaptersPanda.build_safe_server`
   on a stub policy; the ESDF build is replaced on the instance (cuRobo TSDF / ESDF needs CUDA) and records the
   `CameraDepth` list it would integrate (`replay_cameras`: depth + the TSDF mask);
2. the request is rebuilt from the geometry npz row (`wire_pack.wire_from_geom` -> `wire.unpack_camera_observations`), the
   GT code images from its `segmentation_geom` with the episode's recorded code table (`oracle_codes.body_code_table`);
3. the grasp state is the recorded one (`grasp.state`; `AG3S.attach` on the recorded attach call with the recorded
   parent / hold link and the sticky reach pass), then `process_multi_debug` (phase = the recorded contact phase).
Approximation: the attach snapshot / held filter follow the replayed TargetConfirm, so after an attach the fused code-1
count can differ from the record by a few points (SafeLIBERO t1e1: 12 vs 7).
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import Optional

import numpy as np

STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage")) / "results/benchmark"
GT_LIBERO = "20261104T235058_89b424595b57_gtto-gt-libero9-vla_v11ab-AG3S-commonto-gt-C"
GT_SAFE_T1 = "20261105T000549_89b424595b57_gtto-gt-safelibero11-t1-LI-vla_v11ab-AG3S-commonto-gt-C"
GT_SAFE_T3 = "20261105T003449_89b424595b57_gtto-gt-safelibero11-t3-LI-vla_v11ab-AG3S-commonto-gt-C"


def available(run: str, stem: str) -> bool:
    ep = STORAGE / run / "episodes"
    return (ep / f"{stem}_calls.jsonl").is_file() and (ep / f"{stem}_geom.npz").is_file()


def _server_argv(manifest: dict) -> list[str]:
    stack = [manifest]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            if "server_argv" in d:
                return list(d["server_argv"])
            stack.extend(d.values())
        elif isinstance(d, list):
            stack.extend(d)
    raise KeyError("server_argv not in the run manifest")


class Episode:
    def __init__(self, run: str, stem: str, extra=()):
        from AG3S.adapters.panda import profile as pf
        from AG3S.adapters.panda import server as ps
        from AG3S.trajopt import gpu_seeds
        from benchmark.adapters.libero_panda import oracle_codes as oc

        self.run, self.stem = STORAGE / run, stem
        ep = self.run / "episodes"
        self.rows = [json.loads(line) for line in open(ep / f"{stem}_calls.jsonl")]
        with np.load(str(ep / f"{stem}_geom.npz"), allow_pickle=False) as z:
            self.geom = {k: z[k] for k in z.files}
        legend = None
        for line in open(self.run / "episodes.jsonl"):
            d = json.loads(line)
            if stem.endswith(f"t{d['task_id']}_e{d['episode_idx']}"):
                legend = {k: int(v) for k, v in d["oracle_seg"]["codes"].items()}
        names = [str(b) for b in self.geom["model_body_names"]]
        self.target = next(k for k, v in legend.items() if v == oc.CODE_TARGET)
        objects = [k for k, v in sorted(legend.items(), key=lambda kv: kv[1]) if v != oc.CODE_TARGET]
        codes, got, _ = oc.body_code_table(names, np.arange(len(names)), [], self.target, objects)
        assert {k: int(v) for k, v in got.items()} == legend, "recorded code table not reproduced"
        self.lut = oc.geom_lut(codes, self.geom["model_geom_bodyid"])

        os.environ.setdefault("AG3S_PANDA_CONVERSION", "C")
        real = gpu_seeds.available
        gpu_seeds.available = lambda: (True, "")
        try:
            args = pf.parse_v11ab_args(argv=_server_argv(json.loads((self.run / "run_manifest.json").read_text()))[3:]
                                       + list(extra))

            class _Stub:
                def infer(self, obs, **kw):  # pragma: no cover
                    raise AssertionError

            served, _ = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
        finally:
            gpu_seeds.available = real
        self.ag3s = ag = served._policy.ag3s
        self.replay_cameras: list = []

        def _no_esdf(depth_cameras, *_a, **_k):
            self.replay_cameras = list(depth_cameras)
            return None, ["SL-GTTO2 replay: grounding only (no ESDF on CPU)"]
        ag._build_esdf = _no_esdf

    def frame(self, k: int):
        from AG3S.adapters.panda import model as pm
        from AG3S.ag3s.experiments.sources.mujoco_source import CAMERA_MOUNTS
        from AG3S.trajopt import wire
        from benchmark.adapters.libero_panda import oracle_codes as oc
        from benchmark.adapters.libero_panda import wire_pack

        gi = list(self.geom["obs_index"]).index(self.rows[k]["obs_index"])
        _, scene = wire.strip_request(dict(wire_pack.wire_from_geom(self.geom, gi)))
        obs = wire.unpack_camera_observations(scene, robot_model=self.ag3s.robot_model,
                                              attention={c: np.zeros((16, 16), np.float32) for c in scene["cameras"]})
        q7, fingers = pm.split_robot_state(np.asarray(obs[0].robot_state, np.float64))
        obs = [dataclasses.replace(o, robot_state=q7.copy()) for o in obs]
        labels = {str(CAMERA_MOUNTS.get(c, (None, c))[1]): oc.code_image(self.geom[f"segmentation_geom/{c}"][gi], self.lut)
                  for c in scene["cameras"]}
        return obs, fingers, {"labels": labels, "meta": {"target": self.target}}

    def step(self, k: int, *, grasp: bool = True):
        """Replay call `k` (in order). Returns `(metrics, debug, seg)`."""
        ag, r = self.ag3s, self.rows[k]
        obs, fingers, seg = self.frame(k)
        ag.set_finger_joints(fingers)
        g = r.get("grasp") or {}
        state = g.get("state") if grasp else "searching"
        if grasp and g.get("attach"):
            sticky = (g.get("evidence") or {}).get("reach_sticky")
            ag.attach(ag.manipulated.geometry, robot_state=np.asarray(obs[0].robot_state, np.float64),
                      parent_link=g["parent_link"], label="manipulated", reach=0.12, hold_link=g["hold_link"],
                      allowed_contact_links=("panda_leftfinger", "panda_rightfinger", "panda_hand"),
                      **({"reach_sticky": sticky} if sticky and sticky.get("same_object") else {}))
        if grasp and g.get("detach"):
            ag.detach()
        ag.set_grasp_active(state in ("closing", "held"))
        ag.set_grasp_latched(state in ("latched", "closing"))
        ag.set_subtask(None)
        ag.set_placed(state == "placed")
        ag.set_oracle_segmentation(seg)
        phase = (((r.get("to") or {}).get("contact_phase") or {}).get("applied") or "approach") if grasp else "approach"
        cs, dbg = ag.process_multi_debug(obs, phase=phase, active_manipulators=["left"])
        return cs.metrics, dbg, seg

    def tsdf_code1(self, seg) -> tuple[int, int]:
        """`(code-1 pixels with depth, code-1 pixels the TSDF would integrate)` of the last replayed call."""
        n = kept = 0
        for cam in self.replay_cameras:
            lab = np.asarray(seg["labels"][cam.name]) == 1
            valid = np.asarray(cam.depth) > 0
            mask = (np.zeros(lab.shape, bool) if cam.robot_mask is None
                    else np.asarray(cam.robot_mask, bool).reshape(lab.shape))
            n += int((lab & valid).sum())
            kept += int((lab & valid & ~mask).sum())
        return n, kept


def admissibility(ep: Episode) -> Optional[list]:
    adm = ep.ag3s._target_confirm.last_admissibility
    return None if adm is None else [a.record() for a in adm]
