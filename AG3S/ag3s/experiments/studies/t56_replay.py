"""T56 — the MuJoCo replay half: T54's serving replay plus the geometry the self scores need.

Every grounding call is T54's (`t54_replay.run_frames` → `ground_arm` → `t47b_replay._ground`, the live
serving path; conditions ``current`` / ``hold`` and profiles ``t52`` / ``c1a`` exactly as T54). The only
addition is a **capturing** ``ground_fn`` (`Capture`): it calls `t47b_replay._ground` unchanged and keeps,
per call, what a self score is computed from — the frame's filtered cloud (deduplicated by content),
the pipeline's support mask, the chosen cluster as cloud indices (`t47b_metrics.membership_mask`, the
same call T54 rows use) and, per frame, the robot geometry of the posed MuJoCo scene:

* fingertip pads of both hands (the finger body's largest collision geom — `t51_selflabel.pad_geom`,
  the T51 self-label definition; the two fingers of a hand are used as an unordered pair, so the MJCF /
  URDF left-finger name swap does not matter there),
* the finger-body rotation of each hand (hand frame orientation for S2),
* the grasp parent-link origin of each hand for the offline latch's ``reach`` — the serving parent links
  are URDF names (`safe_policy` ``grasp_links``: left ``ee_finger_l1``, right ``ee_finger_r1``); in the
  MJCF the **left names are swapped** (T43 OC: URDF ``ee_finger_l1`` = MJCF ``ee_finger_l2``), so the
  MJCF bodies read are ``ee_finger_l2`` / ``ee_finger_r1`` (`PARENT_LINK_MJCF`),
* the physical contact state of the target (`t51_selflabel.contact_state`: ``both_contact`` hand, ``held``
  hand) — S0's event (sim only), and GT diagnostics (target body position).

GT point labels of each stored cloud are the replay's own (`fr["label_of"]`, the labeller every T47–T54
row uses) — used only for the GT checks (HELD-event label error), never by a self score.

Outputs per job (record, condition, candidate kind, batch): ``rows`` jsonl (the T54 row fields of every
call, minus the per-cluster list) and a geometry ``npz`` (`save_npz` / `load_npz`).

CPU, osmesa. Run by the verifier; the implementer only smoke-runs a few frames.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

HANDS = ("left", "right")
#: Serving grasp parent links (URDF) → the MJCF body holding that link (T43 OC left swap).
PARENT_LINK_URDF = {"left": "ee_finger_l1", "right": "ee_finger_r1"}
PARENT_LINK_MJCF = {"left": "ee_finger_l2", "right": "ee_finger_r1"}
#: Row keys dropped from the stored rows (large, not read by T56).
DROP_ROW_KEYS = ("clusters", "ms_profile")
MEMBERSHIP_CODE = {"none": 0, "indices": 1, "proximity": 2, "empty": 3}


def _digest(pts: np.ndarray, support: np.ndarray) -> str:
    h = hashlib.sha1()
    a = np.ascontiguousarray(np.asarray(pts, np.float64))
    h.update(np.int64(a.shape[0]).tobytes())
    h.update(a.tobytes())
    h.update(np.ascontiguousarray(np.asarray(support, bool)).tobytes())
    return h.hexdigest()


def robot_frame(scene, target: str, cache: dict) -> dict:
    """Robot / contact geometry of the posed scene (module doc). `cache` keeps body / geom ids."""
    from AG3S.ag3s.experiments.studies import t51_selflabel as sl

    model, data = scene.model, scene.data
    if not cache:
        cache["tb"] = sl._bid(model, target)
        cache["fingers"] = {h: tuple(sl._bid(model, n) for n in sl.FINGER_BODIES[h]) for h in HANDS}
        cache["support"] = {sl._bid(model, n) for n in sl.SUPPORT_BODIES}
        cache["pads"] = {h: tuple(sl.pad_geom(model, b) for b in cache["fingers"][h]) for h in HANDS}
        cache["pad_half"] = {h: float(min(np.min(model.geom_size[g]) for g in cache["pads"][h]))
                             for h in HANDS}
        cache["parent"] = {h: sl._bid(model, PARENT_LINK_MJCF[h]) for h in HANDS}
    touch = sl.touching_bodies(model, data, cache["tb"])
    bh, hh = sl.contact_state(touch, cache["fingers"], cache["support"])
    code = {None: -1, "left": 0, "right": 1}
    return {
        "T_world_base": np.asarray(scene.body_pose("base"), np.float64),
        "pads_world": np.asarray([[data.geom_xpos[g] for g in cache["pads"][h]] for h in HANDS],
                                 np.float64),
        "finger_R_world": np.asarray([data.xmat[cache["fingers"][h][0]].reshape(3, 3)
                                      for h in HANDS], np.float64),
        "parent_world": np.asarray([data.xpos[cache["parent"][h]] for h in HANDS], np.float64),
        "both_hand": code[bh], "held_hand": code[hh],
        "target_world": np.asarray(data.xpos[cache["tb"]], np.float64),
    }


class Capture:
    """``ground_fn`` for `t54_replay.run_frames` that keeps the self-score geometry (module doc).

    `run_frames` calls ``ground_fn`` once per (frame, candidate) with the candidates in `names` order,
    so the k-th call belongs to ``names[k % len(names)]``; `check_order` verifies that against the rows.
    """

    def __init__(self, names: Sequence[str], target: str, *, membership_radius: float = 0.005,
                 ground: Optional[Callable] = None):
        self.names = list(names)
        self.target = target
        self.membership_radius = float(membership_radius)
        self._ground = ground
        self.k = 0
        self.order: list[tuple[int, str]] = []
        self.robot: dict[int, dict] = {}
        self.step: dict[int, Any] = {}
        self._cache: dict = {}
        self.cloud_ids: dict[str, int] = {}
        self.clouds: list[dict] = []
        self.calls: dict[tuple[int, str], dict] = {}

    def __call__(self, ag, fr: dict, att: dict):
        from AG3S.ag3s.experiments.studies import t47b_metrics as bm
        from AG3S.ag3s.experiments.studies import t47b_replay as br

        ground = self._ground or br._ground
        cs, debug, ms = ground(ag, fr, att)
        name = self.names[self.k % len(self.names)]
        self.k += 1
        i = int(fr["i"])
        self.order.append((i, name))
        if i not in self.robot and fr.get("scene") is not None:
            self.robot[i] = robot_frame(fr["scene"], self.target, self._cache)
            self.step[i] = fr["step"]
        cloud = debug.get("filtered_cloud")
        pts = np.zeros((0, 3)) if cloud is None else np.asarray(cloud.points, np.float64)
        sup = debug.get("support_mask")
        sup = np.zeros(len(pts), bool) if sup is None else np.asarray(sup, bool)
        dg = _digest(pts, sup)
        cid = self.cloud_ids.get(dg)
        if cid is None:
            cid = len(self.clouds)
            self.cloud_ids[dg] = cid
            labels = np.asarray(fr["label_of"](pts), np.int8) if len(pts) else np.zeros(0, np.int8)
            self.clouds.append({"pts": pts.astype(np.float32), "support": sup, "labels": labels,
                                "digest": dg, "frame": i})
        has = bool(cs.has_target)
        chosen = np.zeros(len(pts), bool)
        member = "none"
        tpts = np.zeros((0, 3), np.float32)
        if has:
            tp = np.asarray(cs.target.points, np.float64)
            chosen, member = bm.membership_mask(pts, tp, np.asarray(cs.target.point_indices),
                                                radius=self.membership_radius)
            if member != "indices":
                tpts = tp.astype(np.float32)
        self.calls[(i, name)] = {"cloud": cid, "has": has, "member": member,
                                 "idx": np.flatnonzero(chosen).astype(np.int32), "tpts": tpts}
        return cs, debug, ms

    def check_order(self, rows: Mapping[str, Sequence[dict]]) -> None:
        """The calls seen == the rows written, candidate for candidate, frame for frame."""
        seen = sorted(self.order)
        want = sorted((int(r["frame"]), str(r["arm"])) for v in rows.values() for r in v)
        if seen != want:
            raise RuntimeError("capture order does not match the replay rows")
        for v in rows.values():
            for r in v:
                c = self.calls[(int(r["frame"]), str(r["arm"]))]
                if bool(r["has_target"]) != c["has"]:
                    raise RuntimeError(f"capture / row has_target differ at {r['frame']} {r['arm']}")


# ------------------------------------------------------------------------------- storage


def arrays_of(cap: Capture, frames: Sequence[int], names: Sequence[str]) -> dict[str, np.ndarray]:
    """The geometry npz payload of one job (module doc)."""
    frames = [int(i) for i in frames]
    names = list(names)
    n_c, n_f = len(names), len(frames)
    off = [0]
    for c in cap.clouds:
        off.append(off[-1] + len(c["pts"]))
    cat = (lambda key, shape, dt: np.concatenate([c[key] for c in cap.clouds]).astype(dt)
           if cap.clouds else np.zeros(shape, dt))
    call_cloud = np.full((n_c, n_f), -1, np.int32)
    call_has = np.zeros((n_c, n_f), bool)
    call_member = np.zeros((n_c, n_f), np.int8)
    idx_parts, idx_off, tp_parts, tp_off = [], [0], [], [0]
    for a, nm in enumerate(names):
        for j, i in enumerate(frames):
            c = cap.calls.get((i, nm))
            if c is None:
                idx_off.append(idx_off[-1])
                tp_off.append(tp_off[-1])
                continue
            call_cloud[a, j] = c["cloud"]
            call_has[a, j] = c["has"]
            call_member[a, j] = MEMBERSHIP_CODE[c["member"]]
            idx_parts.append(c["idx"])
            idx_off.append(idx_off[-1] + len(c["idx"]))
            tp_parts.append(c["tpts"])
            tp_off.append(tp_off[-1] + len(c["tpts"]))
    rb = [cap.robot[i] for i in frames]
    steps = [cap.step[i] for i in frames]
    st = np.asarray([np.asarray(s.state, np.float64) for s in steps])
    act = [np.asarray(s.actions, np.float64) for s in steps]
    h = min(len(a) for a in act) if act else 0
    return {
        "frames": np.asarray(frames, np.int64),
        "t_step": np.asarray([int(s.t_step) for s in steps], np.int64),
        "cand": np.asarray(names),
        "cloud_pts": cat("pts", (0, 3), np.float32),
        "cloud_support": cat("support", (0,), bool),
        "cloud_labels": cat("labels", (0,), np.int8),
        "cloud_off": np.asarray(off, np.int64),
        "cloud_digest": np.asarray([c["digest"] for c in cap.clouds]),
        "cloud_frame": np.asarray([c["frame"] for c in cap.clouds], np.int64),
        "call_cloud": call_cloud, "call_has": call_has, "call_member": call_member,
        "chosen_idx": (np.concatenate(idx_parts).astype(np.int32) if idx_parts
                       else np.zeros(0, np.int32)),
        "chosen_off": np.asarray(idx_off, np.int64),
        "tpts": np.concatenate(tp_parts).astype(np.float32) if tp_parts else np.zeros((0, 3), np.float32),
        "tpts_off": np.asarray(tp_off, np.int64),
        "T_world_base": np.asarray([r["T_world_base"] for r in rb]),
        "pads_world": np.asarray([r["pads_world"] for r in rb]),
        "finger_R_world": np.asarray([r["finger_R_world"] for r in rb]),
        "parent_world": np.asarray([r["parent_world"] for r in rb]),
        "both_hand": np.asarray([r["both_hand"] for r in rb], np.int8),
        "held_hand": np.asarray([r["held_hand"] for r in rb], np.int8),
        "target_world": np.asarray([r["target_world"] for r in rb]),
        "pad_half": np.asarray([cap._cache["pad_half"][hh] for hh in HANDS], np.float64)
        if cap._cache else np.full(2, np.nan),
        "state": st,
        "actions": np.asarray([a[:h] for a in act]) if act else np.zeros((0, 0, 0)),
    }


class Geometry:
    """Read access to one job's npz (or several merged by `merge`)."""

    def __init__(self, z: Mapping[str, np.ndarray]):
        self.z = {k: np.asarray(v) for k, v in z.items()}
        self.frames = [int(i) for i in self.z["frames"]]
        self.fpos = {i: j for j, i in enumerate(self.frames)}
        self.cands = [str(c) for c in self.z["cand"]]
        self.cpos = {c: a for a, c in enumerate(self.cands)}
        self._clouds: dict[int, dict] = {}

    def cloud(self, cid: int) -> dict:
        cid = int(cid)
        c = self._clouds.get(cid)
        if c is None:
            o = self.z["cloud_off"]
            s, e = int(o[cid]), int(o[cid + 1])
            c = {"pts": self.z["cloud_pts"][s:e].astype(np.float64),
                 "support": self.z["cloud_support"][s:e], "labels": self.z["cloud_labels"][s:e]}
            if len(self._clouds) > 256:
                self._clouds.clear()
            self._clouds[cid] = c
        return c

    def call(self, cand: str, frame: int) -> Optional[dict]:
        a, j = self.cpos[cand], self.fpos[int(frame)]
        cid = int(self.z["call_cloud"][a, j])
        if cid < 0:
            return None
        k = a * len(self.frames) + j
        o, t = self.z["chosen_off"], self.z["tpts_off"]
        idx = self.z["chosen_idx"][int(o[k]):int(o[k + 1])].astype(np.int64)
        tp = self.z["tpts"][int(t[k]):int(t[k + 1])].astype(np.float64)
        return {"cloud": cid, "has": bool(self.z["call_has"][a, j]),
                "member": int(self.z["call_member"][a, j]), "idx": idx, "tpts": tp}

    def chosen_points(self, cand: str, frame: int) -> np.ndarray:
        """The chosen cluster's points (cloud[idx]; the stored target points when the membership was
        not by index — an occluded / anchor stand-in geometry)."""
        c = self.call(cand, frame)
        if c is None or not c["has"]:
            return np.zeros((0, 3))
        if c["member"] == MEMBERSHIP_CODE["indices"]:
            return self.cloud(c["cloud"])["pts"][c["idx"]]
        return c["tpts"]

    def frame_cloud(self, frame: int, cand: Optional[str] = None) -> Optional[dict]:
        """The cloud of a frame (of `cand`'s call, else the first stored for that frame)."""
        if cand is not None:
            c = self.call(cand, frame)
            return None if c is None else self.cloud(c["cloud"])
        cf = np.flatnonzero(self.z["cloud_frame"] == int(frame))
        return None if not cf.size else self.cloud(int(cf[0]))


def save_npz(path, arrays: Mapping[str, np.ndarray]) -> None:
    import pathlib

    p = pathlib.Path(path)
    tmp = p.with_name(p.name + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(p)


def load_npz(path) -> Geometry:
    with np.load(path, allow_pickle=False) as z:
        return Geometry({k: z[k] for k in z.files})


ROBOT_KEYS = ("frames", "t_step", "T_world_base", "pads_world", "finger_R_world", "parent_world",
              "both_hand", "held_hand", "target_world", "pad_half", "state", "actions")


def merge(parts: Sequence[Geometry]) -> Geometry:
    """Merge the batches of one (record, condition, kind): candidates concatenated, clouds re-indexed,
    robot arrays checked equal (they come from the same posed scene)."""
    if not parts:
        raise ValueError("nothing to merge")
    base = parts[0].z
    for p in parts[1:]:
        for k in ROBOT_KEYS:
            a, b = np.asarray(base[k]), np.asarray(p.z[k])
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                raise ValueError(f"batches differ in robot array {k}")
    out = {k: base[k] for k in ROBOT_KEYS}
    n_f = len(parts[0].frames)
    pts, sup, lab, off, dig, cfr = [], [], [], [0], [], []
    cand, cc, ch, cm, idx, ioff, tp, toff = [], [], [], [], [], [0], [], [0]
    seen: dict[str, int] = {}
    for p in parts:
        z = p.z
        remap = np.full(len(z["cloud_digest"]), -1, np.int64)
        for c in range(len(z["cloud_digest"])):
            d = str(z["cloud_digest"][c])
            if d in seen:  # the same cloud from another batch (content digest) — stored once
                remap[c] = seen[d]
                continue
            s, e = int(z["cloud_off"][c]), int(z["cloud_off"][c + 1])
            pts.append(z["cloud_pts"][s:e])
            sup.append(z["cloud_support"][s:e])
            lab.append(z["cloud_labels"][s:e])
            off.append(off[-1] + (e - s))
            seen[d] = len(dig)
            remap[c] = len(dig)
            dig.append(d)
            cfr.append(int(z["cloud_frame"][c]))
        cand += [str(c) for c in z["cand"]]
        cl = np.asarray(z["call_cloud"]).astype(np.int64)
        cl = np.where(cl >= 0, remap[np.maximum(cl, 0)], -1).astype(np.int32)
        cc.append(cl)
        ch.append(z["call_has"])
        cm.append(z["call_member"])
        o = z["chosen_off"]
        idx.append(z["chosen_idx"])
        ioff += list(np.asarray(o[1:]) + ioff[-1])
        t = z["tpts_off"]
        tp.append(z["tpts"])
        toff += list(np.asarray(t[1:]) + toff[-1])
    if len(set(cand)) != len(cand):
        raise ValueError("a candidate appears in two batches")
    out.update({
        "cand": np.asarray(cand),
        "cloud_pts": np.concatenate(pts) if pts else np.zeros((0, 3), np.float32),
        "cloud_support": np.concatenate(sup) if sup else np.zeros(0, bool),
        "cloud_labels": np.concatenate(lab) if lab else np.zeros(0, np.int8),
        "cloud_off": np.asarray(off, np.int64), "cloud_digest": np.asarray(dig),
        "cloud_frame": np.asarray(cfr, np.int64),
        "call_cloud": np.concatenate(cc, axis=0), "call_has": np.concatenate(ch, axis=0),
        "call_member": np.concatenate(cm, axis=0),
        "chosen_idx": np.concatenate(idx), "chosen_off": np.asarray(ioff, np.int64),
        "tpts": np.concatenate(tp), "tpts_off": np.asarray(toff, np.int64),
    })
    if out["call_cloud"].shape[1] != n_f:
        raise ValueError("frame count differs between batches")
    return Geometry(out)


# ------------------------------------------------------------------------------- one job


def slim_row(row: Mapping[str, Any]) -> dict:
    return {k: v for k, v in row.items() if k not in DROP_ROW_KEYS}


def run_job(record_path, *, target: str, kind: str, names: Sequence[str], maps: Mapping[str, np.ndarray],
            condition: str, build: Callable[[], Any], freeze: Optional[Sequence[Mapping]] = None,
            frames: Optional[Sequence[int]] = None, label_radius: float = 0.003,
            log: Callable[[str], None] = print, frame_iter=None, specs=None) -> tuple[dict, dict]:
    """``(rows by candidate, geometry arrays)`` of one (record, condition, batch) — `t54_replay.run_frames`
    with the `Capture` ground function. `frame_iter` / `specs` are test hooks (default: the MuJoCo
    replay of the record and the candidate ArmSpecs from `t56_cells.candidate_maps`)."""
    from AG3S.ag3s.experiments.studies import t54_replay as rp

    if specs is None:
        raise ValueError("specs (ArmSpec list) required")
    if frame_iter is None:
        from AG3S.ag3s.experiments.sources.policy_record import POLICY_CAMERA_NAMES, load_run
        from AG3S.ag3s.experiments.studies import t47b_replay as br

        n_f = len(load_run(record_path))
        br._check_sources({s: maps[s] for s in maps}, n_f, len(POLICY_CAMERA_NAMES))
        frame_iter = br._frames(record_path, target=target, frames=frames, label_radius=label_radius,
                                log=log)
    else:
        n_f = None
    cap = Capture([s.name for s in specs], target)
    t0 = time.time()
    fr_list = []

    def tap(it):
        for fr in it:
            fr_list.append(int(fr["i"]))
            yield fr

    rows = rp.run_frames(tap(frame_iter), n_f=n_f if n_f is not None else (len(freeze) if freeze else 0),
                         maps=maps, arms=specs, condition=condition, build=build, freeze=freeze,
                         ground_fn=cap, log=log)
    cap.check_order(rows)
    rows = {k: [{"kind": kind, **slim_row(r)} for r in v] for k, v in rows.items()}
    arrays = arrays_of(cap, fr_list, [s.name for s in specs])
    log(f"  t56 job done: {len(fr_list)} frames x {len(specs)} {kind}(s), {len(cap.clouds)} clouds, "
        f"{time.time() - t0:.0f}s")
    return rows, arrays


__all__ = ["Capture", "Geometry", "HANDS", "MEMBERSHIP_CODE", "PARENT_LINK_MJCF", "PARENT_LINK_URDF",
           "ROBOT_KEYS", "arrays_of", "load_npz", "merge", "robot_frame", "run_job", "save_npz",
           "slim_row"]
