"""T77 — closed-loop ground-truth labels for the 5-stage HSR classifier, per request (`handoff/T77.task.md` §1, §2).

Two sub-commands:

``spec``   find every AG3S closed-loop run (client ``manifest.json`` with ``--safe-remote`` + ``ag3s/run_0000`` step
           npz) under the given kit roots, pair it with its server chunk records (``server_*/server_constraints``,
           segmented by ``policy_seed.episode`` and t_step restarts, restricted by the server's ``arm`` file), and
           write a spec in the T75 ``spec.json`` entry format (``label · client_dir · chunks · obstacle · side`` +
           ``scene · scene_number · split · stratum · episode · seed · kit``). Scene split (§2, fixed): scene numbers
           ``TEST_SCENES`` = {211, 954, 122, 119} -> test, all others -> train; a scene number lands on one side only.
           VLA-only runs and runs without matching server chunks are listed under ``skipped`` with the reason.

``labels`` for every spec entry and request: the client step qpos put into the scene XML (obstacle slots activated as
           the client did, `task_dynamic_map_eval.scene_truth`, the `module_eval.py` path), MuJoCo forward, and

           held     = both fingers of the grasping hand touch the apple ∧ apple centre rose >= 20 mm above request 0
                      (base frame) — `module_eval.py` GT held, unchanged
           phase5   retreat  : held earlier, not held now
                    place    : held ∧ apple centre inside the destination (``crate``) floor's xy range
                    carry    : held ∧ not place
                    grasp    : never held yet ∧ grasping-hand palm site (MJCF ``<side>_ee``) – apple centre <= 60 mm
                               (the A6 window `grasp_set_gate`; an empty-hand close lands here)
                    approach : never held yet ∧ > 60 mm

           plus the serving signals recorded in the chunk ``summary_json``: 3-stage ``subtask`` (label / argmax / p)
           and ``hsr6`` (phase / p), and their 5-stage / coarse translations (`HSR6_TO_5`, `COARSE_OF_5`).

The "basket floor xy range" is the xy box of the destination body's **floor geom** (the collision geom of that body
with the lowest centre, `crate_floor`, ±90 × ±150 mm in the crate frame); the apple centre is taken into the crate
frame, so a yawed crate is handled. ``--basket-xy all`` uses every collision geom of the body instead (handles
included) — a sensitivity option, not the fixed definition.

usage (MuJoCo; .venv-ag3s or .venv-openpi-live, CPU):
  SNAP=<snapshot_t67> PYTHONPATH=$PWD:$SNAP/integration/rby1/manipulation/src MUJOCO_GL=osmesa \\
    python -m AG3S.ag3s.experiments.studies.t77_labels spec   OUT_spec.json [ROOT ...]
  SNAP=... python -m AG3S.ag3s.experiments.studies.t77_labels labels OUT_labels.json SPEC.json [--only LABEL ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import sys
from typing import Any, Iterable, Optional, Sequence

import numpy as np

PHASES5 = ("approach", "grasp", "carry", "place", "retreat")
#: expert ``phase_index`` -> 5 stages (T64 MAP6 with lift · transport merged); other indices are excluded (counted).
MAP5 = {0: 0, 1: 0, 2: 0, 20: 0, 21: 0, 22: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3,
        10: 4, 11: 4, 12: 4, 13: 4}
#: serving HSR6 phase -> 5 stages (lift and transport -> carry).
HSR6_TO_5 = {"approach": "approach", "grasp": "grasp", "lift": "carry", "transport": "carry",
             "placement": "place", "retreat": "retreat"}
#: 5 stages -> the 3-stage serving vocabulary (pick / place / home), for comparing the 3-stage signal.
COARSE_OF_5 = {"approach": "pick", "grasp": "pick", "carry": "place", "place": "place", "retreat": "home"}

LIFT_MM = 20.0
GRASP_WINDOW_M = 0.060           # A6 window, `AG3S/trajopt/config.py` grasp_set_gate
DESTINATION_BODY = "crate"
TARGET_BODY = "apple"
TEST_SCENES = ("211", "954", "122", "119")
#: occlusion kits: episode 1345, policy seed -> scene number (T75 `make_spec.py` SCENE map: s13453 -> 가림121 ...)
OCC_SCENE = {13453: "121", 23453: "122", 33453: "123"}

R = "/mnt/dev/AG3S_VLA-storage/results/development/"
DEFAULT_ROOTS = {
    "CE": R + "20261028T224856+0900_b3b69311f948_8784a1265958/verify/AG3S-CORE-EVAL",
    "PT": R + "20261025T191319+0900_35b909cf096a_cc215238f581/verify/AG3S-PRE-CORE-TO/PT",
    "T71": R + "20261029T232703+0900_8ab3c7ef7757_7199b750125b/verify/T71/S1",
    "T74": R + "20261030T065754+0900_8ab3c7ef7757_4aea29e44498/verify/T74/S1",
    "T72b": R + "20261030T004052+0900_8ab3c7ef7757_1c61eaabd79c/verify/T72/CL",
}
XML_REL = "robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"


# ================================================================================ pure rules (tested)


def gt_held(fingers_touch: int, rise_mm: float) -> bool:
    """`module_eval.py` GT held: both active fingers touch the apple and it rose >= 20 mm."""
    return bool(fingers_touch >= 2 and rise_mm >= LIFT_MM)


def phase5_sequence(held: Sequence[bool], palm_dist_m: Sequence[float], in_basket: Sequence[bool]) -> list[str]:
    """§1 closed-loop 5-stage GT for one run's requests in order (None entries = no measurement -> None)."""
    out, ever = [], False
    for h, d, b in zip(held, palm_dist_m, in_basket, strict=True):
        if h is None:
            out.append(None)
            continue
        if h:
            ever = True
            out.append("place" if b else "carry")
        elif ever:
            out.append("retreat")
        else:
            out.append("grasp" if (d is not None and d <= GRASP_WINDOW_M) else "approach")
    return out


# --- T85 — begin
class CausalPhase5:
    """The closed-loop 5-stage GT of one run, one request at a time (T85, the client's `--safe-gt-stage`).

    It is `phase5_sequence` itself: every request appends (held, palm distance, in-basket) to the run's lists and the
    last entry of `phase5_sequence` over them is the answer — causal, because entry i depends on entries <= i only
    (`ever` is the running OR of held). `held` is `gt_held` with the rise measured from request 0 (`z0`)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.held: list = []
        self.palm: list = []
        self.basket: list = []
        self.z0: Optional[float] = None

    def update(self, measure: dict) -> Optional[str]:
        """This request's stage from `SceneProbe.measure` (a measurement with an "error" key -> None, as the offline
        labels leave such a request unlabelled)."""
        if "error" in measure:
            return None
        if self.z0 is None:
            self.z0 = measure["apple_z_base"]
        rise = (measure["apple_z_base"] - self.z0) * 1e3
        self.held.append(gt_held(measure["fingers_touch"], rise))
        self.palm.append(measure["palm_dist_m"])
        self.basket.append(measure["in_basket"])
        return phase5_sequence(self.held, self.palm, self.basket)[-1]
# --- T85 — end


def serving_to5(hsr6_phase: Optional[str]) -> Optional[str]:
    return HSR6_TO_5.get(hsr6_phase) if hsr6_phase else None


def scene_of(run_name: str, episode: Optional[int], seed: Optional[int], profile: Optional[str]) -> tuple[str, str]:
    """``(stratum, scene_number)``. Hurdle runs -> (허들, episode); occlusion runs on episode 1345 (pillar /
    halfpost profiles, ``AG3S_*`` names) -> (가림, `OCC_SCENE`[seed]); otherwise (회귀, episode)."""
    if run_name.startswith("HURDLE"):
        return "허들", str(episode)
    if episode == 1345 and (run_name.startswith("AG3S_") or "case1" in (profile or "")):
        kind = "가림-halfpost" if "halfpost" in (profile or run_name) else "가림"
        return kind, OCC_SCENE.get(int(seed) if seed is not None else -1, f"1345s{seed}")
    return "회귀", str(episode)


def split_of(scene_number: str) -> str:
    return "test" if str(scene_number) in TEST_SCENES else "train"


def check_split(entries: Iterable[dict]) -> dict:
    """Asserts §2: every scene number on one side only, test numbers never in train, and no (episode, seed) pair
    on both sides. Returns per-split scene lists."""
    by = collections.defaultdict(set)
    pairs = collections.defaultdict(set)
    for e in entries:
        by[e["split"]].add(e["scene_number"])
        pairs[(e.get("episode"), e.get("seed"))].add(e["split"])
    assert not (by["train"] & by["test"]), by["train"] & by["test"]
    assert not (by["train"] & set(TEST_SCENES)), by["train"] & set(TEST_SCENES)
    both = [p for p, s in pairs.items() if len(s) > 1]
    assert not both, f"(episode, seed) on both sides: {both}"
    return {k: sorted(v) for k, v in by.items()}


# ================================================================================ spec (discovery)


def _argv_value(argv: list, key: str) -> Optional[str]:
    return argv[argv.index(key) + 1] if key in argv and argv.index(key) + 1 < len(argv) else None


def _client_info(d: pathlib.Path) -> dict:
    man = json.loads((d / "manifest.json").read_text())
    argv = (man.get("extra") or {}).get("argv") or []
    start = pathlib.Path(str(d) + ".start")
    arm = None
    if start.exists():
        for line in start.read_text().splitlines():
            if line.startswith("arm="):
                arm = line[4:].strip()
    if arm is None and d.parent.name not in ("runs",):
        arm = d.parent.name
    ep = _argv_value(argv, "--episode-index")
    seed = _argv_value(argv, "--policy-seed")
    return {"dir": d, "argv": argv, "safe": "--safe-remote" in argv, "arm": arm,
            "episode": int(ep) if ep is not None else None, "seed": int(seed) if seed is not None else None,
            "obstacle": [_argv_value(argv, "--obstacle-config"), _argv_value(argv, "--obstacle-profile") or "clear"],
            "side": (_argv_value(argv, "--safe-manipulators") or "left").split(",")[0]}


def _step_tsteps(run: pathlib.Path) -> list[tuple[int, str]]:
    out = []
    for f in sorted(run.glob("step_*.npz")):
        with np.load(f) as z:
            out.append((int(z["t_step"]), str(f)))
    return sorted(out)


def server_segments(sdir: pathlib.Path) -> list[dict]:
    """Requests of one server, split into runs where ``policy_seed.episode`` changes or t_step restarts."""
    segs = []
    for rdir in sorted((sdir / "server_constraints").glob("run_*")):
        cur = None
        for f in sorted(rdir.glob("chunk_*.npz")):
            with np.load(f, allow_pickle=True) as z:
                t = int(z["t_step"])
                s = json.loads(str(z["summary_json"]))
            seed = (s.get("policy_seed") or {}).get("episode")
            if cur is None or seed != cur["seed"] or t <= cur["t_steps"][-1]:
                cur = {"seed": seed, "chunks": [], "t_steps": [], "server": str(sdir)}
                segs.append(cur)
            cur["chunks"].append(str(f))
            cur["t_steps"].append(t)
    return segs


def discover(roots: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """``(entries, skipped)`` over the kit roots (see module docstring)."""
    entries, skipped = [], []
    for tag, root in roots.items():
        root = pathlib.Path(root)
        clients, servers = [], []
        for dp, dn, fn in os.walk(root):
            p = pathlib.Path(dp)
            if "manifest.json" in fn and (p / "ag3s" / "run_0000").is_dir():
                clients.append(_client_info(p))
                dn[:] = []
            elif p.name.startswith("server_") and "server_constraints" in dn:
                servers.append(p)
                dn[:] = []
        seg_of: dict[pathlib.Path, list] = collections.defaultdict(list)
        for sdir in servers:
            arm_f = sdir / "arm"
            arm = arm_f.read_text().strip() if arm_f.exists() else None
            scope = sdir.parent
            cands = [c for c in clients if c["safe"] and scope in c["dir"].parents
                     and (arm is None or c["arm"] is None or c["arm"] == arm)]
            for seg in server_segments(sdir):
                hit = [c for c in cands if c["seed"] == seg["seed"]]
                if len(hit) == 1:
                    seg_of[hit[0]["dir"]].append(seg)
                elif len(hit) > 1:
                    skipped.append({"server": str(sdir), "seed": seg["seed"], "n": len(seg["chunks"]),
                                    "reason": f"server segment matches {len(hit)} client runs"})
        for c in clients:
            rel = c["dir"].relative_to(root)
            base = {"label": f"{tag}/{rel}", "client_dir": str(c["dir"])}
            if not c["safe"]:
                skipped.append({**base, "reason": "VLA-only run (no --safe-remote)"})
                continue
            segs = seg_of.get(c["dir"], [])
            if not segs:
                skipped.append({**base, "reason": "no server chunk segment with this policy seed"})
                continue
            steps = _step_tsteps(c["dir"] / "ag3s" / "run_0000")
            st = [t for t, _ in steps]
            # the server's chunk t_step is its request counter (1, 2, 3 ...), the client's is the control step
            # (0, 8, 16 ...): requests pair by order (i-th chunk <-> i-th client step, as module_eval.py)
            same = [s for s in segs if len(s["chunks"]) == len(st)]
            seg = (same or segs)[-1]
            warn = []
            if len(segs) > 1:
                warn.append(f"{len(segs)} server segments for this run; chose the last "
                            f"{'with the client request count' if same else 'one'}")
            if len(seg["chunks"]) != len(st):
                warn.append(f"request count mismatch: client {len(st)} steps, server {len(seg['chunks'])} chunks")
            if seg["t_steps"] != list(range(1, len(seg["t_steps"]) + 1)) and seg["t_steps"] != st:
                warn.append("server chunk t_steps are neither 1..n nor the client t_steps")
            stratum, num = scene_of(c["dir"].name, c["episode"], c["seed"], c["obstacle"][1])
            entries.append({**base, "kit": tag, "config": str(rel.parts[0]) if rel.parts else "",
                            "chunks": seg["chunks"], "chunk_t_steps": seg["t_steps"], "client_t_steps": st,
                            "obstacle": c["obstacle"], "side": c["side"], "episode": c["episode"], "seed": c["seed"],
                            "stratum": stratum, "scene": f"{stratum}{num}", "scene_number": num,
                            "split": split_of(num), "warnings": warn})
    return entries, skipped


def build_spec(out: pathlib.Path, roots: dict[str, str]) -> dict:
    entries, skipped = discover(roots)
    split = check_split(entries)
    summ = collections.Counter((e["split"], e["scene"]) for e in entries)
    spec = {"step": "T77", "roots": roots, "test_scenes": list(TEST_SCENES), "split_scenes": split,
            "n_runs": {"train": sum(e["split"] == "train" for e in entries),
                       "test": sum(e["split"] == "test" for e in entries)},
            "runs_per_scene": {f"{k[0]}:{k[1]}": v for k, v in sorted(summ.items())},
            "n_skipped": dict(collections.Counter(s["reason"].split(" (")[0] for s in skipped)),
            "runs": entries, "skipped": skipped}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    return spec


# ================================================================================ labels (MuJoCo)


class SceneProbe:
    """MuJoCo measurements of one scene (one obstacle configuration): finger contacts with the apple, apple centre
    (base frame), palm site distance, apple-in-basket test."""

    def __init__(self, model_xml: str, obstacle: Sequence, basket_xy: str = "floor"):
        import mujoco

        from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde

        self.mj = mujoco
        self.gt = tde.scene_truth(model_xml, (obstacle[0], obstacle[1]))
        m = self.m = self.gt.m
        self.d = self.gt.d
        self._bind(model_xml, basket_xy)

    def _bind(self, where: str, basket_xy: str, target_body: str = TARGET_BODY) -> None:
        mujoco, m = self.mj, self.m
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.apple = bid(target_body)
        self.crate = bid(DESTINATION_BODY)
        if self.apple < 0 or self.crate < 0:
            raise RuntimeError(f"{where}: no '{target_body}' / '{DESTINATION_BODY}' body")
        self.fingers = {s: {bid(f"ee_finger_{s[0]}{k}") for k in (1, 2)} for s in ("left", "right")}
        self.site = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_ee") for s in ("left", "right")}
        self.basket_box = self._basket_box(basket_xy)

    # --- T85 — begin
    @classmethod
    def live(cls, model, data, *, target_body: str = TARGET_BODY, basket_xy: str = "floor") -> "SceneProbe":
        """A probe on the running simulation's `model` (obstacles already activated by the client) with a **private**
        `MjData`, so `measure` never touches the control loop's state (it sets qpos and runs `mj_forward`)."""
        import types

        import mujoco

        self = cls.__new__(cls)
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base < 0:
            raise RuntimeError("live scene: no 'base' body")
        self.gt = types.SimpleNamespace(m=model, d=self.d, base=base)
        self._bind("live scene", basket_xy, target_body)
        return self

    def measure_live(self, live_data, side: str) -> dict:
        """`measure` of the live `data`'s qpos (and mocap), as the offline labels measure a step npz's qpos."""
        self.d.mocap_pos[:] = live_data.mocap_pos
        self.d.mocap_quat[:] = live_data.mocap_quat
        return self.measure(np.asarray(live_data.qpos, float), side)
    # --- T85 — end

    def _basket_box(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """xy box (crate body frame) of the destination floor geom (lowest collision geom) or of all of them."""
        mj, m = self.mj, self.m
        geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.crate
                 and (m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0)]
        if mode == "floor":
            geoms = [min(geoms, key=lambda g: float(m.geom_pos[g][2]))]
        lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
        for g in geoms:
            Rg = np.zeros(9)
            mj.mju_quat2Mat(Rg, m.geom_quat[g])
            Rg = Rg.reshape(3, 3)
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
            pts = corners @ Rg.T + m.geom_pos[g]
            lo, hi = np.minimum(lo, pts[:, :2].min(0)), np.maximum(hi, pts[:, :2].max(0))
        return lo, hi

    def measure(self, qpos: np.ndarray, side: str) -> dict:
        mj, m, d = self.mj, self.m, self.d
        qpos = np.asarray(qpos, float)
        if len(qpos) != m.nq:
            return {"error": f"qpos {len(qpos)} != nq {m.nq}"}
        d.qpos[:] = qpos
        mj.mj_forward(m, d)
        Rb, ob = d.xmat[self.gt.base].reshape(3, 3), d.xpos[self.gt.base]
        apple_w = d.xpos[self.apple].copy()
        apple_b = (apple_w - ob) @ Rb
        touch = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if self.apple in (b1, b2):
                touch.add(b2 if b1 == self.apple else b1)
        palm = float(np.linalg.norm(d.site_xpos[self.site[side]] - apple_w))
        Rc, oc = d.xmat[self.crate].reshape(3, 3), d.xpos[self.crate]
        a_c = (apple_w - oc) @ Rc
        lo, hi = self.basket_box
        in_basket = bool(np.all(a_c[:2] >= lo) and np.all(a_c[:2] <= hi))
        return {"fingers_touch": len(self.fingers[side] & touch), "apple_z_base": float(apple_b[2]),
                "palm_dist_m": palm, "in_basket": in_basket, "apple_crate_xy_mm": (a_c[:2] * 1e3).round(1).tolist()}


def serving_of(summary: dict) -> dict:
    sub = summary.get("subtask") or {}
    h6 = summary.get("hsr6") or {}
    gr = summary.get("grasp") or {}
    lab3 = sub.get("label")
    return {"subtask_label": lab3, "subtask_argmax": sub.get("argmax"), "subtask_p": sub.get("p"),
            "hsr6_phase": h6.get("phase"), "hsr6_p": h6.get("p"), "hsr6_to5": serving_to5(h6.get("phase")),
            "grasp_state": gr.get("state")}


def label_run(probe: SceneProbe, entry: dict) -> dict:
    side = entry["side"]
    steps = sorted(_step_tsteps(pathlib.Path(entry["client_dir"], "ag3s", "run_0000")))
    rows, z0 = [], None
    for j, cf in enumerate(entry["chunks"]):
        with np.load(cf, allow_pickle=True) as z:
            ct = int(z["t_step"])
            summ = json.loads(str(z["summary_json"]))
        # i-th server chunk <-> i-th client step (sorted by t_step), as module_eval.py; `t_step` = the client's
        row = {"request": j, "chunk_t_step": ct, "serving": serving_of(summ)}
        if j >= len(steps):
            row["missing_step"] = True
            rows.append(row)
            continue
        with np.load(steps[j][1]) as st:
            qpos = np.asarray(st["qpos"], float)
            row["t_step"] = int(st["t_step"])
        mes = probe.measure(qpos, side)
        if "error" in mes:
            row["error"] = mes["error"]
            rows.append(row)
            continue
        if z0 is None:
            z0 = mes["apple_z_base"]
        rise = (mes["apple_z_base"] - z0) * 1e3
        row["gt"] = {"fingers_touch": mes["fingers_touch"], "apple_rise_mm": round(rise, 2),
                     "held": gt_held(mes["fingers_touch"], rise), "palm_dist_mm": round(mes["palm_dist_m"] * 1e3, 2),
                     "in_basket": mes["in_basket"], "apple_crate_xy_mm": mes["apple_crate_xy_mm"]}
        rows.append(row)
    ok = [r for r in rows if "gt" in r]
    seq = phase5_sequence([r["gt"]["held"] for r in ok], [r["gt"]["palm_dist_mm"] / 1e3 for r in ok],
                          [r["gt"]["in_basket"] for r in ok])
    for r, p in zip(ok, seq, strict=True):
        r["gt"]["phase5"] = p
        r["gt"]["coarse"] = COARSE_OF_5[p]
    counts = collections.Counter(r["gt"]["phase5"] for r in ok)
    return {k: entry.get(k) for k in ("label", "client_dir", "kit", "config", "scene", "scene_number", "split",
                                       "stratum", "episode", "seed", "side", "obstacle")} | {
        "n_requests": len(rows), "n_labelled": len(ok), "phase5_counts": {p: counts.get(p, 0) for p in PHASES5},
        "n_missing": sum(1 for r in rows if "gt" not in r), "rows": rows}


def build_labels(out: pathlib.Path, spec_path: pathlib.Path, model_xml: str, only: Optional[list] = None,
                 basket_xy: str = "floor") -> dict:
    spec = json.loads(spec_path.read_text())
    entries = spec["runs"] if isinstance(spec, dict) else spec
    if only:
        entries = [e for e in entries if e["label"] in only]
    probes: dict[tuple, SceneProbe] = {}
    runs = []
    for e in entries:
        key = tuple(e["obstacle"])
        if key not in probes:
            probes[key] = SceneProbe(model_xml, key, basket_xy)
        r = label_run(probes[key], e)
        runs.append(r)
        print(f"{r['label']}: {r['n_labelled']}/{r['n_requests']} labelled, {r['phase5_counts']}", flush=True)
    res = {"step": "T77", "spec": str(spec_path), "model_xml": model_xml, "basket_xy": basket_xy,
           "definitions": __doc__, "phases5": list(PHASES5), "lift_mm": LIFT_MM, "grasp_window_m": GRASP_WINDOW_M,
           "basket_box_crate_frame_m": {str(k): [p.basket_box[0].tolist(), p.basket_box[1].tolist()]
                                        for k, p in probes.items()},
           "runs": runs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=0, default=float))
    return res


def default_xml() -> str:
    snap = os.environ.get("SNAP")
    if snap and pathlib.Path(snap, XML_REL).exists():
        return str(pathlib.Path(snap, XML_REL))
    return str(pathlib.Path(__file__).resolve().parents[4] / XML_REL)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("spec")
    a.add_argument("out")
    a.add_argument("roots", nargs="*", help="TAG=DIR (default: CORE-EVAL, PT, T71, T74, T72b)")
    b = sub.add_parser("labels")
    b.add_argument("out")
    b.add_argument("spec")
    b.add_argument("--model-xml", default=None, help=f"default $SNAP/{XML_REL}")
    b.add_argument("--only", nargs="+", default=None)
    b.add_argument("--basket-xy", default="floor", choices=("floor", "all"))
    args = ap.parse_args(argv)
    if args.cmd == "spec":
        roots = dict(r.split("=", 1) for r in args.roots) if args.roots else DEFAULT_ROOTS
        spec = build_spec(pathlib.Path(args.out), roots)
        print(json.dumps({k: spec[k] for k in ("n_runs", "split_scenes", "n_skipped", "runs_per_scene")},
                         ensure_ascii=False, indent=1))
    else:
        build_labels(pathlib.Path(args.out), pathlib.Path(args.spec), args.model_xml or default_xml(), args.only,
                     args.basket_xy)


if __name__ == "__main__":
    main()
