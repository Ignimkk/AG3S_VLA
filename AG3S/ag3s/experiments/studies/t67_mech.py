"""T67 — mechanism metrics of the stage-1 / stage-2 closed-loop runs (session B, read-only on session A's records).

Per run, the same quantities as T66b on the final configuration (pushdiag · closepose · agg):
  grasped            apple rise ≥ 20 mm at any recorded request (MuJoCo body height, client qpos)
  n_win / max push   requests before the first closing with the VLA-planned palm ≤ 60 mm from the apple, and the largest
                     palm shift refined − VLA reference over the executed 8 rows (MuJoCo FK of the `<side>_ee` site)
  close_dist_mm      palm site to the apple's t=0 centre at the first closing command (traj.npz executed_actions)
  hsr6 objective     requests in the window whose `hsr6.detail.objective` withdrew NBC / A4 / G1
  multi-seed         requests in the window where the GPU multi-seed solution was used
The final-configuration baseline (T66b `agg.json`) is printed next to each run.

usage: MUJOCO_GL=osmesa SN=<snapshot> PYTHONPATH=$SN:$SN/integration/rby1/manipulation/src \
       t67_mech.py STAGE_DIR BASELINE_AGG_JSON OUT_JSON      (.venv-openpi-live)
"""

from __future__ import annotations

import ast
import glob
import json
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402

sys.path.insert(0, os.environ["SN"])
from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde  # noqa: E402

CE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261028T224856+0900_b3b69311f948_8784a1265958"
                  "/verify/AG3S-CORE-EVAL")
XML = os.environ["SN"] + "/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
SPEC = {s["label"]: s for s in json.load(open(CE / "module/spec_fin.json"))}
ITEMS = {line.split(":")[1]: line.strip().split(":") for line in open(CE / "hurdle/hurdle_items.txt")}
OCC_SCENE = {"AG3S_case1_L2_s13453": "가림121", "AG3S_case1_L2_s23453": "가림122", "AG3S_case1_L2_s33453": "가림123"}


def lit(v):
    try:
        return ast.literal_eval(v) if isinstance(v, str) else v
    except (ValueError, SyntaxError):
        return v


def segments(run_dir: pathlib.Path) -> dict:
    """{run name: chunk files} over every server directory under `run_dir` (a kit may run several servers)."""
    out = {}
    for sd in sorted(run_dir.glob("server_*")):
        if not (sd / "order.txt").exists():
            continue
        order = (sd / "order.txt").read_text().split()
        fs = sorted((sd / "server_constraints/run_0000").glob("chunk_*.npz"))
        if not fs:
            continue
        ts = [int(np.load(f)["t_step"]) for f in fs]
        segs = [[fs[0]]]
        for f, a, b in zip(fs[1:], ts, ts[1:]):
            if b <= a:
                segs.append([])
            segs[-1].append(f)
        out.update(zip(order, segs))
    return out


def runs(stage: pathlib.Path):
    """(config, label, side, obstacle, client_dir, chunks) for every finished run under the stage directory."""
    for kit in sorted(stage.glob("REG_C*_k*")):
        c, k = kit.name.split("_")[1], kit.name.split("_")[2]
        for ep, (pdir, ch) in tde.episodes_of(kit, "B_ROI_FPR").items():
            lab = f"회귀{ep}_{k}"
            sp = SPEC[lab]
            yield c, lab, sp["side"], tuple(sp["obstacle"]), pdir.parents[1], [f for _, f, _ in sorted(ch, key=lambda x: x[0])]
    for occ in sorted(stage.glob("OCC_C*")):
        c = occ.name.split("_")[1]
        for kd in sorted(occ.glob("k*")):
            for name, seg in segments(kd).items():
                lab = f"{OCC_SCENE[name]}_{kd.name}"
                sp = SPEC[lab]
                yield c, lab, sp["side"], tuple(sp["obstacle"]), kd / "runs" / name, seg
    for hd in sorted(stage.glob("hurdle_C*")):
        c = hd.name.split("_")[1]
        for kd in sorted((hd / "ag3s").glob("k*")):
            for name, seg in segments(kd).items():
                it = ITEMS[name.split("_ep")[1].split("_")[0]]
                yield c, f"hurdle{it[1]}_{kd.name}", it[3], (it[4], it[5]), kd / "runs" / name, seg


def one(side, obst, cdir, chunks, cache):
    key = json.dumps(obst)
    if key not in cache:
        cache[key] = tde.scene_truth(XML, obst)
    gt = cache[key]
    m = gt.m
    d = mujoco.MjData(m)
    a0 = 0 if side == "left" else 8
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_arm_{i}")] for i in range(7)]
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
    ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    steps = sorted(glob.glob(str(cdir / "ag3s/run_0000/step_*.npz")), key=lambda p: int(np.load(p)["t_step"]))
    q0 = np.asarray(np.load(steps[0])["qpos"], float)
    gt.at(q0)
    R, o = gt.base_pose
    apple0 = (gt.d.xpos[ab] - o) @ R
    rise, win, closed = 0.0, [], False
    for j, f in enumerate(chunks[:len(steps)]):
        q = np.asarray(np.load(steps[j])["qpos"], float)
        gt.at(q)
        apple = (gt.d.xpos[ab] - o) @ R
        rise = max(rise, float(apple[2] - apple0[2]))
        if closed:
            continue
        z = np.load(f, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        st = (s.get("grasp") or {}).get("state")
        if st not in ("latched", "searching"):
            closed = st is not None and st != "latched"
            continue
        P = {}
        for nm, key in (("ref", "reference_chunk"), ("fin", "refined_chunk")):
            pts = []
            for r_ in np.asarray(z[key], float)[:8]:
                d.qpos[:] = q
                d.qpos[adr] = r_[a0:a0 + 7]
                mujoco.mj_kinematics(m, d)
                pts.append(((d.site_xpos[site] - o) @ R).copy())
            P[nm] = np.array(pts)
        if np.linalg.norm(apple - P["ref"][-1]) > 0.060:
            continue
        met = (s.get("to") or {}).get("metrics") or {}
        ms = lit(met.get("multi_seed")) or {}
        ob = (((s.get("hsr6") or {}).get("detail") or {}).get("objective")) or {}
        ga = lit(met.get("grasp_anchor"))
        win.append({"t": int(z["t_step"]), "shift_mm": float(np.linalg.norm(P["fin"] - P["ref"], axis=1).max() * 1e3),
                    "multi_seed": bool(ms.get("triggered")) if isinstance(ms, dict) else False,
                    "withdrawn": sorted(k for k, v in ob.items() if k.endswith("_withdrawn") and v),
                    "goal": (ga.get("variant") or "A4") if isinstance(ga, dict) and ga.get("goal_rows") else None})
    tz = np.load(cdir / "traj.npz", allow_pickle=True)
    ea, mq = tz["executed_actions"], tz["measured_qpos"]
    ic = np.where(ea[:, a0 + 7] < 0.9)[0]
    close = None
    if len(ic):
        d.qpos[:] = q0
        d.qpos[adr] = mq[int(ic[0]), a0:a0 + 7]
        mujoco.mj_kinematics(m, d)
        close = float(np.linalg.norm(apple0 - (d.site_xpos[site] - o) @ R) * 1e3)
    return {"grasped": rise >= 0.020, "apple_rise_mm": round(rise * 1e3, 1), "n_win": len(win),
            "max_shift_win_mm": round(max([w["shift_mm"] for w in win], default=0.0), 1),
            "n_push8_win": sum(w["shift_mm"] >= 8 for w in win), "n_ms_win": sum(w["multi_seed"] for w in win),
            "n_withdrawn_win": sum(bool(w["withdrawn"]) for w in win), "n_goal_win": sum(bool(w["goal"]) for w in win),
            "close_dist_mm": None if close is None else round(close, 1),
            "window": win}


def main(stage: pathlib.Path, base_json: pathlib.Path, out: pathlib.Path) -> None:
    base = {x["run"]: x for x in json.load(open(base_json))}
    res, cache = {}, {}
    for c, lab, side, obst, cdir, chunks in runs(stage):
        if not (cdir / "traj.npz").exists():
            continue
        r = one(side, obst, cdir, chunks, cache)
        res[f"{c}|{lab}"] = r
        b = base.get(lab, {})
        print(f"{c} {lab:16} grasp {'Y' if r['grasped'] else 'n'} (rise {r['apple_rise_mm']:>6}) | max push {r['max_shift_win_mm']:>5} "
              f"(final {b.get('max_shift_win_mm')}) | close {r['close_dist_mm']} (final {b.get('close_dist_mm')}) | "
              f"withdrawn {r['n_withdrawn_win']}/{r['n_win']} goal {r['n_goal_win']} ms {r['n_ms_win']}", flush=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
