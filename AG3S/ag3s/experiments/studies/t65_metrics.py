"""T65 metrics — ATS indicators of the fixed-input replays, per attention source (`handoff/T65.task.md`).

Definitions copied from session A (`AG3S-CORE-EVAL/ats_only.py`, `module_eval.py` LTE block 89–107), not imported, so A's
run directory stays untouched. Pre-grasp window: requests before the first two-finger contact or a 20 mm apple rise.
Replay chunk j <-> recorded client step j (same qpos), as in `module_eval.py`.

Per request:
    has_target · majority_object (10 mm) · precision_5mm · recall_5mm / vis_centroid_mm (visible GT apple, any camera)
    apple_left_frac (seen = |main| <= 1 voxel, left = seen ∧ |target-free| <= 1 voxel)
    removed_by_class (target_free_removed_ijk centres, nearest surface label within 8 mm)
Per run: rates over all pre-grasp requests, means over the requests that have the value, wrong-removed fraction over the
run's pooled removed voxels. Per scene: mean of the 2 draws. Decision: `decide()` (pre-registered rule, T65.task.md).

usage: MUJOCO_GL=osmesa t65_metrics.py RUN65 OUT_DIR      (.venv-openpi-live; mujoco · scipy)
"""

from __future__ import annotations

import collections
import glob
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

CORE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261028T224856+0900_b3b69311f948_8784a1265958")
V = CORE / "verify/AG3S-CORE-EVAL"
SNAP = CORE / "snapshot_core"
sys.path.insert(0, str(SNAP))
from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde  # noqa: E402

# the AG3S package above resolves to the snapshot, so load this checkout's sibling by path
_spec = __import__("importlib.util").util.spec_from_file_location("t65_parity", pathlib.Path(__file__).with_name("t65_parity.py"))
_par = __import__("importlib.util").util.module_from_spec(_spec)
_spec.loader.exec_module(_par)
replay_segments = _par.replay_segments

XML = str(SNAP / "robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml")
SOURCES = ("noun_target_L10H7", "action_last_L8H2", "action_mean_L10H7")
BASE = SOURCES[0]
GROUPS = ("REG_k0", "REG_k1", "OCC_k0", "OCC_k1")
PREC_MM, CLASS_MM, LIFT_MM, MAJ_MM = 5.0, 8.0, 20.0, 10.0
ERR_CLASSES = ("fruit", "obstacle", "table", "crate")
# metric -> +1 if higher is better, -1 if lower is better
DIRECTION = {"identity": +1, "precision_5mm": +1, "other_object": -1, "recall_5mm": +1, "vis_centroid_mm": -1,
             "none_object": -1, "no_target": -1, "apple_left_frac": -1, "wrong_removed_frac": -1}
PRIMARY = ("identity", "precision_5mm", "other_object")
IMPROVE_MIN, SCENES_MIN = 0.02, 5


def grid(z, pre):
    if f"{pre}_distance" not in z.files:
        return None
    return np.asarray(z[f"{pre}_distance"], np.float32), np.asarray(z[f"{pre}_origin"], float), float(z[f"{pre}_voxel_size"])


def lookup(g, p):
    D, o, vs = g
    ijk = np.floor((p - o) / vs).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(D.shape)), axis=1)
    out = np.full(len(p), np.nan, np.float32)
    out[ok] = D[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    return out


def visible(ap, st, head_only=False):
    if not head_only:
        return tde._visible(ap, st)
    sub = {k: st[k] for k in st.files if k.endswith("zed_left")}
    S = type("S", (), {"files": list(sub), "__getitem__": lambda self, k: sub[k]})
    return tde._visible(ap, S())


def run_rows(spec: dict, chunks: list, cache: dict) -> list:
    key = tuple(spec["obstacle"])
    if key not in cache:
        gt = tde.scene_truth(XML, key)
        cache[key] = (gt, tde.RobotSurface(gt))
    gt, rob = cache[key]
    m = gt.m
    ch = "l" if spec["side"] == "left" else "r"
    ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    fingers = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{ch}{k}") for k in (1, 2)}
    ai = [i for i, k in enumerate(gt.kind) if k[1] == "apple"]
    apple_r = float(max(m.geom_size[g].max() for g in range(m.ngeom)
                        if int(m.geom_bodyid[g]) == ab and m.geom_contype[g] != 0))
    apple_vis = [(g, gt._sample_local(g)) for g in range(m.ngeom) if int(m.geom_bodyid[g]) == ab
                 and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and float(m.geom_size[g].max()) < 0.02]
    steps = sorted(glob.glob(spec["client_dir"] + "/ag3s/run_0000/step_*.npz"), key=lambda p: int(np.load(p)["t_step"]))
    rows, z0 = [], None
    for j, f in enumerate(chunks):
        if j >= len(steps):
            break
        st = np.load(steps[j])
        samples, _ = gt.at(np.asarray(st["qpos"], float))
        R, o = gt.base_pose
        ac = (gt.d.xpos[ab] - o) @ R
        z0 = ac[2] if z0 is None else z0
        touch = {int(m.geom_bodyid[c.geom2]) if int(m.geom_bodyid[c.geom1]) == ab else int(m.geom_bodyid[c.geom1])
                 for c in gt.d.contact[:gt.d.ncon] if ab in (int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2]))}
        if fingers <= touch or (ac[2] - z0) * 1e3 >= LIFT_MM:
            break
        z = np.load(f, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        tp = np.asarray(z["target_points"], float) if "target_points" in z.files else np.zeros((0, 3))
        ap = np.vstack([samples[i] for i in ai])
        vis = visible(ap, st)
        row = {"request": j, "t_step": int(z["t_step"]), "has_target": bool(len(tp)),
               "head_visible_frac": round(float(visible(ap, st, True).mean()), 3),
               "attention_source": s.get("attention_source"), "grounding": s.get("grounding_status")}
        allp = np.vstack([p for p in samples if len(p)])
        if len(tp):
            row["precision_5mm"] = float(np.mean(cKDTree(ap).query(tp)[0] * 1e3 <= PREC_MM))
            lab = np.concatenate([[gt.kind[i][1]] * len(p) for i, p in enumerate(samples) if len(p)])
            dd, ii = cKDTree(allp).query(tp)
            row["majority_object"] = collections.Counter(np.where(dd * 1e3 <= MAJ_MM, lab[ii], "none").tolist()).most_common(1)[0][0]
            if vis.sum():
                row["recall_5mm"] = float(np.mean(cKDTree(tp).query(ap[vis])[0] * 1e3 <= PREC_MM))
                row["vis_centroid_mm"] = float(np.linalg.norm(tp.mean(0) - ap[vis].mean(0)) * 1e3)
        # LTE block (module_eval.py 89-107)
        main_g, free_g = grid(z, "esdf"), grid(z, "esdf_free")
        row["free_layer"] = free_g is not None
        if free_g is not None and main_g is not None:
            vs = main_g[2]
            dm, df = lookup(main_g, ap), lookup(free_g, ap)
            seen = np.abs(dm) <= vs
            left = seen & (df <= vs)
            row["apple_left_frac"] = float(left.sum() / seen.sum()) if seen.sum() else None
        if "target_free_removed_ijk" in z.files and len(z["target_free_removed_ijk"]):
            P = np.asarray(z["target_free_trace_origin"], float) + (np.asarray(z["target_free_removed_ijk"], float) + .5) \
                * float(z["target_free_trace_voxel_size"])
            vis_m = [((loc @ gt.d.geom_xmat[g].reshape(3, 3).T + gt.d.geom_xpos[g]) - o) @ R for g, loc in apple_vis]
            rp = rob.at()
            pts = np.vstack([p for p in samples if len(p)] + [rp] + vis_m)
            cls = np.concatenate([[("apple" if gt.kind[i][1] == "apple" else gt.kind[i][0]) for _ in range(len(p))]
                                  for i, p in enumerate(samples) if len(p)]
                                 + [["robot"] * len(rp)] + [["apple_stem_leaf"] * len(v) for v in vis_m])
            dd, ii = cKDTree(pts).query(P)
            c = np.where(dd * 1e3 <= CLASS_MM, cls[ii], "none")
            c = np.where((c == "none") & (np.linalg.norm(P - ac, axis=1) <= apple_r), "apple_inside", c)
            row["removed_by_class"] = dict(collections.Counter(c.tolist()))
        rows.append(row)
    return rows


def run_summary(rows: list) -> dict:
    n = len(rows)

    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.mean(v)) if v else None

    maj = [r.get("majority_object") for r in rows]
    rem = collections.Counter()
    for r in rows:
        rem.update(r.get("removed_by_class") or {})
    tot = sum(rem.values())
    err = sum(rem.get(c, 0) for c in ERR_CLASSES)
    return {"n_pre_grasp": n,
            "identity": sum(x == "apple" for x in maj) / n if n else None,
            "other_object": sum(x not in (None, "apple", "none") for x in maj) / n if n else None,
            "none_object": sum(x == "none" for x in maj) / n if n else None,
            "no_target": sum(not r["has_target"] for r in rows) / n if n else None,
            "precision_5mm": mean("precision_5mm"), "recall_5mm": mean("recall_5mm"),
            "vis_centroid_mm": mean("vis_centroid_mm"), "apple_left_frac": mean("apple_left_frac"),
            "wrong_removed_frac": err / tot if tot else None, "removed_total": tot, "removed_wrong": err,
            "removed_by_class": dict(rem), "head_visible_frac": mean("head_visible_frac"),
            "other_objects": dict(collections.Counter(x for x in maj if x not in (None, "apple", "none")))}


def scene_means(runs: dict) -> dict:
    """{label: summary} -> {scene: {metric: mean of the draws that have it}}."""
    by = collections.defaultdict(list)
    for lab, s in runs.items():
        by[lab.rsplit("_k", 1)[0]].append(s)
    out = {}
    for sc, ss in by.items():
        out[sc] = {}
        for k in DIRECTION:
            v = [s[k] for s in ss if s.get(k) is not None]
            out[sc][k] = float(np.mean(v)) if v else None
    return out


def decide(scenes: dict) -> dict:
    """Pre-registered rule: switch from the noun source only if every PRIMARY metric improves by >= 2 %p (scene mean),
    no other metric gets worse (point estimate), and every PRIMARY metric improves in >= 5 of the 9 scenes."""
    base = scenes[BASE]
    res = {}
    for src in SOURCES[1:]:
        cur, d = scenes[src], {}
        for k, sgn in DIRECTION.items():
            pairs = [(sc, cur[sc][k] - base[sc][k]) for sc in base if sc in cur
                     and cur[sc][k] is not None and base[sc][k] is not None]
            good = [sgn * x for _, x in pairs]
            d[k] = {"diff_mean": float(np.mean([x for _, x in pairs])) if pairs else None,
                    "good_direction_mean": float(np.mean(good)) if good else None,
                    "scenes_improved": int(sum(g > 0 for g in good)), "scenes_worse": int(sum(g < 0 for g in good)),
                    "n_scenes": len(pairs), "per_scene": dict(pairs)}
        c1 = all(d[k]["good_direction_mean"] is not None and d[k]["good_direction_mean"] >= IMPROVE_MIN for k in PRIMARY)
        c2 = all(d[k]["good_direction_mean"] is None or d[k]["good_direction_mean"] >= 0 for k in DIRECTION if k not in PRIMARY)
        c3 = all(d[k]["scenes_improved"] >= SCENES_MIN for k in PRIMARY)
        res[src] = {"diff": d, "c1_primary_improve_2pp": c1, "c2_no_other_worse": c2, "c3_5of9_scenes": c3,
                    "ats_pass": bool(c1 and c2 and c3),
                    "worse_others": [k for k in DIRECTION if k not in PRIMARY and d[k]["good_direction_mean"] is not None
                                     and d[k]["good_direction_mean"] < 0]}
    return res


def main(run65: pathlib.Path, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    spec = {}
    for s in json.loads((V / "module/spec_core.json").read_text()):
        spec[(s["draw"], pathlib.Path(s["client_dir"]).name)] = s
    cache, allrows, runs = {}, {}, {}
    for src in SOURCES:
        for g in GROUPS:
            d = run65 / "verify/T65" / src / g
            if not (d / "order.txt").exists():
                print("missing", d, flush=True)
                continue
            draw = int(g[-1])
            for name, chunks in replay_segments(d).items():
                sp = spec[(draw, name)]
                rows = run_rows(sp, chunks, cache)
                allrows.setdefault(src, {})[sp["label"]] = rows
                runs.setdefault(src, {})[sp["label"]] = run_summary(rows)
                print(src, sp["label"], len(rows), flush=True)
    scenes = {src: scene_means(r) for src, r in runs.items()}
    res = {"definitions": __doc__, "runs": runs, "scenes": scenes}
    if all(s in scenes for s in SOURCES):
        res["decision_ats"] = decide(scenes)
    (out / "t65_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (out / "t65_rows.json").write_text(json.dumps(allrows, ensure_ascii=False, indent=0, default=float))
    print(json.dumps({s: {k: (v["ats_pass"], v["worse_others"]) for k, v in [(s, res["decision_ats"][s])]}
                      for s in res.get("decision_ats", {})}, ensure_ascii=False))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
