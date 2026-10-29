"""T32b replay — T31b replay (t31b/replay.py) extended: variants x runs, hand spheres per chunk.
Runs: T28 E3 (12), T30 E3 (15), T14, T17. Per chunk: TargetConfirm(max_opening 71.46 mm, overlap 5 mm) +
ground_target on the server chunk npz; freeze = recorded exclusion.grasp_active (E3) or from the grasp-attempt
chunk (T14/T17). Hand = npz sphere_centres/radii (constraint model = gripper links ee_left/ee_right/fingers,
trajectory row 0 ~ q_now). Variants = config overrides (keys that do not exist yet are skipped) + hand on/off.
usage: replay.py OUT.json VARIANT[,VARIANT...]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections, multiprocessing as mp, os
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
V = pathlib.Path(_ag3s_workspace_path('outputs/verify'))
FRUIT = ("apple", "banana", "orange", "pear")
MO = 0.07146; FINE = 0.005; R_APPLE = 0.034
LIVE = {"t14": (_ag3s_workspace_path('outputs/live_test/20260928_t14'), "execute_ep1807"),
        "t17": (_ag3s_workspace_path('outputs/live_test/20260928_t17'), "shadow_ep1807")}
HAND_LINKS = ("ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
VARIANTS = {
    "t30": ({"attention_split_radius": None, "split_anchor_on_release": False, "hand_occlusion_reach": None}, False),
    "t31b": ({"split_anchor_on_release": False, "hand_occlusion_reach": None}, False),
    "s2": ({"hand_occlusion_reach": None}, False),
    "s3": ({"split_anchor_on_release": False}, True),
    "t32b": ({}, True),
}

def jobs():
    out = []
    for step, an_dir in (("T28", V / "T28/p3"), ("T30", V / "T30/p")):
        for cond in ("E3a", "E3b"):
            for rn in json.load(open(an_dir / f"{cond}_analysis.json"))["runs"]:
                out.append(("e3", step, cond, rn))
    for k in LIVE:
        out.append(("live", None, None, k))
    return out

def items_of(job):
    kind, step, cond, rn = job
    items = []
    if kind == "e3":
        an_dir = V / ("T28/p3" if step == "T28" else "T30/p")
        r = json.load(open(an_dir / f"{cond}_analysis.json"))["runs"][rn]
        rows = [json.loads(l) for l in open(V / step / f"{cond}_rec/{rn}/frames.jsonl") if l.strip()]
        plan = {x["seq"]: x for x in rows if x["kind"] == "planning"}
        for c in r["chunks"]:
            if c.get("server_chunk"):
                items.append((V / step / f"{cond}_rec" / c["server_chunk"], plan[c["seq"]], c["phase"], c["t"], None))
        return f"{step}:{rn}", items
    dd, loc = LIVE[rn]
    rows = [json.loads(l) for l in open(f"{dd}/{loc}/frames.jsonl") if '"planning"' in l]
    plan = {int(x["seq"]): x for x in rows if x["kind"] == "planning"}
    files = sorted(pathlib.Path(f"{dd}/server_constraints/run_0000").glob("chunk_*.npz"))
    attempt = None
    for f in files:
        k = int(f.stem.split("_")[1]); pr = plan.get(k + 1)
        if pr is not None and np.asarray(pr["actions_reference"])[:8, 7].min() < 0.5:
            attempt = k; break
    for f in files:
        k = int(f.stem.split("_")[1]); pr = plan.get(k + 1)
        if pr is None:
            continue
        fr = attempt is not None and k >= attempt
        items.append((f, pr, "post" if fr else "pre_grasp", k * 8, fr))
    return rn, items

def hand_of(z, s):
    names = s.get("sphere_link_names")
    if "sphere_centres" not in z.files or names is None:
        return None
    c = np.asarray(z["sphere_centres"], np.float64); r = np.asarray(z["sphere_radii"], np.float64)
    keep = np.array([n in HAND_LINKS for n in names])
    if not keep.any():
        return None
    return np.concatenate([c[keep], r[keep, None]], axis=1)

def gap(hand, pts):
    """min over hand spheres and points of |p - c| - r (m)."""
    if hand is None or pts is None or len(pts) == 0:
        return None
    from scipy.spatial import cKDTree
    t = cKDTree(np.asarray(pts, np.float64).reshape(-1, 3))
    d, _ = t.query(hand[:, :3], k=1)
    return float(np.min(d - hand[:, 3]))

def one(arg):
    job, variant = arg
    import dataclasses, mujoco
    import p12_records as P
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.types import PointCloud, AttentionPointCloud
    from AG3S.ag3s.stages.target_grounding import ground_target, TargetConfirm
    from AG3S.ag3s.stages.admissibility import destination_overlap, principal_extents
    cfg = AG3SConfig()
    import re
    TC = TargetConfirm
    mv = re.fullmatch(r"(s3|t32b)r(\d+)(nc)?", variant)
    if mv:   # parametric: s3r50 = hand only at reach 50 mm; t32br50 = S2 + S3; nc = no challenger-side test
        over = {"hand_occlusion_reach": int(mv.group(2)) / 1000.0}
        if mv.group(1) == "s3":
            over["split_anchor_on_release"] = False
        use_hand = True
        if mv.group(3):
            class TC(TargetConfirm):
                def _hidden_by_hand(self, *, hidden, challenger, points):
                    return TargetConfirm._hidden_by_hand(self, hidden=hidden, challenger=None, points=points)
    else:
        over, use_hand = VARIANTS[variant]
    fields = {f.name for f in dataclasses.fields(cfg.clustering)}
    ccfg = dataclasses.replace(cfg.clustering, **{k: v for k, v in over.items() if k in fields})
    confirm = TC(ccfg, max_opening=MO, overlap_distance=FINE)
    has_hand_api = hasattr(confirm, "note_hand")
    m, d, probe, obj, fing = P.build()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    label, items = items_of(job)
    switched = False
    rows = []
    lab_cache = {}
    for f, pr, phase, t, frozen_live in items:
        z = np.load(f, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        frozen = frozen_live if frozen_live is not None else bool((s.get("exclusion") or {}).get("grasp_active"))
        P.set_q(m, d, pr["qpos"]); op = pr["object_poses"]
        R = d.xmat[bid].reshape(3, 3).copy(); p0 = d.xpos[bid].copy()
        gt = {b: R.T @ (np.asarray(op[b]["pos"], float) - p0) for b in op if b in FRUIT or b == "crate"}
        hand = hand_of(z, s)
        e = {"t": int(t), "phase": phase, "frozen": bool(frozen), "file": str(f), "hand": hand is not None}
        if hand is not None and "apple" in gt:
            e["hand_gap_apple_mm"] = round(1000 * (float(np.min(np.linalg.norm(hand[:, :3] - gt["apple"], axis=1) - hand[:, 3])) - R_APPLE), 1)
        confirm.freeze(frozen)
        if has_hand_api:
            from AG3S.ag3s.stages.admissibility import HandSpheres
            confirm.note_hand(HandSpheres(hand[:, :3], hand[:, 3]) if use_hand and hand is not None else None)
        # hand gap to the anchor *before* this frame's decision (what the rule reads)
        pre_geom = confirm.last_geometry
        e["hand_gap_anchor_pre_mm"] = None if pre_geom is None or hand is None else round(1000 * gap(hand, pre_geom.points), 1)
        apc = AttentionPointCloud(PointCloud(z["attention_cloud"].astype(np.float64)), z["attention_values"],
                                  raw_attention=z["attention_raw"])
        g = ground_target(apc, ccfg, seed_percentile=cfg.attention.seed_percentile,
                          seed_threshold=cfg.attention.seed_threshold, exclude_mask=z["support_mask"],
                          min_radius=cfg.geometry.min_radius, timestamp=float(z["t_step"]), confirm=confirm)
        dec = confirm.last
        if dec is not None and dec.mode == "switch":
            switched = True
        man = confirm.manipulated
        # the admissible leader that is not the held object (the challenger) and its GT
        chal = None
        if dec is not None and g.clusters and confirm.last_admissibility is not None:
            adm = confirm.last_admissibility
            for i, c in enumerate(g.clusters):
                if adm[i].admissible:
                    if dec.cluster is not None and c is dec.cluster:
                        break
                    dd_ = {b: float(np.linalg.norm(gt[b] - c.centroid)) for b in gt}
                    chal = {"score": round(float(c.target_score), 4), "nearest": min(dd_, key=dd_.get),
                            "split_role": (c.split or {}).get("role")}
                    break
        v = {"mode": None if dec is None else dec.mode, "reason": None if dec is None else dec.reason,
             "streak": None if dec is None else dec.streak, "after_switch": switched,
             "n_split": sum(1 for x in g.splits if x.accepted), "n_triggered": len(g.splits),
             "challenger": chal,
             "hand_occluded": None if dec is None else getattr(dec, "hand_occluded", None)}
        if man is not None:
            rec = man.record()
            dd_ = {b: float(np.linalg.norm(gt[b] - man.centroid)) for b in gt}
            v.update({"id": rec["id"], "state": rec["state"], "nearest": min(dd_, key=dd_.get),
                      "centroid": [float(x) for x in man.centroid],
                      "anchor_from_split": (rec.get("anchor") or {}).get("from_split"),
                      "anchor_source": (rec.get("anchor") or {}).get("source"),
                      "release_split": (rec.get("anchor") or {}).get("release_split"),
                      "hand": rec.get("hand"),
                      "from_split": (rec.get("split") or {}).get("manipulated_from_split"),
                      "assoc": None if rec.get("association") is None else rec["association"]["outcome"]})
            pts = man.points
            if hand is not None and len(pts):
                v["hand_gap_anchor_mm"] = round(1000 * gap(hand, pts), 1)
            if len(pts) >= 3:
                ext = principal_extents(pts)
                v["extents_mm"] = [round(1000 * x, 1) for x in ext]
                v["all_axes_le_opening"] = bool(ext[-1] <= MO)
                labs = []
                for q in pts:
                    key = tuple(np.round(q, 6))
                    if key not in lab_cache:
                        body, cat, per = P.nearest(m, d, probe, obj, p0 + R @ q)
                        lab_cache[key] = body if per[body] <= 0.010 else "none"
                    labs.append(lab_cache[key])
                P.set_q(m, d, pr["qpos"])
                cnt = collections.Counter(labs); n = len(labs)
                big = [b for b in FRUIT if cnt[b] >= 10 and cnt[b] / n >= 0.10]
                v["fruits"] = {b: cnt[b] for b in FRUIT if cnt[b]}
                v["merged"] = bool("apple" in big and len(big) >= 2)
        ball = None
        if man is not None and man.usable and man.admissibility is not None and man.admissibility.admissible \
                and man.admissibility.narrowest_extent_m is not None and man.admissibility.narrowest_extent_m <= MO + 1e-12:
            dest = confirm.destination
            ov = destination_overlap(man.points, dest, FINE)
            if dest is None or ov < cfg.clustering.destination_overlap_fraction:
                pts = man.points; c = pts.mean(0)
                r = float(np.quantile(np.linalg.norm(pts - c, axis=1), cfg.esdf.target_ball_quantile)) \
                    + cfg.esdf.target_ball_pad_voxels * FINE
                da = float(np.linalg.norm(gt["apple"] - c))
                ball = {"radius_mm": 1000 * r, "apple_centre_mm": 1000 * da,
                        "apple_centre_inside": bool(da <= r), "apple_whole_inside": bool(da + R_APPLE <= r),
                        "neighbour_centres_inside": [b for b in gt if b not in ("apple", "crate")
                                                     and np.linalg.norm(gt[b] - c) <= r]}
        v["ball"] = ball
        e[variant] = v
        rows.append(e)
    return label, variant, rows

PART = pathlib.Path(__file__).parent / "parts"

def part_name(job, variant):
    kind, step, cond, rn = job
    return PART / f"{variant}__{(step or 'live')}_{rn}.json"

def one_saved(arg):
    job, variant = arg
    f = part_name(job, variant)
    if f.exists():
        return str(f)
    label, variant, rows = one(arg)
    tmp = f.with_suffix(".tmp")
    json.dump({"label": label, "variant": variant, "rows": rows}, open(tmp, "w"))
    tmp.rename(f)
    print("part", f.name, flush=True)
    return str(f)

def merge(vs, out_path):
    out = {}
    for j in jobs():
        for v in vs:
            d = json.load(open(part_name(j, v)))
            if d["label"] not in out:
                out[d["label"]] = d["rows"]
            else:
                for a, b in zip(out[d["label"]], d["rows"]):
                    assert a["t"] == b["t"]
                    a[v] = b[v]
    json.dump({"doc": __doc__, "variants": vs, "runs": out}, open(out_path, "w"))

if __name__ == "__main__":
    # partial results per (run, variant) in parts/ — a resumed run skips finished ones.
    # usage: replay.py OUT.json VARIANTS [NPROC]
    PART.mkdir(exist_ok=True)
    vs = sys.argv[2].split(",")
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    args = [(j, v) for j in jobs() for v in vs if not part_name(j, v).exists()]
    # longest runs first
    if args:
        with mp.Pool(min(len(args), nproc)) as pool:
            for _ in pool.imap_unordered(one_saved, args):
                pass
    merge(vs, sys.argv[1])
    print("done", len(jobs()), vs, flush=True)
