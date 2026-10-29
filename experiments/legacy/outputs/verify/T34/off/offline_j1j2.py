"""T34 verifier offline replay on the T33 records (working tree = T34 code).

J1 (every T33 attach, found from p/<COND>_analysis.json grasp_latch.attach):
  rebuild  : AG3S(AG3SConfig + clustering.gripper_max_opening 0.07146, esdf voxel 0.02 / fine 0.005 / tsdf 0.005 / max 0.4 as serve_safe) with the constraint robot of
             fixed_input_ablation.build_robot; `_last_support_surfaces` = TLS plane of the attach-1 chunk's
             support_mask points with base z > 0.5 m; `_last_esdf_field` = recorded coarse+fine of the attach-1 chunk
             (fixed_input_ablation.build_field, outside 1.0 m); AG3S.attach(TargetGeometry(snapshot chunk target_points),
             robot_state=q_now (attach chunk planning row), parent_link ee_finger_l1, geometry_state=q_snap (planning
             row of the chunk with t_step = snapshot frame + 1)).
  spheres  : T33 = recorded held.spheres_parent_frame (fit + pad); T34 = rebuilt held.spheres_parent_frame (lifted).
  bottoms  : n.c - off - r (mm) against the TLS plane, and GT: MuJoCo mj_geomDistance of the sphere centre to the
             table body minus r (probe helper of T28 P1-2), at q_snap and at q_now.
  row      : held<->field row = d_field(c) - r (esdf_margin 0) with the recorded field of the chunk, spheres at q_now
             (chunk planning row) and at the reference chunk rows 0-7 (ChunkLayout.full_q); min over spheres and rows.
             Computed for the attach chunk and for every later chunk of that run while T33 was attached (grasp state
             held), per chunk: min row T33 spheres, min row T34 spheres, and the GT nearest body of the worst T34 row's
             sphere centre (surface distance - r).
J2 (every T33 run): GraspLatch(LatchConfig(), evidence=True, external_destination=True) replayed from the first chunk
  whose recorded grasp.state_before != 'searching' (phase set to the recorded state_before, _last_opening = recorded
  evidence opening_previous); per chunk: signal = grasp_signal_from_feedback(recorded exec_feedback, 0, closed_below),
  reach = recorded evidence reach_mm, manipulated id/state = recorded summary.grasp.evidence manipulated_id/state,
  placed = False, held_observation = from the previous chunk's recorded summary.manipulated when last_seen_frame ==
  frame >= attach frame: held_object.held_observation_check(centroid, FK(q of that chunk's planning row, frame_link)
  @ rebuilt primary fit centre (palm), primary fit at attach pose, attach geometry centroid) + held_radius = fit
  radius + pad. On a replayed attach the J1 rebuild is run at that chunk (same inputs as above).
  variants: default ; observation_only (revoke_use_opening False).
usage: offline_j1j2.py OUT.json
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, re, pathlib, copy, dataclasses
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P
import AG3S.trajopt.experiments.fixed_input_ablation as T
from AG3S.trajopt.types import ChunkLayout
from AG3S.trajopt.grasp_latch import GraspLatch, LatchConfig, GraspPhase, grasp_signal_from_feedback
from AG3S.ag3s.robot_models.held_object import held_observation_check

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33'))
PORT = {"E3a": 8204, "E3b": 8206}
MAX_OPENING = 0.07146
MB = P.build()
m, d, probe, obj, _ = MB
_ROB = {}


def robot(cond, run):
    if cond not in _ROB:
        rec = V / f"{cond}_rec"
        args = T.build_parser().parse_args([
            "--constraints", str(rec / f"server_{PORT[cond]}/server_constraints/run_0000"),
            "--frames", str(rec / run / "frames.jsonl"),
            "--server-log", str(rec / f"serve_safe_{PORT[cond]}.log"), "--out", "/dev/null"])
        settings = T.resolve_settings(args, T.parse_server_log(args.server_log),
                                      json.loads((rec / run / "manifest.json").read_text()))
        rb, qadr, _ = T.build_robot(settings)
        _ROB[cond] = (rb, qadr)
    return _ROB[cond]


def chunks(cond, run):
    A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
    rows = {int(x["seq"]): x for x in (json.loads(l) for l in (V / f"{cond}_rec" / run / "frames.jsonl")
                                       .read_text().splitlines() if l.strip()) if x.get("kind") == "planning"}
    out = []
    for c in A["chunks"]:
        if c.get("server_chunk"):
            f = V / f"{cond}_rec" / c["server_chunk"]
            z = np.load(f, allow_pickle=False)
            out.append({"t": int(c["t"]), "file": f, "row": rows[int(c["seq"])], "t_step": int(z["t_step"]),
                        "s": json.loads(str(z["summary_json"]))})
    return out


def npz(c):
    return np.load(c["file"], allow_pickle=False)


class _Surf:
    def __init__(self, n, off):
        self.normal = np.asarray(n, float); self.offset = float(off); self.id = 0


def tls_plane(z):
    pts = np.asarray(z["raw_cloud"], float)[np.asarray(z["support_mask"], bool)]
    tab = pts[pts[:, 2] > 0.5]
    c = tab.mean(0)
    n = np.linalg.svd(tab - c, full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    return n, float(n @ c), int(len(tab)), float(np.sqrt(np.mean((tab @ n - n @ c) ** 2)))


def make_ag3s(rb):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    # server-equivalent for what attach reads: pad = esdf.fine_voxel_size (serve_safe --fine-voxel 0.005)
    cfg = AG3SConfig().with_overrides({"clustering": {"gripper_max_opening": MAX_OPENING},
                                       "esdf": {"voxel_size": 0.02, "fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005,
                                                "max_distance": 0.4}})
    return AG3S(cfg, robot_model=rb, constraint_robot_model=rb, attached_parent_links=("ee_left", "ee_right"))


def target(points, tid=0):
    from AG3S.ag3s.stages.geometry import fit_sphere
    from AG3S.ag3s.types import TargetGeometry
    pts = np.asarray(points, np.float64)
    return TargetGeometry(id=tid, points=pts, point_indices=np.arange(len(pts)), centroid=pts.mean(0),
                          bounding_geometry=fit_sphere(pts), attention_score=0.9, confidence=0.9)


def rebuild(cond, run, ch, a):
    rb, qadr = robot(cond, run)
    s_a = ch[a]["s"]
    rec_held = ((s_a.get("grasp") or {}).get("attached") or {}).get("held")
    f = int(re.match(r"frame (\d+)", rec_held["snapshot_state"]).group(1)) if rec_held else None
    isnap = next(i for i, c in enumerate(ch) if c["t_step"] == f + 1)
    z_snap, z_prev = npz(ch[isnap]), npz(ch[a - 1])
    q_snap = T.q_now_from_qpos(ch[isnap]["row"]["qpos"], qadr)
    q_now = T.q_now_from_qpos(ch[a]["row"]["qpos"], qadr)
    n, off, npts, rms = tls_plane(z_prev)
    ag = make_ag3s(rb)
    ag._last_support_surfaces = [_Surf(n, off)]
    ag._last_esdf_field = T.build_field(z_prev, outside_distance=1.0)
    geo = target(np.asarray(z_snap["target_points"], float), tid=int((s_a.get("manipulated") or {}).get("id", 0)))
    ag.attach(geo, robot_state=q_now, parent_link="ee_finger_l1", geometry_state=q_snap)
    return ag, dict(rb=rb, qadr=qadr, q_snap=q_snap, q_now=q_now, n=n, off=off, npts=npts, rms=rms,
                    rec_held=rec_held, snap_frame=f, geo=geo)


def local_spheres(lst):
    return [(np.asarray(x["centre_m"], float), float(x["radius_mm"]) / 1000.0) for x in lst]


def world_spheres(rb, q, link, loc):
    Tq = rb.link_pose(q, link)
    return [(Tq[:3, :3] @ c + Tq[:3, 3], r) for c, r in loc]


def gt_gap(row_qpos, c_base, r):
    P.set_q(m, d, row_qpos)
    pw = P.base_to_world(m, d, c_base)
    body, cat, per = P.nearest(m, d, probe, obj, pw)
    return {b: round(1000 * (v - r), 2) for b, v in per.items()}


def rows_min(rb, qadr, c, link, loc):
    z = npz(c)
    field = T.build_field(z, outside_distance=1.0)
    lay = ChunkLayout.rby1(rb.joint_names)
    q_now = T.q_now_from_qpos(c["row"]["qpos"], qadr)
    traj = lay.chunk_to_trajectory(np.asarray(z["reference_chunk"], float))[:, :8]
    Q = [q_now] + list(lay.full_q(traj, q_now).T)
    best = (np.inf, None, None)
    per_q = []
    for qi, q in enumerate(Q):
        ws = world_spheres(rb, q, link, loc)
        dd = np.asarray(field.distance(np.array([w[0] for w in ws])), float)
        vals = dd - np.array([w[1] for w in ws])
        per_q.append(float(vals.min()))
        k = int(np.argmin(vals))
        if vals[k] < best[0]:
            best = (float(vals[k]), qi, k)
    ws0 = world_spheres(rb, Q[best[1]], link, loc) if best[1] is not None else None
    surf = None
    if ws0 is not None:
        c_ = ws0[best[2]][0]
        dc = float(np.asarray(field.distance(c_[None]), float)[0]); g = np.asarray(field.gradient(c_[None]), float)[0]
        gn = float(np.linalg.norm(g))
        surf = c_ - dc * g / gn if gn > 1e-9 else None
    return best, per_q, ws0, surf


def j1_event(cond, run, ch, a):
    ag, k = rebuild(cond, run, ch, a)
    rb, qadr = k["rb"], k["qadr"]
    held = ag._held_record
    link = held["frame_link"]
    rec = k["rec_held"]
    loc_t33 = local_spheres(rec["spheres_parent_frame"])
    loc_t34 = local_spheres(held["spheres_parent_frame"])
    loc_fit = local_spheres(held["filter_spheres_parent_frame"])
    n, off = k["n"], k["off"]
    snap_row = next(c for c in ch if c["t_step"] == k["snap_frame"] + 1)["row"]
    out = {"t_attach": ch[a]["t"], "snapshot_frame": k["snap_frame"],
           "fit_centre_vs_T33_record_mm": round(1000 * float(np.linalg.norm(
               np.asarray(held.get("fit", {}).get("centre_m", [np.nan] * 3)) - np.asarray(rec["fit"]["centre_m"]))), 3),
           "fit_radius_mm": {"rebuilt": held.get("fit", {}).get("radius_mm"), "T33": rec["fit"]["radius_mm"]},
           "plane_tls": {"normal": [round(float(v), 5) for v in n], "offset_m": round(off, 5), "n_pts": k["npts"],
                         "rms_mm": round(1000 * k["rms"], 2)},
           "support_record": held.get("support"),
           "radii_mm": {"T33": [round(1000 * r, 3) for _, r in loc_t33], "T34": [round(1000 * r, 3) for _, r in loc_t34]},
           "filter_spheres_equal_T33_fit_mm": round(1000 * max(float(np.linalg.norm(a_[0] - b_[0])) for a_, b_ in
                                                               zip(loc_fit, loc_t33)), 3)}
    for pose, q, row in (("snapshot", k["q_snap"], snap_row), ("attach", k["q_now"], ch[a]["row"])):
        for name, loc in (("T33", loc_t33), ("T34", loc_t34)):
            ws = world_spheres(rb, q, link, loc)
            out[f"bottom_above_tls_plane_mm_{pose}_{name}"] = [round(1000 * float(n @ c - off - r), 2) for c, r in ws]
            out[f"gt_table_gap_mm_{pose}_{name}"] = [gt_gap(row["qpos"], c, r).get("table") for c, r in ws]
    # rows: attach chunk + every later chunk that T33 recorded as held
    per_chunk = []
    global k_n, k_off
    k_n, k_off = n, off
    for j in range(a, len(ch)):
        g = ch[j]["s"].get("grasp") or {}
        if j > a and g.get("state") != "held" and not g.get("detach"):
            break
        rec_ = {"t": ch[j]["t"], "state_T33": g.get("state"), "detach_T33": bool(g.get("detach"))}
        for name, loc in (("T33", loc_t33), ("T34", loc_t34)):
            (v, qi, kk), per_q, ws, surf = rows_min(rb, qadr, ch[j], link, loc)
            rec_[f"row_min_mm_{name}"] = round(1000 * v, 2)
            rec_[f"row_min_mm_q_now_{name}"] = round(1000 * per_q[0], 2)
            if name == "T34" and ws is not None:
                c_, r_ = ws[kk]
                gg = gt_gap(ch[j]["row"]["qpos"], c_, r_)
                ga = gg.pop("apple", None)
                nb = min(gg, key=gg.get)
                rec_["worst_T34"] = {"sphere": kk, "row_index": qi, "gt_nearest_body_excl_apple": nb,
                                     "gt_gap_mm": gg[nb], "gt_table_gap_mm": gg.get("table"), "gt_apple_gap_mm": ga}
                if surf is not None:
                    gs = gt_gap(ch[j]["row"]["qpos"], surf, 0.0)
                    rec_["worst_T34"]["field_surface_point"] = {
                        "def": "c - d(c) grad/|grad| (field nearest-surface estimate) -> GT surface distance per body (mm)",
                        "gt_nearest_body": min(gs, key=gs.get), "gt_nearest_mm": gs[min(gs, key=gs.get)],
                        "apple_mm": gs.get("apple"), "table_mm": gs.get("table"),
                        "height_above_tls_plane_mm": round(1000 * float(k_n @ surf - k_off), 2)}
        per_chunk.append(rec_)
        if g.get("detach"):
            break
    out["rows_per_chunk"] = per_chunk
    out["attach_chunk_row_min_mm"] = {"T33": per_chunk[0]["row_min_mm_T33"], "T34": per_chunk[0]["row_min_mm_T34"]}
    return out, ag, k


def j2_replay(cond, run, ch, use_opening=True):
    cfg = LatchConfig() if use_opening else dataclasses.replace(LatchConfig(), revoke_use_opening=False)
    L = GraspLatch(cfg, evidence=True, external_destination=True)
    start = next((i for i, c in enumerate(ch) if ((c["s"].get("grasp") or {}).get("state_before") or "searching")
                  != "searching"), None)
    events = []
    if start is None:
        return {"events": events, "note": "no latched chunk in T33 record"}
    g0 = ch[start]["s"]["grasp"]
    L.phase = GraspPhase(g0["state_before"])
    L._last_opening = (g0.get("evidence") or {}).get("opening_previous")
    ref = None
    for j in range(start, len(ch)):
        s = ch[j]["s"]; g = s.get("grasp") or {}; ev = g.get("evidence") or {}
        sig = grasp_signal_from_feedback(s.get("exec_feedback"), 0, cfg.gripper_closed_below)
        reach = None if ev.get("reach_mm") is None else ev["reach_mm"] / 1000.0
        obs = None
        if L.holding and ref is not None and j > 0:
            pm = ch[j - 1]["s"].get("manipulated")
            if pm and int(pm.get("last_seen_frame", -1)) == int(pm.get("frame", -2)) and int(pm["frame"]) >= ref["frame"]:
                rb, qadr = robot(cond, run)
                q = T.q_now_from_qpos(ch[j - 1]["row"]["qpos"], qadr)
                Tq = rb.link_pose(q, ref["link"])
                centre = Tq[:3, :3] @ ref["local"] + Tq[:3, 3]
                obs = held_observation_check(observed_centroid=pm["centroid"], held_centre=centre,
                                             held_centre_at_attach=ref["at_attach"],
                                             object_centroid_at_attach=ref["centroid"])
                obs.update({"held_radius_m": ref["radius"], "frame": int(pm["frame"])})
        e = L.observe_grasp(sig, reach_m=reach, manipulated_id=ev.get("manipulated_id"),
                            manipulated_state=ev.get("manipulated_state"), placed=False, held_observation=obs)
        if e.attach:
            ag, k = rebuild_at(cond, run, ch, j)
            hc = ag._held_check_ref
            ref = {"frame": ch[j]["t_step"] - 1,
                   "local": hc["primary_fit_local"], "radius": hc["primary_fit_radius"],
                   "at_attach": hc["primary_fit_at_attach"], "centroid": hc["object_centroid_at_attach"],
                   "link": ag._held_record["frame_link"]}
        if e.attach or e.detach or (e.evidence or {}).get("hold_checks"):
            events.append({"t": ch[j]["t"], "attach": e.attach, "detach": e.detach, "revoked": e.revoked,
                           "state": L.phase.value, "hold_window": (e.evidence or {}).get("hold_window"),
                           "hold_checks": (e.evidence or {}).get("hold_checks"),
                           "revoke": (e.evidence or {}).get("revoke"), "held_observation": obs,
                           "T33_recorded": {"attach": bool(g.get("attach")), "detach": bool(g.get("detach")),
                                            "state": g.get("state")}})
    return {"start_t": ch[start]["t"], "events": events,
            "attach_t": [x["t"] for x in events if x["attach"]], "detach_t": [x["t"] for x in events if x["detach"]],
            "revoke_t": [x["t"] for x in events if x["revoked"]],
            "revoke_reasons": [x["revoke"]["reasons"] for x in events if x["revoked"]],
            "T33_attach_t": [c["t"] for c in ch if (c["s"].get("grasp") or {}).get("attach")],
            "T33_detach_t": [c["t"] for c in ch if (c["s"].get("grasp") or {}).get("detach")]}


def rebuild_at(cond, run, ch, j):
    # the replayed attach happens at chunk j; the T33 record at j (if it attached there) has the snapshot frame.
    if ((ch[j]["s"].get("grasp") or {}).get("attached") or {}).get("held") is None:
        # replayed attach where T33 did not attach: snapshot = manipulated last_seen_frame of the previous chunk
        rb, qadr = robot(cond, run)
        pm = ch[j - 1]["s"].get("manipulated") or {}
        f = int(pm.get("last_seen_frame", ch[j]["t_step"] - 1))
        s2 = copy.deepcopy(ch[j]["s"])
        s2.setdefault("grasp", {}).setdefault("attached", {})["held"] = {"snapshot_state": f"frame {f}"}
        ch = list(ch); ch[j] = dict(ch[j], s=s2)
        ag, k = rebuild(cond, run, ch, j)
        return ag, k
    return rebuild(cond, run, ch, j)


def main():
    out = {"doc": __doc__, "j1": {}, "j2": {}}
    for cond in ("E3a", "E3b"):
        A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"]
        for run in sorted(A):
            ch = chunks(cond, run)
            for a, c in enumerate(ch):
                if (c["s"].get("grasp") or {}).get("attach"):
                    ev, _, _ = j1_event(cond, run, ch, a)
                    out["j1"][f"{run}@{c['t']}"] = ev
                    print("J1", run, c["t"], json.dumps({k: ev[k] for k in ev if k not in ("rows_per_chunk", "support_record")})[:1500], flush=True)
            out["j2"][run] = {"default": j2_replay(cond, run, ch, True),
                              "observation_only": j2_replay(cond, run, ch, False)}
            print("J2", run, {k: out["j2"][run]["default"].get(k) for k in ("attach_t", "detach_t", "revoke_t", "revoke_reasons", "T33_attach_t", "T33_detach_t")},
                  "obs_only", out["j2"][run]["observation_only"].get("revoke_t"), flush=True)
            pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    out["n_invalid_geomdistance"] = P.N_INVALID[0]
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
