"""T43 S2 analysis (verifier, CPU): demo replay vs dataset vs policy rollouts, per episode.

Runs     : DEMO = raw/replay/DEMO_ep<EP> (replay_demo.py); policy E0 / E3b = every finished run of the episode in
           outputs/verify/T42/1/runs/{E0,E3b}/noobs (.done) and outputs/verify/T39/runs/{E0,E3b} (.done).
Outcome  : T42-1 grasp_robust.robust(run_dir) = analyze_t42 (T39/T28 rule generalized: grasp = target z >= z0 + 0.050 m at a
           planning row; place = target centre inside the crate interior at the LAST planning row; success = both) + close
           command rows (applied arm gripper crossing 0.9 downward), lift_t, robust grasp (T42-1 definition B).
Geometry : T43 S measure_pick.geom_at at the first close command row (onset) and onset + 16 control rows (settled):
           tip_dz_mm / tip_dxy_mm = fingertip midpoint vs the fruit body centre, tilt_deg = approach axis vs -z,
           approach_yaw_deg_mirrored, closing_yaw_deg_mirrored_mod180, closing_elev_deg, grasp_dz/dxy (finger bodies), contacts.
           Dataset row: T43 S raw/demo_grasp.json (dataset observation.state at grasp_close_start_frame + 16; fruits at the
           recorded initial / settled poses).
Tracking : DEMO control row t qpos (commanded-at pose) vs dataset observation.state[t]: used-arm joints max |diff| (deg) over
           t <= min(len, onset + 16) and over all frames; gripper opening (|finger qpos| / 0.045) vs state col 7 / 15;
           final target xy vs the dataset's final_fruit_poses (mm).
usage: analyze_s2.py EP [EP ...]  -> raw/s2_analysis.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib, glob
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/S/p'))
import grasp_robust as GR  # noqa: E402
import measure_pick as MP  # noqa: E402

S2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S2'))
DS = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1'))
META = {r["episode_index"]: r for r in (json.loads(l) for l in (DS / "meta/randomized_episodes.jsonl").read_text().splitlines() if l.strip())}
DEMO_G = {e["episode"]: e for e in json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw/demo_grasp.json')).read_text())["episodes"]}
KEEP = ("tip_dz_mm", "tip_dxy_mm", "tip_spread_mm", "tilt_deg", "approach_yaw_deg_mirrored", "closing_yaw_deg_mirrored_mod180",
        "closing_elev_deg", "grasp_dz_mm", "grasp_dxy_mm", "opening_meas")


def policy_runs(ep):
    out = []
    for cond in ("E0", "E3b"):
        for root in (f"{_ag3s_workspace_path()}/outputs/verify/T42/1/runs/{cond}/noobs", f"{_ag3s_workspace_path()}/outputs/verify/T39/runs/{cond}"):
            for d in sorted(glob.glob(f"{root}/{cond}_ep{ep}_s*")):
                if os.path.isdir(d) and os.path.exists(d + ".done"):
                    out.append((cond, d))
    seen, uniq = set(), []
    for cond, d in out:  # a seed present in both roots: keep the first (T42-1)
        k = (cond, pathlib.Path(d).name)
        if k not in seen:
            seen.add(k); uniq.append((cond, d))
    return uniq


def geom_rows(rd, arm, fruit, close_t):
    rows = [json.loads(l) for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = {c["t_step"]: c for c in rows if c["kind"] == "control"}
    res = {}
    if close_t is None:
        return res, ctrl
    for nm, t in (("onset", close_t), ("settled", close_t + 16)):
        if t in ctrl:
            g = MP.geom_at(np.asarray(ctrl[t]["qpos"], float), arm, fruit)
            res[nm] = {"t": t, **{k: g[k] for k in KEEP}, "n_contacts": len(g.get("contacts", [])),
                       "contact_bodies": sorted({c["body"] for c in g.get("contacts", [])}),
                       "tip_rel_fruit_mm": ((g["tip_mid"] - g["fruit_pos"]) * 1e3).tolist(),
                       "fruit_pos": g["fruit_pos"].tolist()}
    return res, ctrl


def one_run(cond, rd, arm, fruit):
    r = GR.robust(rd)
    ct = (r.get("close_cmd_t") or [None])[0]
    g, ctrl = geom_rows(rd, arm, fruit, ct)
    keep = ("success", "grasp", "place", "grasp_t", "place_t", "close_cmd_t", "n_close_cycles", "lift_t", "regrasp", "robust_grasp",
            "robust_grasp_strict", "release_t")
    return {"cond": cond, "run_dir": rd, **{k: r.get(k) for k in keep}, "geom": g}, ctrl


def tracking(ep, arm, ctrl, onset):
    df = pd.read_parquet(DS / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet", columns=["observation.state"])
    S = np.stack(df["observation.state"].to_numpy()).astype(float)
    J = MP.model()["J"]; s = arm
    cols = list(range(0, 7)) if arm == "left" else list(range(8, 15)); gcol = 7 if arm == "left" else 15
    adr = [J[f"{s}_arm_{i}"] for i in range(7)]; gadr = J[f"gripper_finger_{s[0]}1"]
    T = [t for t in sorted(ctrl) if t < len(S)]
    D = np.array([np.abs(np.asarray(ctrl[t]["qpos"])[adr] - S[t, cols]) for t in T])
    G = np.array([abs(abs(ctrl[t]["qpos"][gadr]) / MP.OPEN_M - S[t, gcol]) for t in T])
    lim = len(T) if onset is None else min(len(T), onset + 17)
    rec = META[ep]; fruit = rec["target_fruit"]
    last = ctrl[max(ctrl)]["qpos"]; fin = np.asarray(last[MP.QADR[fruit]:MP.QADR[fruit] + 3])
    dfin = np.asarray(rec["final_fruit_poses"][fruit][:3])
    return {"n_frames_compared": len(T), "arm_joint_maxabs_deg_to_settled": float(np.degrees(D[:lim].max())),
            "arm_joint_maxabs_deg_all": float(np.degrees(D.max())), "arm_joint_median_deg_all": float(np.degrees(np.median(D))),
            "gripper_opening_maxabs_all": float(G.max()),
            "final_target_xy_vs_dataset_final_mm": float(1e3 * np.linalg.norm(fin[:2] - dfin[:2])),
            "final_target_z_vs_dataset_final_mm": float(1e3 * (fin[2] - dfin[2]))}


def main(eps):
    out = {"definitions": __doc__, "episodes": {}}
    p = S2 / "raw/s2_analysis.json"
    if p.exists():
        out = json.loads(p.read_text()); out["definitions"] = __doc__
    for ep in eps:
        rec = META[ep]; arm, fruit = rec["used_arm"], rec["target_fruit"]
        e = {"episode": ep, "arm": arm, "fruit": fruit, "split": rec["split"], "dataset_success": rec.get("success"),
             "dataset_events": {k: (rec.get("events") or {}).get(k) for k in ("grasp_close_start_frame", "grasp_verified_frame",
                                                                              "lift_from_table_start_frame", "success_frame")}}
        dg = DEMO_G.get(ep)
        e["dataset_geom"] = None if dg is None else {nm: {k: dg["rows"][nm].get(k) for k in KEEP if k in dg["rows"][nm]} | {"frame": dg["rows"][nm]["frame"]}
                                                     for nm in ("onset", "settled") if nm in dg["rows"]}
        rd = str(S2 / f"raw/replay/DEMO_ep{ep}")
        demo, ctrl = one_run("DEMO", rd, arm, fruit)
        e["demo_replay"] = demo
        e["demo_tracking"] = tracking(ep, arm, ctrl, demo["close_cmd_t"][0] if demo["close_cmd_t"] else None)
        e["policy_runs"] = [one_run(c, d, arm, fruit)[0] for c, d in policy_runs(ep)]
        out["episodes"][str(ep)] = e
        print(ep, arm, fruit, "DEMO", "S" if demo["success"] else ("g" if demo["grasp"] else "-"),
              {k: round(v, 1) for k, v in (demo["geom"].get("settled") or {}).items() if k in ("tip_dz_mm", "tip_dxy_mm", "tilt_deg")},
              "policy", [(r["cond"], "S" if r["success"] else ("g" if r["grasp"] else "-")) for r in e["policy_runs"]], flush=True)
        p.write_text(json.dumps(MP.jl(out), indent=1))


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
