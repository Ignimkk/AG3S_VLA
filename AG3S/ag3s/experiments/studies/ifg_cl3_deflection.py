"""IFG-CL3 failure analysis — how much did the trajectory optimizer (TO) change the VLA's action, and when did HOLD stop it?

CPU only, read only. For every run of the IFG-CL3 paired re-evaluation and every planning chunk, the server record holds the
policy output (`reference_chunk`, 50 x 16) and the TO output (`refined_chunk`). The client executes the first
`n_control / n_planning` rows (8) of the chunk, or a HOLD target instead when the gate holds.

Per executed row: TCP deflection = |TCP(refined row) - TCP(reference row)| (mm) of the used arm, by forward kinematics on the
snapshot model (`integration.rby1.bringup.avoidance_arrow.TcpKinematics`, site left_ee / right_ee) with every other DoF at the
model keyframe `teleop` (torso / base / head are not commanded by the 16-D action). This is "how far the TO pushed the gripper
away from what the policy wanted at that state" — the same quantity the green avoidance arrow draws.

Windows per run (analysis/<run>.json of analyze_cl.py):
  approach = chunks whose first control row is before the first gripper close (pre_grasp.close_seq; no close -> all chunks)
  all      = every chunk
Per run: max / median / p90 of the per-chunk max deflection, number of chunks with max deflection > 10 mm and > 20 mm,
HOLD chunks (analysis hold list) in the window, and outcome (primary = contact-free success, failure_mode).

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_cl3_deflection --cl3 <run>/verify/IFG-CL3 --out <json>
"""
import argparse, json, pathlib, re, sys

import numpy as np

ARMS = ("C1aN", "SURF")
THRESH_MM = (10.0, 20.0)


def chunk_paths(seg):
    """Server record files of one run, in order (first..last of server_segment; a missing file stays None)."""
    first, last = pathlib.Path(seg["first"]), pathlib.Path(seg["last"])
    a = int(re.search(r"chunk_(\d+)", first.name).group(1)); b = int(re.search(r"chunk_(\d+)", last.name).group(1))
    out = []
    for k in range(a, b + 1):
        p = first.parent / f"chunk_{k:05d}.npz"
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cl3", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--model-xml", default=None)
    ap.add_argument("--segment", nargs="*", default=[
        "SURF_ep194_s1943=runs/server_8292_20261018T093310/server_constraints/run_0000/chunk_00675.npz:chunk_00749.npz"],
        help="NAME=FIRST:LAST for runs whose analysis has no server_segment (verified by t_step 1..75 of the boundary chunks)")
    ap.add_argument("--skip", nargs="*", default=["chunk_00720.npz@server_8292_20261018T093310"],
                    help="FILE@SERVERDIR records never opened (NFS-quarantined, absent from the run)")
    a = ap.parse_args()
    import mujoco
    from integration.rby1.bringup.avoidance_arrow import TcpKinematics, arm_targets

    V = a.cl3
    seeds = json.loads((V / "seeds.json").read_text())
    snap = V.parent.parent / "snapshot"
    xml = a.model_xml or str(snap / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    base = np.array(m.key_qpos[kid] if kid >= 0 else m.qpos0, np.float64)
    kin = TcpKinematics(m)
    skip = {tuple(s.split("@")) for s in a.skip}
    rows = []
    for it in seeds["items"]:
        for arm in ARMS:
            name = f"{arm}_ep{it['episode']}_s{it['seed']}"
            an = json.loads((V / "analysis" / f"{name}.json").read_text())
            side = an["used_arm"]
            n_exec = int(an["n_control"] // an["n_planning"])
            hold_seqs = {h["seq"] for h in an.get("hold") or []}
            close_seq = (an.get("pre_grasp") or {}).get("close_seq")
            per = []
            missing = 0
            seg = an.get("server_segment")
            ov = dict(x.split("=", 1) for x in a.segment)
            if seg is None and name in ov:
                f, l = ov[name].split(":"); f = V / f
                seg = {"first": str(f), "last": str(f.parent / l)}
            if seg is None:
                raise SystemExit(f"{name}: no server_segment and no --segment override")
            for i, p in enumerate(chunk_paths(seg)):
                seq = i + 1
                if (p.name, p.parent.parent.parent.name) in skip or not p.exists():
                    missing += 1; per.append(None); continue
                z = np.load(p, allow_pickle=True)
                ref, rfn = z["reference_chunk"][:n_exec], z["refined_chunk"][:n_exec]
                d = [float(np.linalg.norm(kin.displacement(base, side, arm_targets("rby1_16d", r0)[side], arm_targets("rby1_16d", r1)[side]))) * 1000
                     for r0, r1 in zip(ref, rfn)]
                per.append({"seq": seq, "t": int(z["t_step"]), "max_mm": max(d), "mean_mm": float(np.mean(d)), "hold": seq in hold_seqs})
            ok = [c for c in per if c]

            def summ(cs):
                mx = np.array([c["max_mm"] for c in cs]) if cs else np.zeros(0)
                return {"n_chunks": len(cs), "n_hold": sum(c["hold"] for c in cs),
                        "max_mm": float(mx.max()) if mx.size else None, "median_mm": float(np.median(mx)) if mx.size else None,
                        "p90_mm": float(np.percentile(mx, 90)) if mx.size else None,
                        **{f"n_over_{int(t)}mm": int((mx > t).sum()) for t in THRESH_MM}}
            appr = [c for c in ok if close_seq is None or c["seq"] < close_seq]
            rows.append({"name": name, "arm": arm, "episode": it["episode"], "seed": it["seed"], "stratum": it["stratum"], "side": side,
                         "success": an["success"], "grasp": an["grasp"], "failure_mode": an.get("failure_mode"), "close_seq": close_seq,
                         "grasp_t": an.get("grasp_t"), "n_missing_records": missing,
                         "approach": summ(appr), "all": summ(ok),
                         "per_chunk": [None if c is None else {k: c[k] for k in ("seq", "t", "max_mm", "hold")} for c in per]})
            print(name, an["success"], an.get("failure_mode"), rows[-1]["approach"]["max_mm"], rows[-1]["approach"]["n_hold"], flush=True)
    # group summaries: arm x outcome (success / grasp failure / other failure)
    def grp(r):
        return "success" if r["success"] else ("no_grasp" if not r["grasp"] else "fail_after_grasp")
    groups = {}
    for r in rows:
        g = groups.setdefault(f"{r['arm']}|{grp(r)}", {"n": 0, "approach_max_mm": [], "approach_median_mm": [], "approach_n_over_10mm": [],
                                                       "approach_n_hold": [], "all_n_hold": []})
        g["n"] += 1
        for k, src in (("approach_max_mm", ("approach", "max_mm")), ("approach_median_mm", ("approach", "median_mm")),
                       ("approach_n_over_10mm", ("approach", "n_over_10mm")), ("approach_n_hold", ("approach", "n_hold")), ("all_n_hold", ("all", "n_hold"))):
            v = r[src[0]][src[1]]
            if v is not None:
                g[k].append(v)
    gs = {k: {"n": v["n"], **{f"{kk}_median": (float(np.median(vv)) if vv else None) for kk, vv in v.items() if kk != "n"},
              **{f"{kk}_max": (float(np.max(vv)) if vv else None) for kk, vv in v.items() if kk != "n"},
              "runs_with_approach_hold": sum(1 for x in v["approach_n_hold"] if x > 0)} for k, v in sorted(groups.items())}
    # paired: episodes where the arms disagree on grasp, deflection of the failing arm
    a.out.write_text(json.dumps({"definitions": __doc__, "cl3": str(V), "model_xml": xml, "keyframe": "teleop" if kid >= 0 else "qpos0",
                                 "groups": gs, "rows": rows}, indent=1, default=float))
    print(json.dumps(gs, indent=1, default=float))


if __name__ == "__main__":
    main()
