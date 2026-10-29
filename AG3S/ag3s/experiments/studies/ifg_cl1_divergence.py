"""IFG-CL1 diagnostic — where do a C1aN run and an oracle-path run (ORAC / ORACS / SURF / SURFS) of the same seed diverge, and what did
TrajOpt do there?

Per run (analysis/<run>.json server_segment -> chunk k = client planning step k):
  deflection_k   = max |refined_chunk - reference_chunk| over the first 8 rows (executed part), radians, per arm side
  gripper_cmd_k  = reference / refined gripper column of the active side (row 0)
  min clearance of the authorized links (ee_<side>, ee_finger_<side>1/2) and of the other robot spheres, step 0..7 (mm)
  verdict status from summary_json (status / verdict / to)
Pair: first chunk where the executed rows (refined[:8]) differ by > 1e-4 rad between the two arms.

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_cl1_divergence --cl1 <run>/verify/IFG-CL1 --seeds 391 764 ... --arms ORAC SURF
"""
import argparse, glob, json, pathlib

import numpy as np

GRIP = {"left": 15, "right": 7}  # 16-D action layout: right arm 0-6, right gripper 7, left arm 8-14, left gripper 15
ARM = {"left": slice(8, 15), "right": slice(0, 7)}


def chunks(cl1, name):
    a = json.loads((cl1 / "analysis" / f"{name}.json").read_text())
    seg = a["server_segment"]; d0 = pathlib.Path(seg["first"]).parent
    i0 = int(pathlib.Path(seg["first"]).stem.split("_")[1])
    return a, [d0 / f"chunk_{i0 + k:05d}.npz" for k in range(int(seg["n"]))]


def per_chunk(f, side):
    z = np.load(f, allow_pickle=True)
    s = json.loads(str(z["summary_json"]))
    ref, refd = np.asarray(z["reference_chunk"], float), np.asarray(z["refined_chunk"], float)
    names = s.get("sphere_link_names") or []
    cl = np.asarray(z["clearance"], float) if "clearance" in z.files else None
    sd = "l" if side == "left" else "r"
    auth = {f"ee_{side}", f"ee_finger_{sd}1", f"ee_finger_{sd}2"}
    out = dict(t_step=int(z["t_step"]), defl=float(np.abs(refd[:8, ARM[side]] - ref[:8, ARM[side]]).max()),
               grip_ref=float(ref[0, GRIP[side]]), grip_exec=float(refd[0, GRIP[side]]), exec=refd[:8].copy(),
               status=s.get("status"), verdict=(s.get("verdict") or {}).get("decision") if isinstance(s.get("verdict"), dict) else s.get("verdict"))
    if cl is not None and names and len(names) == cl.shape[1]:
        a_idx = [i for i, n in enumerate(names) if n in auth]; o_idx = [i for i, n in enumerate(names) if n not in auth]
        out["clr_auth_mm"] = float(cl[:8, a_idx].min() * 1e3) if a_idx else None
        out["clr_other_mm"] = float(cl[:8, o_idx].min() * 1e3) if o_idx else None
        if a_idx:
            j = int(np.argmin(cl[:8, a_idx].min(0))); out["clr_auth_link"] = names[a_idx[j]]
    te = s.get("target_exclusion") or {}
    out["te_mode"] = te.get("mode"); out["n_removed"] = te.get("n_removed"); out["n_kept"] = te.get("n_kept_nontarget")
    return out


def run(cl1, name):
    a, fs = chunks(cl1, name)
    side = a.get("used_arm") or "left"
    rows = [per_chunk(f, side) for f in fs]
    return a, side, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cl1", type=pathlib.Path, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True); ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    res = {}
    for ep in a.seeds:
        base = glob.glob(str(a.cl1 / "analysis" / f"C1aN_ep{ep}_s*.json"))[0]
        bn = pathlib.Path(base).stem
        ab, side, rb = run(a.cl1, bn)
        res[ep] = {"C1aN": dict(success=ab["success"], grasp_t=ab["grasp_t"], failure=ab["failure_mode"])}
        for arm in a.arms:
            f = glob.glob(str(a.cl1 / "analysis" / f"{arm}_ep{ep}_s*.json"))
            if not f:
                continue
            ax, _, rx = run(a.cl1, pathlib.Path(f[0]).stem)
            k0 = next((k for k in range(min(len(rb), len(rx))) if np.abs(rb[k]["exec"] - rx[k]["exec"]).max() > 1e-4), None)
            pre = lambda rows, gt: [r for r in rows if gt is None or r["t_step"] < gt]
            g = ab["grasp_t"]
            rbp, rxp = pre(rb, g), pre(rx, g)
            summ = lambda rows: dict(n=len(rows), defl_med=float(np.median([r["defl"] for r in rows])) if rows else None,
                                     defl_gt_0p05=int(sum(r["defl"] > 0.05 for r in rows)),
                                     clr_auth_med=float(np.median([r["clr_auth_mm"] for r in rows if r.get("clr_auth_mm") is not None])) if rows else None)
            d = dict(success=ax["success"], failure=ax["failure_mode"], first_divergence_chunk=k0,
                     at_div=None if k0 is None else {"t_step": rb[k0]["t_step"],
                                                     "C1aN": {k: rb[k0].get(k) for k in ("defl", "grip_ref", "grip_exec", "clr_auth_mm", "clr_auth_link", "clr_other_mm", "status", "te_mode", "n_removed", "n_kept")},
                                                     arm: {k: rx[k0].get(k) for k in ("defl", "grip_ref", "grip_exec", "clr_auth_mm", "clr_auth_link", "clr_other_mm", "status", "te_mode", "n_removed", "n_kept")}},
                     pre_grasp_C1aN=summ(rbp), pre_grasp_arm=summ(rxp),
                     grip_close_cmd_chunks={"C1aN": int(sum(r["grip_exec"] < 0.5 for r in rbp)), arm: int(sum(r["grip_exec"] < 0.5 for r in rxp))},
                     grip_ref_vs_exec_diff_chunks={arm: int(sum(abs(r["grip_exec"] - r["grip_ref"]) > 1e-3 for r in rx))})
            res[ep][arm] = d
    a.out.write_text(json.dumps(res, indent=1, default=float))
    for ep, d in res.items():
        print("ep", ep, "C1aN", d["C1aN"])
        for arm, x in d.items():
            if arm == "C1aN":
                continue
            print(f"  {arm}: {x['failure']} first_div={x['first_divergence_chunk']} at_div={json.dumps(x['at_div'], default=float)[:420]}")
            print(f"     pre-grasp C1aN {x['pre_grasp_C1aN']}  {arm} {x['pre_grasp_arm']}  grip-close chunks {x['grip_close_cmd_chunks']}  grip changed by TO {x['grip_ref_vs_exec_diff_chunks']}")


if __name__ == "__main__":
    main()
