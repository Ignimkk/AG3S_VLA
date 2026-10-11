"""T40 J analysis (verifier, CPU). usage: analyze_j.py OUT_JSON RUNS_ROOT [--no-contact]
RUNS_ROOT = outputs/verify/T40/J/runs (J reruns) or outputs/verify/T40/G/runs (re-measure the G records with the fixed distance).
Same as T40/p/analyze_t40.py (T39 rules, content-paired server segments) but hurdle distances use hurdle_metrics_j (snapshot
${AG3S_PROJECT_ROOT}-t40h, fixed client distance). For J runs: bit identity vs the T40 G record of the same stage/cond/episode up to the old
stop step k_old = G n_control - 1 (exclusive of the stop row itself): executed_actions[:k_old], control rows 0..k_old-1 qpos and
applied_ctrl, planning rows whose seq <= ceil(k_old / 8) actions / actions_reference / qpos / object_poses, and (E3b) server
refined_chunk / reference_chunk for those chunks."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/J/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/p'))
import hurdle_metrics_j as HMJ
sys.modules['hurdle_metrics'] = HMJ   # analyze_t40 imports hurdle_metrics: use the fixed-distance copy
import analyze_t40 as AT
AT.HM = HMJ
import compare_pair as CP

GREC = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/runs'))


def prefix_identity(rj, rg, segj, segg):
    zg, zj = np.load(rg / "traj.npz"), np.load(rj / "traj.npz")
    k = int(zg["executed_actions"].shape[0]) - 1
    out = {"k_old_stop_step": k}
    ea, eb = zj["executed_actions"][:k], zg["executed_actions"][:k]
    out["executed_actions_prefix"] = CP.cmp_arr(ea, eb)
    pj, cj = CP.load_frames(rj); pg, cg = CP.load_frames(rg)
    out["control_qpos_prefix"] = CP.series(cj[:k], cg[:k], "qpos")
    out["control_applied_ctrl_prefix"] = CP.series(cj[:k], cg[:k], "applied_ctrl")
    nch = math.ceil(k / 8)
    for key, sub in (("actions", None), ("actions_reference", None), ("qpos", None), ("object_poses", True)):
        out[f"planning_{key}_prefix"] = CP.series(pj[:nch], pg[:nch], key, sub=sub)
    if segj is not None and segg is not None:
        out["server_prefix"] = AT.cmp_server(segj[:nch], segg[:nch])
    flat = [out["executed_actions_prefix"]["equal"]] + [v["equal"] for kk, v in out.items() if kk.endswith("_prefix") and isinstance(v, dict) and "equal" in v and kk != "executed_actions_prefix"]
    if "server_prefix" in out:
        flat += [v["equal"] for v in out["server_prefix"].values() if isinstance(v, dict)]
    out["all_bit_identical_prefix"] = bool(all(flat))
    return out


def main():
    out_p, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); do_contact = "--no-contact" not in sys.argv
    AT.G = root
    if "/J/" in str(root):
        orig = AT.bit_identity
    AT.main.__globals__["G"] = root
    sys.argv = [sys.argv[0], str(out_p)] + (["--no-contact"] if not do_contact else [])
    AT.main()
    res = json.loads(out_p.read_text())
    res["hurdle_metrics_module"] = HMJ.__file__
    if "/J/" in str(root):
        for k, r in res["runs"].items():
            cond, stage, name = k.split("/")
            rj, rg = root / cond / stage / name, GREC / cond / stage / name
            segj = segg = None
            if cond == "E3b":
                segj = [pathlib.Path(r["server_segment"]["first"]).parent / f for f in []] if False else None
                # re-pair by content for both records
                for base, which in ((root, "j"), (GREC, "g")):
                    segs = AT.segments_by_time(base / cond)
                    plan = [json.loads(l) for l in ((rj if which == "j" else rg) / "frames.jsonl").read_text().splitlines() if l.strip()]
                    acts = [np.asarray(p["actions"], np.float64) for p in plan if p["kind"] == "planning"]
                    hit = next((sg for (mt, sg) in segs if len(sg) == len(acts) and all(np.abs(np.asarray(np.load(f, allow_pickle=True)["refined_chunk"], np.float64) - a).max() <= 1e-6 for f, a in zip(sg, acts))), None)
                    if which == "j":
                        segj = hit
                    else:
                        segg = hit
            r["prefix_identity_vs_G"] = prefix_identity(rj, rg, segj, segg)
            print(k, "prefix identical", r["prefix_identity_vs_G"]["all_bit_identical_prefix"], "k_old", r["prefix_identity_vs_G"]["k_old_stop_step"], flush=True)
        out_p.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
