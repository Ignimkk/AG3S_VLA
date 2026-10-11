"""T40 G — analysis of the B (no obstacle), C (hurdle) and D (adjusted hurdle) runs (verifier, CPU only).
usage: analyze_t40.py OUT_JSON [--no-contact]

Per run: analyze_t39.run_metrics (T39 rules, unchanged: success / grasp / place = analyze_e0; HOLD chunk = planning row with
not all control rows executed, kinds = gate.hold_kinds; verdicts; chunk times excl. first; contact settle) + hurdle_metrics.run_hurdle.
E3b server chunk records are paired to a run by CONTENT (a server segment = chunks from t_step 1; the segment with as many chunks
as the run's planning rows whose every refined_chunk equals the planning-row actions within 1e-6); B and C runs of one episode share the seed, so the T39 seed pairing alone is ambiguous here.
B runs: hurdle distances use the C config of the episode (virtual hurdle, no physics).
B vs T39 bit identity: traj.npz arrays (executed_actions, measured_qpos, predicted_chunks, inference_states, chunk_start_steps),
frames.jsonl planning actions / actions_reference / qpos / object_poses and control qpos / applied_ctrl (compare_pair.py
functions), and for E3b the server record refined_chunk / reference_chunk / clearance per chunk (T39 segment by seed in
T39/runs/E3b/server_*, B segment by time)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, datetime, re
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/p'))
import analyze_t39 as A39  # noqa: E402  (copy of T39/p/analyze_t39.py)
import compare_pair as CP  # noqa: E402
import hurdle_metrics as HM  # noqa: E402

G = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/runs'))
T39 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39/runs'))
CFG = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/configs'))
NAME = re.compile(r"^(E0|E3b)_ep(\d+)_s(\d+)$")


def iso(p):
    try:
        return datetime.datetime.fromisoformat(p.read_text().splitlines()[0].strip()).timestamp()
    except Exception:
        return None


def obstacle_of(start_file):
    for l in start_file.read_text().splitlines():
        if l.startswith("obstacle="):
            v = l.split("=", 1)[1]
            if v and v != ":":
                c, p = v.split(":", 1)
                return c, p
    return None


def segments_by_time(cond_dir):
    segs = []
    for rec in sorted(cond_dir.glob("server_*/server_constraints/run_*")):
        cur = []
        for f in sorted(rec.glob("chunk_*.npz")):
            ts = int(np.load(f, allow_pickle=True)["t_step"])
            if ts == 1 and cur:
                segs.append(cur); cur = []
            cur.append(f)
        if cur:
            segs.append(cur)
    return [(s[0].stat().st_mtime, s) for s in segs]


def t39_segment(ep, seed):
    for rec in sorted((T39 / "E3b").glob("server_*/server_constraints/run_*")):
        cur = []; out = []
        for f in sorted(rec.glob("chunk_*.npz")):
            z = np.load(f, allow_pickle=True)
            if int(z["t_step"]) == 1 and cur:
                out.append(cur); cur = []
            cur.append(f)
        if cur:
            out.append(cur)
        for sg in out:
            s0 = json.loads(str(np.load(sg[0], allow_pickle=True)["summary_json"]))
            if (s0.get("policy_seed") or {}).get("applied") == seed:
                return sg
    return None


def cmp_server(sa, sb):
    rs = {"n_chunks": [len(sa), len(sb)]}
    for key in ("refined_chunk", "reference_chunk", "clearance"):
        eq, mx, first = len(sa) == len(sb), 0.0, None
        for i in range(min(len(sa), len(sb))):
            c = CP.cmp_arr(np.load(sa[i], allow_pickle=True)[key], np.load(sb[i], allow_pickle=True)[key])
            if not c["equal"]:
                eq = False; first = i if first is None else first
            if c["max_abs"] is not None:
                mx = max(mx, c["max_abs"])
        rs[key] = {"equal": eq, "max_abs": mx, "first_diff_chunk": first}
    return rs


def bit_identity(rb, rt, seg_b, seg_t):
    res = {}
    za, zb = np.load(rb / "traj.npz"), np.load(rt / "traj.npz")
    res["traj_npz"] = {k: CP.cmp_arr(za[k], zb[k]) for k in ("executed_actions", "measured_qpos", "predicted_chunks", "inference_states", "chunk_start_steps")}
    ex = np.abs(za["executed_actions"] - zb["executed_actions"]).max(axis=1) if za["executed_actions"].shape == zb["executed_actions"].shape else None
    res["first_executed_step_diff"] = (int(np.argmax(ex > 0)) if (ex > 0).any() else None) if ex is not None else "shape"
    pa, ca = CP.load_frames(rb); pb, cb = CP.load_frames(rt)
    res["frames"] = {"planning_actions": CP.series(pa, pb, "actions"), "planning_actions_reference": CP.series(pa, pb, "actions_reference"),
                     "planning_qpos": CP.series(pa, pb, "qpos"), "planning_object_poses": CP.series(pa, pb, "object_poses", sub=True),
                     "control_qpos": CP.series(ca, cb, "qpos"), "control_applied_ctrl": CP.series(ca, cb, "applied_ctrl")}
    flat = [v["equal"] for v in res["traj_npz"].values()] + [v["equal"] for v in res["frames"].values()]
    if seg_b is not None and seg_t is not None:
        res["server_record"] = cmp_server(seg_b, seg_t)
        flat += [v["equal"] for k, v in res["server_record"].items() if isinstance(v, dict)]
    res["all_bit_identical"] = bool(all(flat))
    return res


def main():
    out_p = pathlib.Path(sys.argv[1]); do_contact = "--no-contact" not in sys.argv
    res = {"definitions": __doc__, "hurdle_definitions": HM.__doc__, "runs": {}}
    for cond in ("E0", "E3b"):
        cdir = G / cond
        if not cdir.exists():
            continue
        segs = segments_by_time(cdir) if cond == "E3b" else []
        for stage in ("B", "C", "D"):
            sd = cdir / stage
            if not sd.exists():
                continue
            for rd in sorted(p for p in sd.iterdir() if p.is_dir() and NAME.match(p.name)):
                if not (sd / f"{rd.name}.done").exists():
                    continue
                m = NAME.match(rd.name); ep, seed = int(m.group(2)), int(m.group(3))
                st, en = iso(sd / f"{rd.name}.start"), iso(sd / f"{rd.name}.end")
                # pairing by content: the segment with as many chunks as the run has planning rows whose every refined_chunk equals the
                # run's planning-row actions (<= 1e-6); ties (none seen) -> nearest first-chunk mtime to the run start.
                mine = []
                if cond == "E3b":
                    plan = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
                    acts = [np.asarray(p["actions"], np.float64) for p in plan if p["kind"] == "planning"]
                    hits = []
                    for (mt, sg) in segs:
                        if len(sg) != len(acts):
                            continue
                        if all(np.asarray(np.load(f, allow_pickle=True)["refined_chunk"], np.float64).shape == a.shape and
                               np.abs(np.asarray(np.load(f, allow_pickle=True)["refined_chunk"], np.float64) - a).max() <= 1e-6 for f, a in zip(sg, acts)):
                            hits.append((abs(mt - (st or mt)), sg))
                    mine = [sorted(hits, key=lambda x: x[0])[0][1]] if hits else []
                    r_pair_n = len(hits)
                r, _ = A39.run_metrics(cond, ep, "s", seed, rd, mine if cond == "E3b" else None, set(), do_contact)
                r["stage"] = stage; r["n_server_segments_matching_content"] = r_pair_n if cond == "E3b" else None
                ob = obstacle_of(sd / f"{rd.name}.start")
                if ob is None:  # B: virtual hurdle of the episode's C config
                    cfgs = sorted(CFG.glob(f"t40f_ep{ep}_s{seed}_r1.json"))
                    ob = (str(cfgs[0]), f"t40f_ep{ep}_s{seed}_r1") if cfgs else None
                    physical = False
                else:
                    physical = True
                if ob is not None:
                    r["hurdle"] = HM.run_hurdle(rd, ob[0], ob[1], r["grasp_t"], r["place_t"], physical, mine[0] if (cond == "E3b" and mine) else None)
                if stage == "B":
                    rt = T39 / cond / rd.name
                    r["bit_identity_vs_T39"] = bit_identity(rd, rt, mine[0] if mine else None, t39_segment(ep, seed) if cond == "E3b" else None) if rt.exists() else None
                res["runs"][f"{cond}/{stage}/{rd.name}"] = r
                h = r.get("hurdle") or {}
                print(cond, stage, rd.name, "S" if r["success"] else "-", "grasp", r["grasp"], "place", r["place"], "hold", r["n_hold_chunks"],
                      "hurdle_min A/B/C", [round(1e3 * ((h.get("phases") or {}).get("all") or {}).get(k, {}).get("min_m", np.nan), 1) if ((h.get("phases") or {}).get("all") or {}).get(k) else None for k in "ABC"],
                      "bit_id", (r.get("bit_identity_vs_T39") or {}).get("all_bit_identical"), flush=True)
    out_p.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
