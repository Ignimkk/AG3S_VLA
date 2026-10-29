"""T42-2 analysis of the obstacle runs (verifier, CPU only). usage: analyze_t42_2.py OUT_JSON
Runs: outputs/verify/T42/2/runs/<COND>/<KIND>/<COND>_ep<EP>_s<SEED> with NAME.done (check_run passed); KIND = place | pick | both;
obstacle config / profile from NAME.start ('obstacle=CFG:PROF').
Per run: analyze_t42 (success / grasp / place, T28 rule generalized), grasp_robust (definition B), HOLD chunks / gate actions,
obst_metrics.run_obst (distances per slot x group per phase, client contacts and stop, AG3S target vs obstacle for E3b).
E3b server chunks are paired to a run by content (a server segment = chunks from t_step 1; the segment with as many chunks as the
run's planning rows whose every refined_chunk equals the planning-row actions within 1e-6).
Reference: the episode's no-obstacle runs of the same seed from T42-1 (T39 for apple + left rep 1-2).
Facts per E3b run that is not a success (listed, not judged): stop phase / stop_t, the contact pair(s), HOLD chunk times and kinds
(and the longest consecutive HOLD streak before the stop), obstacle-as-target counts, target dropped (grasp_robust slip_B)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, re
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/2/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))
import analyze_t42 as AN  # noqa: E402
import grasp_robust as GR  # noqa: E402
import obst_metrics as OM  # noqa: E402

R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/2/runs'))
NAME = re.compile(r"^(E0|E3b|E3m)_ep(\d+)_s(\d+)$")


def obstacle_of(start):
    for l in start.read_text().splitlines():
        if l.startswith("obstacle="):
            v = l.split("=", 1)[1]
            if v and v != ":":
                c, p = v.split(":", 1); return c, p
    return None


def segments(cdir):
    segs = []
    for rec in sorted(cdir.glob("server_*/server_constraints/run_*")):
        cur = []
        for f in sorted(rec.glob("chunk_*.npz")):
            if int(np.load(f, allow_pickle=True)["t_step"]) == 1 and cur:
                segs.append(cur); cur = []
            cur.append(f)
        if cur:
            segs.append(cur)
    return segs


def pair(segs, rd):
    plan = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    acts = [np.asarray(p["actions"], np.float64) for p in plan if p["kind"] == "planning"]
    hits = []
    for sg in segs:
        if len(sg) != len(acts):
            continue
        ok = True
        for f, a in zip(sg, acts):
            rc = np.asarray(np.load(f, allow_pickle=True)["refined_chunk"], np.float64)
            if rc.shape != a.shape or np.abs(rc - a).max() > 1e-6:
                ok = False; break
        if ok:
            hits.append(sg)
    return hits


def holds(rd):
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = [r for r in rows if r["kind"] == "planning"]; ctrl = [r for r in rows if r["kind"] == "control"]
    ht = []; kinds = {}
    for p in plan:
        cs = [c for c in ctrl if c["chunk_seq"] == p["seq"]]
        if cs and not all(c.get("executed", True) for c in cs):
            ht.append(p["t_step"])
            for k in (p.get("gate") or {}).get("hold_kinds") or []:
                kinds[k] = kinds.get(k, 0) + 1
    streak = best = 0; prev = None
    for x in ht:
        streak = streak + 1 if (prev is not None and x - prev == 8) else 1; best = max(best, streak); prev = x
    return {"hold_t": ht, "n_hold": len(ht), "hold_kinds": kinds, "max_consecutive_hold": best}


def main():
    out = {"definitions": __doc__, "obst_definitions": OM.__doc__, "robust_definitions": GR.__doc__, "runs": {}}
    for cdir in sorted(p for p in R.glob("*") if p.is_dir()):
        segs = segments(cdir) if cdir.name.startswith("E3") else []
        for kdir in sorted(p for p in cdir.glob("*") if p.is_dir() and not p.name.startswith(("server_", "failed"))):
            for rd in sorted(p for p in kdir.iterdir() if p.is_dir() and NAME.match(p.name)):
                if not (kdir / f"{rd.name}.done").exists():
                    continue
                m = NAME.match(rd.name); cond, ep, seed = m.group(1), int(m.group(2)), int(m.group(3))
                ob = obstacle_of(kdir / f"{rd.name}.start")
                a = AN.analyze(rd); rb = GR.robust(rd); rb.pop("series", None)
                hits = pair(segs, rd) if segs else []
                r = {"cond": cond, "kind": kdir.name, "episode": ep, "seed": seed, "obstacle": ob,
                     **{k: a[k] for k in ("target_fruit", "used_arm", "success", "grasp", "place", "grasp_t", "place_t", "n_control", "n_planning",
                                          "non_target_in_crate_final", "gate_action_counts", "policy_infer_ms_excl_first")},
                     "robust": rb, "holds": holds(rd), "n_server_segments_matching": len(hits) if segs else None}
                r["obst"] = OM.run_obst(rd, ob[0], ob[1], hits[0] if len(hits) == 1 else None) if ob else None
                out["runs"][f"{cond}/{kdir.name}/{rd.name}"] = r
                o = r["obst"] or {}
                print(cond, kdir.name, rd.name, "S" if r["success"] else "-", "grasp", r["grasp"], "place", r["place"], "stop", o.get("obstacle_stop"),
                      o.get("stop_phase"), "hold", r["holds"]["n_hold"], "contact", (o.get("client_obstacle_safety") or {}).get("contact_pairs"), flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
