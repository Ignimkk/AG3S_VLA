"""Every attach in an index: per held chunk the TO held rows (old = flag off rebuild, new = flag on) on the
recorded field + refined plan. Output JSON per run."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, resource, numpy as np, pathlib
sys.path.insert(0, "/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/r2")
from multiprocessing import Pool
OUT = pathlib.Path("/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/r2/scan4")
OUT.mkdir(exist_ok=True)

def one(args):
    idx_path, key = args
    from replay import Run, plane_surface, FIA
    IDX = json.load(open(idx_path))
    R = Run(key, IDX[key])
    out = {"run": key, "attaches": []}
    for a in R.attaches():
        try:
            ag0, att0, rec = R.rebuild(a, False)
            ag1, att1, _ = R.rebuild(a, True)
        except Exception as e:
            out["attaches"].append({"attach": a, "error": repr(e)}); continue
        hr0 = ag0.held_record
        A = {"attach": a, "hand": R.summ[a]["grasp"].get("hand"), "parent": att0.parent_link,
             "fit_dc_mm": float(1e3*np.linalg.norm(np.array(hr0["fit"]["centre_m"])-np.array(rec["fit"]["centre_m"]))),
             "off_matches_record": bool(np.allclose([s["centre_m"] for s in hr0["spheres_parent_frame"]], [s["centre_m"] for s in rec["spheres_parent_frame"]], atol=2e-6)
                                         and np.allclose([s["radius_mm"] for s in hr0["spheres_parent_frame"]], [s["radius_mm"] for s in rec["spheres_parent_frame"]], atol=2e-3)),
             "body": ag1.held_record["body_cover"], "chunks": []}
        pl = (rec.get("support") or {}).get("plane")
        planes = [] if pl is None else [plane_surface(pl)]
        seqs = R.held_seqs(a)
        for s in seqs:
            field, chunk = R.field(s)
            old, oldrows = R.held_rows(s, att0, field, chunk)
            with np.load(R.npz[s], allow_pickle=True) as zz:
                ref = np.asarray(zz["reference_chunk"], float)
            path = R.G["layout"].full_q(R.G["layout"].chunk_to_trajectory(ref)[:, :8], R.q_now(s)).T
            fa = ag1._frame_held_attached(R.q_now(s), planes, path)
            fr = ag1.held_record["body_cover"]["frame"]
            new, newrows = R.held_rows(s, fa, field, chunk)
            g = R.summ[s].get("grasp") or {}
            A["chunks"].append({"seq": s, "t": R.plan[s]["t_step"], "near": fr["near_support"], "extra_lift_mm": fr.get("extra_lift_mm"), "gap_mm": fr["min_bottom_gap_mm"], "gap_now_mm": fr["bottom_gap_now_mm"], "near_row": fr["near_row"],
                                "old_mm": (1e3*old).tolist(), "new_mm": (1e3*new).tolist(),
                                "detach_next": bool((R.summ.get(s+1, {}).get("grasp") or {}).get("detach")),
                                "last_held": s == seqs[-1],
                                "gate": (R.plan[s].get("gate") or {}).get("action"),
                                "to_clear_mm": None if (R.summ[s].get("to") or {}).get("metrics", {}).get("clearance_m") is None else 1e3*R.summ[s]["to"]["metrics"]["clearance_m"]})
        out["attaches"].append(A)
    out["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    name = key.replace(_ag3s_workspace_path('outputs/verify/'), "").replace("/", "_")
    (OUT / f"{name}.json").write_text(json.dumps(out))
    return key, out["maxrss_mb"], len(out["attaches"])

if __name__ == "__main__":
    idx_path = sys.argv[1]; nw = int(sys.argv[2]); subs = sys.argv[3:]
    keys = [k for k in json.load(open(idx_path)) if not subs or any(x in k for x in subs)]
    if nw <= 1:
        for k in keys: print(one((idx_path, k)), flush=True)
    else:
        with Pool(nw, maxtasksperchild=4) as p:
            for r in p.imap_unordered(one, [(idx_path, k) for k in keys]): print(r, flush=True)
