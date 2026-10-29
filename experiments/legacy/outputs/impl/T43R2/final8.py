from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np
sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import test_t43r2_held_body_cover_replay as T
out = {}
for rel in T.RUNS:
    R = T.Run(rel); ag_on, _ = R.on; _, att_off = R.off
    bc = ag_on.held_record["body_cover"]
    won, woff, fin = [], [], []
    for s in R.held_seqs():
        fa, fr = R.frame_attached(s, "on")
        v = T._apple_in_palm(R, s, fa.parent_link)
        woff.append(T._gap(v, att_off).max())
        if not fr["near_support"]:
            g = T._gap(v, fa); won.append(g.max()); fin.append(np.mean(g <= 0))
    t_c = R.truth["apple_bar_contact_t"]; s_c = max(s for s in R.held_seqs() if R.plan[s]["t_step"] <= t_c)
    by = {c["seq"]: c for c in R.truth["held_chunks"]}
    fa, _ = R.frame_attached(s_c, "on")
    on = 1e3 * R.held_rows(s_c, fa); off = 1e3 * R.held_rows(s_c, att_off); tr = np.array(by[s_c]["true_apple_bar_mm_per_step"])
    near = sum(R.frame_attached(s, "on")[1]["near_support"] for s in R.held_seqs())
    out[rel] = dict(radii=[x["radius_mm"] for x in bc["spheres"]], mode=bc["mode"], shift=bc.get("bottom_shift_mm"), raised=bc.get("radius_raised_mm"),
                    lift=[x["lift_mm"] for x in bc["lift"]["spheres"]], n_held=len(R.held_seqs()), n_near=near,
                    cov_on_max_mm=1e3*max(won), cov_on_min_frac=min(fin), cov_off_max_mm=1e3*max(woff),
                    contact_seq=s_c, on_min=on.min(), off_min=off.min(), true_min=tr.min(), steps_true_le0=np.flatnonzero(tr<=0).tolist(), steps_on_le0=np.flatnonzero(on<=0).tolist())
    print(rel, json.dumps({k: (np.round(v,1).tolist() if isinstance(v,(list,float,np.floating)) else v) for k,v in out[rel].items()}))
json.dump(out, open("/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/r2/final8.json","w"), default=float)
