from common.workspace import workspace_path as _ag3s_workspace_path
import json
from collections import defaultdict
from pathlib import Path
OUT = Path(_ag3s_workspace_path('outputs/verify/T20'))
rows = json.load(open(OUT / "synth_rows.json"))
groups = defaultdict(list)
for r in rows:
    key = (r["sequence"], r["min_score"], r["lost_frames"], r.get("crate_level"), r.get("K"), r.get("reobserve_offset_mm"), r.get("occluder"))
    groups[key].append(r)
out = {"A": [], "B": [], "C": []}
for key, rs in groups.items():
    seq, ms, lost, lc, K, off, occl = key
    rs.sort(key=lambda r: r["frame"])
    st = [(r["manipulated"] or {}).get("state") for r in rs]
    ids = [(r["manipulated"] or {}).get("id") for r in rs]
    ages = [(r["manipulated"] or {}).get("age_frames") for r in rs]
    sw = [r["frame"] for r in rs if r["decision_mode"] == "switch"]
    if seq.startswith("A") or seq.startswith("B"):
        crate_scores = sorted({round(r["leader_score"], 4) for r in rs if r["phase"] != "apple_visible"})
        crate_score = [s for r in rs if r["phase"] != "apple_visible" for s, c in r["scores"] if c[1] > 0.4][0]
        out[seq[0]].append(dict(min_score=ms, lost_frames=lost, crate_level=lc, crate_score=crate_score,
                                switch_frames=sw, switch_after_n_attention_frames=(sw[0] - 2 if sw else None),
                                final_id=ids[-1], final_state=st[-1], states=st, ids=ids, ages=ages,
                                modes=[r["decision_mode"] for r in rs]))
    else:
        back = [r for r in rs if r["phase"] == "apple_back"]
        occ = [r for r in rs if r["phase"] == "occluded"]
        out["C"].append(dict(min_score=ms, lost_frames=lost, K=K, reobserve_offset_mm=off, occluder=occl,
                             switch_frames=sw, ids=ids, states=st, ages=ages,
                             occluded_phase_states=sorted(set(x for x in st[3:3 + K])),
                             occluded_phase_status=sorted({r["grounding_status"] for r in occ}),
                             max_age=max(a or 0 for a in ages),
                             id_before=ids[2], id_on_return=ids[3 + K], id_final=ids[-1],
                             same_id_on_return=(ids[3 + K] == ids[2]),
                             state_on_return=st[3 + K], mode_on_return=back[0]["decision_mode"],
                             modes=[r["decision_mode"] for r in rs]))
json.dump(out, open(OUT / "synth_summary.json", "w"), indent=1)
print("A/B: switch frame index (frames 0-2 apple; attention frames start at 3) -> after n attention frames")
for s in "AB":
    for d in sorted(out[s], key=lambda d: (d["lost_frames"] is not None, d["lost_frames"] or 0, d["min_score"], d["crate_level"])):
        if d["lost_frames"] is None:
            print(s, d["min_score"], round(d["crate_score"], 4), d["switch_after_n_attention_frames"], d["final_id"], d["final_state"])
print("C")
for d in sorted(out["C"], key=lambda d: (d["lost_frames"] is not None, d["lost_frames"] or 0, d["min_score"], d["occluder"], d["K"], d["reobserve_offset_mm"])):
    print(d["min_score"], d["lost_frames"], d["occluder"], d["K"], d["reobserve_offset_mm"], "sw", d["switch_frames"], "same", d["same_id_on_return"], d["id_before"], d["id_on_return"], d["state_on_return"], d["mode_on_return"], d["occluded_phase_states"], d["max_age"])
