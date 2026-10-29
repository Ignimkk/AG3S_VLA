from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob
from pathlib import Path
OUT = Path(_ag3s_workspace_path('outputs/verify/T20'))
summ = {}
for ms in [0.0, 0.05, 0.1, 0.3]:
    for lost in [None, 10, 20]:
        rows = json.load(open(OUT / f"t14_replay_min{ms}_lost{lost}.json"))
        key = f"min{ms}_lost{lost}"
        seq = {r["seq"]: r for r in rows}
        def m(r, k): return (r["manipulated"] or {}).get(k)
        sw = [(r["seq"], m(r, "switched_from"), m(r, "id"), r["manip_nearest"][0]) for r in rows if r["decision_mode"] == "switch"]
        w = [seq[s] for s in range(12, 22)]
        summ[key] = dict(
            min_score=ms, lost_frames=lost,
            ids_seq10_22=[m(seq[s], "id") for s in range(10, 23)],
            states_seq10_22=[m(seq[s], "state") for s in range(10, 23)],
            ages_seq10_22=[m(seq[s], "age_frames") for s in range(10, 23)],
            modes_seq10_22=[seq[s]["decision_mode"] for s in range(10, 23)],
            status_seq10_22=[seq[s]["grounding_status"] for s in range(10, 23)],
            manip_to_apple_mm_seq10_22=[seq[s]["manip_to_apple_mm"] for s in range(10, 23)],
            manip_nearest_seq10_22=[(seq[s]["manip_nearest"] or [None])[0] for s in range(10, 23)],
            seq12_21_all_id0=all(m(r, "id") == 0 for r in w),
            seq12_21_all_occluded=all(m(r, "state") == "occluded" for r in w),
            seq12_21_n_occluded=sum(m(r, "state") == "occluded" for r in w),
            seq12_21_max_manip_to_apple_mm=max(r["manip_to_apple_mm"] for r in w),
            switches_all_seq=sw,
            first_switch_seq=(sw[0][0] if sw else None),
            max_occluded_age_all=max([m(r, "age_frames") for r in rows if m(r, "state") == "occluded"] or [0]),
            max_age_all=max(m(r, "age_frames") or 0 for r in rows),
            lost_seqs=[r["seq"] for r in rows if m(r, "state") == "lost"],
            lost_status_seqs=[r["seq"] for r in rows if r["grounding_status"] == "lost"],
            seq22=dict(status=seq[22]["grounding_status"], state=m(seq[22], "state"), age=m(seq[22], "age_frames"), id=m(seq[22], "id")),
            seq23_31=[(r["seq"], m(r, "id"), m(r, "state"), m(r, "age_frames"), r["decision_mode"], round(r["manip_to_apple_mm"], 1)) for r in rows if r["seq"] >= 23],
            final_id=m(rows[-1], "id"), final_state=m(rows[-1], "state"),
            final_to_apple_mm=rows[-1]["manip_to_apple_mm"],
            n_frames=len(rows),
        )
json.dump(summ, open(OUT / "t14_sweep_summary.json", "w"), indent=1)
for k, v in summ.items():
    print(k, "id0", v["seq12_21_all_id0"], "occ", v["seq12_21_n_occluded"], "sw", v["switches_all_seq"], "maxocc", v["max_occluded_age_all"], "lost", v["lost_seqs"], "s22", v["seq22"])
    print("   ", v["seq23_31"])
