"""T35-diag E — E3b_ep1800_r1 attach (t=144) / attach_revoked (t=160): GT around t=136..176 from a/a_runs.json + T34 revoke detail.
Per control row: grip cmd, measured opening, d_l1/d_l2 (GT finger-apple, mm), contact midpoint - apple centre (world mm, witness),
apple dz / xy displacement from run start (mm), apple world position. 'in_hand' = both d_l1 and d_l2 <= 1.0 mm.
Latch records (T34 p/E3b_revoke_detail.json): attach / hold_checks / revoke reasons with their feedback seq and chunk t.
Feedback rows of a chunk decision = control rows of the previous chunk (feedback seq = chunk seq - 1).
usage: revoke_gt.py -> e/revoke_gt.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json
R = json.load(open("a/a_runs.json"))["runs"]["E3b_rec/E3b_ep1800_r1"]["timeline"]
RD = json.load(open(_ag3s_workspace_path('outputs/verify/T34/p/E3b_revoke_detail.json')))["runs"]["E3b_ep1800_r1"]
rows = []
for e in R:
    if 136 <= e["t"] <= 176:
        rows.append({k: e[k] for k in ("t", "seq", "k", "grip_cmd", "open_meas", "d_l1_mm", "d_l2_mm", "mid_rel_apple_mm", "dz_mm", "disp_xy_mm", "apple")}
                    | {"in_hand": e["d_l1_mm"] <= 1.0 and e["d_l2_mm"] <= 1.0})
def seg(a, b):
    s = [r for r in rows if a <= r["t"] <= b]
    return {"t": [a, b], "n_in_hand": sum(r["in_hand"] for r in s), "n": len(s), "max_dz_mm": max(r["dz_mm"] for r in s), "min_dz_mm": min(r["dz_mm"] for r in s),
            "open_first": s[0]["open_meas"], "open_last": s[-1]["open_meas"], "max_d_l1_mm": max(r["d_l1_mm"] for r in s), "max_d_l2_mm": max(r["d_l2_mm"] for r in s),
            "first_not_in_hand_t": next((r["t"] for r in s if not r["in_hand"]), None)}
out = {"definitions": __doc__, "latch_records": RD, "rows": rows,
       "feedback_of_attach_chunk_t144": seg(136, 143), "attach_chunk_exec_t144_151": seg(144, 151),
       "feedback_of_revoke_chunk_t160": seg(152, 159), "after_revoke_t160_176": seg(160, 176),
       "last_in_hand_t": max((r["t"] for r in rows if r["in_hand"]), default=None),
       "apple_dz_max_mm_136_176": max(r["dz_mm"] for r in rows), "t_apple_dz_max": max(rows, key=lambda r: r["dz_mm"])["t"]}
json.dump(out, open("e/revoke_gt.json", "w"), indent=1)
for k in ("feedback_of_attach_chunk_t144", "attach_chunk_exec_t144_151", "feedback_of_revoke_chunk_t160", "after_revoke_t160_176"):
    print(k, out[k])
print("last_in_hand_t", out["last_in_hand_t"], "dzmax", out["apple_dz_max_mm_136_176"], "at", out["t_apple_dz_max"])
