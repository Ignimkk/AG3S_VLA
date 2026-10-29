"""T7a s2 — s1 의 행들을 **joint 축**과 **run 대조 축**으로 접는다.

읽는 것: `s1_active_rows.json` (행 단위 측정). 새로 재지 않는다.
내는 것:
  1. joint 축 표 — `left_arm_0..6` · `right_arm_0..6` 마다 활성 제약에 걸린 chunk 수.
     link -> joint 사슬은 `s0_model_identity.json` 에서 온 것 (MuJoCo kinematic tree).
  2. link 축 표 — rank 1 횟수 · band 별 chunk 수 · tightest-24 점유 slot 수.
     (`ee_finger_l1` / `ee_finger_l2` 포함)
  3. shadow(성공, executed=reference) 대 closed-loop(실패, executed=refined) 를 같은 chunk 축에.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib

import numpy as np

SRC = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a/s1_active_rows.json'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a/s2_joint_axis.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a/s2_tables.npz'))
BANDS = (0.0, 5.0, 10.0)
SEL = "no_target__left_spheres"
SEL_ALL = "no_target__all_spheres"


def main() -> None:
    src = json.loads(SRC.read_text())
    meta = src["meta"]
    chain = meta["joint_chain_by_link"]
    joints = meta["arm_joints"]
    links = meta["link_axis"]

    out: dict = {"source": str(SRC), "selection_note": {
        "no_target": "apple geom 제외 (authorized target 은 ESDF 에서 파낸다)",
        "left_spheres": "left arm + ee_finger_l1/l2 의 구만",
        "exec8": "chunk 50 스텝 중 실제 실행된 앞 8 스텝 (T6d m3b 실측)",
    }, "bands_mm": list(BANDS), "runs": {}, "comparison": {}}

    tables: dict = {}
    for run, rd in src["runs"].items():
        out["runs"][run] = {}
        for variant in rd["chunks"][0]["variants"]:
            for sel in (SEL, SEL_ALL):
                jt = {j: {"chunks_rank1": 0, "tightest24_slots": 0,
                          **{f"chunks_min_below_{b:g}mm": 0 for b in BANDS}} for j in joints}
                lk = {ln: {"chunks_rank1": 0, "tightest24_slots": 0,
                           **{f"chunks_min_below_{b:g}mm": 0 for b in BANDS}} for ln in links}
                n_rows_below = {f"{b:g}mm": 0 for b in BANDS}
                obst_rank1: dict[str, int] = {}
                mat = np.full((rd["n_chunks"], len(links)), np.nan)
                for c in rd["chunks"]:
                    s = c["variants"][variant][sel]
                    top = s["top_rows_exec8"][0]
                    lk[top["link"]]["chunks_rank1"] += 1
                    obst_rank1[top["obstacle_body"]] = obst_rank1.get(top["obstacle_body"], 0) + 1
                    for j in chain.get(top["link"], []):
                        jt[j]["chunks_rank1"] += 1
                    for ln, v in s["per_link_min_margin_slack_mm_exec8"].items():
                        mat[c["chunk"], links.index(ln)] = v
                        for b in BANDS:
                            if v <= b:
                                lk[ln][f"chunks_min_below_{b:g}mm"] += 1
                                for j in chain.get(ln, []):
                                    jt[j][f"chunks_min_below_{b:g}mm"] += 1
                    for ln, n in s["tightest24_link_counts_exec8"].items():
                        lk[ln]["tightest24_slots"] += int(n)
                        for j in chain.get(ln, []):
                            jt[j]["tightest24_slots"] += int(n)
                    for b in BANDS:
                        n_rows_below[f"{b:g}mm"] += int(s[f"n_rows_below_{b:g}mm_exec8"])
                out["runs"][run][f"{variant}__{sel}"] = {
                    "n_chunks": rd["n_chunks"],
                    "joint_axis": jt,
                    "link_axis": lk,
                    "rank1_obstacle_counts": dict(sorted(obst_rank1.items())),
                    "total_rows_below_band_exec8": n_rows_below,
                    "chunk_min_slack_mm": [
                        round(c["variants"][variant][sel]["min_margin_slack_mm_exec8"], 4)
                        for c in rd["chunks"]],
                    "chunk_rank1_link": [c["variants"][variant][sel]["top_rows_exec8"][0]["link"]
                                         for c in rd["chunks"]],
                    "chunk_rank1_obstacle": [
                        c["variants"][variant][sel]["top_rows_exec8"][0]["obstacle_body"]
                        for c in rd["chunks"]],
                    "chunk_rank1_step": [
                        c["variants"][variant][sel]["top_rows_exec8"][0]["step"]
                        for c in rd["chunks"]],
                }
                tables[f"{run}__{variant}__{sel}"] = mat

    # shadow(성공) 대 closed-loop(실패) — 같은 chunk 축
    a = out["runs"]["shadow_approach"][f"reference__{SEL}"]
    b = out["runs"]["closed_loop_approach"][f"refined__{SEL}"]
    sa = np.asarray(a["chunk_min_slack_mm"], np.float64)
    sb = np.asarray(b["chunk_min_slack_mm"], np.float64)
    diff_link = [i for i in range(len(sa)) if a["chunk_rank1_link"][i] != b["chunk_rank1_link"][i]]
    diff_obs = [i for i in range(len(sa))
                if a["chunk_rank1_obstacle"][i] != b["chunk_rank1_obstacle"][i]]
    ma = tables[f"shadow_approach__reference__{SEL}"]
    mb = tables[f"closed_loop_approach__refined__{SEL}"]
    tables["divergence_per_link_mm"] = mb - ma
    per_link_diff = {ln: {
        "mean_mm": round(float(np.nanmean(mb[:, i] - ma[:, i])), 4),
        "max_abs_mm": round(float(np.nanmax(np.abs(mb[:, i] - ma[:, i]))), 4),
        "argmax_chunk": int(np.nanargmax(np.abs(mb[:, i] - ma[:, i]))),
        "shadow_min_over_chunks_mm": round(float(np.nanmin(ma[:, i])), 4),
        "closed_loop_min_over_chunks_mm": round(float(np.nanmin(mb[:, i])), 4),
    } for i, ln in enumerate(links) if not np.all(np.isnan(ma[:, i]))}
    out["comparison"] = {
        "axis": "chunk 0..74 (planning seq 1..75)",
        "shadow_executed": "reference", "closed_loop_executed": "refined",
        "selection": SEL,
        "shadow_chunks_min_slack_below_0mm": int((sa <= 0).sum()),
        "closed_loop_chunks_min_slack_below_0mm": int((sb <= 0).sum()),
        "shadow_min_slack_mm_over_all_chunks": round(float(sa.min()), 4),
        "closed_loop_min_slack_mm_over_all_chunks": round(float(sb.min()), 4),
        "shadow_median_min_slack_mm": round(float(np.median(sa)), 4),
        "closed_loop_median_min_slack_mm": round(float(np.median(sb)), 4),
        "n_chunks_rank1_link_differs": len(diff_link),
        "first_chunk_rank1_link_differs": (diff_link[0] if diff_link else None),
        "n_chunks_rank1_obstacle_differs": len(diff_obs),
        "first_chunk_rank1_obstacle_differs": (diff_obs[0] if diff_obs else None),
        "per_link": per_link_diff,
        "first_chunk_shadow_negative_closed_loop_positive": next(
            (i for i in range(len(sa)) if sa[i] <= 0.0 < sb[i]), None),
        "n_chunks_shadow_negative_closed_loop_positive": int(((sa <= 0) & (sb > 0)).sum()),
    }
    # shadow 안에서 reference 대 refined (TO 가 바꾼 양) — 같은 chunk 축
    c = out["runs"]["shadow_approach"][f"refined__{SEL}"]
    sc = np.asarray(c["chunk_min_slack_mm"], np.float64)
    out["comparison"]["shadow_reference_vs_refined"] = {
        "n_chunks_refined_larger": int((sc > sa).sum()),
        "mean_gain_mm": round(float(np.mean(sc - sa)), 4),
        "max_gain_mm": round(float(np.max(sc - sa)), 4),
        "min_gain_mm": round(float(np.min(sc - sa)), 4),
        "refined_chunks_below_0mm": int((sc <= 0).sum()),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ, link_axis=np.asarray(links), **tables)

    print("=== joint axis · closed_loop_approach / refined /", SEL)
    jt = out["runs"]["closed_loop_approach"][f"refined__{SEL}"]["joint_axis"]
    for j in [x for x in joints if x.startswith("left")]:
        print(f"  {j:14s} rank1={jt[j]['chunks_rank1']:3d}/75  "
              f"<=0mm={jt[j]['chunks_min_below_0mm']:3d}  <=5mm={jt[j]['chunks_min_below_5mm']:3d}  "
              f"<=10mm={jt[j]['chunks_min_below_10mm']:3d}  t24_slots={jt[j]['tightest24_slots']}")
    print("=== link axis · closed_loop_approach / refined /", SEL)
    for ln, v in out["runs"]["closed_loop_approach"][f"refined__{SEL}"]["link_axis"].items():
        if v["tightest24_slots"] or v["chunks_rank1"]:
            print(f"  {ln:18s} rank1={v['chunks_rank1']:3d}  <=0mm={v['chunks_min_below_0mm']:3d}  "
                  f"<=10mm={v['chunks_min_below_10mm']:3d}  t24={v['tightest24_slots']:5d}")
    print("=== comparison")
    print(json.dumps({k: v for k, v in out["comparison"].items() if k != "per_link"},
                     indent=1, ensure_ascii=False))
    print("=== per link diff (closed_loop refined - shadow reference)")
    for ln, v in per_link_diff.items():
        print(f"  {ln:18s} mean={v['mean_mm']:9.3f}  max_abs={v['max_abs_mm']:9.3f} "
              f"@chunk {v['argmax_chunk']:2d}  shadow_min={v['shadow_min_over_chunks_mm']:9.3f}  "
              f"cl_min={v['closed_loop_min_over_chunks_mm']:9.3f}")
    print("wrote", OUT, NPZ)


if __name__ == "__main__":
    main()
