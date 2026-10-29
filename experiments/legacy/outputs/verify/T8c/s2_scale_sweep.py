"""T8c s2 — 배율 sweep. 새 거리 계산 없이 T8a tensor 를 반지름만 바꿔 다시 읽는다.

T8a tensor 는 clearance = surface_distance - r(scale 1.0) 이다. 중심이 배율에 안 움직이므로
  surface_distance = clearance_1.0 + r_1.0[i]          (구마다)
  clearance_s      = surface_distance - r_s[i]         (s1 이 실제 클래스에서 읽은 r_s)
가 정확하다. r_s 는 s1_radii_by_scale.npz 에서 온다 — 배율에 선형이 아니라서 직접 곱하지 않는다.

조건 (T8a M2 와 같다): 사과 행은 authorized 셋 {ee_finger_l1, ee_finger_l2, link_left_arm_5}
에서만 지운다. margin 10 mm. exec8 과 full50 둘 다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib

import numpy as np

T8A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8a'))
T8C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c'))
OUT = T8C / "s2_scale_sweep.json"
NPZ = T8C / "s2_scale_curves.npz"
MARGIN = 10.0
EXEC_H = 8
AUTH = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")


def main() -> None:
    t = np.load(T8A / "m0_clearance_tensor.npz", allow_pickle=True)
    clear1 = t["clearance_m"].astype(np.float64) * 1000.0          # (F, H, S, B) mm
    bodies = [str(b) for b in t["bodies"]]
    link_names = [str(n) for n in t["link_names"]]
    r1 = np.asarray(t["radii_m"], np.float64) * 1000.0             # (S,)
    F, H, S, B = clear1.shape
    uniq = sorted(set(link_names))
    link_of = np.asarray([uniq.index(n) for n in link_names], np.int64)
    ai = bodies.index("apple")
    auth_sphere = np.asarray([n in AUTH for n in link_names], bool)

    # surface_distance 로 되돌린다 (구·step·body 마다)
    sd = clear1 + r1[None, None, :, None]                          # (F, H, S, B) mm

    rz = np.load(T8C / "s1_radii_by_scale.npz", allow_pickle=True)
    assert [str(x) for x in rz["link_names"]] == link_names
    s1 = json.loads((T8C / "s1_radii_by_scale.json").read_text())
    by_key = {c["key"]: c for c in s1["combos"]}

    keep = np.ones((S, B), bool)
    keep[auth_sphere, ai] = False

    windows = {"full50": slice(0, H), "exec8": slice(0, EXEC_H)}
    results = []
    curves = {}
    for c in s1["combos"]:
        key = c["key"]
        rs = np.asarray(rz[key], np.float64) * 1000.0              # (S,)
        clear = sd - rs[None, None, :, None]
        entry = {"key": key, "arm5_scale": c["arm5_scale"], "finger_scale": c["finger_scale"],
                 "capsule_radius_scale_by_link": c["capsule_radius_scale_by_link"],
                 "radius_mm_link_left_arm_5": c["radius_mm_link_left_arm_5"],
                 "radius_mm_ee_finger_l1": c["radius_mm_ee_finger_l1"],
                 "radius_mm_ee_finger_l2": c["radius_mm_ee_finger_l2"],
                 "n_coverage_shortfall": c["n_coverage_shortfall"],
                 "worst_shortfall_mm": c["worst_shortfall_mm"],
                 "coverage_shortfall": c["coverage_shortfall"],
                 "windows": {}}
        for wname, w in windows.items():
            cw = np.where(keep[None, None, :, :], clear[:, w], np.inf)
            le0 = cw - MARGIN <= 0.0
            pairs = []
            for li, ln in enumerate(uniq):
                sm = link_of == li
                for bi, bn in enumerate(bodies):
                    if not keep[sm, bi].any():
                        continue
                    v = cw[:, :, sm, bi]
                    n = int((v - MARGIN <= 0.0).sum())
                    if n == 0:
                        continue
                    chunk_min = v.min(axis=(1, 2))
                    pairs.append({
                        "link": ln, "obstacle_body": bn, "n_rows_slack_le0": n,
                        "n_rows_total": int(v.size),
                        "n_chunks_slack_le0": int((chunk_min - MARGIN <= 0.0).sum()),
                        "min_clearance_mm": round(float(v.min()), 4),
                        "min_slack_mm": round(float(v.min() - MARGIN), 4),
                        "argmin_chunk": int(np.argmin(chunk_min))})
            pairs.sort(key=lambda p: p["min_slack_mm"])
            entry["windows"][wname] = {
                "n_rows_slack_le0": int(le0.sum()),
                "n_rows_total": int(np.isfinite(cw).sum()),
                "n_chunks_slack_le0": int(le0.any(axis=(1, 2, 3)).sum()),
                "n_chunks": F,
                "global_min_slack_mm": round(float(cw.min()) - MARGIN, 4),
                "feasible_at_10mm": bool(le0.sum() == 0),
                "violating_pairs": pairs,
                "per_chunk_min_slack_mm": [round(float(x), 4)
                                           for x in (cw.min(axis=(1, 2, 3)) - MARGIN)],
            }
        # 참고: 이 배율에서 사과 하나만 상대로 <=0 인 link (authorized 셋은 고정이지만
        # 반지름이 바뀌면 이 집합도 바뀐다 — 사실만 기록)
        ap = clear[:, :, :, ai]
        entry["apple_only_links_le0_at_10mm_full50"] = [
            ln for li, ln in enumerate(uniq)
            if bool((ap[:, :, link_of == li] - MARGIN <= 0.0).any())]
        results.append(entry)
        f50 = entry["windows"]["full50"]
        e8 = entry["windows"]["exec8"]
        print(f"{key:26s} L5 r={c['radius_mm_link_left_arm_5'][0]:6.2f} mm | "
              f"full50 rows {f50['n_rows_slack_le0']:5d} chunks {f50['n_chunks_slack_le0']:3d} "
              f"min {f50['global_min_slack_mm']:8.2f} | "
              f"exec8 rows {e8['n_rows_slack_le0']:4d} chunks {e8['n_chunks_slack_le0']:3d} "
              f"min {e8['global_min_slack_mm']:8.2f} | shortfall {c['n_coverage_shortfall']} "
              f"worst {c['worst_shortfall_mm']:5.1f} mm")
        for wn in windows:
            curves[f"{key}__{wn}"] = np.asarray(
                entry["windows"][wn]["per_chunk_min_slack_mm"], np.float64)

    # --- 0 에 닿는 최소 배율 --------------------------------------------------------------
    def first_feasible(finger, wname):
        cand = [e for e in results if e["finger_scale"] == finger
                and e["windows"][wname]["feasible_at_10mm"]]
        if not cand:
            return None
        return max(e["arm5_scale"] for e in cand)   # 가장 덜 줄이고도 되는 값

    feas = {f"finger_{f:g}__{w}": first_feasible(f, w)
            for f in s1["finger_scales"] for w in windows}
    worst_combo = min(results, key=lambda e: e["arm5_scale"] + e["finger_scale"])
    surviving = {w: worst_combo["windows"][w]["violating_pairs"] for w in windows}

    out = {
        "method": "T8a tensor -> surface_distance = clearance + r(scale 1.0) -> clearance(s) = "
                  "surface_distance - r(s); r(s) read from UrdfSphereChain per scale (s1). "
                  "sphere centres are invariant to the scale (verified, drift 0.0 m). "
                  "No new distance computation, no MuJoCo, no rollout.",
        "condition": {
            "apple_rows_removed_for_links": list(AUTH),
            "margin_mm": MARGIN, "windows": list(windows),
            "record": "outputs/live_test/local_shadow_from_pc/frames.jsonl actions_reference",
            "n_chunks": F, "horizon": H,
        },
        "centre_invariance_check": s1["centre_invariance_check"],
        "radius_rule": s1["radius_rule"],
        "arm5_links_scaled": s1["arm5_links_scaled"],
        "finger_links_scaled": s1["finger_links_scaled"],
        "largest_arm5_scale_that_reaches_zero_violations": feas,
        "most_aggressive_combo": worst_combo["key"],
        "pairs_surviving_the_most_aggressive_combo": surviving,
        "combos": results,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ, **curves)
    print("\nlargest arm5 scale reaching zero violations:", json.dumps(feas))
    print("most aggressive combo:", worst_combo["key"])
    for w in windows:
        print(f"  surviving [{w}]:")
        for p in surviving[w]:
            print(f"    {p['link']:18s} {p['obstacle_body']:8s} rows {p['n_rows_slack_le0']:5d} "
                  f"chunks {p['n_chunks_slack_le0']:3d} min_clr {p['min_clearance_mm']:8.2f} "
                  f"slack {p['min_slack_mm']:8.2f} mm")
    print("wrote", OUT, NPZ)


if __name__ == "__main__":
    main()
