"""T8a m1·m2·m3 — m0 tensor 에서 표를 뽑는다. MuJoCo 를 다시 돌리지 않는다.

m1 : link 별 **사과 하나만** 상대 clearance. margin 0 / 10 / 50 mm 에서 <=0 인 chunk 수와 최소값.
     margin 10 mm 에서 <=0 인 link 집합 = authorized set A.
m2 : A 의 link 에서 사과 행을 지우고 **남은 모든 행**을 margin 10 mm 로 검사. (link, body) 쌍별로.
m3 : A 의 link 이 사과를 상대로 <=0 인 그 질의점에서 **table 까지의 거리**.

row 하나 = (chunk, step, sphere, body).  slack = clearance - margin.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib

import numpy as np

T8A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8a'))
NPZ = T8A / "m0_clearance_tensor.npz"
OUT = T8A / "m1m2m3_analysis.json"
FIGNPZ = T8A / "m1m2m3_curves.npz"
T7A_NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a/s1_curves.npz'))
MARGINS_MM = (0.0, 10.0, 50.0)
PRIMARY_MARGIN_MM = 10.0
EXEC_H = 8


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    clear = z["clearance_m"].astype(np.float64) * 1000.0      # (F, H, S, B) mm
    bodies = [str(b) for b in z["bodies"]]
    link_names = [str(n) for n in z["link_names"]]
    radii_mm = z["radii_m"] * 1000.0
    F, H, S, B = clear.shape
    uniq_links = sorted(set(link_names))
    link_of_sphere = np.asarray([uniq_links.index(n) for n in link_names], np.int64)
    ai = bodies.index("apple")
    ti = bodies.index("table")
    ci = bodies.index("crate")
    non_apple = np.asarray([b != "apple" for b in bodies], bool)
    windows = {"full50": slice(0, H), "exec8": slice(0, EXEC_H)}
    out: dict = {
        "shape": {"n_chunks": F, "horizon": H, "n_spheres": S, "bodies": bodies,
                  "link_axis": uniq_links},
        "margins_mm": list(MARGINS_MM), "primary_margin_mm": PRIMARY_MARGIN_MM,
        "row_definition": "row = (chunk, step, sphere, body); clearance = surface_distance - "
                          "sphere_radius; slack = clearance - margin",
    }

    # ---- 0. T7a 재현 확인: exec8 · apple 제외 · margin 50 mm 의 link 별 곡선 ---------------
    t7 = np.load(T7A_NPZ, allow_pickle=True)
    t7_links = [str(x) for x in t7["link_axis"]]
    ref = t7["shadow_approach__reference__no_target"]           # (75, 8, 18) margin_slack mm
    mine = np.full_like(ref, np.nan)
    sub = clear[:, :EXEC_H][:, :, :, non_apple].min(axis=3)     # (F, 8, S)
    for li, ln in enumerate(t7_links):
        mine[:, :, li] = sub[:, :, link_of_sphere == uniq_links.index(ln)].min(axis=2) - 50.0
    diff = np.abs(mine - ref)
    out["t7a_reproduction_check"] = {
        "compared": "T7a s1_curves shadow_approach__reference__no_target (margin 50 mm, exec8)",
        "link_axis_identical": t7_links == uniq_links,
        "max_abs_diff_mm": float(np.nanmax(diff)),
        "n_cells": int(diff.size),
    }
    print("T7a reproduction max|diff| =", float(np.nanmax(diff)), "mm")

    # ---- 1. M1 — 사과 하나만 -----------------------------------------------------------
    apple = clear[:, :, :, ai]                                  # (F, H, S) mm
    m1: dict = {}
    for wname, w in windows.items():
        aw = apple[:, w]                                        # (F, Hw, S)
        per: dict = {}
        for li, ln in enumerate(uniq_links):
            sm = link_of_sphere == li
            v = aw[:, :, sm]                                    # (F, Hw, ns)
            chunk_min = v.min(axis=(1, 2))                      # (F,)
            e = {"n_spheres": int(sm.sum()),
                 "sphere_radius_mm": sorted({round(float(r), 4) for r in radii_mm[sm]}),
                 "min_clearance_mm": round(float(v.min()), 4),
                 "argmin_chunk": int(np.argmin(chunk_min)),
                 "n_rows_total": int(v.size)}
            for mg in MARGINS_MM:
                k = f"{mg:g}mm"
                e[f"n_chunks_slack_le0_at_{k}"] = int((chunk_min - mg <= 0.0).sum())
                e[f"n_rows_slack_le0_at_{k}"] = int((v - mg <= 0.0).sum())
                e[f"min_slack_mm_at_{k}"] = round(float(v.min() - mg), 4)
            per[ln] = e
        m1[wname] = per
    out["m1_apple_only_per_link"] = m1

    authorized = {w: [ln for ln in uniq_links
                      if m1[w][ln][f"n_chunks_slack_le0_at_{PRIMARY_MARGIN_MM:g}mm"] > 0]
                  for w in windows}
    out["m1_authorized_link_set"] = {
        "rule": f"link with >=1 chunk whose apple-only slack <= 0 at margin "
                f"{PRIMARY_MARGIN_MM:g} mm",
        **{f"at_{PRIMARY_MARGIN_MM:g}mm_{w}": authorized[w] for w in windows},
        **{f"at_{mg:g}mm_full50": [ln for ln in uniq_links
                                   if m1["full50"][ln][f"n_chunks_slack_le0_at_{mg:g}mm"] > 0]
           for mg in MARGINS_MM},
    }
    A = authorized["full50"]
    A_mask_sphere = np.asarray([link_names[s] in A for s in range(S)], bool)
    print("authorized (full50, 10mm):", A)

    # ---- 2. M2 — A 에서 사과를 빼고 남은 모든 행 ------------------------------------------
    m2: dict = {}
    for wname, w in windows.items():
        cw = clear[:, w]                                        # (F, Hw, S, B)
        # keep mask over (S, B): drop (sphere in A, apple)
        keep = np.ones((S, B), bool)
        keep[A_mask_sphere, ai] = False
        pairs = []
        n_rows_total = 0
        n_rows_le0 = 0
        worst = None
        for li, ln in enumerate(uniq_links):
            sm = link_of_sphere == li
            for bi, bn in enumerate(bodies):
                if not keep[sm, bi].any():
                    continue
                v = cw[:, :, sm, bi]                            # (F, Hw, ns)
                chunk_min = v.min(axis=(1, 2))
                n_rows_total += int(v.size)
                nle = int((v - PRIMARY_MARGIN_MM <= 0.0).sum())
                n_rows_le0 += nle
                rec = {"link": ln, "obstacle_body": bn,
                       "n_rows_total": int(v.size),
                       "n_rows_slack_le0": nle,
                       "n_chunks_slack_le0": int((chunk_min - PRIMARY_MARGIN_MM <= 0.0).sum()),
                       "min_clearance_mm": round(float(v.min()), 4),
                       "min_slack_mm": round(float(v.min() - PRIMARY_MARGIN_MM), 4),
                       "argmin_chunk": int(np.argmin(chunk_min))}
                pairs.append(rec)
                if worst is None or rec["min_slack_mm"] < worst["min_slack_mm"]:
                    worst = rec
        viol = sorted([p for p in pairs if p["n_rows_slack_le0"] > 0],
                      key=lambda p: p["min_slack_mm"])
        chunk_any = np.zeros(F, bool)
        cwk = np.where(keep[None, None, :, :], cw, np.inf)
        chunk_any = (cwk - PRIMARY_MARGIN_MM <= 0.0).any(axis=(1, 2, 3))
        m2[wname] = {
            "apple_rows_removed_for_links": A,
            "n_pairs_evaluated": len(pairs),
            "n_rows_total": n_rows_total,
            "n_rows_slack_le0": n_rows_le0,
            "n_chunks_with_any_slack_le0": int(chunk_any.sum()),
            "n_chunks": F,
            "global_min_slack_mm": round(float(np.nanmin(cwk)) - PRIMARY_MARGIN_MM, 4),
            "worst_pair": worst,
            "violating_pairs": viol,
            "per_chunk_min_slack_mm": [round(float(x), 4)
                                       for x in (cwk.min(axis=(1, 2, 3)) - PRIMARY_MARGIN_MM)],
        }
    out["m2_apple_removed_remaining_rows_at_10mm"] = m2

    # M2 대조: 사과를 아무 link 에서도 빼지 않은 경우, 그리고 전 link 에서 뺀 경우
    contrast = {}
    for tag, km in (("no_removal", np.ones((S, B), bool)),
                    ("removed_for_all_links", None)):
        keep = np.ones((S, B), bool)
        if tag == "removed_for_all_links":
            keep[:, ai] = False
        cwk = np.where(keep[None, None, :, :], clear, np.inf)
        contrast[tag] = {
            "n_rows_slack_le0": int((cwk - PRIMARY_MARGIN_MM <= 0.0).sum()),
            "n_chunks_with_any_slack_le0": int(
                (cwk - PRIMARY_MARGIN_MM <= 0.0).any(axis=(1, 2, 3)).sum()),
            "global_min_slack_mm": round(float(np.nanmin(cwk)) - PRIMARY_MARGIN_MM, 4),
        }
    out["m2_contrast_full50_at_10mm"] = contrast

    # ---- 3. M3 — 사과가 음수인 그 질의점에서 table 까지 -----------------------------------
    hit = (apple - PRIMARY_MARGIN_MM <= 0.0) & A_mask_sphere[None, None, :]   # (F, H, S)
    table = clear[:, :, :, ti]
    crate = clear[:, :, :, ci]
    other = clear[:, :, :, non_apple].min(axis=3)
    m3: dict = {"selection": f"query points (chunk, step, sphere) with sphere on an authorized "
                             f"link and apple slack <= 0 at margin {PRIMARY_MARGIN_MM:g} mm",
                "n_query_points": int(hit.sum()),
                "n_chunks_with_any": int(hit.any(axis=(1, 2)).sum())}
    for nm, arr in (("table", table), ("crate", crate), ("min_over_non_apple_bodies", other)):
        vals = arr[hit]
        m3[nm] = {
            "min_clearance_mm": round(float(vals.min()), 4) if vals.size else None,
            "median_clearance_mm": round(float(np.median(vals)), 4) if vals.size else None,
            "max_clearance_mm": round(float(vals.max()), 4) if vals.size else None,
            "n_query_points_slack_le0_at_10mm": int((vals - PRIMARY_MARGIN_MM <= 0.0).sum()),
            "n_query_points_below_0mm": int((vals <= 0.0).sum()),
        }
    per_chunk = []
    for i in range(F):
        hi = hit[i]
        if not hi.any():
            continue
        per_chunk.append({
            "chunk": i, "n_query_points": int(hi.sum()),
            "min_apple_clearance_mm": round(float(apple[i][hi].min()), 4),
            "min_table_clearance_mm": round(float(table[i][hi].min()), 4),
            "min_crate_clearance_mm": round(float(crate[i][hi].min()), 4),
            "min_non_apple_clearance_mm": round(float(other[i][hi].min()), 4),
            "links": sorted({link_names[s] for s in np.unique(np.nonzero(hi)[1])}),
        })
    m3["per_chunk"] = per_chunk
    per_link = {}
    for li, ln in enumerate(uniq_links):
        sm = link_of_sphere == li
        hi = hit[:, :, sm]
        if not hi.any():
            continue
        per_link[ln] = {
            "n_query_points": int(hi.sum()),
            "n_chunks": int(hi.any(axis=(1, 2)).sum()),
            "min_apple_clearance_mm": round(float(apple[:, :, sm][hi].min()), 4),
            "min_table_clearance_mm": round(float(table[:, :, sm][hi].min()), 4),
            "median_table_clearance_mm": round(float(np.median(table[:, :, sm][hi])), 4),
            "min_crate_clearance_mm": round(float(crate[:, :, sm][hi].min()), 4),
        }
    m3["per_link"] = per_link
    out["m3_table_distance_at_apple_negative_points"] = m3

    # ---- figure 용 곡선 ----------------------------------------------------------------
    L = len(uniq_links)
    curves = {}
    for wname, w in windows.items():
        cw = clear[:, w]
        ap = np.empty((F, L)); noap = np.empty((F, L)); tb = np.empty((F, L)); cr = np.empty((F, L))
        for li in range(L):
            sm = link_of_sphere == li
            ap[:, li] = cw[:, :, sm, ai].min(axis=(1, 2))
            noap[:, li] = cw[:, :, sm][:, :, :, non_apple].min(axis=(1, 2, 3))
            tb[:, li] = cw[:, :, sm, ti].min(axis=(1, 2))
            cr[:, li] = cw[:, :, sm, ci].min(axis=(1, 2))
        curves[f"apple_only_{wname}"] = ap
        curves[f"no_apple_{wname}"] = noap
        curves[f"table_only_{wname}"] = tb
        curves[f"crate_only_{wname}"] = cr
    np.savez_compressed(FIGNPZ, link_axis=np.asarray(uniq_links),
                        bodies=np.asarray(bodies), **curves)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT, "and", FIGNPZ)
    # 콘솔 요약
    print("\n--- M1 (apple only, full50) ---")
    print(f"{'link':18s} {'minClr':>9s} {'<=0@0':>7s} {'<=0@10':>7s} {'<=0@50':>7s}")
    for ln in uniq_links:
        e = m1["full50"][ln]
        print(f"{ln:18s} {e['min_clearance_mm']:9.3f} {e['n_chunks_slack_le0_at_0mm']:7d} "
              f"{e['n_chunks_slack_le0_at_10mm']:7d} {e['n_chunks_slack_le0_at_50mm']:7d}")
    print("\n--- M2 (full50, 10 mm) violating pairs ---")
    for p in m2["full50"]["violating_pairs"]:
        print(f"{p['link']:18s} {p['obstacle_body']:8s} rows={p['n_rows_slack_le0']:6d} "
              f"chunks={p['n_chunks_slack_le0']:3d} min_slack={p['min_slack_mm']:9.3f} mm")
    print("\n--- M3 ---")
    print(json.dumps({k: v for k, v in m3.items() if k not in ("per_chunk", "per_link")},
                     indent=1, ensure_ascii=False))
    print(json.dumps(m3["per_link"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
