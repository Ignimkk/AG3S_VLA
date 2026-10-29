"""T43 FR — one row per option: build (fr_build.json) + offline (offline_summary.json) + CPU TO time (fr_time_A/B.json).
Times: per run (C1807 / B1968) the mean of reps A and B of (median, p90 over chunks of the min over passes); ratio = mean of the
paired per-chunk ratio to fc25 medians.  usage: fr_table.py OUT_JSON  (prints the markdown table)"""
import json, sys
B = json.load(open("fr_build.json")); O = json.load(open("offline_summary.json"))["options"]
T = [json.load(open(f"fr_time_{r}.json"))["summary"] for r in "AB"]
LABEL = {"bare": "축선 사슬 (finger-cover 끔)", "fc25": "**FC 기본 (greedy, overflow 2.5)**", "ov3": "greedy overflow 3", "ov4": "greedy overflow 4",
         "ov5": "greedy overflow 5", "ov6": "greedy overflow 6", "k8": "greedy K 8 (2.5)", "k12": "greedy K 12 (2.5)",
         "k16": "greedy K 16 (2.5)", "k24": "greedy K 24 (2.5)", "gap05": "greedy gap 0.5 (2.5)", "gap10": "greedy gap 1.0 (2.5)",
         "ilp25": "ilp overflow 2.5", "ilp3": "ilp overflow 3", "ilp4": "ilp overflow 4", "ilp5": "ilp overflow 5", "ilp6": "ilp overflow 6",
         "ilp25gap05": "ilp gap 0.5 (2.5)", "ilp25gap10": "ilp gap 1.0 (2.5)"}
rows = {}
for nm in LABEL:
    t = {r: {k: sum(x[nm][r][k] for x in T) / 2 for k in ("median_ms", "p90_ms", "ratio_to_fc25_median")} for r in ("C1807", "B1968")}
    spread = max(abs(T[0][nm][r]["median_ms"] - T[1][nm][r]["median_ms"]) / T[0][nm][r]["median_ms"] for r in ("C1807", "B1968"))
    if nm == "bare":
        b = {"per_finger": 19, "n_spheres": B["bare"]["n_spheres"], "max_opening_mm": B["bare"]["max_opening_mm"]["ee_left"]}
        l1 = {"uncovered": None, "max_uncovered_mm": 13.58, "overflow_mm": 5.05}
        oa, ob = O["old"]["a"], O["old"]["b"]["all"]
    else:
        b = {"per_finger": B[nm]["per_finger"], "n_spheres": B[nm]["n_spheres"], "max_opening_mm": B[nm]["max_opening_mm"]["ee_left"]}
        l1 = B[nm]["links"]["ee_finger_l1"]
        oa, ob = O[nm]["a"], O[nm]["b"]["all"]
    rows[nm] = {"label": LABEL[nm], **b, "uncovered": l1["uncovered"], "n_targets": 20774, "max_uncovered_mm": l1["max_uncovered_mm"],
                "vertices_outside": l1.get("vertices_outside"), "overflow_mm": l1["overflow_mm"], "time": t, "time_spread_AB": spread,
                "a_geo_sees": oa["geo_sees"], "a_geo_le_mesh": oa["geo_le_mesh"], "a_esdf_sees": oa["esdf_sees"],
                "b_margin0": ob["margin0"], "b_margin0_new": ob["margin0_new_vs_chain"], "b_margin0_new_not_held": ob["margin0_new_vs_chain_not_held"],
                "b_band10": ob["band10"], "b_band10_new_not_held": ob["band10_new_vs_chain_not_held"],
                "b_band10_new_vs_fc25_not_held": ob["band10_new_vs_fc25_not_held"]}
json.dump(rows, open(sys.argv[1], "w"), indent=1)
print("| 선택지 | 구 / 손가락 | 전체 | 안 덮인 표본 (/20,774) | 최대 미덮음 mm | 넘침 mm | 최대 개도 mm | CPU TO 중앙 C/B ms | p90 C/B ms | ×FC 기본 C/B | (a) 실제 기하 본다 /57 | (a) mesh 까지 닿음 /57 | (b) margin 0: 청크 · 사슬 대비 새 (서버 미HOLD) | (b) 10 mm 띠: 청크 · 사슬 대비 새 미HOLD · FC 대비 새 미HOLD |")
print("|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---|")
for nm, r in rows.items():
    t = r["time"]
    unc = "—" if r["uncovered"] is None else f"{r['uncovered']:,}"
    print(f"| {r['label']} | {r['per_finger']} | {r['n_spheres']} | {unc} | {r['max_uncovered_mm']:.2f} | {r['overflow_mm']:.2f} | {r['max_opening_mm']:.1f} | "
          f"{t['C1807']['median_ms']:.1f} / {t['B1968']['median_ms']:.1f} | {t['C1807']['p90_ms']:.1f} / {t['B1968']['p90_ms']:.1f} | "
          f"{t['C1807']['ratio_to_fc25_median']:.2f} / {t['B1968']['ratio_to_fc25_median']:.2f} | {r['a_geo_sees']} | {r['a_geo_le_mesh']} | "
          f"{r['b_margin0']} · {r['b_margin0_new']} ({r['b_margin0_new_not_held']}) | {r['b_band10']} · {r['b_band10_new_not_held']} · {r['b_band10_new_vs_fc25_not_held']} |")
print("max A/B spread of medians:", round(max(r["time_spread_AB"] for r in rows.values()) * 100, 1), "%")
