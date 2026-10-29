"""T60 — collect every number of steps 1–3 into `AG3S/ag3s/docs/handoff/T60.verify.json` (numbers only).

usage: t60_build_verify.py RUN_DIR
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402

STORE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development")
T47_PUB = STORE / "20261012T141412+0900_bd778e5e3453_d7a51b456f36/verify/T47"
T47B_PUB = STORE / "20261012T141412+0900_bd778e5e3453_d7a51b456f36/verify/T47b"
T49_PUB = STORE / "20261012T190547+0900_28b981600bf7_c39a577332fe/verify/T49"
T51_PUB = STORE / "20261012T180318+0900_28b981600bf7_852011fdcae0/verify/T51"


def rd(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def r6(x):
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, dict):
        return {k: r6(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r6(v) for v in x]
    return x


def step1(run: pathlib.Path) -> dict:
    s1 = run / "verify/T60/step1"
    ts = rd(s1 / "timestamps.json")
    fig = rd(ROOT / "AG3S/ag3s/docs/figures/t60/t60-removed-table.json")
    gap = rd(ROOT / "AG3S/ag3s/docs/figures/t60/t60-ready-gap.json")
    R = list(ts["runs"].values())
    prim = "8|0.35|0.12|8"
    changes = {}
    for k in R[0]["grid"]:
        dP = sum((r["grid"][k][0] is None) != (r["grid"][prim][0] is None) for r in R)
        dH = sum((r["grid"][k][1] is None) != (r["grid"][prim][1] is None) for r in R)
        changes[k] = [dP, dH]
    placed = [r for r in R if r["t_P"] is not None]
    hp = [r["t_H"] - r["t_P"] for r in placed if r["t_H"] is not None]
    pp = [r["t_P"] - r["recorded"]["place_t"] for r in placed]
    return {"params": ts["primary"], "per_set": fig, "ready_gap": {k: v for k, v in gap.items() if k != "per_run"},
            "grid_class_changes_vs_primary": {"n_settings": len(changes),
                                              "settings_with_P_change": {k: v[0] for k, v in changes.items() if v[0]},
                                              "settings_with_H_change": {k: v[1] for k, v in changes.items() if v[1]}},
            "rows_tP_minus_recorded_place_t": {"min": min(pp), "max": max(pp), "median": sorted(pp)[len(pp) // 2]},
            "rows_tH_minus_tP": {"min": min(hp), "max": max(hp), "median": sorted(hp)[len(hp) // 2], "n": len(hp)},
            "hand_target_contact_after_tP_runs": sum(r["hand_target_after_P"] > 0 for r in placed),
            "target_out_of_crate_after_tP_runs": sum(r["out_of_crate_after_P"] > 0 for r in placed),
            "replay_qpos_mismatch_rows_total": sum((r["recorded"]["qpos_mismatch_rows"] or 0) for r in R)}


def step2(run: pathlib.Path) -> dict:
    cl = rd(run / "verify/T60/step2/closed_loop.json")
    t54 = rd(run / "verify/T60/step2/t54.json")
    out = {"closed_loop": {}, "t54": {}}
    for s, v in cl["sets"].items():
        out["closed_loop"][s] = {"arms": v["arms"], "pairs": {
            p: {w: {f: {k: d[w][f][k] for k in ("A_total", "B_total", "A_only", "B_only", "mcnemar_exact_p")}
                    for f in ("success", "success_ct")} for w in ("full", "P", "H")} for p, d in v["pairs"].items()},
            "full_reproduced": v["full_reproduced"]}
        pre = {"pre8_close_hold_chunks": 0, "after_tP": 0}
        for r in v["runs"].values():
            for q in r["pre8_close_hold_seqs"]:
                pre["pre8_close_hold_chunks"] += 1
                pre["after_tP"] += int(r["t_P"] is not None and (q - 1) * 8 > r["t_P"])
        out["closed_loop"][s]["pre8_close_hold_vs_tP"] = pre
    for w, v in t54["windows"].items():
        out["t54"][w] = {"frames_per_arm_cond": v["n_frames_all"],
                         "current": v["summary"]["current"], "hold": {k: v["summary"]["hold"][k] for k in ("n_target_frames", "metrics")},
                         "t57_entry_A": v["t57_entry_A"]}
    out["t54"]["t53_endpoints"] = t54["t53_endpoints"]
    out["t54"]["full_reproduced"] = "current / hold means of identity, no_target, purity, coverage for 8 arms within 1e-6; entry rows equal"
    return out


def t51_window(root: pathlib.Path) -> dict:
    if not (root / "heldout/heldout__scene.json").is_file():
        return {"status": "missing"}
    o = {}
    for lab in ("gt", "self"):
        s3 = rd(root / lab / "score3d/score3d__scene.json")
        ks = rd(root / lab / "ksel/ksel__scene.json")
        bs = rd(root / lab / "bootstrap/bootstrap__scene.json")
        s2 = rd(root / lab / "score2d/score2d__scene.json")
        o[lab] = {"score2d_global_top": s2["global"].get("top") if s2 else None,
                  "forced": s2.get("forced") if s2 else None,
                  "score3d_ranking_top5": s3["ranking"][:5] if s3 else None,
                  "ksel_selected": ks["selected"] if ks else None, "ksel_choice": ks["choice"] if ks else None,
                  "bootstrap_top1": bs["full_sample_top1"] if bs else None,
                  "bootstrap_top1_p_rank1": bs["full_sample_top1_p_rank1"] if bs else None,
                  "bootstrap_stable_single_cell": bs["stable_single_cell"] if bs else None}
    ho = rd(root / "heldout/heldout__scene.json")
    o["heldout"] = {"selected": ho["selected"], "serving": ho["serving"],
                    "summary": {src: {"record_mean_gt_identity": v.get("record_mean_gt_identity"),
                                      "scene_mean_gt_identity": v.get("scene_mean_gt_identity"),
                                      "by_scene": v.get("by_scene")} for src, v in ho["summary"].items()}}
    ab = rd(root / "ablation/ablation__scene.json")
    if ab:
        vals = sorted(ab["random_distribution"]["values"])
        o["ablation"] = {"named": ab["named"], "scene_mean_gt_identity": ab["scene_mean_gt_identity"],
                         "random_n": len(vals), "random_median": (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2,
                         "oracle_restricted": {k: ab["oracle_restricted"][k] for k in ("cell", "value")}}
    cmp_ = rd(root / "compare/compare__scene.json")
    if cmp_:
        o["compare"] = {k: cmp_[k] for k in ("p2_top1", "p2_top1_same", "p2_top3_overlap", "p4_same_combo")}
    flt = rd(root / "filter/filter.json")
    o["filter"] = {"n_eligible": len(flt["eligible"]), "n_frames": flt["n_frames"]}
    return o


def step3(run: pathlib.Path) -> dict:
    s3 = run / "verify/T60/step3"
    out = {"sampled_check": {k: v for k, v in rd(s3 / "sampled_check.json").items() if k != "per_run"},
           "record_endpoints": rd(s3 / "records/record_endpoints.json")["records"]}
    t47 = rd(s3 / "t47_select.json")
    out["t47_select"] = {w: {"n_frames": t47["windows"][w]["n_frames"],
                             "d0_last_cam_common_top3": t47["windows"][w]["settings"]["d0_last"]["cam"]["common_topk"]["3"],
                             "d0_last_cam_per_camera_top3": t47["windows"][w]["settings"]["d0_last"]["cam"]["per_camera_top3"],
                             "common_top3_sets": {n: {src: sorted(map(tuple, v[src]["common_topk"]["3"])) for src in ("cam", "img")}
                                                  for n, v in t47["windows"][w]["settings"].items()}}
                         for w in ("full", "P", "H")}
    out["t47_select"]["settings_whose_common_top3_set_changed"] = {
        w: sorted(f"{n}|{src}" for n in out["t47_select"]["full"]["common_top3_sets"] for src in ("cam", "img")
                  if out["t47_select"]["full"]["common_top3_sets"][n][src] != out["t47_select"][w]["common_top3_sets"][n][src])
        for w in ("P", "H")}
    e2e = {}
    for name, d in (("published_s0", T47_PUB), ("P_s0", s3 / "T47_P"), ("H_s0", s3 / "T47_H"), ("P_s1", s3 / "T47_P_s1")):
        s = rd(d / "e2e/summary__heldout.json")
        o = rd(d / "validate/oracle.json")
        if s:
            e2e[name] = {"source_cells": s["source_cells"],
                         "pooled": {src: {k: v[k] for k in ("has_target", "on_target", "silent_failure") if k in v}
                                    for src, v in s["pooled"].items()},
                         "oracle_d0_last_top3": o["settings"]["d0_last"]["top3"] if o else None}
    out["t47_e2e_heldout"] = e2e
    t49 = {}
    for w in ("full", "P", "H"):
        d = rd(s3 / f"T49_{w}/select/select.json")
        t49[w] = {q: {"gt_top3": v["gt_topk"]["3"], "n_samples": v["n_samples"],
                      "label_free_cam_top3": v.get("label_free", {}).get("cam", {}).get("common", {}).get("topk", {}).get("3")}
                  for q, v in d["queries"].items()}
    t49["full_equals_published"] = rd(s3 / "T49_full/select/select.json")["queries"] == rd(T49_PUB / "select/select.json")["queries"]
    out["t49_select"] = t49
    out["t48a_ep1807"] = rd(s3 / "t48a_ep1807.json")
    out["t51"] = {w: t51_window(s3 / f"T51_{w}") for w in ("full", "P", "H")}
    out["t51"]["published"] = t51_window(T51_PUB)
    out["t51"]["full_equals_published"] = "every scene-chain json equal within 1e-9 except ksel replayed_for_ksel (cache)"
    out["t52_scope"] = "held-out headfix / ep1800 / ep1850 / ep1900, 15 frames each, none reaches P (record_endpoints)"
    return out


def main(run: pathlib.Path) -> None:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    base = rd(run / "verify/T60/baseline/legacy.json")
    fr = base["frames"]
    out = {
        "_comment": "numbers only; interpretation is in the log",
        "step": "T60", "date": _dt.date.today().isoformat(), "run_dir": str(run),
        "protocol": T.PROTOCOL_VERSION,
        "runs": [
            {"id": "step1", "command": f".venv-ag3s/bin/python AG3S/ag3s/experiments/studies/t60_timestamps.py {run}/verify/T60/step1"},
            {"id": "step1_fig", "command": f".venv-openpi-live/bin/python AG3S/ag3s/experiments/studies/t60_figures.py {run}/verify/T60/step1 AG3S/ag3s/docs/figures/t60"},
            {"id": "step2_closed_loop", "command": f".venv-ag3s/bin/python AG3S/ag3s/experiments/studies/t60_closed_loop.py {run}/verify/T60/step1 {run}/verify/T60/step2/closed_loop.json"},
            {"id": "step2_t54", "command": f".venv-ag3s/bin/python AG3S/ag3s/experiments/studies/t60_t54.py {run}/verify/T60/step1 {run}/verify/T60/step2/t54.json"},
            {"id": "step3_sampled_check", "command": f".venv-ag3s/bin/python AG3S/ag3s/experiments/studies/t60_sampled_check.py {run}/verify/T60/step1 {run}/verify/T60/step3/sampled_check.json"},
            {"id": "step3_records", "command": f"MUJOCO_GL=osmesa .venv-openpi-live/bin/python AG3S/ag3s/experiments/studies/t60_records.py {run}/verify/T60/step3/records"},
            {"id": "step3_t47", "command": f"{run}/verify/T60/step3/chain_t47.sh {{P,H}}; t60_t47_run.py --window P ... -- --stage e2e --sigma 1 --sources l8h2 topk_common (root T47_P_s1); t60_t47.py"},
            {"id": "step3_t49", "command": "t60_t49.py --window {full,P,H} ... -- --stage select"},
            {"id": "step3_t51", "command": f"{run}/verify/T60/step3/chain_t51.sh full; {run}/verify/T60/step3/run_t51_windows.sh"},
            {"id": "step3_t48a", "command": "t60_t48a.py <T48 run> records/ep1807.npz t48a_ep1807.json"},
        ],
        "code_state": {"head": head, "new_files": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "AG3S/ag3s/experiments/studies").glob("t60_*.py")) + ["tests/ag3s/test_t60_termination.py"],
                       "existing_code_modified": False, "existing_records_modified": False},
        "baseline_check": {"matched": sum(x["clearance_before_mm"] < 0 for x in fr) == 14 and sum(bool(x["target"]) for x in fr) == 15
                           and fr[0]["clearance_before_mm"] == -29.031048280806342,
                           "violated_start": sum(x["clearance_before_mm"] < 0 for x in fr),
                           "has_target": sum(bool(x["target"]) for x in fr),
                           "frame0_clearance_before_mm": fr[0]["clearance_before_mm"],
                           "raw": str(run / "verify/T60/baseline/legacy.json")},
        "numbers": r6({"step1": step1(run), "step2": step2(run), "step3": step3(run)}),
        "figures": [{"path": f"AG3S/ag3s/docs/figures/t60/{p.name}", "sidecar": f"AG3S/ag3s/docs/figures/t60/{p.stem}.json"}
                    for p in sorted((ROOT / "AG3S/ag3s/docs/figures/t60").glob("*.png"))],
        "not_measured": [
            "T54b / T54c / T55 / T56 (other session, verify not final): masks handed over only",
            "T48a val / T48b / T50 (HSR dataset demos): no per-frame object state in the dataset; P / H need re-simulation",
            "T49 validate / e2e of the changed label-free (secondary) noun / basket sets",
            "T47b label-based 3D metrics (identity_correct, purity) for the new T47 common top-3 (T47 e2e rows report on_target only)",
            "runtime termination detector (new rollout)",
        ],
    }
    p = ROOT / "AG3S/ag3s/docs/handoff/T60.verify.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=T._default) + "\n")
    print(p)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
