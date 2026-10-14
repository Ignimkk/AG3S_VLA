"""T60c — collect the T51b (noun / combined Ω) window comparison into `handoff/T60c.verify.json` (numbers only).

usage: t60c_build_verify.py RUN_DIR
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
PUB = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261012T180318+0900_28b981600bf7_852011fdcae0/verify/T51b")


def r6(x):
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, dict):
        return {k: r6(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r6(v) for v in x]
    return x


def window(base: pathlib.Path) -> dict:
    s = json.loads((base / "summary/summary__scene.json").read_text())
    h = json.loads((base / "heldout_compare/heldout_compare__scene.json").read_text())
    out = {"heldout_named": h["named"], "heldout_scene_mean_gt_identity": h["scene_mean_gt_identity"],
           "heldout_by_scene": {k: {g: v.get("gt_identity") for g, v in (b or {}).items()} for k, b in h["by_scene"].items()}
           if isinstance(h.get("by_scene"), dict) else None, "omegas": {}}
    for om, v in s["omegas"].items():
        out["omegas"][om] = {}
        for lab, x in v.items():
            sc = x.get("score3d") or {}
            top = (sc.get("ranking_top10") or [{}])[0]
            out["omegas"][om][lab] = {
                "selected": x["ksel"]["selected"]["combo"], "selected_family": x["ksel"].get("family"),
                "ksel_nested_mean": x["ksel"].get("nested_mean"),
                "score3d_top1": top.get("cell"), "score3d_top1_value": top.get("gt_identity") if lab == "gt" else top.get("self_identity"),
                "bootstrap_top1": x["bootstrap"]["full_sample_top1"], "bootstrap_p_rank1": x["bootstrap"]["full_sample_top1_p_rank1"],
                "bootstrap_stable": x["bootstrap"]["stable_single_cell"],
                "n_candidates_3d": sc.get("n_candidates")}
    return out


def main(run: pathlib.Path) -> None:
    s = run / "verify/T60c"
    res = {w: window(s / f"T51b_{w}") for w in ("full", "P", "H")}
    res["published"] = window(PUB)
    res["full_equals_published"] = "every T51b scene json equal within 1e-9 (heldout_compare differs only in path strings)"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"_comment": "numbers only", "step": "T60c", "date": _dt.date.today().isoformat(), "run_dir": str(run),
           "runs": [{"id": "chain", "command": f"{s}/chain_t51b.sh {{full,P,H}} (t60_t51b.py wrapping t51b_study; "
                                              "--t51-dir = T60 step3 T51_<W>, --t47-dir = T60 step3 T47_<W> for P / H)"}],
           "code_state": {"head": head, "new_files": ["AG3S/ag3s/experiments/studies/t60_t51b.py",
                                                      "AG3S/ag3s/experiments/studies/t60c_build_verify.py"]},
           "cache_checks": ["T51b row cache copied per window; only cell-named sources (no paper3_s1) — checked at start",
                            "replay options checked against the T60 T51 __scene payloads (check_replay_opts)",
                            "P rows copied to H before H (cp -n), paper3_s1 deleted"],
           "numbers": r6(res),
           "not_measured": ["T61 (pre-registered head selection) — draft only, awaiting decisions"]}
    p = ROOT / "AG3S/ag3s/docs/handoff/T60c.verify.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(p)
    for w in ("published", "full", "P", "H"):
        r = res[w]
        print(w, {k: v for k, v in r["heldout_named"].items()}, r["heldout_scene_mean_gt_identity"])
        for om, v in r["omegas"].items():
            print("   ", om, {lab: (x["selected"], x["bootstrap_p_rank1"], x["bootstrap_stable"]) for lab, x in v.items()})


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
