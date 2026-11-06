"""SL16 hvc -- 거리장에서 손이 주인인 점유 voxel 이 어디서 오는지 가른다 (CPU, 읽기만, GPU 불필요).

    PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python -m AG3S.trajopt.experiments.panda_hvc_origin \\
        --run-dir ${AG3S_STORAGE_ROOT}/results/benchmark/<DREC run> --geomsnap-dir <dir of *_geomsnap.npz> --out SL16.hvc.data.json

`panda_sfhand_leak` (SL15) 은 깊이 영상에서 새는 손 점이 손 구에서 p95 4 mm 인데, 같은 call 의 fine 거리장에서 손이 주인인 점유 voxel 은
`panda_hand` 구에서 중앙값 13.6 mm / p95 29 mm 라는 것을 보였다. 이 도구는 그 차이가 어디서 오는지 가른다 (규칙은 결과 전에 고정, SL16.task.md 3.1).

CLOSING / HELD call 마다 fine 계층의 점유 voxel (`d < 0`) 중 MuJoCo 가장 가까운 collision geom 이 손 · 손가락 body 이고 그 geom 까지 `ghost_mm` 이내인 것을
(= **손 voxel**) 셋으로 나눈다. 거리 sd = 서버 self-filter 모델의 `panda_hand` · `panda_leftfinger` · `panda_rightfinger` 구 합집합까지의 `|p - c| - r`.

* `current` : 지금 관절 상태 (q_now · 손가락) 의 구까지 `sd <= tol_mm` (기본 4)
* `trail`   : current 가 아니고, 직전 `n_prev` (기본 8) 요청 중 어느 요청의 관절 상태에서 `sd <= tol_mm`
* `smear`   : 둘 다 아님

고정 규칙: trail 비율 >= 0.5 이면 HVC 모드 `trail`, 아니면 `current` (r = 손 voxel 의 current-sd p90, 상한 `cap_mm` = 20).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.trajopt.experiments.panda_sfhand_leak import (FINGER_BODIES, FINGER_LINKS, HAND_BODIES, HAND_LINK, QUANTILES, filter_model_at,
                                                        nearest_geom, sphere_signed)

TOOL = "AG3S.trajopt.experiments.panda_hvc_origin"
HAND_FINGER_LINKS = (HAND_LINK,) + tuple(FINGER_LINKS)
CAP_MM = 20.0
TRAIL_SHARE_THRESHOLD = 0.5
#: `trail` 모드가 현재 손 구에서 이만큼보다 먼 voxel 은 빼지 않는다 (task 3.1)
TRAIL_NEAR_CURRENT_MM = 20.0
CLASSES = ("current", "trail", "smear")


def stats_mm(values_m: Sequence[float]) -> dict[str, Any]:
    v = np.asarray(values_m, np.float64).reshape(-1) * 1000.0
    if not v.size:
        return {"n": 0}
    out = {"n": int(v.size), "min": float(v.min())}
    for q, x in zip(QUANTILES, np.percentile(v, QUANTILES)):
        out["median" if q == 50 else ("max" if q == 100 else f"p{q}")] = float(x)
    return out


def classify_hand_voxels(sd_cur: np.ndarray, sd_trail: np.ndarray, tol_m: float) -> np.ndarray:
    """`(N,)` 0 = current, 1 = trail, 2 = smear. `sd_trail` = 직전 요청들의 구까지 거리의 최솟값 (없으면 +inf)."""
    cls = np.full(len(sd_cur), 2, np.int8)
    cls[sd_trail <= tol_m] = 1
    cls[sd_cur <= tol_m] = 0
    return cls


def decide_mode(trail_share: float, hand_sd_p90_m: float, *, cap_mm: float = CAP_MM, threshold: float = TRAIL_SHARE_THRESHOLD) -> dict[str, Any]:
    """고정 규칙: 잔상 비율 >= 0.5 이면 `trail`, 아니면 `current` (r = p90 상한 `cap_mm`, mm 로 올림하지 않는다)."""
    mode = "trail" if trail_share >= threshold else "current"
    return {"mode": mode, "trail_share": float(trail_share), "trail_threshold": threshold, "hand_voxel_sd_p90_mm": float(hand_sd_p90_m * 1000.0),
            "cap_mm": cap_mm, "r_mm": float(min(max(hand_sd_p90_m * 1000.0, 0.0), cap_mm)) if mode == "current" else None}


def hand_spheres(model, q: np.ndarray, applied: dict) -> tuple[np.ndarray, np.ndarray]:
    """self-filter 구 중 손 · 손가락 link 의 (중심, 반지름)."""
    centres, radii = filter_model_at(model, q, applied)
    links = np.asarray([str(n) for n in model.sphere_link_names])
    m = np.isin(links, HAND_FINGER_LINKS)
    return centres[m], radii[m]


def per_call(model, call, z, summary: dict, geom, snap: dict, literal, *, ghost_m: float, reach_m: float, tol_m: float, n_prev: int) -> Optional[dict]:
    from AG3S.adapters.panda import model as pm
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field
    from AG3S.trajopt.experiments.panda_fixed_input_ablation import neighbour_voxels

    applied = ((literal(summary.get("finger_joints")) or {}).get("applied") or {}).get("self_filter") or {}
    cc, cr = hand_spheres(model, call.q_now, applied)
    field = build_field(z, outside_distance=0.45)
    links = np.asarray([str(n) for n in model.sphere_link_names])
    c0 = filter_model_at(model, call.q_now, applied)[0][links == HAND_LINK].mean(axis=0)
    vox = neighbour_voxels(field, c0, reach_m, layer=-1)
    if not len(vox):
        return None
    T = np.asarray(call.T_world_base, np.float64)
    body, dist = nearest_geom(vox @ T[:3, :3].T + T[:3, 3], snap)
    targets = [str(b) for b in (snap.get("target_bodies") or [])]
    owner = np.full(len(vox), "other", dtype=object)
    owner[np.isin(body, HAND_BODIES + FINGER_BODIES) & (dist <= ghost_m)] = "hand"
    owner[np.isin(body, targets) & (dist <= ghost_m)] = "target"
    owner[dist > ghost_m] = "unowned"
    sd_cur = sphere_signed(vox, cc, cr)
    # 직전 요청들의 구: 같은 episode 의 seq - 1, seq - 2 ... (연속 seq 만)
    prevs = []
    p = call.prev
    while p is not None and len(prevs) < n_prev:
        prevs.append(p)
        p = p.prev
    sd_trail = np.full(len(vox), np.inf)
    need = (owner == "hand") | (sd_cur <= TRAIL_NEAR_CURRENT_MM / 1000.0)
    for pc in prevs:
        _, fingers = pm.split_robot_state(geom["robot_state/left_wrist"][pc.obs_index] if "robot_state/left_wrist" in geom.files
                                          else geom["robot_state/head"][pc.obs_index])
        pc_c, pc_r = hand_spheres(model, pc.q_now, fingers)
        sd_trail[need] = np.minimum(sd_trail[need], sphere_signed(vox[need], pc_c, pc_r))
    fingers_now = pm.split_robot_state(geom["robot_state/left_wrist"][call.obs_index] if "robot_state/left_wrist" in geom.files
                                       else geom["robot_state/head"][call.obs_index])[1]
    finger_gap = max(abs(float(applied.get(k, 0.0)) - float(v)) for k, v in fingers_now.items()) if applied else None
    # 안전 조건 확인용: 지지면 평면 아래 voxel (평면 = [a b c d], a x + b y + c z + d = 0, 기록 좌표)
    planes = np.asarray(z["support_planes"], np.float64) if "support_planes" in z.files else np.zeros((0, 4))
    below = np.zeros(len(vox), bool)
    for pl in planes:
        if np.isfinite(pl).all() and np.linalg.norm(pl[:3]) > 0:
            below |= (vox @ pl[:3] + pl[3]) / np.linalg.norm(pl[:3]) < 0.0
    return {"owner": owner, "sd_cur": sd_cur, "sd_trail": sd_trail, "below_support": below, "n_prev": len(prevs),
            "voxel_size_m": float(field.layers[-1].grid.voxel_size), "finger_gap": finger_gap}


def table(parts: list[dict], tol_m: float, sub_name: str = "hand") -> dict[str, Any]:
    """call 모음 -> 손 voxel 분류 표 (개수 · 비율 · class 별 sd 분위수)."""
    sd_cur = np.concatenate([p["sd_cur"][p["owner"] == "hand"] for p in parts]) if parts else np.zeros(0)
    sd_tr = np.concatenate([p["sd_trail"][p["owner"] == "hand"] for p in parts]) if parts else np.zeros(0)
    cls = classify_hand_voxels(sd_cur, sd_tr, tol_m)
    out: dict[str, Any] = {"n_calls": len(parts), "n_hand_voxels": int(len(sd_cur)), "sd_cur_all_mm": stats_mm(sd_cur)}
    for i, name in enumerate(CLASSES):
        m = cls == i
        out[name] = {"n": int(m.sum()), "share": float(m.mean()) if len(m) else None, "sd_cur_mm": stats_mm(sd_cur[m]),
                     "sd_trail_min_mm": stats_mm(sd_tr[m][np.isfinite(sd_tr[m])])}
    # call 별 비율 (큰 call 이 전체를 끌지 않는지)
    shares = []
    for p in parts:
        m = p["owner"] == "hand"
        if m.sum() >= 5:
            c = classify_hand_voxels(p["sd_cur"][m], p["sd_trail"][m], tol_m)
            shares.append([(c == i).mean() for i in range(3)])
    if shares:
        s = np.asarray(shares)
        out["per_call_share"] = {n: {"median": float(np.median(s[:, i])), "mean": float(s[:, i].mean())} for i, n in enumerate(CLASSES)}
        out["per_call_share"]["n_calls"] = len(shares)
    return out


def risk(parts: list[dict], mode: str, r_m: float, tol_m: float) -> dict[str, Any]:
    """고른 모드로 지웠을 때 (안전 조건 전 · 후) 손이 아닌 주인의 voxel 이 얼마나 지워지나 (측정 단계의 참고)."""
    out: dict[str, Any] = {}
    for owner in ("hand", "target", "other", "unowned"):
        n = freed = freed_safe = 0
        for p in parts:
            m = p["owner"] == owner
            if mode == "current":
                f = p["sd_cur"] <= r_m
            else:
                f = (p["sd_trail"] <= tol_m) & (p["sd_cur"] <= TRAIL_NEAR_CURRENT_MM / 1000.0)
                f |= p["sd_cur"] <= tol_m
            n += int(m.sum())
            freed += int((f & m).sum())
            freed_safe += int((f & m & ~p["below_support"]).sum())
        out[owner] = {"n": n, "freed": freed, "freed_above_support": freed_safe, "share_freed": (freed / n) if n else None}
    return out


def run(run_dir: pathlib.Path, snapdir: pathlib.Path, *, ghost_mm: float = 30.0, reach_mm: float = 220.0, tol_mm: float = 4.0, n_prev: int = 8,
        limit: Optional[int] = None) -> dict[str, Any]:
    from AG3S.adapters.panda import model as pm
    from AG3S.trajopt.experiments import panda_fixed_input_ablation as T

    model = pm.build_panda_self_filter_model()
    stems = sorted(p.name[:-len("_calls.jsonl")] for p in (run_dir / "episodes").glob("*_calls.jsonl"))
    calls = []
    for st in stems:
        calls += T.load_episode_calls(run_dir / "episodes", st)
    T.attach_chunks(calls, run_dir / "server_constraints/run_0000")
    sel = [c for c in calls if c.chunk_path is not None and c.phase in ("closing", "held")][:limit]
    parts, rows, geoms = [], [], {}
    for c in sel:
        snap = T.load_geom_snapshot(T.geom_snapshot_dir_path(snapdir, c.stem), c.obs_index)
        if snap is None:
            rows.append({"id": c.id, "phase": c.phase, "skipped": "no geom snapshot"})
            continue
        z = np.load(c.chunk_path, allow_pickle=False)
        summary = json.loads(str(z["summary_json"]))
        if c.stem not in geoms:
            geoms[c.stem] = np.load(run_dir / "episodes" / f"{c.stem}_geom.npz")
        m = per_call(model, c, z, summary, geoms[c.stem], snap, T._literal, ghost_m=ghost_mm / 1000.0, reach_m=reach_mm / 1000.0,
                     tol_m=tol_mm / 1000.0, n_prev=n_prev)
        if m is None:
            rows.append({"id": c.id, "phase": c.phase, "skipped": "no occupied voxel near the hand"})
            continue
        m.update(id=c.id, phase=c.phase, latch=(c.row.get("grasp") or {}).get("state"))
        parts.append(m)
        rows.append({"id": c.id, "phase": c.phase, "n_hand_voxels": int((m["owner"] == "hand").sum()), "n_prev": m["n_prev"],
                     "finger_gap_m": m["finger_gap"]})
    tol_m = tol_mm / 1000.0
    res: dict[str, Any] = {
        "tool": TOOL, "run_dir": str(run_dir), "tol_mm": tol_mm, "n_prev": n_prev, "ghost_mm": ghost_mm, "reach_mm": reach_mm,
        "rule": f"trail share >= {TRAIL_SHARE_THRESHOLD} -> trail; else current with r = p90(hand voxel sd) capped at {CAP_MM} mm",
        "voxel_size_mm": parts[0]["voxel_size_m"] * 1000.0 if parts else None,
        "n_calls_closing_held_with_chunk": len(sel), "n_calls_measured": len(parts),
        "max_finger_gap_between_summary_and_geom_robot_state_m": max([m["finger_gap"] or 0.0 for m in parts] or [0.0]),
        "tables": {"closing_and_held": table(parts, tol_m), "closing": table([p for p in parts if p["phase"] == "closing"], tol_m),
                   "held": table([p for p in parts if p["phase"] == "held"], tol_m)}}
    for name in ("closing", "held"):
        sub = [p for p in parts if p["phase"] == name]
        res["tables"][name + "_by_latch_state"] = {}
        for st in sorted({str(p["latch"]) for p in sub}):
            res["tables"][name + "_by_latch_state"][st] = table([p for p in sub if str(p["latch"]) == st], tol_m)
    main = res["tables"]["closing_and_held"]
    if main["n_hand_voxels"]:
        sd_all = np.concatenate([p["sd_cur"][p["owner"] == "hand"] for p in parts])
        res["decision"] = decide_mode(main["trail"]["share"], float(np.percentile(sd_all, 90)))
        # 감도 (규칙은 위 한 가지; 아래는 참고): 손 voxel 의 p90 을 다르게 정의하면
        res["decision_sensitivity"] = {
            "p90_of_sd_cur_positive_only_mm": float(np.percentile(sd_all[sd_all > 0], 90) * 1000.0) if (sd_all > 0).any() else None,
            "p90_closing_mm": main and res["tables"]["closing"]["sd_cur_all_mm"].get("p90"),
            "p90_held_mm": res["tables"]["held"]["sd_cur_all_mm"].get("p90"),
            "trail_share_closing": res["tables"]["closing"]["trail"]["share"] if res["tables"]["closing"]["n_hand_voxels"] else None,
            "trail_share_held": res["tables"]["held"]["trail"]["share"] if res["tables"]["held"]["n_hand_voxels"] else None}
        d = res["decision"]
        res["carve_risk_if_applied_in_this_data"] = risk(parts, d["mode"], (d["r_mm"] or 0.0) / 1000.0, tol_m)
        res["hand_voxel_below_support_share"] = float(np.mean(np.concatenate([p["below_support"][p["owner"] == "hand"] for p in parts])))
    else:
        res["decision"] = None
    res["per_call"] = rows
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--geomsnap-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--ghost-mm", type=float, default=30.0)
    ap.add_argument("--reach-mm", type=float, default=220.0)
    ap.add_argument("--tol-mm", type=float, default=4.0)
    ap.add_argument("--n-prev", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rby1-note", default=None, help="RB-Y1 비교 가능 여부를 사람이 적는다 (이 도구는 Panda 기록만 읽는다)")
    a = ap.parse_args(argv)
    res = run(a.run_dir, a.geomsnap_dir, ghost_mm=a.ghost_mm, reach_mm=a.reach_mm, tol_mm=a.tol_mm, n_prev=a.n_prev, limit=a.limit)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.rby1_note:
        res["rby1_comparison"] = a.rby1_note
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("n_calls_measured", "decision", "decision_sensitivity") if k in res}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
