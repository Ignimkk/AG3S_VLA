"""SL15 sfhand -- 로봇 자기 손이 self-filter 를 새어 나가는 거리를 기록에서 잰다 (CPU, GPU 불필요).

    PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python -m AG3S.trajopt.experiments.panda_sfhand_leak \\
        --run-dir ${AG3S_STORAGE_ROOT}/results/benchmark/<DREC run> --geomsnap-dir <dir of *_geomsnap.npz> --out leak.json

`<DREC run>` = `--record-constraints full` 로 돈 vla_v11ab 실행 하나 (`panda_fixed_input_ablation` 의 입력과 같다).

## 무엇을 재나

CLOSING / HELD call 마다 점을 MuJoCo geom 으로 가른다 (`panda_fixed_input_ablation.geom_signed_distance`; 가장 가까운 collision
geom 의 body; mesh 는 AABB 상자로 근사된다):

* `hand`   = body `gripper0_right_gripper` (손바닥), `finger` = body `gripper0_leftfinger / rightfinger / *_tip`
* `target` = snapshot 의 BDDL target body (그릇)

그 점에서 **서버가 실제로 쓴 self-filter 모델** (`model.build_panda_self_filter_model()`, 손가락은 기록의
`finger_joints.applied.self_filter` 값) 의 구까지 부호 있는 거리 `|p - c| - r` 를 잰다. 필터를 통과한 점은 (inflation 0 에서)
모든 구의 바깥에 있으므로 이 값이 0 보다 크다: **그 link 에 inflation δ 를 주면 `sd <= δ` 인 점이 지워진다.**

* 새는 점의 거리 (leak) = 새는 점 (가장 가까운 geom 이 hand / finger 이고 그 geom 까지 `<= ghost_mm`) 에서
  그 link (hand → `panda_hand` 구, finger → 두 손가락 구) 까지의 `sd`.
* target 점의 거리 = target 점 (가장 가까운 geom 이 target body 이고 `<= ghost_mm`, 그리고 손가락/손 구에서
  `near_mm` 안) 에서 손가락 구 · 손 구까지의 `sd`. 이 값보다 큰 inflation 은 쥔 물체를 지우기 시작한다.

## 점의 출처 (세 가지, `primary_source` = `depth`)

* `depth`  -- `<stem>_geom.npz` 의 전해상도 깊이 (두 카메라, 요청 시각의 로봇 상태) 를 서버와 같은 `backproject` 로 점으로 만들고,
  inflation 0 의 self-filter 가 지우는 점 (`sd <= 0`) 을 뺀 것. 서버의 깊이 마스크 (TSDF 로 가는 것) 가 남기는 점과 같은 점이다.
  서버의 점군 voxel 축소 · cap 이 없어 표본이 크다.
* `cloud`  -- 기록의 `filtered_cloud` (융합 · voxel 로 줄어든 점군; 손 점이 몇 개뿐이다).
* `voxels` -- 기록된 fine 거리장의 점유 voxel (`d < 0`). TSDF 두께 (truncation) 와 이전 프레임에서 적분되어 남은 것이 섞여 거리가 커진다 (참고).

이 도구는 서버 · 기록기를 바꾸지 않는다 (읽기만).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Optional, Sequence

import numpy as np

TOOL = "AG3S.trajopt.experiments.panda_sfhand_leak"
HAND_BODIES = ("gripper0_right_gripper",)
FINGER_BODIES = ("gripper0_leftfinger", "gripper0_rightfinger", "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip")
HAND_LINK = "panda_hand"
FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger")
INFLATION_GRID_MM = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
QUANTILES = (50, 90, 95, 99, 100)
#: 고정 규칙의 상한 (mm)
CAP_MM = 15.0


def stats_mm(values_m: Sequence[float]) -> dict[str, Any]:
    """분포 요약 (mm): n · min · median · p90 · p95 · p99 · max."""
    v = np.asarray(values_m, np.float64).reshape(-1) * 1000.0
    if not v.size:
        return {"n": 0}
    out = {"n": int(v.size), "min": float(v.min())}
    for q, x in zip(QUANTILES, np.percentile(v, QUANTILES)):
        out["median" if q == 50 else ("max" if q == 100 else f"p{q}")] = float(x)
    return out


def ceil_mm(value_m: float, cap_mm: float = CAP_MM) -> float:
    """고정 규칙: 1 mm 로 올림, `cap_mm` 로 자름 (mm)."""
    return float(min(np.ceil(round(value_m * 1000.0, 9)), cap_mm))


def sphere_signed(points: np.ndarray, centres: np.ndarray, radii: np.ndarray, chunk: int = 256) -> np.ndarray:
    """`(N,)` -- 점에서 구들까지의 부호 있는 거리 `min_j(|p - c_j| - r_j)` (m)."""
    out = np.full(len(points), np.inf)
    for i in range(0, len(points), chunk):
        d = np.linalg.norm(points[i:i + chunk, None, :] - centres[None, :, :], axis=2) - radii[None, :]
        out[i:i + chunk] = d.min(axis=1)
    return out


def nearest_geom(points_world: np.ndarray, snap: dict) -> tuple[np.ndarray, np.ndarray]:
    """각 점의 가장 가까운 collision geom 의 (body 이름, 그 geom 까지 거리 (m))."""
    from AG3S.trajopt.experiments.panda_fixed_input_ablation import geom_signed_distance

    cols = np.flatnonzero((np.asarray(snap["geom_contype"]) | np.asarray(snap["geom_conaffinity"])) > 0)
    names = np.asarray([str(n) for n in snap["body_names"]])
    d = geom_signed_distance(points_world, np.asarray(snap["geom_xpos"])[cols], np.asarray(snap["geom_xmat"])[cols],
                             np.asarray(snap["geom_size"])[cols], np.asarray(snap["geom_type"])[cols])
    j = d.argmin(axis=1)
    return names[np.asarray(snap["geom_bodyid"])[cols][j]], d[np.arange(len(d)), j]


def filter_model_at(model, q: np.ndarray, applied: dict) -> tuple[np.ndarray, np.ndarray]:
    """기록된 손가락 값으로 self-filter 구 중심 · 반지름 (base 좌표)."""
    params = model.param_vector({k: float(v) for k, v in applied.items()}) if applied else None
    return model.sphere_centers_numeric(np.asarray(q, np.float64), params)


def measure_points(pts: np.ndarray, centres: np.ndarray, radii: np.ndarray, links: np.ndarray, T_world_base: np.ndarray,
                   snap: dict, *, ghost_m: float, near_m: float, survivors_only: bool, reach_m: float = 0.22,
                   extra: Optional[dict] = None) -> dict:
    """base 좌표의 점들을 hand / finger / target 으로 가르고 각각의 self-filter 구까지 거리 (m) 를 낸다.

    `survivors_only`: 기록의 점이 필터를 이미 통과한 점이 아니면 (깊이 픽셀) 어느 구 안의 점 (`sd <= 0`) 을 뺀다 --
    inflation 0 의 필터가 지웠을 점이다 (`hit <=> sq <= R^2`). `pts` 는 모두 손 구 중심에서 `reach_m` 안이므로, 그 밖에 있는
    구 (`|c - c0| - r > reach_m`; 로봇 밑동의 수천 구) 는 어느 점에도 닿지 않아 빼도 결과가 같다.

    `extra` = `{이름: (중심, 반지름)}` -- hand 점에서 이 구 집합까지의 거리도 잰다 (제약 모델의 palm shell 등).
    """
    hand, fing = links == HAND_LINK, np.isin(links, FINGER_LINKS)
    c0 = centres[hand].mean(axis=0)
    reachable = np.linalg.norm(centres - c0, axis=1) - radii <= reach_m
    sd_all = sphere_signed(pts, centres[reachable], radii[reachable])
    n_masked = int((sd_all <= 0.0).sum())
    if survivors_only:
        keep = sd_all > 0.0
        pts, sd_all = pts[keep], sd_all[keep]
    empty = {"sd_hand": np.zeros(0), "sd_finger": np.zeros(0), "geom": np.zeros(0), "extra": {k: np.zeros(0) for k in (extra or {})}}
    if not len(pts):
        return {"hand": empty, "finger": empty, "target": empty, "n_masked_by_filter": n_masked, "n_points": 0, "n_inside_filter": 0}
    world = pts @ T_world_base[:3, :3].T + T_world_base[:3, 3]
    body, dist = nearest_geom(world, snap)
    sd_hand = sphere_signed(pts, centres[hand], radii[hand])
    sd_fing = sphere_signed(pts, centres[fing], radii[fing])
    targets = [str(b) for b in (snap.get("target_bodies") or [])]
    is_hand = np.isin(body, HAND_BODIES) & (dist <= ghost_m)
    is_fing = np.isin(body, FINGER_BODIES) & (dist <= ghost_m)
    near = np.minimum(sd_hand, sd_fing) <= near_m
    is_tgt = np.isin(body, targets) & (dist <= ghost_m) & near

    def pick(m, with_extra=False):
        out = {"sd_hand": sd_hand[m], "sd_finger": sd_fing[m], "geom": dist[m]}
        if with_extra:
            out["extra"] = {k: sphere_signed(pts[m], c, r) for k, (c, r) in (extra or {}).items()}
        return out

    return {"hand": pick(is_hand, True), "finger": pick(is_fing), "target": pick(is_tgt), "n_masked_by_filter": n_masked,
            "n_points": int(len(pts)), "n_inside_filter": int((sd_all <= 0.0).sum())}


def model_state(model, call, summary: dict, literal, q: Optional[np.ndarray] = None):
    applied = ((literal(summary.get("finger_joints")) or {}).get("applied") or {}).get("self_filter") or {}
    centres, radii = filter_model_at(model, call.q_now if q is None else q, applied)
    return centres, radii, np.asarray([str(n) for n in model.sphere_link_names])


def near_hand(pts: np.ndarray, centres: np.ndarray, links: np.ndarray, reach_m: float) -> np.ndarray:
    c0 = centres[links == HAND_LINK].mean(axis=0)
    return pts[np.linalg.norm(pts - c0, axis=1) <= reach_m]


def constraint_sets(cmodel, call, summary: dict, literal, q: np.ndarray) -> dict:
    """제약 모델 (TrajOpt 의 구) 의 손바닥 구 집합: `palm_shell` (껍질 구) 과 `franka_hand` (franka.yml 의 panda_hand 구)."""
    applied = ((literal(summary.get("finger_joints")) or {}).get("applied") or {}).get("self_filter") or {}
    centres, radii = filter_model_at(cmodel, q, applied)
    links, roles = np.asarray([str(n) for n in cmodel.sphere_link_names]), np.asarray(list(cmodel.sphere_roles))
    shell, franka = (links == HAND_LINK) & (roles == "palm_shell"), (links == HAND_LINK) & (roles != "palm_shell")
    return {"palm_shell": (centres[shell], radii[shell]), "franka_hand": (centres[franka], radii[franka])}


def source_cloud(model, call, z, summary, literal, *, reach_m: float, snap: dict, **kw) -> Optional[dict]:
    """융합 `filtered_cloud` (서버가 필터한 뒤의 점; 6000 점 안팎으로 voxel 로 줄어 있다)."""
    centres, radii, links = model_state(model, call, summary, literal)
    pts = near_hand(np.asarray(z["filtered_cloud"], np.float64).reshape(-1, 3), centres, links, reach_m)
    if not len(pts):
        return None
    return measure_points(pts, centres, radii, links, np.asarray(call.T_world_base, np.float64), snap, survivors_only=False,
                          reach_m=reach_m, **kw)


def source_voxels(model, call, z, summary, literal, *, reach_m: float, snap: dict, **kw) -> Optional[dict]:
    """기록된 fine 거리장의 점유 voxel (`d < 0`)."""
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field
    from AG3S.trajopt.experiments.panda_fixed_input_ablation import neighbour_voxels

    centres, radii, links = model_state(model, call, summary, literal)
    field = build_field(z, outside_distance=0.45)
    vox = neighbour_voxels(field, centres[links == HAND_LINK].mean(axis=0), reach_m, layer=-1)
    if not len(vox):
        return None
    return measure_points(vox, centres, radii, links, np.asarray(call.T_world_base, np.float64), snap, survivors_only=False,
                          reach_m=reach_m, **kw)


def source_depth(model, call, summary, geom, literal, *, reach_m: float, snap: dict, cmodel=None, **kw) -> Optional[dict]:
    """전해상도 깊이 픽셀 (두 카메라, 카메라 시각의 로봇 상태) 의 역투영 -- 서버 깊이 마스크가 보는 점과 같다."""
    from AG3S.adapters.panda import model as pm
    from AG3S.ag3s.config import PointCloudConfig
    from AG3S.ag3s.stages.reconstruction import backproject

    # `depth_mm` is uint16 with `depth_scale_mm` counts per metre (wire.DEPTH_SCALE_MM); depth_min / depth_max are the
    # server's PointCloudConfig defaults (the render near / far planes in the file are not the sensor limits)
    cfg = PointCloudConfig(depth_scale=1.0 / float(geom["depth_scale_mm"]))
    T_wb = np.asarray(call.T_world_base, np.float64)
    ms = []
    for cam in ("head", "left_wrist"):
        q, _ = pm.split_robot_state(geom[f"robot_state/{cam}"][call.obs_index])
        centres, radii, links = model_state(model, call, summary, literal, q=q)
        cloud = backproject(geom[f"depth_mm/{cam}"][call.obs_index], geom[f"K/{cam}"][call.obs_index],
                            geom[f"T_base_cam/{cam}"][call.obs_index], cfg)
        pts = near_hand(np.asarray(cloud.points, np.float64), centres, links, reach_m)
        if len(pts):
            extra = constraint_sets(cmodel, call, summary, literal, q) if cmodel is not None else None
            ms.append(measure_points(pts, centres, radii, links, T_wb, snap, survivors_only=True, reach_m=reach_m, extra=extra, **kw))
    return merge(ms) if ms else None


def merge(ms: list[dict]) -> dict:
    out: dict[str, Any] = {k: {f: np.concatenate([m[k][f] for m in ms]) for f in ("sd_hand", "sd_finger", "geom")}
                           for k in ("hand", "finger", "target")}
    names = set().union(*(m["hand"].get("extra", {}).keys() for m in ms))
    out["hand"]["extra"] = {n: np.concatenate([m["hand"]["extra"][n] for m in ms if n in m["hand"].get("extra", {})]) for n in names}
    for k in ("n_points", "n_masked_by_filter", "n_inside_filter"):
        out[k] = sum(m[k] for m in ms)
    return out


def block(sub: list[dict]) -> dict[str, Any]:
    """한 점 출처의 call 모음 -> 분포 · inflation 격자."""
    def cat(key, field):
        return np.concatenate([p[key][field] for p in sub]) if sub else np.zeros(0)

    out: dict[str, Any] = {"n_calls": len(sub)}
    hand_pts, fing_pts = cat("hand", "sd_hand"), cat("finger", "sd_finger")
    out["hand_points_to_panda_hand_spheres_mm"] = stats_mm(hand_pts)
    out["finger_points_to_finger_spheres_mm"] = stats_mm(fing_pts)
    out["hand_points_to_finger_spheres_mm"] = stats_mm(cat("hand", "sd_finger"))
    out["hand_points_geom_distance_mm"] = stats_mm(cat("hand", "geom"))
    # 감도: 충돌 상자 위 · 안 (<= 5 mm) 의 점만 -- 확실히 로봇인 점
    sure = [p["hand"]["sd_hand"][p["hand"]["geom"] <= 0.005] for p in sub]
    out["hand_points_within_5mm_of_geom_to_panda_hand_mm"] = stats_mm(np.concatenate(sure) if sure else [])
    sure_f = [p["finger"]["sd_finger"][p["finger"]["geom"] <= 0.005] for p in sub]
    out["finger_points_within_5mm_of_geom_to_fingers_mm"] = stats_mm(np.concatenate(sure_f) if sure_f else [])
    # 새는 손 점이 TrajOpt 의 손바닥 구 (palm shell · franka.yml) 에서 얼마나 떨어져 있나: shell 이 self-filter 에 없다는 것이 문제의 일부인가
    ex = {n: np.concatenate([p["hand"]["extra"][n] for p in sub if n in p["hand"].get("extra", {})] or [np.zeros(0)])
          for n in ("palm_shell", "franka_hand")}
    for n, v in ex.items():
        if v.size:
            out[f"hand_points_to_constraint_{n}_spheres_mm"] = stats_mm(v)
    if ex["palm_shell"].size and ex["palm_shell"].size == ex["franka_hand"].size:
        out["hand_points_nearest_constraint_sphere_is_shell"] = float((ex["palm_shell"] <= ex["franka_hand"]).mean())
        out["hand_points_within_clearance_of_shell"] = {f"{d}mm": float((ex["palm_shell"] <= d / 1000.0).mean()) for d in (0, 2, 5, 10)}
    # call 별 p95 의 분포 (점이 많은 call 이 전체를 끌지 않는지)
    out["per_call_p95_of_hand_points_mm"] = stats_mm([np.percentile(p["hand"]["sd_hand"], 95) for p in sub if len(p["hand"]["sd_hand"]) >= 5])
    tf, th = cat("target", "sd_finger"), cat("target", "sd_hand")
    out["target_points_to_finger_spheres_mm"] = stats_mm(tf)
    out["target_points_to_hand_spheres_mm"] = stats_mm(th)
    per_f = [float(p["target"]["sd_finger"].min()) for p in sub if len(p["target"]["sd_finger"])]
    per_h = [float(p["target"]["sd_hand"].min()) for p in sub if len(p["target"]["sd_hand"])]
    out["target_nearest_to_finger_per_call_mm"] = stats_mm(per_f)
    out["target_nearest_to_hand_per_call_mm"] = stats_mm(per_h)
    grid = {}
    for d in INFLATION_GRID_MM:
        dm = d / 1000.0
        grid[str(d)] = {
            "hand_points_caught_by_panda_hand": float((hand_pts <= dm).mean()) if hand_pts.size else None,
            "finger_points_caught_by_fingers": float((fing_pts <= dm).mean()) if fing_pts.size else None,
            "target_points_deleted_by_fingers": float((tf <= dm).mean()) if tf.size else None,
            "target_points_deleted_by_hand": float((th <= dm).mean()) if th.size else None,
            "calls_losing_target_to_fingers": float(np.mean([m <= dm for m in per_f])) if per_f else None,
            "calls_losing_target_to_hand": float(np.mean([m <= dm for m in per_h])) if per_h else None}
    out["inflation_grid_mm"] = grid
    out["n_points"] = {"hand": int(hand_pts.size), "finger": int(fing_pts.size), "target_near": int(tf.size),
                       "scanned": int(sum(p["n_points"] for p in sub)), "masked_by_filter_at_0": int(sum(p["n_masked_by_filter"] for p in sub))}
    out["n_points_inside_a_filter_sphere"] = int(sum(p["n_inside_filter"] for p in sub))
    return out


def run(run_dir: pathlib.Path, snapdir: pathlib.Path, *, ghost_mm: float = 30.0, near_mm: float = 60.0, reach_mm: float = 220.0,
        sources: Sequence[str] = ("depth", "cloud", "voxels"), limit: Optional[int] = None) -> dict[str, Any]:
    from AG3S.adapters.panda import model as pm
    from AG3S.trajopt.experiments import panda_fixed_input_ablation as T

    kw = dict(ghost_m=ghost_mm / 1000.0, near_m=near_mm / 1000.0)
    reach = reach_mm / 1000.0
    model = pm.build_panda_self_filter_model()
    # the constraint model of `-pc` (--links arms, --palm-shell on): its palm-shell spheres are the ones TrajOpt keeps clear of the field
    from AG3S.trajopt import serve_safe as ss

    cmodel = pm.build_panda_constraint_model(pm.panda_constraint_link_filter("arms"),
                                             palm_shell=ss.palm_shell_options(argparse.Namespace(palm_shell="on", palm_shell_overflow=None)))
    stems = sorted(p.name[:-len("_calls.jsonl")] for p in (run_dir / "episodes").glob("*_calls.jsonl"))
    calls = []
    for st in stems:
        calls += T.load_episode_calls(run_dir / "episodes", st)
    T.attach_chunks(calls, run_dir / "server_constraints/run_0000")

    def dev(c) -> float:
        return float(np.abs(np.asarray(c.joint_refined)[:, :7] - np.asarray(c.joint_reference)[:, :7]).max())

    sel = [c for c in calls if c.chunk_path is not None and c.phase in ("closing", "held")][:limit]
    parts: dict[str, list[dict]] = {s: [] for s in sources}
    rows, geoms = [], {}
    for c in sel:
        snap = T.load_geom_snapshot(T.geom_snapshot_dir_path(snapdir, c.stem), c.obs_index)
        if snap is None:
            rows.append({"id": c.id, "phase": c.phase, "skipped": "no geom snapshot"})
            continue
        z = np.load(c.chunk_path, allow_pickle=False)
        summary = json.loads(str(z["summary_json"]))
        if c.stem not in geoms:
            geoms[c.stem] = np.load(run_dir / "episodes" / f"{c.stem}_geom.npz")
        row = {"id": c.id, "phase": c.phase, "dev_rad": dev(c)}
        for s in sources:
            if s == "depth":
                m = source_depth(model, c, summary, geoms[c.stem], T._literal, reach_m=reach, snap=snap, cmodel=cmodel, **kw)
            elif s == "cloud":
                m = source_cloud(model, c, z, summary, T._literal, reach_m=reach, snap=snap, **kw)
            else:
                m = source_voxels(model, c, z, summary, T._literal, reach_m=reach, snap=snap, **kw)
            if m is None:
                continue
            m.update(id=c.id, phase=c.phase, dev_rad=row["dev_rad"])
            parts[s].append(m)
            row[f"n_hand_{s}"], row[f"n_finger_{s}"] = int(len(m["hand"]["sd_hand"])), int(len(m["finger"]["sd_finger"]))
        rows.append(row)

    result: dict[str, Any] = {
        "tool": TOOL, "run_dir": str(run_dir), "ghost_mm": ghost_mm, "target_near_mm": near_mm, "reach_mm": reach_mm,
        "self_filter_model": {"n_spheres": int(model.n_spheres),
                              "spheres_per_link": {k: int(v) for k, v in zip(*np.unique(np.asarray(model.sphere_link_names), return_counts=True))},
                              "roles": sorted(set(model.sphere_roles)), "palm_shell_in_model": "palm_shell" in set(model.sphere_roles)},
        "n_calls_total": len(calls), "n_calls_closing_held_with_chunk": len(sel),
        "primary_source": "depth", "sources": {}}
    for s in sources:
        P = parts[s]
        result["sources"][s] = {"closing_and_held": block(P), "closing": block([p for p in P if p["phase"] == "closing"]),
                                "held": block([p for p in P if p["phase"] == "held"]),
                                "closing_and_held_dev_gt_0.02rad": block([p for p in P if p["dev_rad"] > 0.02])}
    result["per_call"] = rows
    return result


def decide(block_: dict, *, cap_mm: float = CAP_MM) -> dict[str, Any]:
    """고정 규칙 (lead, 2026-11-05): panda_hand = p95(hand 점) 을 1 mm 로 올림, `cap_mm` 로 자름.
    손가락은 (손가락 leak 이 크면서) 그 p95 가 median(call 별 target→손가락 최근접 거리) 보다 작을 때만 자기 p95 로 같이 넣는다."""
    hand_p95 = block_["hand_points_to_panda_hand_spheres_mm"]["p95"] / 1000.0
    fing = block_["finger_points_to_finger_spheres_mm"]
    tgt = block_["target_nearest_to_finger_per_call_mm"]
    out = {"cap_mm": cap_mm, "panda_hand_p95_mm": hand_p95 * 1000.0, "panda_hand_mm": ceil_mm(hand_p95, cap_mm),
           "finger_p95_mm": fing.get("p95"), "median_target_to_finger_per_call_mm": tgt.get("median")}
    include = bool("p95" in fing and tgt.get("n") and 0.0 < fing["p95"] < tgt["median"])
    out["fingers_included"] = include
    out["finger_mm"] = ceil_mm(fing["p95"] / 1000.0, cap_mm) if include else None
    return out


def add_alternatives(res: dict) -> dict:
    """The same fixed rule on the other point sources (sensitivity only; `decision` is the `primary_source`'s)."""
    res["decision_by_source"] = {
        s: decide(b["closing_and_held"]) for s, b in res["sources"].items()
        if "p95" in b["closing_and_held"]["hand_points_to_panda_hand_spheres_mm"]
        and "median" in b["closing_and_held"]["target_nearest_to_finger_per_call_mm"]}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--geomsnap-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--ghost-mm", type=float, default=30.0)
    ap.add_argument("--target-near-mm", type=float, default=60.0)
    ap.add_argument("--reach-mm", type=float, default=220.0)
    ap.add_argument("--limit", type=int, default=None, help="first N closing/held calls only (smoke)")
    ap.add_argument("--sources", nargs="+", default=["depth", "cloud", "voxels"], choices=["depth", "cloud", "voxels"])
    a = ap.parse_args(argv)
    res = run(a.run_dir, a.geomsnap_dir, ghost_mm=a.ghost_mm, near_mm=a.target_near_mm, reach_mm=a.reach_mm, sources=a.sources, limit=a.limit)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    primary = res["sources"].get(res["primary_source"])
    ok = primary is not None and "p95" in primary["closing_and_held"]["hand_points_to_panda_hand_spheres_mm"]
    res["decision"] = decide(primary["closing_and_held"]) if ok else None
    add_alternatives(res)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("self_filter_model", "decision")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
