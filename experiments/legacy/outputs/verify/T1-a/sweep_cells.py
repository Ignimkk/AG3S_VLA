"""T1-a 1 차 — attention 셀 1296 조합 전수 sweep. verifier(A2) 측정용, benchmark/ 밖.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1-a/sweep_cells.py --npz <attention npz> --records <run dir> --out <json>

## 왜 두 단계인가 — 실측 비용

`ground_target` 은 58,717 점 클라우드에서 **한 번에 4.82 s** 다 (`lift` 는 0.005 s).
1296 조합 x 15 프레임을 그대로 돌면 npz 하나에 약 26 시간이다. 그래서 나눈다.

* **A 단계 (전수 1296)** — 클라우드를 만들지 않고 16x16 격자에서만 잰다. MuJoCo 분할
  마스크를 같은 16x16 로 내려 각 칸의 target 픽셀 비율을 구해 두고, 조합마다
  `peak_on_target`(attention 최대 칸에 target 이 있나) 과 `target_mass`(attention 질량 중
  target 몫) 를 센다. 14D 선택 기준(`step-01-attention.json` 의 peak_on_target·hit_rate)과
  같은 종류의 양이다. 조합당 마이크로초.
* **B 단계 (상위 N 개만)** — A 에서 고른 조합만 실제 `lift -> ground_target` 을 태워
  `has_target` 을 낸다. 이것이 판정값이고, A 는 **순위를 매기는 데만** 쓴다.

축 순서는 생산자(`ag3s/experiments/sources/pi05_attention.py:234`)가 정한
`(frame, denoise, agg, layer, head, camera, 16, 16)` 이다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np

from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.stages.attention_lifting import lift
from AG3S.ag3s.stages.reconstruction import reconstruct
from AG3S.ag3s.stages.robot_filter import filter_robot_points
from AG3S.ag3s.stages.support_surface import fit_support_surfaces
from AG3S.ag3s.stages.target_grounding import ground_target
from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model

CAM = "zed_left"
G = 16


def target_body_ids(scene, target: str) -> set[int]:
    """`target` 이름을 포함하는 body 의 id 들 — 분할 마스크에서 target 픽셀을 고르는 데 쓴다."""
    import mujoco
    ids = set()
    for i in range(scene.model.nbody):
        name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if target in name:
            ids.add(i)
    return ids


def cell_target_fraction(body_ids_img: np.ndarray, tids: set[int]) -> np.ndarray:
    """(16,16) — 각 attention 칸에서 target 픽셀이 차지하는 비율."""
    h, w = body_ids_img.shape
    mask = np.isin(body_ids_img, list(tids)) if tids else np.zeros((h, w), bool)
    ys = (np.arange(h) * G // h).clip(0, G - 1)
    xs = (np.arange(w) * G // w).clip(0, G - 1)
    out = np.zeros((G, G), np.float64)
    cnt = np.zeros((G, G), np.float64)
    np.add.at(out, (ys[:, None].repeat(w, 1), xs[None, :].repeat(h, 0)), mask)
    np.add.at(cnt, (ys[:, None].repeat(w, 1), xs[None, :].repeat(h, 0)), 1.0)
    return out / np.maximum(cnt, 1.0)


def frame_context(scene, step, robot_model, cfg, tids):
    pose_scene(scene, step)
    frame = scene.capture(CAM)
    depth = np.asarray(frame.depth, np.float64)
    K = np.asarray(frame.camera_intrinsics, np.float64)
    T = np.asarray(frame.T_base_cam, np.float64)
    q = np.asarray(frame.robot_state, np.float64)
    bid = np.asarray(frame.body_ids)

    cloud, _ = reconstruct(depth=depth, pointcloud=None, camera_intrinsics=K, T_base_cam=T,
                           uv=None, config=cfg.pointcloud, frame_id=cfg.frame_id)
    cloud, _ = filter_robot_points(cloud, robot_model, q, cfg.pointcloud)
    surfaces, support_mask = fit_support_surfaces(cloud, cfg.support_surface, timestamp=0.0)
    return {"cloud": cloud, "support_mask": support_mask,
            "hw": tuple(int(x) for x in depth.shape[:2]), "K": K, "T": T,
            "n_points": len(cloud), "n_surfaces": len(surfaces),
            "tfrac": cell_target_fraction(bid, tids),
            "n_target_px": int(np.isin(bid, list(tids)).sum()) if tids else 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--top", type=int, default=40, help="B 단계로 보낼 상위 조합 수")
    ap.add_argument("--hit-radius", type=float, default=0.10,
                    help="grounding 중심이 참 사과에서 이 안이면 '맞게 짚었다'로 센다 (m)")
    ap.add_argument("--always-include", default="0,last,8,2",
                    help="A 순위와 무관하게 B 로 보낼 셀 'denoise,agg,layer,head' (14D 의 L8H2)")
    args = ap.parse_args()

    blob = np.load(args.npz, allow_pickle=False)
    att = np.asarray(blob["attention"], np.float32)
    denoise = [int(d) for d in blob["denoise_steps"]]
    aggs = [str(a) for a in blob["aggregations"]]
    cams = [str(c) for c in blob["cameras"]]
    ci = cams.index("cam_high")
    n_fr_npz, n_d, n_a, n_l, n_h, _ = att.shape[:6]
    print(f"npz {args.npz}\n  shape {att.shape}  denoise={denoise} aggs={aggs} cams={cams}")

    cfg = AG3SConfig()
    run = load_run(args.records, limit=args.frames)
    scene = replay_scene(run)
    robot_model = build_robot_model(scene)
    tids = target_body_ids(scene, args.target)
    n_frames = min(args.frames, len(run.steps), n_fr_npz)
    print(f"  target={args.target} body ids {sorted(tids)}")

    contexts, truth = [], []
    t0 = time.time()
    for i in range(n_frames):
        ctx = frame_context(scene, run.steps[i], robot_model, cfg, tids)
        truth.append(np.asarray(scene.body_position_in_base(args.target), float).reshape(3))
        contexts.append(ctx)
        print(f"  frame {i:3d}  points {ctx['n_points']:7d}  surfaces {ctx['n_surfaces']}  "
              f"target px {ctx['n_target_px']:6d}")
    print(f"MuJoCo + 클라우드 {time.time() - t0:.1f}s ({n_frames} 프레임)")

    # ---------------------------------------------------------------- A 단계: 전수 1296
    tfrac = np.stack([c["tfrac"] for c in contexts])              # (F,16,16)
    grids = att[:n_frames, :, :, :, :, ci]                        # (F,D,A,L,H,16,16)
    flatg = grids.reshape(n_frames, n_d, n_a, n_l, n_h, G * G).astype(np.float64)
    flatt = tfrac.reshape(n_frames, G * G)

    peak_idx = flatg.argmax(axis=-1)                              # (F,D,A,L,H)
    on_target = np.take_along_axis(
        flatt[:, None, None, None, None, :],
        peak_idx[..., None], axis=-1)[..., 0] > 0.0               # (F,D,A,L,H)
    gsum = flatg.sum(axis=-1)
    tmass = (flatg * flatt[:, None, None, None, None, :]).sum(axis=-1) / np.maximum(gsum, 1e-12)

    peak_frames = on_target.sum(axis=0)                           # (D,A,L,H)
    mass_mean = tmass.mean(axis=0)
    print(f"A 단계 끝. peak_on_target 최대 {int(peak_frames.max())}/{n_frames} 프레임")

    rowsA = []
    for di in range(n_d):
        for ai in range(n_a):
            for li in range(n_l):
                for hi in range(n_h):
                    rowsA.append({"denoise": denoise[di], "agg": aggs[ai], "layer": li, "head": hi,
                                  "_ix": (di, ai, li, hi),
                                  "peak_on_target_frames": int(peak_frames[di, ai, li, hi]),
                                  "target_mass_mean": float(mass_mean[di, ai, li, hi])})
    rowsA.sort(key=lambda r: (-r["peak_on_target_frames"], -r["target_mass_mean"]))

    # ---------------------------------------------------------------- B 단계: 상위 N + 14D 셀
    want = [tuple(r["_ix"]) for r in rowsA[: args.top]]
    d0, a0, l0, h0 = args.always_include.split(",")
    fixed = (denoise.index(int(d0)), aggs.index(a0), int(l0), int(h0))
    if fixed not in want:
        want.append(fixed)
    print(f"B 단계: {len(want)} 조합 x {n_frames} 프레임 "
          f"(약 {len(want) * n_frames * 4.9 / 60:.0f} 분 예상)")

    byix = {tuple(r["_ix"]): r for r in rowsA}
    rowsB = []
    t0 = time.time()
    for k, ix in enumerate(want):
        di, ai, li, hi = ix
        ok, hit, statuses, scores, dists = 0, 0, {}, [], []
        for i, ctx in enumerate(contexts):
            ac = lift(ctx["cloud"], att[i, di, ai, li, hi, ci], cfg.attention,
                      image_hw=ctx["hw"], camera_intrinsics=ctx["K"], T_base_cam=ctx["T"])
            g = ground_target(ac, cfg.clustering,
                              seed_percentile=cfg.attention.seed_percentile,
                              seed_threshold=cfg.attention.seed_threshold,
                              exclude_mask=ctx["support_mask"],
                              min_radius=cfg.geometry.min_radius, timestamp=0.0)
            nm = g.status.name if hasattr(g.status, "name") else str(g.status)
            statuses[nm] = statuses.get(nm, 0) + 1
            scores.append(round(float(g.best_score), 4))
            if g.ok:
                ok += 1
                ctr = np.asarray(g.target.centroid, float).reshape(3)
                dd = float(np.linalg.norm(ctr - truth[i]))
                dists.append(round(dd, 4))
                hit += dd <= args.hit_radius
        base = byix[ix]
        rowsB.append({"denoise": base["denoise"], "agg": base["agg"], "layer": li, "head": hi,
                      "is_14d_cell": ix == fixed,
                      "peak_on_target_frames": base["peak_on_target_frames"],
                      "target_mass_mean": base["target_mass_mean"],
                      "n_has_target": ok, "n_on_apple": hit, "n_frames": n_frames,
                      "dist_to_apple_m": dists, "statuses": statuses, "best_scores": scores})
        el = time.time() - t0
        print(f"  [{k + 1:3d}/{len(want)}] d={base['denoise']} {base['agg']:5} "
              f"L{li:2d}H{hi}  peak {base['peak_on_target_frames']:2d}  "
              f"has_target {ok}/{n_frames}  on_apple {hit}/{n_frames}  ({el:.0f}s)")

    rowsB.sort(key=lambda r: (-r["n_on_apple"], -r["n_has_target"], -r["peak_on_target_frames"]))
    for r in rowsA:
        r.pop("_ix", None)

    out = {
        "npz": args.npz, "records": args.records, "n_frames": n_frames,
        "axis_order": "(frame, denoise, agg, layer, head, camera, 16, 16) — pi05_attention.py:234",
        "camera_used": "cam_high", "target": args.target,
        "target_body_ids": sorted(tids),
        "cost_measured": {"ground_target_s_per_call": 4.82, "lift_s_per_call": 0.005,
                          "cloud_points": contexts[0]["n_points"]},
        "stageA": {"n_combos": len(rowsA),
                   "metric": "peak_on_target_frames = attention 최대 칸에 target 픽셀이 있는 프레임 수; "
                             "target_mass_mean = attention 질량 중 target 칸의 몫 (프레임 평균)",
                   "rows": rowsA},
        "stageB": {"n_combos": len(rowsB),
                   "hit_radius_m": args.hit_radius,
                   "metric": "n_has_target = ground_target 가 성공한 프레임 수; n_on_apple = 그 "
                             "중심이 참 사과에서 hit_radius 안인 프레임 수. has_target 는 "
                             "'무언가를 지었다' 일 뿐 그것이 사과라는 뜻이 아니다.",
                   "rows": rowsB},
        "frame_points": [c["n_points"] for c in contexts],
        "frame_target_px": [c["n_target_px"] for c in contexts],
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    c14 = [r for r in rowsB if r["is_14d_cell"]][0]
    print(f"\nwrote {args.out}")
    print(f"  14D 셀 L{c14['layer']}H{c14['head']} d={c14['denoise']} {c14['agg']}: "
          f"has_target {c14['n_has_target']}/{n_frames}  on_apple {c14['n_on_apple']}/{n_frames}")
    print(f"  B 최선: on_apple {rowsB[0]['n_on_apple']}/{n_frames}  "
          f"has_target {rowsB[0]['n_has_target']}/{n_frames} "
          f"d={rowsB[0]['denoise']} {rowsB[0]['agg']} L{rowsB[0]['layer']}H{rowsB[0]['head']}")
    print(f"  has_target > 0 인 조합 {sum(1 for r in rowsB if r['n_has_target'] > 0)}/{len(rowsB)}")
    print(f"  on_apple  > 0 인 조합 {sum(1 for r in rowsB if r['n_on_apple'] > 0)}/{len(rowsB)}")


if __name__ == "__main__":
    main()
