"""T44 E2 — target exclusion 오프라인 비교: `ball` · `ball_owned` (remove/keep) · **oracle**.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      /mnt/dev/work/.venv-openpi-live/bin/python -m benchmark.ag3s.experiments.live.exclusion_e2 \\
      --run outputs/verify/T43/LB/runs/E0/E0_ep1008_s10081 --frames 6 8 10 \\
      --out outputs/verify/T44/e2/ep1008

    # 후보 frame 고르기 (GPU 불필요): 이웃 과일 거리 · 손–target 거리 표
    ... exclusion_e2 --run RUN --list-candidates

## 무엇을 하나

client 기록 (`frames.jsonl` 의 planning 행: `qpos` 66 + `object_poses`, `manifest.json` 의 씬) 의 한 frame 을
`TransportScene` 으로 되살려 (qpos 를 그대로 넣고 `mj_forward`) 서버와 **같은 경로**로 관측을 만든다 —
`client.capture` 와 같은 카메라 · `wire.pack_request` (uint16 mm depth) → `wire.unpack_camera_observations`.
그 위에서 실제 AG3S (`serve_safe.build_ag3s`, V11AB 의 ESDF 설정: curobo · 20 mm coarse · 5 mm fine ·
5 mm TSDF · `exclude_authorized`) 를 돌려 reconstruction → self-filter → support → grounding → ESDF 를 얻는다.

그 다음 **같은 integrator** (AG3S 의 `CuroboFieldBuilder` 가 이 frame 에 적분한 TSDF) 로 target-free 미세
계층을 넷 만든다 — `_build_tiers(only="fine")` 를 방식만 바꿔 부른다 (같은 창 · 같은 TSDF):

| method | 지우는 것 |
|---|---|
| `ball` | 공 안 (`z ≥ z_min`) seed 전부 — V11AB 기준 |
| `ball_owned_remove` | 공 ∖ non-target 소유 seed, unowned 는 제거 |
| `ball_owned_keep` | 같고, unowned 는 보존 |
| `oracle` | **GT target body** 라벨 seed 만 (ROI 안). 부호 교정 = 지운 seed ∪ GT target 쪽 non-seed |

## identity (`identity_mode`)

attention 은 π0.5 가 아니라 GT target 중심의 Gaussian (`mujoco_source.gaussian_attention`) 이고, **target 을
보는 카메라에만** 싣는다 (`gt_attention` — 안 보는 카메라의 map 은 정규화 때문에 전부 ≈ 1 이 된다).
`--identity gt` (기본) 는 **correct identity 를 강제**하는 진단 옵션이다: pipeline 이 고른 manipulated 기하의 점이
80 % 이상 GT target 이면 `pipeline`, 아니면 GT target 비율이 가장 높은 grounded cluster 가 80 % 이상이면
`nearest_cluster`, 둘 다 아니면 GT label 이 target 인 non-support cloud 점 전부 (`gt_label` — GT 중심에서
키우되 붙은 이웃 과일로 새지 않는다). `--identity gt_label` 은 늘 그 마지막 것 (완전한 identity),
`--identity pipeline` 은 바꾸지 않는다 (틀려도 그대로). 고른 점이 관측된 target 을 얼마나 덮는지
(`identity.identity_completeness`) 를 남긴다 — `ball_owned` 는 빠진 target 점을 non-target 증거로 읽는다.

## GT label (seed 마다)

depth 픽셀의 `body_ids` 를 base 점으로 올린 것 + 과일 · crate 의 **mesh 전체 표면** 표본 (`gt_mesh_points` — depth 는
과일 앞면만 보지만 TSDF 는 truncation 띠의 뒷면도 seed 로 심는다) 을 미세 voxel 로 모아 다수결 — `target` ·
`distractor` (다른 과일) · `support` (table) · `destination` (crate) · `robot` · `other`. 다수가 80 % 미만이면
`shared`. 점이 없는 seed voxel 은 1.5 voxel 안 최근접 점의 label, 그 다음 과일 convex hull 안이면 그 과일, 그것도 아니면 `unlabelled`.

## 입력 (셋 중 하나)

| 입력 | 장면 |
|---|---|
| `--run RUN --frames SEQ…` (E2-lb) | client 기록의 planning 행 `qpos` 66 그대로 |
| `--demo-episodes 0,1,…` (E2-demo) | 16D demonstration: `teleop` keyframe (base · torso · head) + parquet `observation.state` 의 팔 · gripper + `initial_fruit_poses` · `basket_pose` (grasp 전이라 과일은 제자리). frame = `pregrasp_start_frame` ~ `grasp_close_start_frame` 균등 `--demo-frames-per-episode` 개 |
| `--posed-frames F.json` | 위와 같은 posed-frame 목록 (`--write-posed-frames` 가 쓰는 형식) |

## 출력 (frame 마다 `e2_<run>_s<seq>.json` + `.npz`)

- `seeds` — ROI (= 공 + `--shell` 껍질) 안 seed 수: method × label × (removed / kept), 공 안만 따로 (`in_ball`).
- `metrics` — method 마다 target recall (지운 target seed / target seed, ROI) · distractor seed 손실 ·
  support seed 손실 · ambiguous / unowned 수 · build ms.
- `queries` — 같은 `q` 의 robot 구 (`--query-links`, 기본 양팔+손가락; authorized = 그 팔의 contact links):
  main 거리, method 별 target-free 거리, oracle 대비 overestimation, falsely-feasible
  (`d − r ≥ margin` 인데 oracle 은 `< margin`). `roi_probe` 는 공 둘레 껍질 위 점 (r + 0/5/10/20 mm) 의 같은 비교.
- npz: ROI seed ijk · label · method 별 처리 (0 kept · 1 removed) · origin · voxel (figure 용).

**실행 환경**: `.venv-openpi-live` + GPU (cuRobo). 첫 `_build_tiers` 는 JIT 라 frame 마다 한 번 버리고 잰다
(curobo-esdf 함정 3).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Any, Optional

import numpy as np

METHODS = ("ball", "ball_owned_remove", "ball_owned_keep", "oracle")
FRUITS = ("apple", "banana", "orange", "pear")
LABELS = ("target", "distractor", "support", "destination", "robot", "other", "shared",
          "unlabelled")


# ----------------------------------------------------------------------------- record → scene


def load_planning_rows(run: pathlib.Path) -> list[dict]:
    rows = [json.loads(line) for line in open(run / "frames.jsonl") if line.strip()]
    return [r for r in rows if r.get("kind") == "planning" and r.get("qpos") is not None]


def open_scene(manifest: dict):
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene

    scene = TransportScene(manifest["scene"]["model_xml"], settle_steps=0)
    return scene


def pose(scene, row: dict) -> float:
    q = np.asarray(row["qpos"], np.float64)
    if q.shape[0] != int(scene.model.nq):
        raise ValueError(f"planning row qpos has {q.shape[0]} entries, model nq={scene.model.nq} — "
                         "a different scene would be replayed")
    scene.data.qpos[:] = q
    scene.data.qvel[:] = 0.0
    scene.mujoco.mj_forward(scene.model, scene.data)
    # object_poses 와 qpos 가 같은 씬을 말하는지 (free joint 를 qpos 가 이미 담는다) 확인한다. 두 값은
    # 같은 순간의 것이 아니다 (실측: crate 0.25 mm · apple 0.004 mm) — 5 mm 를 넘으면 다른 씬이다.
    worst = 0.0
    for name, p in (row.get("object_poses") or {}).items():
        try:
            got = scene.body_position(name)
        except KeyError:
            continue
        worst = max(worst, float(np.linalg.norm(got - np.asarray(p["pos"], np.float64))))
    if worst > 0.005:
        raise ValueError(f"an object is {worst * 1000:.1f} mm from the recorded pose — "
                         "a different scene would be replayed")
    return worst


def body_in_base(scene, name: str) -> np.ndarray:
    return scene.body_position_in_base(name)


# ----------------------------------------------------------------------------- demonstration poses

#: The 16-D demonstration dataset (T44 E2-demo). `meta/randomized_episodes.jsonl` + `data/chunk-*/`.
DEMO_DATASET = pathlib.Path("/mnt/dev/work/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1")
#: 16-D state layout (`meta/info.json` `observation.state`): [left 7, left gripper, right 7, right gripper].
STATE16_ARM = {"left": slice(0, 7), "right": slice(8, 15)}
STATE16_GRIP = {"left": 7, "right": 15}
#: `pi05_infer.RBY1_GRIPPER_OPEN` — the gripper ctrl/qpos the dataset was collected with (1.0 = open).
RBY1_GRIPPER_OPEN = -0.045


def load_demo_meta(dataset: pathlib.Path) -> dict[int, dict]:
    out = {}
    for line in open(dataset / "meta" / "randomized_episodes.jsonl"):
        if line.strip():
            r = json.loads(line)
            out[int(r["episode_index"])] = r
    return out


def demo_state16(dataset: pathlib.Path, episode: int) -> np.ndarray:
    import pyarrow.parquet as pq

    path = dataset / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    col = pq.read_table(path, columns=["observation.state"]).column(0).to_pylist()
    return np.asarray(col, np.float64)


def demo_frames(meta: dict, n: int) -> list[int]:
    """`n` frames evenly spaced over `[pregrasp_start_frame, grasp_close_start_frame]` (ends included)."""
    ev = meta["events"]
    a, b = int(ev["pregrasp_start_frame"]), int(ev["grasp_close_start_frame"])
    return sorted(set(int(round(v)) for v in np.linspace(a, b, max(1, n))))


def posed_frame_from_demo(meta: dict, state: np.ndarray, frame: int) -> dict:
    """A posed-frame record (the `--posed-frames` JSON schema): everything `pose_posed` needs."""
    return {"source": "demo", "episode_index": int(meta["episode_index"]), "frame": int(frame),
            "split": meta.get("split"), "success": meta.get("success"),
            "target_fruit": meta["target_fruit"], "used_arm": meta["used_arm"],
            "state16": [float(v) for v in state[frame]],
            "initial_robot_state": meta.get("initial_robot_state"),
            "fruit_poses": meta["initial_fruit_poses"], "basket_pose": meta["basket_pose"]}


def pose_posed(scene, pf: dict) -> float:
    """Scene from a posed frame: `teleop` keyframe (base · torso · head · everything else), the 16-D arm
    state (arms + grippers, `pi05_infer.rby1_state` inverse), fruits at their (initial) pose, crate at
    `basket_pose`. No settling — a static pose (the fruits have not been touched before the grasp)."""
    m, d, mj = scene.model, scene.data, scene.mujoco
    key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "teleop")
    if key >= 0:
        mj.mj_resetDataKeyframe(m, d, key)
    else:
        mj.mj_resetData(m, d)

    def adr(name):
        j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise ValueError(f"joint {name!r} missing from the model")
        return int(m.jnt_qposadr[j])

    st = np.asarray(pf["state16"], np.float64).reshape(16)
    for side, tag in (("left", "l"), ("right", "r")):
        for i, v in enumerate(st[STATE16_ARM[side]]):
            d.qpos[adr(f"{side}_arm_{i}")] = float(v)
        g = float(st[STATE16_GRIP[side]]) * RBY1_GRIPPER_OPEN
        d.qpos[adr(f"gripper_finger_{tag}1")] = g
        try:   # the second finger mirrors the first (recorded qpos: −0.045 / +0.045 when open)
            d.qpos[adr(f"gripper_finger_{tag}2")] = -g
        except ValueError:
            pass
    poses = {f"{k}_free": v for k, v in pf["fruit_poses"].items()}
    poses["crate_free"] = pf["basket_pose"]
    for joint, pose7 in poses.items():
        p = np.asarray(pose7, np.float64).reshape(7)
        a = adr(joint)
        d.qpos[a:a + 7] = p
    d.qvel[:] = 0.0
    mj.mj_forward(m, d)
    return 0.0


def nearest_other_xy(pf: dict) -> float:
    t = np.asarray(pf["fruit_poses"][pf["target_fruit"]][:2], np.float64)
    return min(float(np.linalg.norm(np.asarray(v[:2], np.float64) - t))
               for k, v in pf["fruit_poses"].items() if k != pf["target_fruit"])


# ----------------------------------------------------------------------------- GT labels


def label_classes(scene, target: str) -> dict[int, str]:
    from benchmark.ag3s.experiments.sources.mujoco_source import is_robot_body

    out = {}
    for bid, name in scene._body_names.items():
        if is_robot_body(name):
            out[bid] = "robot"
        elif name == target:
            out[bid] = "target"
        elif name in FRUITS:
            out[bid] = "distractor"
        elif name == "table":
            out[bid] = "support"
        elif name == "crate":
            out[bid] = "destination"
        else:
            out[bid] = "other"
    return out


def gt_points(frames, classes: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
    """All depth pixels of all cameras → base points + class index into `LABELS`."""
    pts, lab = [], []
    for f in frames:
        depth = np.asarray(f.depth, np.float64)
        ids = np.asarray(f.body_ids)
        v, u = np.nonzero((depth > 1e-6) & np.isfinite(depth) & (ids >= 0))
        z = depth[v, u]
        K, T = np.asarray(f.camera_intrinsics), np.asarray(f.T_base_cam)
        cam = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        pts.append(cam @ T[:3, :3].T + T[:3, 3])
        lab.append(np.asarray([LABELS.index(classes.get(int(b), "other")) for b in ids[v, u]],
                              np.int16))
    return np.vstack(pts), np.concatenate(lab)


def gt_mesh_points(scene, classes: dict[int, str], *, spacing: float = 0.003):
    """The **whole** surface (mesh samples, `mujoco_source._body_surface_points`) of every fruit and the
    crate, in the base frame, with its class, and each fruit's convex hull (`(Delaunay, class)`).

    Depth points only see the front of a fruit; the TSDF also seeds inside it (measured, demo ep 677:
    ~190 seeds per frame 8–18 mm from the ball centre, i.e. inside the apple, with no depth point
    within 1.5 voxels). Without the mesh and the hull those would be `unlabelled` and the oracle would
    keep the target's inside as an obstacle."""
    from scipy.spatial import Delaunay

    from benchmark.ag3s.experiments.sources.mujoco_source import _body_surface_points

    base = np.linalg.inv(scene.body_pose("base"))
    pts, lab, hulls = [], [], []
    for bid, cls in classes.items():
        name = scene._body_names.get(bid)
        if cls not in ("target", "distractor", "destination"):
            continue
        local = _body_surface_points(scene.model, bid, spacing)
        if not len(local):
            continue
        T = base @ scene.body_pose(name)
        w = local @ T[:3, :3].T + T[:3, 3]
        pts.append(w)
        lab.append(np.full(len(w), LABELS.index(cls), np.int16))
        if cls != "destination":          # the crate is hollow — its hull is not its volume
            try:
                hulls.append((Delaunay(w[::max(1, len(w) // 4000)]), LABELS.index(cls)))
            except Exception:  # noqa: BLE001 — degenerate mesh: surface samples only
                pass
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, np.int16), hulls
    return np.vstack(pts), np.concatenate(lab), hulls


def voxel_labels(points, labels, origin, vs, ijk, *, share=0.8, hulls=()) -> np.ndarray:
    """`ijk` (K, 3) voxel 마다 GT label index — 다수결 (`share` 미만이면 `shared`), 점 없는 voxel 은
    1.5 vs 안 최근접 점의 label, 그것도 없으면 `unlabelled`."""
    from scipy.spatial import cKDTree

    out = np.full(len(ijk), LABELS.index("unlabelled"), np.int16)
    if not len(ijk) or not len(points):
        return out
    lo = ijk.min(axis=0) - 2
    hi = ijk.max(axis=0) + 2
    g = np.floor((points - origin) / vs + 0.5).astype(np.int64)
    near = np.all((g >= lo) & (g <= hi), axis=1)
    g, lab, p = g[near], labels[near], points[near]
    key = lambda a: (a[:, 0] << 42) | (a[:, 1] << 21) | a[:, 2]   # noqa: E731
    gk = key(g - lo)
    want = key(ijk - lo)
    n_lab = len(LABELS)
    counts = {}
    for k_, l_ in zip(gk.tolist(), lab.tolist()):
        c = counts.setdefault(k_, np.zeros(n_lab, np.int64))
        c[l_] += 1
    centres = origin + ijk * vs
    tree = cKDTree(p) if len(p) else None
    for i, k_ in enumerate(want.tolist()):
        c = counts.get(k_)
        if c is not None:
            top = int(np.argmax(c))
            out[i] = top if c[top] >= share * c.sum() else LABELS.index("shared")
        elif tree is not None:
            d, j = tree.query(centres[i], k=1)
            if d <= 1.5 * vs:
                out[i] = lab[j]
    # inside a fruit (its convex hull) — the TSDF's interior seeds of an object belong to it.
    un = np.flatnonzero(out == LABELS.index("unlabelled"))
    for hull, cls in hulls:
        if not len(un):
            break
        inside = hull.find_simplex(centres[un]) >= 0
        out[un[inside]] = cls
        un = un[~inside]
    return out


# ----------------------------------------------------------------------------- identity


def gt_attention(scene, frames: dict, target: str, centre, *, min_px: int = 20) -> dict:
    """GT stand-in attention, **only for cameras that see the target** (≥ `min_px` target pixels).

    `gaussian_attention` normalises by its own max, so for a camera whose projection of the target
    falls far outside the image (behind it, or the other wrist) the floor dominates and the whole map
    becomes ≈ 1 — every point of that camera turns into an attention seed (measured: wrist_cam_r on
    demo ep 677 seeded the pear/banana side, 600 mm from the target). A camera without a map
    contributes geometry only (`CameraObservation` docstring)."""
    from benchmark.ag3s.experiments.sources.mujoco_source import gaussian_attention

    tid = scene.mujoco.mj_name2id(scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, target)
    out = {}
    for cam, f in frames.items():
        if int((np.asarray(f.body_ids) == tid).sum()) >= min_px:
            out[cam] = gaussian_attention(f, centre)
    return out


def point_labeler(gt_pts, gt_lab, radius: float):
    """`f(points) -> label index` — the class of the nearest GT depth point within `radius`
    (`unlabelled` otherwise). The cloud is the same depth, so a cloud point's own pixel is ≤ 1 mm away."""
    from scipy.spatial import cKDTree

    tree = cKDTree(gt_pts) if len(gt_pts) else None

    def f(points):
        p = np.asarray(points, np.float64).reshape(-1, 3)
        out = np.full(len(p), LABELS.index("unlabelled"), np.int16)
        if tree is None or not len(p):
            return out
        d, j = tree.query(p, k=1)
        ok = d <= radius
        out[ok] = gt_lab[j[ok]]
        return out

    return f


def choose_identity(ag3s, debug, gt_centre, mode: str, *, support_mask, label_of,
                    min_fraction: float = 0.8):
    """`(geometry, identity_mode, record)` — the target geometry the four fields exclude.

    `--identity gt` (correct identity forced, diagnostic): the pipeline's manipulated geometry if
    ≥ `min_fraction` of its points are GT-target points (`pipeline`); else the grounded cluster with the
    highest GT-target fraction if it passes (`nearest_cluster`); else every non-support cloud point whose
    GT label is the target (`gt_label` — the 'grow from the GT centre' fallback without leaking into a
    touching neighbour). `--identity pipeline` takes the pipeline's choice as it is.
    """
    from benchmark.ag3s.types import TargetGeometry

    t_idx = LABELS.index("target")
    cloud = debug.get("filtered_cloud")
    pts = np.asarray(cloud.points, np.float64)
    grounding = debug.get("grounding")
    clusters = list(getattr(grounding, "clusters", ()) or ())
    ident = getattr(ag3s, "_gated_manipulated", None)
    chosen = None if ident is None else ident.geometry

    def frac(points):
        lab = label_of(points)
        return float((lab == t_idx).mean()) if len(lab) else 0.0

    def geom_of(idx):
        idx = np.asarray(idx, np.int64)
        return TargetGeometry(id=-1, points=pts[idx], point_indices=idx,
                              centroid=pts[idx].mean(axis=0), bounding_geometry=None,
                              attention_score=0.0, confidence=0.0)

    rec: dict[str, Any] = {"pipeline_id": None if ident is None else int(ident.id),
                           "pipeline_state": None if ident is None else str(ident.state),
                           "grounding_status": getattr(getattr(grounding, "status", None),
                                                       "value", None),
                           "n_clusters": len(clusters), "min_target_fraction": min_fraction}
    if chosen is not None:
        rec["pipeline_centroid_err_mm"] = round(float(np.linalg.norm(
            np.asarray(chosen.centroid) - gt_centre)) * 1000.0, 2)
        rec["pipeline_target_fraction"] = round(frac(chosen.points), 4)
    keep = label_of(pts) == t_idx
    if support_mask is not None and len(support_mask) == len(pts):
        keep &= ~np.asarray(support_mask, bool)
    gt_idx = np.flatnonzero(keep)
    rec["n_gt_target_cloud_points"] = int(len(gt_idx))

    def done(geom, how):
        # completeness: how much of the observed target the chosen point set covers — `ball_owned`
        # keeps the seeds of target points left out of it (they are non-target evidence).
        if geom is not None and len(gt_idx):
            got = np.intersect1d(np.asarray(geom.point_indices, np.int64), gt_idx).size
            rec["identity_completeness"] = round(got / len(gt_idx), 4)
        return geom, how, rec

    if mode == "pipeline":
        return done(chosen, "pipeline")
    if mode == "gt_label":
        if not len(gt_idx):
            return None, "none", rec
        rec["chosen_target_fraction"] = 1.0
        return done(geom_of(gt_idx), "gt_label")
    if chosen is not None and rec["pipeline_target_fraction"] >= min_fraction:
        rec["chosen_target_fraction"] = rec["pipeline_target_fraction"]
        return done(chosen, "pipeline")
    if clusters:
        fr = [frac(pts[np.asarray(c.point_indices, np.int64)]) for c in clusters]
        k = int(np.argmax(fr))
        rec["best_cluster_target_fraction"] = round(fr[k], 4)
        rec["best_cluster_err_mm"] = round(float(np.linalg.norm(
            np.asarray(clusters[k].centroid) - gt_centre)) * 1000.0, 2)
        if fr[k] >= min_fraction:
            rec["chosen_target_fraction"] = round(fr[k], 4)
            return done(geom_of(clusters[k].point_indices), "nearest_cluster")
    if not len(gt_idx):
        return None, "none", rec
    rec["chosen_target_fraction"] = 1.0
    return done(geom_of(gt_idx), "gt_label")


# ----------------------------------------------------------------------------- the four tiers


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_free_tier(builder, ball, *, owned=None, exclusion=None):
    """같은 integrator · 같은 창에서 target-free 미세 계층 하나. `(tier, ms)`."""
    import torch

    itg = builder._mapper.integrator
    itg_fine = None if builder._fine_mapper is None else builder._fine_mapper.integrator
    dev = builder._mapper._device if hasattr(builder._mapper, "_device") else "cuda:0"
    if exclusion is not None:
        builder._ball_exclusion = exclusion          # instance attribute shadows the method
    try:
        _sync()
        t0 = time.perf_counter()
        (tier,) = builder._build_tiers(itg, None, dev, torch, only="fine",
                                       tier_name="fine_no_target", ball=ball, itg_fine=itg_fine,
                                       owned=owned)
        _sync()
        ms = (time.perf_counter() - t0) * 1e3
    finally:
        if exclusion is not None:
            del builder._ball_exclusion
    return tier, ms


def build_coarse_free_tier(builder, ball, *, owned=None, nontarget=None, target=None,
                           sign_floor=None):
    """T44c A5 — 같은 integrator 에서 target 을 지운 **거친** 계층 (`coarse_no_target`). `(tier, ms)`.

    `CuroboFieldBuilder.update` 가 `esdf.target_free_coarse` 에서 만드는 것과 같은 호출이다
    (`_coarse_free_spec` → `_build_tiers(only="coarse", coarse_free=...)`). `owned` = 그 방식의
    `OwnedEvidence` (`ball` 이면 None), `nontarget` = 공유 판정의 non-target 점 (방식과 무관).
    거친 TSDF (truncation 60 mm) 를 쓰므로 미세 target-free 계층과 버퍼가 다르다 — 순서는 상관없다.

    T44c A6: `target` = target 점 (`plane` 띠의 target 쪽 판정, pipeline 은 `target_free_points` 를 넘긴다),
    `sign_floor` = `"guard"` (A5) · `"plane"` — None 이면 builder config (`esdf.target_free_coarse_sign_floor`,
    기본 `"plane"` — T44c A7).
    """
    import dataclasses as _dc

    import torch

    itg = builder._mapper.integrator
    itg_fine = None if builder._fine_mapper is None else builder._fine_mapper.integrator
    dev = builder._mapper._device if hasattr(builder._mapper, "_device") else "cuda:0"
    spec = builder._coarse_free_spec(ball, owned, nontarget, target_points=target)
    if sign_floor is not None:
        spec = _dc.replace(spec, sign_floor=str(sign_floor))
    _sync()
    t0 = time.perf_counter()
    (tier,) = builder._build_tiers(itg, None, dev, torch, only="coarse",
                                   tier_name="coarse_no_target", ball=ball, itg_fine=itg_fine,
                                   coarse_free=spec)
    _sync()
    return tier, (time.perf_counter() - t0) * 1e3


def roi_seeds(builder, ball, shell: float):
    """ROI (공 + 껍질) 안의 seed voxel ijk — fine 계층의 seed 를 **다시** 심어 읽는다 (읽기만)."""
    import torch

    itg = (builder._fine_mapper or builder._mapper).integrator
    origin = itg._last_esdf_origin.clone()
    itg._seed_esdf_impl(origin, itg._esdf_voxel_size)
    vs = float(itg._esdf_voxel_size[0])
    shape = tuple(int(v) for v in itg._site_index.shape)
    org = builder._origin_of(itg, shape, vs)
    big = dataclasses.replace(ball, radius=float(ball.radius) + shell, z_min=-np.inf)
    cand = builder._ball_candidates(big, org, vs, shape)
    if cand is None:
        return np.zeros((0, 3), np.int64), org, vs
    seed = builder._seed_mask(itg, cand)
    _ = torch  # noqa: F841
    return cand[seed], org, vs


def make_oracle(builder, ball, gt_pts, gt_lab, shell: float, *, hulls=()):
    """`_ball_exclusion` 자리에 들어갈 oracle — GT target 라벨 seed 만 (ROI 안) 지운다."""
    from benchmark.ag3s.fields.curobo_builder import _nearest

    t_idx = LABELS.index("target")

    def oracle(itg, _ball, origin, vs, shape, *, owned=None):
        t0 = time.monotonic()
        big = dataclasses.replace(ball, radius=float(ball.radius) + shell, z_min=-np.inf)
        cand = builder._ball_candidates(big, origin, vs, shape)
        empty = np.zeros((0, 3), np.int64)
        counts = {"exclusion_method": "oracle", "n_seeds_in_ball": 0, "n_removed": 0,
                  "n_kept_nontarget": 0, "n_ambiguous": 0, "n_unowned_removed": 0,
                  "n_unowned_kept": 0}
        trace = {"removed_ijk": empty, "nontarget_kept_ijk": empty, "ambiguous_ijk": empty,
                 "unowned_kept_ijk": empty}
        if cand is None:
            return 0, None, {"counts": counts, "trace": trace}
        seed = builder._seed_mask(itg, cand)
        lab = voxel_labels(gt_pts, gt_lab, origin, vs, cand, hulls=hulls)
        removed = cand[seed & (lab == t_idx)]
        n = 0
        if len(removed):
            n, _ = builder._clear_seeds(itg, removed, shape)
        centres = origin + cand * vs
        tp = gt_pts[gt_lab == t_idx]
        op = gt_pts[(gt_lab != t_idx) & (gt_lab != LABELS.index("robot"))]
        near = np.linalg.norm(op - ball.centre, axis=1) <= float(ball.radius) * 3 + shell
        d_t, d_o = _nearest(tp, centres), _nearest(op[near], centres)
        sign = (seed & (lab == t_idx)) | (~seed & (d_t < d_o - vs))
        counts.update({"n_seeds_in_ball": int(seed.sum()), "n_removed": int(n),
                       "classify_ms": round((time.monotonic() - t0) * 1e3, 3)})
        trace["removed_ijk"] = removed
        sign_idx = cand[sign]
        return n, (sign_idx if len(sign_idx) else None), {"counts": counts, "trace": trace}

    return oracle


# ----------------------------------------------------------------------------- queries


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def compose_free_field(main_field, layer, coarse_layer=None):
    """main 계층 + target-free 미세 계층 (+ T44c 거친 target-free 계층) 의 `CuroboEsdfField`."""
    from benchmark.ag3s.fields.curobo_field import CuroboEsdfField

    f = CuroboEsdfField(tuple(main_field.layers), outside_distance=main_field.outside_distance,
                        target_free_layers=(layer,), stats={},
                        target_free_coarse_layer=coarse_layer)
    f.static_shapes = tuple(getattr(main_field, "static_shapes", ()) or ())
    return f


def query_block(main_field, free_layers: dict, centres, radii, margin: float, oracle="oracle",
                coarse_layers: Optional[dict] = None):
    """`coarse_layers` (T44c): `{method: 거친 target-free 계층}` — 있으면 그 방식의 질의에 합성한다.
    `free_layers` 의 키가 `"<m>+cf"` 이면 `coarse_layers["<m>"]` 를 쓴다 (같은 미세 계층, 식만 새것)."""
    d_main = np.asarray(main_field.distance(centres), np.float64)
    out = {"d_main_mm": (d_main * 1000).round(3).tolist(), "methods": {}}
    d_free = {}
    for name, layer in free_layers.items():
        base = name[:-3] if name.endswith("+cf") else None
        cl = None if base is None else (coarse_layers or {}).get(base)
        if base is not None and cl is None:
            continue
        f = compose_free_field(main_field, layer, cl)
        d_free[name] = np.asarray(f.target_free_distance(centres), np.float64)
    ref = d_free.get(oracle)
    for name, d in d_free.items():
        rec = {"d_free_mm": (d * 1000).round(3).tolist()}
        if ref is not None:
            over = d - ref
            cl, cl_ref = d - radii, ref - radii
            rec.update({
                "overestimation_mm_max": round(float(over.max()) * 1000, 3) if len(over) else None,
                "overestimation_mm_median": round(float(np.median(over)) * 1000, 3)
                if len(over) else None,
                "n_overestimated_gt_1mm": int((over > 0.001).sum()),
                "n_falsely_feasible": int(((cl >= margin) & (cl_ref < margin)).sum()),
                "n_falsely_infeasible": int(((cl < margin) & (cl_ref >= margin)).sum()),
            })
        out["methods"][name] = rec
    return out


# ----------------------------------------------------------------------------- one frame


def run_frame(ctx: dict, item: dict, args) -> dict:
    """One posed frame. `item` = `{"key", "seq", "t_step", "pose": fn(scene) -> err_m, "arm", "meta"}`."""
    from benchmark.ag3s.config import DEFAULT_CONTACT_LINKS
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        gaussian_attention, set_finger_joints_from_scene)
    from benchmark.ag3s.fields.curobo_builder import OwnedEvidence
    from benchmark.ag3s.fields.device_field import layer_from_device
    from benchmark.trajopt import wire

    scene, ag3s, target = ctx["scene"], ctx["ag3s"], ctx["target"]
    arm = item["arm"]
    pose_err = item["pose"](scene)
    gt_centre = body_in_base(scene, target)
    frames = {c: scene.capture(c) for c in args.cameras}
    attention = gt_attention(scene, frames, target, gt_centre)
    req = wire.pack_request(
        {}, cameras=args.cameras, depth={c: f.depth for c, f in frames.items()},
        intrinsics={c: f.camera_intrinsics for c, f in frames.items()},
        extrinsics={c: f.T_base_cam for c, f in frames.items()},
        robot_state={c: f.robot_state for c, f in frames.items()},
        stamps={c: float(item["seq"]) for c in args.cameras}, phase=args.phase,
        active_manipulators=[arm], seq=int(item["seq"]))
    _, sc = wire.strip_request(req)
    set_finger_joints_from_scene(scene, ag3s.robot_model)
    if ag3s.constraint_robot_model is not None:
        set_finger_joints_from_scene(scene, ag3s.constraint_robot_model)
    obs = wire.unpack_camera_observations(sc, robot_model=ag3s.robot_model, attention=attention)
    t0 = time.perf_counter()
    cs, debug = ag3s.process_multi_debug(obs, phase=args.phase, active_manipulators=[arm])
    ag3s_ms = (time.perf_counter() - t0) * 1e3
    support_mask = debug.get("support_mask")
    surfaces = list(cs.support_surfaces)
    gt_depth_pts, gt_depth_lab = gt_points(frames.values(), ctx["classes"])
    label_of = point_labeler(gt_depth_pts, gt_depth_lab, radius=0.003)
    geom, identity_mode, id_rec = choose_identity(ag3s, debug, gt_centre, args.identity,
                                                 support_mask=support_mask, label_of=label_of)
    rec: dict[str, Any] = {"attention_cameras": sorted(attention), "key": item["key"], "seq": int(item["seq"]),
                           "t_step": int(item["t_step"]), "source": item["meta"],
                           "target": target, "arm": arm,
                           "identity_mode": identity_mode, "identity": id_rec,
                           "gt_target_centre_m": gt_centre.round(5).tolist(),
                           "ag3s_ms": round(ag3s_ms, 2), "status": cs.status.value,
                           "object_pose_err_mm": round(pose_err * 1000.0, 3)}
    if geom is None:
        rec["skipped"] = "no target geometry"
        return rec
    builder = ag3s._esdf_builder
    fine_vs = float(ag3s.config.esdf.fine_voxel_size)
    ball = ag3s._target_ball(geom.points, surfaces, voxel=fine_vs)
    if ball is None:
        rec["skipped"] = "no ball (no support plane under the target)"
        return rec
    nontarget = ag3s._nontarget_points(debug.get("filtered_cloud"), support_mask, geom)
    rec["ball"] = ball.summary()
    rec["nontarget"] = {k: v for k, v in nontarget.items() if k != "points"}
    # nearest distractor (GT) for the frame's context
    others = [n for n in FRUITS if n != target]
    rec["nearest_distractor"] = min(
        ((n, round(float(np.linalg.norm(body_in_base(scene, n) - gt_centre)) * 1000, 1))
         for n in others), key=lambda x: x[1])
    gt_pts, gt_lab = gt_depth_pts, gt_depth_lab
    mesh_pts, mesh_lab, hulls = gt_mesh_points(scene, ctx["classes"])
    rec["gt_label_points"] = {"depth": int(len(gt_pts)), "mesh": int(len(mesh_pts))}
    gt_pts, gt_lab = np.vstack([gt_pts, mesh_pts]), np.concatenate([gt_lab, mesh_lab])

    cfgE = ag3s.config.esdf
    owned = {m: OwnedEvidence(target_points=geom.points, nontarget_points=nontarget["points"],
                              tie_voxels=cfgE.owned_tie_voxels,
                              max_dist_voxels=cfgE.owned_max_dist_voxels,
                              unowned=("keep" if m.endswith("keep") else "remove"))
             for m in ("ball_owned_remove", "ball_owned_keep")}
    oracle = make_oracle(builder, ball, gt_pts, gt_lab, args.shell, hulls=hulls)
    build_free_tier(builder, ball)                       # warm-up (JIT on the first frame)
    tiers, ms, ms_all = {}, {}, {}
    for m in args.methods:
        runs = []
        for _ in range(max(1, args.repeats if m != "oracle" else 1)):
            tier, t = build_free_tier(builder, ball, owned=owned.get(m),
                                      exclusion=oracle if m == "oracle" else None)
            runs.append(t)
        tiers[m], ms[m], ms_all[m] = tier, float(np.median(runs)), [round(v, 3) for v in runs]
    seeds, org, vs = roi_seeds(builder, ball, args.shell)
    labels = voxel_labels(gt_pts, gt_lab, org, vs, seeds, hulls=hulls)
    centres = org + seeds * vs
    in_ball = (np.linalg.norm(centres - ball.centre, axis=1) <= ball.radius) \
        & (centres[:, 2] >= ball.z_min)
    key = lambda a: (np.asarray(a, np.int64)[:, 0] << 42) | (np.asarray(a, np.int64)[:, 1] << 21) \
        | np.asarray(a, np.int64)[:, 2]   # noqa: E731
    seed_keys = key(seeds) if len(seeds) else np.zeros(0, np.int64)
    processed = {}
    seeds_rec, metrics = {}, {}
    for m, tier in tiers.items():
        tr = tier.extra.get("exclusion_trace") or {}
        rem = tr.get("removed_ijk", np.zeros((0, 3), np.int64))
        removed = np.isin(seed_keys, key(rem)) if len(rem) else np.zeros(len(seeds), bool)
        processed[m] = removed.astype(np.int8)
        table, table_ball = {}, {}
        for li, name in enumerate(LABELS):
            sel = labels == li
            if sel.any():
                table[name] = {"removed": int((removed & sel).sum()),
                               "kept": int((~removed & sel).sum())}
                sb = sel & in_ball
                if sb.any():
                    table_ball[name] = {"removed": int((removed & sb).sum()),
                                        "kept": int((~removed & sb).sum())}
        seeds_rec[m] = {"roi": table, "in_ball": table_ball}
        tgt = labels == LABELS.index("target")
        dis = labels == LABELS.index("distractor")
        sup = labels == LABELS.index("support")
        ex = tier.extra.get("exclusion") or {}
        metrics[m] = {
            "n_target_seeds_roi": int(tgt.sum()),
            "target_recall": (round(float((removed & tgt).sum()) / float(tgt.sum()), 4)
                              if tgt.any() else None),
            "distractor_seeds_lost": int((removed & dis).sum()),
            "distractor_seeds_in_ball": int((dis & in_ball).sum()),
            "support_seeds_lost": int((removed & sup).sum()),
            "n_removed_total": int(ex.get("n_removed", 0)),
            "n_removed_outside_roi_seed_list": int(len(rem) - int(removed.sum())),
            "n_ambiguous": int(ex.get("n_ambiguous", 0)),
            "n_kept_nontarget": int(ex.get("n_kept_nontarget", 0)),
            "n_unowned_removed": int(ex.get("n_unowned_removed", 0)),
            "n_unowned_kept": int(ex.get("n_unowned_kept", 0)),
            "n_sign_forced": int(tier.extra.get("n_sign_forced", 0)),
            "build_ms": round(ms[m], 3),          # median over --repeats (oracle: 1, not comparable)
            "build_ms_runs": ms_all[m],
            "classify_ms": ex.get("classify_ms"),
        }
    rec["seeds"] = seeds_rec
    rec["metrics"] = metrics
    rec["n_roi_seeds"] = int(len(seeds))

    # ---- T44c A5: target-free coarse tier per method (not oracle — no GT coarse labels)
    coarse_layers = {}
    if getattr(args, "target_free_coarse", False):
        build_coarse_free_tier(builder, ball, nontarget=nontarget["points"])     # warm-up
        crec = {}
        for m in args.methods:
            if m == "oracle":
                continue
            runs, ctier = [], None
            for _ in range(max(1, args.repeats)):
                ctier, t = build_coarse_free_tier(builder, ball, owned=owned.get(m),
                                                  nontarget=nontarget["points"],
                                                  target=geom.points,
                                                  sign_floor=getattr(args, "coarse_sign_floor",
                                                                     None))
                runs.append(t)
            coarse_layers[m] = layer_from_device(ctier.values, ctier.origin, ctier.voxel_size)
            ex = dict(ctier.extra.get("exclusion") or {})
            crec[m] = {**ex, "build_ms": round(float(np.median(runs)), 3),
                       "build_ms_runs": [round(v, 3) for v in runs],
                       "n_sign_forced": int(ctier.extra.get("n_sign_forced", 0)),
                       "sign_threshold_m": ctier.extra.get("threshold_m")}
        rec["coarse_free"] = crec

    # ---- queries: robot spheres at q, and a probe shell around the ball
    free_layers = {m: layer_from_device(t.values, t.origin, t.voxel_size) for m, t in tiers.items()}
    for m in list(coarse_layers):
        free_layers[f"{m}+cf"] = free_layers[m]      # 같은 미세 계층, T44c 식
    main = cs.esdf
    qm = ctx["query_model"]
    set_finger_joints_from_scene(scene, qm)
    q = np.asarray(frames[args.cameras[0]].robot_state, np.float64)
    c, r = qm.sphere_centers_numeric(q)
    names = [str(n) for n in qm.sphere_link_names]
    auth = set(DEFAULT_CONTACT_LINKS[arm])
    is_auth = np.asarray([n in auth for n in names], bool)
    near = np.linalg.norm(c - ball.centre, axis=1) - r <= args.query_radius
    qrec = {"links": names, "authorized": is_auth.tolist(), "radius_mm": (r * 1000).round(2).tolist(),
            "near_target": near.tolist(), "margin_m": args.margin}
    qrec["all"] = query_block(main, free_layers, c, r, args.margin, coarse_layers=coarse_layers)
    for tag, sel in (("authorized", is_auth), ("non_authorized", ~is_auth)):
        if sel.any():
            blk = query_block(main, free_layers, c[sel], r[sel], args.margin,
                              coarse_layers=coarse_layers)
            qrec[tag] = {m: {k: v for k, v in b.items() if k != "d_free_mm"}
                         for m, b in blk["methods"].items()}
    shells = (0.0, 0.005, 0.010, 0.020)
    dirs = fibonacci_sphere(args.probe_dirs)
    probe = np.concatenate([ball.centre + (ball.radius + s) * dirs for s in shells])
    probe = probe[probe[:, 2] >= ball.z_min]
    pb = query_block(main, free_layers, probe, np.zeros(len(probe)), args.margin,
                     coarse_layers=coarse_layers)
    qrec["roi_probe"] = {"n": int(len(probe)), "shells_mm": [s * 1000 for s in shells],
                         **{m: {k: v for k, v in b.items() if k != "d_free_mm"}
                            for m, b in pb["methods"].items()}}
    rec["queries"] = qrec
    rec["_npz"] = {"seed_ijk": seeds.astype(np.int16), "seed_label": labels,
                   "in_ball": in_ball, "origin": org, "voxel_size": np.float64(vs),
                   "ball_centre": ball.centre, "ball_radius": np.float64(ball.radius),
                   "ball_z_min": np.float64(ball.z_min),
                   "target_points": geom.points.astype(np.float32),
                   "nontarget_points_near": nontarget["points"][
                       np.linalg.norm(nontarget["points"] - ball.centre, axis=1)
                       <= ball.radius + 0.05].astype(np.float32),
                   **{f"processed_{m}": v for m, v in processed.items()}}
    return rec


# ----------------------------------------------------------------------------- main


def list_candidates(run: pathlib.Path, manifest: dict, arm: str) -> None:
    scene = open_scene(manifest)
    target = manifest["scene"]["episode_target_fruit"]
    ee = "EE_BODY_L" if arm == "left" else "EE_BODY_R"
    print("seq  t_step  nearest_distractor_mm  hand_target_mm")
    try:
        for row in load_planning_rows(run):
            pose(scene, row)
            c = body_in_base(scene, target)
            nd = min(float(np.linalg.norm(body_in_base(scene, n) - c)) for n in FRUITS
                     if n != target)
            ht = float(np.linalg.norm(body_in_base(scene, ee) - c))
            print(f"{row['seq']:>4} {row['t_step']:>6} {nd * 1000:>10.1f} {ht * 1000:>14.1f}")
    finally:
        scene.close()


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_argument_group("input (one of)")
    src.add_argument("--run", type=pathlib.Path, default=None,
                     help="client 기록 디렉터리 (frames.jsonl · manifest.json) — E2-lb")
    src.add_argument("--frames", type=int, nargs="*", default=(),
                     help="--run: planning 행의 seq 목록 (사전 고정)")
    src.add_argument("--demo-episodes", default=None,
                     help="E2-demo: 16D demonstration episode index 목록 (`0,1,5`). 과일 = initial pose, "
                          "팔 = parquet observation.state, 나머지 관절 = scene 기본 (teleop keyframe)")
    src.add_argument("--demo-frames-per-episode", type=int, default=4,
                     help="pregrasp_start_frame ~ grasp_close_start_frame 균등 간격 frame 수")
    src.add_argument("--demo-dataset", type=pathlib.Path, default=DEMO_DATASET)
    src.add_argument("--posed-frames", type=pathlib.Path, default=None,
                     help="posed-frame JSON (list; 키 = `posed_frame_from_demo` 의 출력: state16 · "
                          "fruit_poses · basket_pose · target_fruit · used_arm · episode_index · frame)")
    src.add_argument("--write-posed-frames", type=pathlib.Path, default=None,
                     help="--demo-episodes 로 만든 posed-frame 목록을 이 JSON 에 쓰고 끝낸다 (GPU 불필요)")
    src.add_argument("--model-xml", default=None,
                     help="demo / posed-frame 입력의 씬 XML (기본: 16D transport model)")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--list-candidates", action="store_true",
                    help="--run: planning 행마다 이웃 과일 거리 · 손–target 거리 표만 찍는다")
    ap.add_argument("--identity", choices=("gt", "gt_label", "pipeline"), default="gt",
                    help="gt (기본) = pipeline → 가장 GT-target 인 cluster → GT label 점 순 (correct identity "
                         "강제) · gt_label = 늘 GT label 이 target 인 cloud 점 전부 (완전한 identity) · "
                         "pipeline = pipeline 이 고른 것 그대로")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--arm", default=None, help="--run: 기본 manifest 의 episode_used_arm")
    ap.add_argument("--phase", default="approach")
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="기본: manifest cameras (--run) 또는 wire.DEFAULT_CAMERAS")
    ap.add_argument("--history", type=int, default=0,
                    help="--run: frame 앞의 planning 행을 이만큼 먼저 AG3S 에 넣는다 (TSDF 누적 · identity)")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--fine-voxel", type=float, default=0.005)
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    ap.add_argument("--owned-tie-voxels", type=float, default=1.0)
    ap.add_argument("--owned-max-dist-voxels", type=float, default=2.0)
    ap.add_argument("--target-free-coarse", action="store_true",
                    help="T44c — method 마다 target 을 지운 거친 계층도 만들고 질의에 `<m>+cf` (새 식) 를 "
                         "더한다 (oracle 제외). 같은 frame · 같은 미세 계층에서 옛 식과 나란히")
    ap.add_argument("--coarse-sign-floor", choices=("guard", "plane"), default=None,
                    help="T44c A6 — `--target-free-coarse` 의 부호 교정 아래 한계 (기본: config "
                         "`esdf.target_free_coarse_sign_floor` = plane, T44c A7)")
    ap.add_argument("--shell", type=float, default=0.020, help="ROI = 공 + 이 껍질 (m)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="falsely-feasible 판정의 clearance 기준 (m): d − r ≥ margin")
    ap.add_argument("--query-radius", type=float, default=0.10,
                    help="near_target 표시: 구 표면이 공 중심에서 이 거리 안 (m)")
    ap.add_argument("--probe-dirs", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3,
                    help="method 마다 target-free 계층을 이만큼 지어 build ms 의 중앙값 (oracle 은 1 — "
                         "GT label 계산이 들어 있어 비교 대상이 아니다)")
    ap.add_argument("--query-links", default="arms", help="query 구 모델 (`serve_safe --links` 값)")
    return ap


def build_items(args) -> tuple[list[dict], dict, str]:
    """`(items, summary_header, model_xml)` from `--run`/`--frames`, `--demo-episodes` or
    `--posed-frames`. Each item poses the scene itself (`item["pose"](scene)`)."""
    from benchmark.trajopt import wire

    if args.run is not None:
        manifest = json.loads((args.run / "manifest.json").read_text())
        arm = args.arm or manifest["scene"].get("episode_used_arm") or "left"
        args.cameras = tuple(args.cameras or manifest.get("cameras") or wire.DEFAULT_CAMERAS)
        rows = load_planning_rows(args.run)
        by_seq = {int(r["seq"]): i for i, r in enumerate(rows)}
        items = []
        for seq in args.frames:
            if seq not in by_seq:
                raise SystemExit(f"seq {seq} 가 planning 행에 없다")
            i = by_seq[seq]
            row = rows[i]
            items.append({
                "key": f"{args.run.name}_s{seq:03d}", "seq": seq, "t_step": int(row["t_step"]),
                "pose": (lambda sc, r=row: pose(sc, r)), "arm": arm,
                "target": manifest["scene"]["episode_target_fruit"],
                "history": [(lambda sc, r=h: pose(sc, r)) for h in
                            rows[max(0, i - args.history):i]],
                "meta": {"source": "run", "run": str(args.run), "seq": seq,
                         "episode_index": manifest["scene"].get("episode_index"),
                         "episode_split": manifest["scene"].get("episode_split")}})
        head = {"input": "run", "run": str(args.run),
                "episode_index": manifest["scene"].get("episode_index"),
                "episode_split": manifest["scene"].get("episode_split")}
        return items, head, manifest["scene"]["model_xml"]
    args.cameras = tuple(args.cameras or wire.DEFAULT_CAMERAS)
    if args.posed_frames is not None:
        pfs = json.loads(args.posed_frames.read_text())
        head = {"input": "posed_frames", "posed_frames": str(args.posed_frames)}
    else:
        pfs = demo_posed_frames(args)
        head = {"input": "demo", "dataset": str(args.demo_dataset),
                "demo_episodes": sorted({p["episode_index"] for p in pfs}),
                "frames_per_episode": args.demo_frames_per_episode,
                "frame_rule": "evenly spaced over [pregrasp_start_frame, grasp_close_start_frame]"}
    items = []
    for pf in pfs:
        ep, fr = int(pf["episode_index"]), int(pf["frame"])
        items.append({
            "key": f"demo_ep{ep:04d}_f{fr:03d}", "seq": fr, "t_step": fr,
            "pose": (lambda sc, p=pf: pose_posed(sc, p)), "arm": pf["used_arm"],
            "target": pf["target_fruit"], "history": [],
            "meta": {"source": pf.get("source", "posed"), "episode_index": ep, "frame": fr,
                     "split": pf.get("split"), "success": pf.get("success"),
                     "nearest_other_fruit_xy_m": round(nearest_other_xy(pf), 4)}})
    xml = args.model_xml or "pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml"
    return items, head, xml


def demo_posed_frames(args) -> list[dict]:
    if not args.demo_episodes:
        raise SystemExit("--run · --demo-episodes · --posed-frames 중 하나가 필요하다")
    eps = [int(v) for v in str(args.demo_episodes).replace(" ", "").split(",") if v]
    meta = load_demo_meta(args.demo_dataset)
    out = []
    for ep in eps:
        m = meta[ep]
        state = demo_state16(args.demo_dataset, ep)
        for fr in demo_frames(m, args.demo_frames_per_episode):
            out.append(posed_frame_from_demo(m, state, fr))
    return out


def main(argv=None) -> None:
    ap = _parser()
    args = ap.parse_args(argv)
    if args.list_candidates:
        if args.run is None:
            ap.error("--list-candidates 는 --run 과 함께")
        manifest = json.loads((args.run / "manifest.json").read_text())
        list_candidates(args.run, manifest,
                        args.arm or manifest["scene"].get("episode_used_arm") or "left")
        return
    if args.write_posed_frames is not None:
        pfs = demo_posed_frames(args)
        args.write_posed_frames.parent.mkdir(parents=True, exist_ok=True)
        args.write_posed_frames.write_text(json.dumps(pfs, indent=1))
        print(f"wrote {len(pfs)} posed frames -> {args.write_posed_frames}")
        return
    if args.out is None:
        ap.error("--out 이 필요합니다")
    if args.run is not None and not args.frames:
        ap.error("--run 에는 --frames 가 필요합니다")

    import torch

    from benchmark.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
    from benchmark.ag3s.runtime.asset_path import resolve_asset
    from benchmark.trajopt.serve_safe import build_ag3s, constraint_link_filter

    if not torch.cuda.is_available():
        raise SystemExit("cuRobo 경로가 필요합니다 (GPU) — CPU mock 만으로 끝내지 않는다")
    items, head, model_xml = build_items(args)
    model_xml = str(resolve_asset(model_xml, what="transport model"))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"stage": "T44_E2_exclusion", **head, "model_xml": model_xml,
               "identity": args.identity, "methods": list(args.methods),
               "cameras": list(args.cameras), "phase": args.phase, "history": args.history,
               "shell_m": args.shell, "margin_m": args.margin,
               "owned_tie_voxels": args.owned_tie_voxels,
               "owned_max_dist_voxels": args.owned_max_dist_voxels, "frames": []}
    scene = TransportScene(model_xml, settle_steps=0)
    try:
        for item in items:
            ag3s = build_ag3s(model_xml, voxel=args.voxel, range_max=2.0, links="gripper",
                              backend="curobo", fine_voxel=args.fine_voxel,
                              tsdf_voxel=args.tsdf_voxel, target_field_policy="exclude-authorized",
                              target_exclusion={"owned_tie_voxels": args.owned_tie_voxels,
                                                "owned_max_dist_voxels":
                                                    args.owned_max_dist_voxels})
            ctx = {"scene": scene, "ag3s": ag3s, "target": item["target"],
                   "classes": label_classes(scene, item["target"]),
                   "query_model": build_constraint_robot_model(
                       scene, link_filter=constraint_link_filter(args.query_links))}
            for h in item["history"]:
                _history_step(ctx, h, item, args)
            rec = run_frame(ctx, item, args)
            npz = rec.pop("_npz", None)
            name = f"e2_{item['key']}"
            if npz is not None:
                np.savez_compressed(args.out / f"{name}.npz", **npz)
            (args.out / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1,
                                                              default=_json_default))
            summary["frames"].append({"key": item["key"], "file": f"{name}.json",
                                      "source": item["meta"],
                                      "identity_mode": rec.get("identity_mode"),
                                      "skipped": rec.get("skipped"),
                                      "metrics": rec.get("metrics")})
            _print_frame(rec)
            del ag3s, ctx
            torch.cuda.empty_cache()
    finally:
        scene.close()
    tag = args.run.name if args.run is not None else (
        "demo" if args.posed_frames is None else args.posed_frames.stem)
    (args.out / f"e2_{tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=_json_default))
    print(f"wrote {args.out}")


def _history_step(ctx, pose_fn, item, args) -> None:
    """Feed an earlier pose through the same AG3S (TSDF accumulation · identity)."""
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        gaussian_attention, set_finger_joints_from_scene)
    from benchmark.trajopt import wire

    scene, ag3s = ctx["scene"], ctx["ag3s"]
    pose_fn(scene)
    c = body_in_base(scene, ctx["target"])
    frames = {cam: scene.capture(cam) for cam in args.cameras}
    req = wire.pack_request(
        {}, cameras=args.cameras, depth={k: f.depth for k, f in frames.items()},
        intrinsics={k: f.camera_intrinsics for k, f in frames.items()},
        extrinsics={k: f.T_base_cam for k, f in frames.items()},
        robot_state={k: f.robot_state for k, f in frames.items()},
        stamps={k: 0.0 for k in args.cameras}, phase=args.phase,
        active_manipulators=[item["arm"]], seq=0)
    _, sc = wire.strip_request(req)
    set_finger_joints_from_scene(scene, ag3s.robot_model)
    obs = wire.unpack_camera_observations(
        sc, robot_model=ag3s.robot_model,
        attention=gt_attention(scene, frames, ctx["target"], c))
    ag3s.process_multi_debug(obs, phase=args.phase, active_manipulators=[item["arm"]])


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _print_frame(rec: dict) -> None:
    print(f"{rec.get('key', '')} seq {rec['seq']:>3} t={rec['t_step']:>4} identity={rec['identity_mode']}"
          + (f" SKIPPED: {rec['skipped']}" if rec.get("skipped") else ""))
    for m, x in (rec.get("metrics") or {}).items():
        print(f"   {m:<18} recall {x['target_recall']}  distractor lost {x['distractor_seeds_lost']:>4}"
              f"  support lost {x['support_seeds_lost']:>3}  amb {x['n_ambiguous']:>3}"
              f"  unowned rm/keep {x['n_unowned_removed']}/{x['n_unowned_kept']}"
              f"  build {x['build_ms']:.2f} ms")


if __name__ == "__main__":
    main()
