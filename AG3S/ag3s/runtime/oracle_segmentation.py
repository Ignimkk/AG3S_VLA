"""T46e A13 — the **visible-target oracle** (DX-01 조건 B): simulator GT segmentation → labelled surface samples.

**무엇을 위한 것인가.** DT-02 의 extent block 은 같은 identity 에서 target 범위 처리만 바꾼 arm 을 비교한다
(`ball` · `ball_owned` · `ball_owned` + F1 · **oracle**). oracle 은 "카메라에 실제로 보인 target 표면" 만 target 으로
지우는 기준이다 — 제안 extent 가 이상적인 경계에 얼마나 가까운가를 closed loop 에서 잰다. 제안이 아니고 진단이다.

## 세 곳이 같은 표를 쓴다

| 곳 | 하는 일 |
|---|---|
| client (`trajopt.client.SafeRemoteClient`, `oracle_target=`) | MuJoCo segmentation (`CameraFrame.body_ids` — `capture` 가 이미 렌더한다, 추가 렌더 0) 을 `label_image` 로 code 영상 (uint8) 으로 바꿔 요청에 싣는다 (`wire.ORACLE_SEG`) |
| server (`runtime.pipeline.AG3S`, `esdf.target_extent = "oracle_visible"`) | 그 code 영상과 같은 요청의 depth 를 `backproject_labels` 로 올려 `OracleMemory` 에 모은다. 지울 물체의 표본과 나머지 표본을 builder 에 넘긴다 |
| offline (`trajopt.experiments.target_free_replay.rebuild_oracle_free`) | 기록 planning 행의 `qpos` 로 같은 카메라를 다시 렌더해 같은 memory 를 짓는다 |

## code 표 (`label_image`)

| code | 무엇 |
|---:|---|
| 0 (`CODE_NONE`) | 아무것도 안 맞음 · **로봇** — 근거로 쓰지 않는다 (로봇은 self-filter · depth mask 가 이미 뺀다) |
| 1 (`CODE_TARGET`) | 과제의 GT target 물체 (episode 의 `target_fruit`) |
| 2 (`CODE_STATIC`) | 로봇도 잡을 물체도 아닌 것 — table · crate · 바닥 · 벽 |
| 3, 4, … (`CODE_OBJECT_BASE` + i) | target 이 아닌 잡을 물체 (이웃 과일), 물체마다 하나 |

## memory (`OracleMemory`) — 왜 이 frame 의 pixel 만으로는 안 되나

TSDF 는 episode 동안 누적된다 (`esdf.time_decay = 1.0`). grasp 직전에는 손이 target 을 가려 이 frame 에 보이는 target pixel 이
줄지만, 앞 frame 들이 본 target 면의 seed 는 TSDF 에 남는다. 이 frame 의 pixel 만 쓰면 oracle 이 바로 그 chunk 에서 덜 지운다.
그래서 memory 는 TSDF 와 같은 규약으로 모은다:

| 단계 | 규칙 |
|---|---|
| 표본 | 유효 depth (`depth_min < d ≤ depth_max`) 이고 code ≠ 0 인 pixel 을 base 로 올린 점. `cell` (기본 2.5 mm) 격자에 하나씩 — **새 관측이 옛 것을 덮는다** (같은 칸의 code 도) |
| 범위 (ROI) | 잡을 물체 (code 1 · ≥ 3) 표본의 AABB + `roi_margin` 밖의 표본은 담지 않는다 — 그 밖의 점은 target 근처 seed 의 분류를 바꾸지 못한다 (`roi_margin ≥ 2 × 후보 반경`) |
| truncation 띠 | 잡을 물체 pixel (`band_stride` 마다) 의 광선을 따라 보인 면 뒤 `band_depth` (= 미세 TSDF truncation) 까지 지나간 칸 (`band_cell`, 기본 = 미세 voxel) 과 그 칸을 지나간 물체 code 의 bit mask. TSDF 는 그 광선으로 음수 띠를 만들고, 띠의 경계가 seed 가 된다 — 물체 밑 · 안쪽의 그림자 seed 는 표면 표본에서 멀어 표면 표본만으로는 주인을 모른다. 지우지 않는다 (TSDF 가 아직 그 seed 를 들고 있으면 주인은 그대로다) |
| 지움 (carving) | 이 frame 의 카메라가 옛 표본 **너머** 를 봤으면 (투영 pixel 주변 `carve_window`² 의 가장 가까운 유효 depth > 표본 깊이 + `carve_tol`) 그 표본은 지운다 — TSDF 의 자유 공간 갱신과 같은 뜻 (물체가 움직였다). 주변 최소를 쓰는 것은 실루엣 가장자리 pixel 이 물체를 비껴 뒤 table 을 맞히는 경우 때문이다 |
| reset | episode 마다 (`AG3S.reset`) |
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Sequence

import numpy as np

CODE_NONE = 0
CODE_TARGET = 1
CODE_STATIC = 2
CODE_OBJECT_BASE = 3
#: The graspable objects of the RB-Y1 transport scene (body names). A body whose name — or whose root body's
#: name — is one of these and is not the target gets its own code `CODE_OBJECT_BASE + i`.
GRASPABLE_OBJECTS = ("apple", "banana", "orange", "pear")
#: Wire format version of the code image (`wire.ORACLE_SEG_META["version"]`).
CODE_VERSION = 1


def code_table(body_names: Mapping[int, str], target: str, *, root_of: Optional[Mapping[int, int]] = None,
               objects: Sequence[str] = GRASPABLE_OBJECTS, is_robot=None) -> tuple[np.ndarray, dict[str, int]]:
    """`(lut, codes)` — `lut[body_id + 1]` = the code of a pixel whose MuJoCo body is `body_id` (`lut[0]` = no hit).

    `root_of` maps a body id to its root body id (MuJoCo `model.body_rootid`) so that a child body of a fruit
    takes the fruit's code. `is_robot(name)` (default `mujoco_source.is_robot_body`) marks robot bodies → 0.
    `codes` = `{object name: code}` for the target and the other graspable objects (the record's legend).
    Raises if `target` is not one of `objects` — a typo would silently label nothing as the target.
    """
    if is_robot is None:
        from AG3S.ag3s.experiments.sources.mujoco_source import is_robot_body as is_robot
    objects = tuple(str(o) for o in objects)
    target = str(target)
    if target not in objects:
        raise ValueError(f"oracle target {target!r} is not a graspable object {objects}")
    codes: dict[str, int] = {target: CODE_TARGET}
    for i, name in enumerate(o for o in objects if o != target):
        codes[name] = CODE_OBJECT_BASE + i
    n = (max(int(b) for b in body_names) + 1) if body_names else 0
    lut = np.full(n + 1, CODE_NONE, np.uint8)
    for bid, name in body_names.items():
        bid = int(bid)
        root = bid if root_of is None else int(root_of.get(bid, bid))
        names = (str(name), str(body_names.get(root, name)))
        if any(nm in codes for nm in names):
            lut[bid + 1] = codes[next(nm for nm in names if nm in codes)]
        elif any(is_robot(nm) for nm in names):
            lut[bid + 1] = CODE_NONE
        else:
            lut[bid + 1] = CODE_STATIC
    return lut, codes


def label_image(body_ids: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """(H, W) uint8 code image from a MuJoCo body-id image (`-1` = no hit → `CODE_NONE`)."""
    b = np.asarray(body_ids, np.int64)
    idx = np.clip(b + 1, 0, len(lut) - 1)
    out = lut[idx]
    out[(b < -1) | (b + 1 >= len(lut))] = CODE_NONE
    return out.astype(np.uint8, copy=False)


def backproject_labels(depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, labels: np.ndarray, *,
                       depth_min: float = 0.0, depth_max: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
    """`(points (N, 3) base frame, codes (N,) uint8)` of the labelled pixels with a valid depth.

    The same pinhole as `reconstruction.backproject` (OpenCV: +z forward, `u` = column). Pixels with code 0, a depth
    outside `(depth_min, depth_max]` or a non-finite depth are dropped.
    """
    d = np.asarray(depth_m, np.float64)
    lab = np.asarray(labels)
    if d.shape != lab.shape:
        raise ValueError(f"oracle labels {lab.shape} do not match the depth image {d.shape}")
    ok = (lab != CODE_NONE) & np.isfinite(d) & (d > float(depth_min)) & (d <= float(depth_max))
    v, u = np.nonzero(ok)
    if not len(v):
        return np.zeros((0, 3)), np.zeros(0, np.uint8)
    z = d[v, u]
    K = np.asarray(K, np.float64)
    T = np.asarray(T_base_cam, np.float64)
    cam = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    return cam @ T[:3, :3].T + T[:3, 3], lab[v, u].astype(np.uint8)


@dataclasses.dataclass(frozen=True)
class OracleView:
    """One camera of one request: metric depth, `K`, `T_base_cam` and the code image (same shape as depth)."""

    depth_m: np.ndarray
    K: np.ndarray
    T_base_cam: np.ndarray
    labels: np.ndarray
    name: str = ""


def _keys(cells: np.ndarray) -> np.ndarray:
    c = np.asarray(cells, np.int64) + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def _is_object(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes)
    return (c == CODE_TARGET) | (c >= CODE_OBJECT_BASE)


class OracleMemory:
    """The episode's labelled surface samples (see the module docstring). Pure numpy; one per `AG3S`."""

    def __init__(self, *, cell: float = 0.0025, roi_margin: float = 0.065, carve_tol: float = 0.010,
                 carve_window: int = 5, depth_min: float = 0.0, depth_max: float = np.inf,
                 band_depth: float = 0.030, band_stride: int = 2, band_cell: Optional[float] = None):
        if not (cell > 0 and roi_margin >= 0 and carve_tol > 0 and int(carve_window) >= 1
                and band_depth >= 0 and int(band_stride) >= 1):
            raise ValueError("cell > 0, roi_margin >= 0, carve_tol > 0, carve_window >= 1, band_depth >= 0, "
                             "band_stride >= 1")
        #: the truncation band behind each object pixel (m) — the fine TSDF's truncation — and the pixel stride
        self.band_depth = float(band_depth)
        self.band_stride = int(band_stride)
        #: the band cell (m): coarser than the sample cell (default 2×, = one fine voxel) so that rays a few mm apart
        #: (head camera ≈ 1.5 mm/px × stride 2 at 0.8 m) fill it without gaps; steps along the ray are half of it
        self.band_cell = float(2.0 * float(cell) if band_cell is None else band_cell)
        if not self.band_cell > 0.0:
            raise ValueError("band_cell > 0")
        self.cell = float(cell)
        self.roi_margin = float(roi_margin)
        self.carve_tol = float(carve_tol)
        self.carve_window = int(carve_window)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.reset()

    def reset(self) -> None:
        self.keys = np.zeros(0, np.int64)
        self.points = np.zeros((0, 3))
        self.codes = np.zeros(0, np.uint8)
        self.frame = np.zeros(0, np.int64)
        self.n_updates = 0
        self.legend: dict[str, int] = {}
        self.last: dict[str, Any] = {}
        #: the episode's truncation band: sorted cell keys and, per cell, the bit mask of the object codes whose pixel
        #: rays passed it behind their visible surface (`band_lookup`). Never carved — it attributes seeds the TSDF
        #: still holds, wherever they came from.
        self.band_keys = np.zeros(0, np.int64)
        self.band_masks = np.zeros(0, np.uint32)

    def __len__(self) -> int:
        return int(len(self.keys))

    # ------------------------------------------------------------------------------------------------
    def _carve(self, views: Sequence[OracleView]) -> np.ndarray:
        """bool (len(self),) — samples current rays passed beyond.

        A sample is seen through when the **nearest** valid depth in the `carve_window` × `carve_window` pixels around
        its projection is farther than the sample + `carve_tol`. The window keeps silhouette samples: a point on an
        object's surface projects inside that object's silhouette in every camera, but a pixel-centre ray at the
        silhouette edge can miss it and hit the table behind — the neighbourhood minimum still sees the object.
        """
        from scipy.ndimage import minimum_filter

        drop = np.zeros(len(self), bool)
        if not len(self):
            return drop
        for v in views:
            d = np.asarray(v.depth_m, np.float64)
            valid_px = np.isfinite(d) & (d > self.depth_min) & (d <= self.depth_max)
            d = minimum_filter(np.where(valid_px, d, np.inf), size=self.carve_window, mode="nearest")
            h, w = d.shape
            T = np.asarray(v.T_base_cam, np.float64)
            K = np.asarray(v.K, np.float64)
            R, t = T[:3, :3], T[:3, 3]
            pc = (self.points - t) @ R            # base → camera (R^T (p - t))
            z = pc[:, 2]
            front = z > 1e-6
            u = np.full(len(z), -1, np.int64)
            vv = np.full(len(z), -1, np.int64)
            u[front] = np.floor(K[0, 0] * pc[front, 0] / z[front] + K[0, 2] + 0.5).astype(np.int64)
            vv[front] = np.floor(K[1, 1] * pc[front, 1] / z[front] + K[1, 2] + 0.5).astype(np.int64)
            inside = front & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
            idx = np.flatnonzero(inside)
            if not len(idx):
                continue
            dm = d[vv[idx], u[idx]]
            seen_through = np.isfinite(dm) & (dm > z[idx] + self.carve_tol)
            drop[idx[seen_through]] = True
        return drop

    def update(self, views: Sequence[OracleView], *, legend: Optional[Mapping[str, int]] = None) -> dict:
        """Add this request's labelled pixels (newest wins per cell), after carving the samples it saw through."""
        self.n_updates += 1
        if legend:
            self.legend = {str(k): int(v) for k, v in legend.items()}
        pts, cds = [], []
        for v in views:
            p, c = backproject_labels(v.depth_m, v.K, v.T_base_cam, v.labels,
                                      depth_min=self.depth_min, depth_max=self.depth_max)
            pts.append(p)
            cds.append(c)
        new_p = np.vstack(pts) if pts else np.zeros((0, 3))
        new_c = np.concatenate(cds) if cds else np.zeros(0, np.uint8)
        n_px = int(len(new_p))
        carved = self._carve(views)
        n_carved = int(carved.sum())
        if n_carved:
            keep = ~carved
            self.keys, self.points = self.keys[keep], self.points[keep]
            self.codes, self.frame = self.codes[keep], self.frame[keep]
        # ROI: the graspable objects (this frame's and the memory's) + margin.
        obj = np.vstack([new_p[_is_object(new_c)], self.points[_is_object(self.codes)]])
        if len(obj):
            lo = obj.min(axis=0) - self.roi_margin
            hi = obj.max(axis=0) + self.roi_margin
            inr = np.all((new_p >= lo) & (new_p <= hi), axis=1)
            new_p, new_c = new_p[inr], new_c[inr]
            mem_in = np.all((self.points >= lo) & (self.points <= hi), axis=1)
            if not mem_in.all():
                self.keys, self.points = self.keys[mem_in], self.points[mem_in]
                self.codes, self.frame = self.codes[mem_in], self.frame[mem_in]
        else:
            new_p, new_c = new_p[:0], new_c[:0]
        cells = np.floor(new_p / self.cell).astype(np.int64)
        k_new = _keys(cells) if len(cells) else np.zeros(0, np.int64)
        all_k = np.concatenate([k_new, self.keys])
        all_p = np.vstack([new_p, self.points])
        all_c = np.concatenate([new_c, self.codes])
        all_f = np.concatenate([np.full(len(k_new), self.n_updates, np.int64), self.frame])
        uk, first = np.unique(all_k, return_index=True)      # first occurrence = this frame's sample
        self.keys, self.points = uk, all_p[first]
        self.codes, self.frame = all_c[first].astype(np.uint8), all_f[first]
        n_band_new = self._update_band(views, lo if len(obj) else None, hi if len(obj) else None)
        self.last = {"n_labelled_px": n_px, "n_new_in_roi": int(len(new_p)), "n_carved": n_carved,
                     "n_samples": int(len(self)), "n_target_samples": int((self.codes == CODE_TARGET).sum()),
                     "n_target_samples_this_frame": int((self.frame == self.n_updates)[self.codes == CODE_TARGET].sum()),
                     "n_band_cells": int(len(self.band_keys)), "n_band_points_this_frame": n_band_new,
                     "update": int(self.n_updates)}
        return dict(self.last)

    def _update_band(self, views: Sequence[OracleView], lo, hi) -> int:
        """Fold this request's object pixels' truncation band (behind the visible surface, `band_depth` along the ray,
        every `band_stride`-th pixel, steps of one cell) into the band cells. Returns the band points this frame."""
        if lo is None or self.band_depth <= 0.0:
            return 0
        steps = np.arange(0.0, self.band_depth + 1e-9, 0.5 * self.band_cell)
        keys, masks = [], []
        for v in views:
            lab = np.asarray(v.labels)[::self.band_stride, ::self.band_stride]
            d = np.asarray(v.depth_m, np.float64)[::self.band_stride, ::self.band_stride]
            ok = (_is_object(lab) & np.isfinite(d) & (d > self.depth_min) & (d <= self.depth_max))
            r, q = np.nonzero(ok)
            if not len(r):
                continue
            u = q * self.band_stride
            vv = r * self.band_stride
            K = np.asarray(v.K, np.float64)
            T = np.asarray(v.T_base_cam, np.float64)
            ray = np.column_stack([(u - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones(len(u))])
            z = d[r, q][:, None] + steps[None, :]                     # camera z behind the surface
            pc = ray[:, None, :] * z[:, :, None]
            p = pc.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
            code = np.repeat(lab[r, q].astype(np.uint32), len(steps))
            inr = np.all((p >= lo) & (p <= hi), axis=1)
            p, code = p[inr], code[inr]
            if not len(p):
                continue
            keys.append(_keys(np.floor(p / self.band_cell).astype(np.int64)))
            masks.append(np.left_shift(np.uint32(1), code))
        if not keys:
            return 0
        k = np.concatenate([np.concatenate(keys), self.band_keys])
        m = np.concatenate([np.concatenate(masks), self.band_masks])
        order = np.argsort(k, kind="stable")
        k, m = k[order], m[order]
        start = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        self.band_keys = k[start]
        self.band_masks = np.bitwise_or.reduceat(m, start).astype(np.uint32)
        return int(sum(len(x) for x in keys))

    def band_lookup(self) -> tuple[np.ndarray, np.ndarray, float]:
        """`(sorted cell keys, per-cell code bit masks, cell)` — `curobo_builder.oracle_band_masks` reads them."""
        return self.band_keys.copy(), self.band_masks.copy(), self.band_cell

    # ------------------------------------------------------------------------------------------------
    def instance_of(self, points: np.ndarray, *, radius: float, min_fraction: float = 0.5) -> tuple[Optional[int], dict]:
        """Which graspable object the points (the manipulated geometry) lie on: `(code | None, info)`.

        Each point takes the code of its nearest sample within `radius`; the majority object code (1 or ≥ 3) among the
        points that found one wins if it is at least `min_fraction` of them. `None` when no point found an object sample
        or no object reaches the fraction.
        """
        p = np.asarray(points, np.float64).reshape(-1, 3)
        info: dict[str, Any] = {"n_points": int(len(p)), "radius_mm": round(float(radius) * 1e3, 3),
                                "min_fraction": float(min_fraction)}
        if not len(p) or not len(self):
            info["why"] = "no_points" if not len(p) else "empty_memory"
            return None, info
        from scipy.spatial import cKDTree

        # only the samples within the points' AABB + radius can be within radius of one of them
        box = np.all((self.points >= p.min(axis=0) - radius) & (self.points <= p.max(axis=0) + radius), axis=1)
        if not box.any():
            info["n_labelled"] = info["n_on_objects"] = 0
            info["why"] = "no_object_sample"
            return None, info
        sp, sc = self.points[box], self.codes[box]
        d, j = cKDTree(sp).query(p, k=1)
        near = d <= float(radius)
        c = sc[j[near]]
        obj = c[_is_object(c)]
        info["n_labelled"] = int(near.sum())
        info["n_on_objects"] = int(len(obj))
        if not len(obj):
            info["why"] = "no_object_sample"
            return None, info
        vals, cnt = np.unique(obj, return_counts=True)
        k = int(np.argmax(cnt))
        frac = float(cnt[k]) / float(near.sum())
        info["votes"] = {int(v): int(n) for v, n in zip(vals, cnt)}
        info["fraction"] = round(frac, 4)
        if frac < float(min_fraction):
            info["why"] = "no_majority"
            return None, info
        return int(vals[k]), info

    def evidence(self, code: int) -> tuple[np.ndarray, np.ndarray]:
        """`(samples of object `code`, every other sample)` — what `oracle_visible_split` compares."""
        on = self.codes == int(code)
        return self.points[on].copy(), self.points[~on].copy()


def oracle_evidence(memory: Optional["OracleMemory"], geometry_points, *, radius: float, identity: str = "pipeline",
                    max_dist_voxels: float = 6.0, z_floor: Optional[float] = None,
                    surface_protect_voxels: float = 1.0) -> tuple[Any, dict]:
    """`(curobo_builder.OracleVisibleEvidence | None, record)` — which object to remove, its samples and the episode's
    truncation band.

    `identity`: `pipeline` = the object the manipulated geometry points lie on (`OracleMemory.instance_of`, nearest
    sample within `radius`, majority ≥ ½) · `gt` = the GT target (code 1). None (→ the estimated extent this frame)
    when the memory is empty, there are no geometry points, the object is not resolved or it has no sample; the record
    says why. `z_floor` = the support guard (`TargetBall.z_min`). The server
    (`AG3S._oracle_evidence`) and the offline replay call this one function.
    """
    from AG3S.ag3s.fields.curobo_builder import OracleVisibleEvidence

    rec: dict[str, Any] = {"applied": False, "identity_mode": str(identity)}
    if memory is None or not len(memory):
        rec["why"] = "no_segmentation" if memory is None or not memory.n_updates else "empty_memory"
        return None, rec
    p = np.asarray(geometry_points, np.float64).reshape(-1, 3)
    if not len(p):
        rec["why"] = "no_manipulated_geometry"
        return None, rec
    code_pipe, inst = memory.instance_of(p, radius=float(radius))
    rec.update({"pipeline_instance": inst, "pipeline_code": code_pipe,
                "identity_matches_gt": None if code_pipe is None else bool(code_pipe == CODE_TARGET)})
    code = CODE_TARGET if identity == "gt" else code_pipe
    if code is None:
        rec["why"] = f"identity_unresolved ({inst.get('why')})"
        return None, rec
    tp, op = memory.evidence(code)
    rec.update({"code": int(code),
                "object": next((k for k, v in memory.legend.items() if int(v) == int(code)), None),
                "n_target_samples": int(len(tp)), "n_other_samples": int(len(op))})
    if not len(tp):
        rec["why"] = "no_visible_sample"
        return None, rec
    rec["applied"] = True
    bk, bm, cell = memory.band_lookup()
    rec["n_band_cells"] = int(len(bk))
    rec["z_floor_m"] = None if z_floor is None else float(z_floor)
    return OracleVisibleEvidence(target_points=tp, other_points=op, max_dist_voxels=float(max_dist_voxels),
                                 info={"code": int(code)}, code=int(code), band_cell=float(cell), band_keys=bk,
                                 band_masks=bm, z_floor=z_floor,
                                 surface_protect_voxels=float(surface_protect_voxels)), rec


__all__ = ["CODE_NONE", "CODE_TARGET", "CODE_STATIC", "CODE_OBJECT_BASE", "GRASPABLE_OBJECTS", "CODE_VERSION",
           "code_table", "label_image", "backproject_labels", "OracleView", "OracleMemory", "oracle_evidence"]
