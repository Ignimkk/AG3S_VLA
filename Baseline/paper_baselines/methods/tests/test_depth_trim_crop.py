"""BASELINE-BENCH-READY L: ``ObstacleCloudSource(crop_mode="depth_trim")`` -- the detector box keeps only the object
at the box centre; per-box clouds are associated in 3D before merging. ``crop_mode="box"`` is the V5 behaviour.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -q \
        Baseline/paper_baselines/methods/tests/test_depth_trim_crop.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.methods import safety_common as SC
from Baseline.paper_baselines.methods.interface import PRIVILEGED_PREFIX, SceneObservation

H = W = 128
K = np.array([[110.0, 0.0, 64.0], [0.0, 110.0, 64.0], [0.0, 0.0, 1.0]])


# --- a tiny ray caster: table plane z = 0 + axis-aligned boxes ----------------------------------------------------
def look_at(eye, target) -> np.ndarray:
    """OpenCV camera (x right, y down, z forward) at ``eye`` looking at ``target`` -> T_base_cam."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def render_depth(T: np.ndarray, boxes: list[tuple[np.ndarray, np.ndarray]], table: bool = True) -> np.ndarray:
    v, u = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)  # z = 1
    R, o = T[:3, :3], T[:3, 3]
    d = d_cam @ R.T  # base-frame ray per unit camera depth
    best = np.full((H, W), np.inf)
    if table:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -o[2] / d[..., 2]
        best = np.where((s > 0) & np.isfinite(s), s, best)
    for c, h in boxes:
        lo, hi = np.asarray(c) - h, np.asarray(c) + h
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        tn = np.nanmax(np.minimum(t1, t2), -1)
        tf = np.nanmin(np.maximum(t1, t2), -1)
        hit = (tf >= tn) & (tn > 0)
        best = np.where(hit & (tn < best), tn, best)
    return np.where(np.isfinite(best), best, 0.0)  # camera z (= depth) because d_cam has z = 1


def project_box(T: np.ndarray, c, h, pad: float = 0.0) -> list[float]:
    c, h = np.asarray(c, float), np.asarray(h, float)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    pc = (corners - T[:3, 3]) @ T[:3, :3]
    uv = (pc[:, :2] / pc[:, 2:]) * [K[0, 0], K[1, 1]] + K[:2, 2]
    (u0, v0), (u1, v1) = uv.min(0), uv.max(0)
    du, dv = pad * (u1 - u0), pad * (v1 - v0)
    return [u0 - du, v0 - dv, u1 + du + 1, v1 + dv + 1]


class BoxDetector:
    """Fixed boxes per camera (keyed by the rgb fill value) for every prompt except the target phrase."""

    def __init__(self, boxes_by_cam: dict[int, list], target: dict[int, list] | None = None):
        self.boxes, self.target = boxes_by_cam, target or {}

    def provenance(self):
        return {"detector": "fixed boxes (test)"}

    def detect(self, rgb, text, box_threshold, text_threshold):
        cam = int(rgb[0, 0, 0])
        bx = self.target.get(cam, []) if text == "target thing" else self.boxes.get(cam, [])
        b = np.asarray(bx, float).reshape(-1, 4)
        return b, np.linspace(0.9, 0.5, len(b)), ["object"] * len(b)


FRONT = (np.array([0.50, 0.00, 0.04]), np.array([0.04, 0.04, 0.04]))  # cube on the table
BACK = (np.array([0.30, 0.02, 0.09]), np.array([0.05, 0.05, 0.09]))  # taller box behind it (peeks over the front one)
HEAD = look_at([1.05, 0.0, 0.55], [0.35, 0.0, 0.02])
WRIST = look_at([0.50, 0.0, 0.35], [0.50, 0.001, 0.0])


def scene(cams=("head",), boxes=(FRONT, BACK)) -> SceneObservation:
    Ts = {"head": HEAD, "left_wrist": WRIST}
    rgb = {c: np.full((H, W, 3), i, np.uint8) for i, c in enumerate(cams)}
    depth = {c: render_depth(Ts[c], list(boxes)) for c in cams}
    return SceneObservation(rgb=rgb, depth=depth, intrinsics={c: K for c in cams}, T_base_cam={c: Ts[c] for c in cams},
                            q=np.zeros(7), ee_pose=np.eye(4), gripper=0.0, instruction="pick up the target thing",
                            target_phrase="target thing", extras={})


def inside(p, c, h, tol=0.006) -> np.ndarray:
    return np.all(np.abs(p - c) <= h + tol, axis=1)


def semi_axes(p) -> np.ndarray:
    from Baseline.paper_baselines.methods.knows.perception.ellipsoid_fit import mvee

    _, Q = mvee(p)
    return np.sqrt(np.clip(np.linalg.eigvalsh(Q), 0, None))


def src(det, mode, **kw):
    return SC.ObstacleCloudSource("detector", detector=det, detector_prompt="all_objects", robot_filter=False,
                                  crop_mode=mode, min_points=20, **kw)


# --- tests -------------------------------------------------------------------------------------------------------
def test_render_sanity():
    d = render_depth(HEAD, [FRONT, BACK])
    p = SC.backproject(d, K, HEAD)
    assert len(p) > 0.7 * H * W and abs(np.percentile(p[:, 2], 30)) < 1e-6  # mostly table at z = 0
    assert inside(p, *FRONT).sum() > 200 and inside(p, *BACK).sum() > 100


def test_depth_trim_keeps_only_the_front_object_of_a_box():
    obs = scene()
    box = project_box(HEAD, *FRONT, pad=0.25)
    det = BoxDetector({0: [box]})
    old = src(det, "box")(obs).points
    new = src(det, "depth_trim")(obs).points
    assert list(old) == ["object#0"] and list(new) == ["object#0"]
    po, pn = old["object#0"], new["object#0"]
    # the old crop holds the table and the box behind; the trimmed crop only the front cube
    assert inside(po, *BACK).sum() > 30 and (np.abs(po[:, 2]) < 1e-6).sum() > 100
    assert inside(pn, *FRONT).all()
    assert inside(pn, *FRONT).sum() > 0.7 * inside(po, *FRONT).sum()  # minus the bottom support_margin_m
    ao, an = semi_axes(po), semi_axes(pn)
    assert an.max() < 0.07 < ao.max()  # ellipsoid covers the front cube only (half-diagonal 0.069)


def test_depth_trim_picks_the_centre_object_when_the_box_is_on_the_back_one():
    obs = scene()
    det = BoxDetector({0: [project_box(HEAD, *BACK, pad=0.15)]})
    pn = src(det, "depth_trim")(obs).points["object#0"]
    assert inside(pn, *BACK).all() and not inside(pn, *FRONT).any()


def test_table_and_empty_boxes_give_no_object():
    obs = scene(boxes=(FRONT,))
    table_only = [2.0, 2.0, 30.0, 30.0]  # far corner of the image: table / nothing
    det = BoxDetector({0: [table_only, project_box(HEAD, *FRONT, pad=0.1)]})
    c = src(det, "depth_trim")(obs)
    assert list(c.points) == ["object#0"] and inside(c.points["object#0"], *FRONT).all()
    assert c.info["boxes"]["head"][0]["name"] is None and c.info["boxes"]["head"][0]["trim"]["n_out"] == 0
    assert c.info["support_z"] == pytest.approx(0.0, abs=0.005)
    assert "object#1" in src(det, "box")(obs).points  # V5: the table crop becomes an obstacle


def test_head_and_wrist_views_of_one_object_are_associated():
    cams = ("head", "left_wrist")
    obs = scene(cams=cams)
    hb, wb = project_box(HEAD, *FRONT, pad=0.1), project_box(WRIST, *FRONT, pad=0.1)
    hb2 = project_box(HEAD, *BACK, pad=0.05)
    det = BoxDetector({0: [hb, hb2], 1: [wb]})
    c = src(det, "depth_trim", cameras=cams)(obs)
    assert sorted(c.points) == ["object#0", "object#1"]
    assert inside(c.points["object#0"], *FRONT).all() and inside(c.points["object#1"], *BACK).all()
    assert c.info["boxes"]["left_wrist"][0]["name"] == "object#0"  # wrist crop joined the head crop of the cube
    assert len(c.info["association"]) == 1
    # the old name merge: wrist ``object#0`` (cube) joins head ``object#0`` (cube) by rank; swap the head order and
    # it joins the box behind instead
    det_sw = BoxDetector({0: [hb2, hb], 1: [wb]})
    old = src(det_sw, "box", cameras=cams)(obs).points["object#0"]
    assert inside(old, *BACK).sum() > 30 and inside(old, *FRONT).sum() > 30
    new = src(det_sw, "depth_trim", cameras=cams)(obs).points
    assert inside(new["object#0"], *BACK).all() and inside(new["object#1"], *FRONT).all()


def test_same_name_far_apart_is_not_merged_obstacle_names():
    cams = ("head", "left_wrist")
    obs = scene(cams=cams)
    obs.extras[PRIVILEGED_PREFIX + "obstacle_names"] = ["thing_1"]
    det = BoxDetector({0: [project_box(HEAD, *BACK, pad=0.05)], 1: [project_box(WRIST, *FRONT, pad=0.1)]})
    s = SC.ObstacleCloudSource("detector", detector=det, detector_prompt="obstacle_names", robot_filter=False,
                               cameras=cams, min_points=20)
    c = s(obs)
    assert sorted(c.points) == ["thing_1", "thing_1@left_wrist"]
    assert inside(c.points["thing_1"], *BACK).all() and inside(c.points["thing_1@left_wrist"], *FRONT).all()


def test_box_mode_is_the_v5_crop():
    """``crop_mode="box"`` = the CLOSED-LOOP-FIX code path (box pixels minus robot spheres, name merge)."""
    obs = scene(cams=("head", "left_wrist"))
    boxes = {0: [project_box(HEAD, *FRONT, pad=0.25)], 1: [project_box(WRIST, *FRONT, pad=0.25)]}
    c = src(BoxDetector(boxes), "box", cameras=("head", "left_wrist"), max_points=10 ** 6)(obs)
    ref = []
    for i, cam in enumerate(("head", "left_wrist")):
        x0, y0, x1, y1 = np.round(boxes[i][0]).astype(int)
        m = np.zeros((H, W), bool)
        m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
        ref.append(SC.backproject(obs.depth[cam], K, obs.T_base_cam[cam], m))
    assert np.array_equal(c.points["object#0"], np.concatenate(ref))
    assert c.info["crop_mode"] == "box"
    with pytest.raises(ValueError):
        SC.ObstacleCloudSource("detector", crop_mode="mask")


def test_crop_mode_reaches_the_filters():
    from Baseline.paper_baselines.methods.interface import get_method

    for m in ("aegis", "knows", "multilink_cbf"):
        f = get_method(m, obstacle_source="detector", detector=BoxDetector({}), detector_prompt="all_objects")
        assert f.source.crop_mode == "depth_trim"
        assert f.provenance()["obstacle"]["crop_mode"] == "depth_trim"
        g = get_method(m, obstacle_source="detector", detector=BoxDetector({}), detector_prompt="all_objects",
                       crop_mode="box")
        assert g.source.crop_mode == "box"


# --- the V5 case: call 0 of SafeLIBERO spatial L I t0e0, Grounding DINO boxes as recorded -------------------------
V5 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
V5_AEGIS = V5 / "20261024T212548_b95c1c878fd6_clm-v5pilot-LI-aegis"
V5_KNOWS = V5 / "20261024T190828_b95c1c878fd6_clm-v5smoke-LI-knows"
needs_v5 = pytest.mark.skipif(not (V5_AEGIS / "episodes").is_dir() or not (V5_KNOWS / "method_calls").is_dir(),
                              reason="V5 runs not available")


def v5_call0():
    """Wire observation of V5 t0e0 call 0 rebuilt from the run files (depth / K / T / robot state / EE pose as
    ``wire_obs.scene_observation`` builds them) + the head boxes the V5 aegis run recorded at that call."""
    from scipy.spatial.transform import Rotation

    stem = "safelibero_spatial_t0_e0"
    with np.load(V5_AEGIS / "episodes" / f"{stem}.npz") as z:
        t = int(z["chunk_t"][0])
        state = z["policy_state"][0].astype(np.float64)
    with np.load(V5_AEGIS / "episodes" / f"{stem}_geom.npz") as g:
        r = int(np.flatnonzero(g["obs_index"] == t)[0])
        cams = [str(c) for c in g["cameras"]]
        depth = {c: g[f"depth_mm/{c}"][r].astype(np.float64) / 1000.0 for c in cams}
        Kc = {c: g[f"K/{c}"][r].astype(np.float64) for c in cams}
        T = {c: g[f"T_base_cam/{c}"][r].astype(np.float64) for c in cams}
        rs = g[f"robot_state/{cams[-1]}"][r].astype(np.float64)
        Twb = g["T_world_base"][r].astype(np.float64)
    Twe = np.eye(4)
    Twe[:3, :3], Twe[:3, 3] = Rotation.from_rotvec(state[3:6]).as_matrix(), state[:3]
    obs = SceneObservation(rgb={c: np.zeros(depth[c].shape + (3,), np.uint8) for c in cams}, depth=depth,
                           intrinsics=Kc, T_base_cam=T, q=rs[:7].copy(), ee_pose=np.linalg.inv(Twb) @ Twe,
                           gripper=float(state[6] - state[7]), instruction="", target_phrase=None,
                           extras={"robot_state": rs, "T_world_base": Twb})
    row = json.loads((V5_AEGIS / "method_calls" / "calls.jsonl").open().readline())
    assert row["episode"] == 0 and row["seq"] == 0
    boxes = row["filter_perception"]["detector"]["boxes"]["head"]
    knows0 = json.loads((V5_KNOWS / "method_calls" / "calls.jsonl").open().readline())["filter_obstacles"]
    return obs, boxes, knows0


class RecordedBoxes:
    def __init__(self, boxes):
        self.b = np.asarray([x["box"] for x in boxes], float)
        self.s = np.asarray([x["score"] for x in boxes], float)

    def detect(self, rgb, text, box_threshold, text_threshold):
        return self.b.copy(), self.s.copy(), ["object"] * len(self.s)


@needs_v5
def test_v5_call0_box_mode_reproduces_the_077m_ellipsoid_and_depth_trim_removes_it():
    from Baseline.paper_baselines.methods.knows.perception.ellipsoid_fit import mvee

    obs, boxes, knows0 = v5_call0()
    old = SC.ObstacleCloudSource("detector", detector=RecordedBoxes(boxes), detector_prompt="all_objects",
                                 crop_mode="box")(obs)
    # object#2 has 3441 points (< max_points: no subsample) -> the V5 KNOWS ellipsoid exactly
    p2 = old.points["object#2"]
    assert len(p2) == 3441
    _, Q = mvee(p2)
    ax = np.sort(np.sqrt(np.linalg.eigvalsh(Q)))
    assert np.allclose(ax, sorted(knows0["object#2"]["semi_axes"]), atol=6e-4)
    assert ax.max() > 0.75
    new = SC.ObstacleCloudSource("detector", detector=RecordedBoxes(boxes), detector_prompt="all_objects")(obs)
    axes_new = {k: float(np.sqrt(np.linalg.eigvalsh(mvee(v)[1])).max()) for k, v in new.points.items()}
    axes_old = {k: float(np.sqrt(np.linalg.eigvalsh(mvee(v)[1])).max()) for k, v in old.points.items()}
    assert max(axes_new.values()) < 0.5 * max(axes_old.values())
    # every remaining crop is above the table and bounded
    for v in new.points.values():
        assert v[:, 2].min() > new.info["support_z"] + 0.01


# --- gt_seg obstacle set (lead decision 2026-10-25) ----------------------------------------------------------------
SL3 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0")


@pytest.mark.skipif(not (SL3 / "episodes").is_dir(), reason="SL3 dev recorded run not available")
def test_gt_obstacle_set_restricts_gt_seg_to_the_named_obstacles():
    obs, _, _ = SC.load_recorded_call(SL3, "libero_spatial_t0_e0", 0)
    full = SC.ObstacleCloudSource("gt_seg")
    assert full.gt_obstacle_set == "all_non_target"
    allc = full(obs).points
    assert {"glazed_rim_porcelain_ramekin_1", "wooden_cabinet_1"} <= set(allc)
    obs.extras[PRIVILEGED_PREFIX + "obstacle_names"] = ["glazed_rim_porcelain_ramekin_1", "akita_black_bowl_1"]
    named = SC.ObstacleCloudSource("gt_seg", gt_obstacle_set="obstacle_names")
    c = named(obs)
    assert list(c.points) == ["glazed_rim_porcelain_ramekin_1"]  # the target stays excluded even if named
    assert np.array_equal(c.points["glazed_rim_porcelain_ramekin_1"], allc["glazed_rim_porcelain_ramekin_1"])
    assert named.provenance()["gt_obstacle_set"] == "obstacle_names"
    assert "obstacle_names" in named.provenance()["privileged_inputs"]
    del obs.extras[PRIVILEGED_PREFIX + "obstacle_names"]
    with pytest.raises(KeyError):
        named(obs)
    with pytest.raises(ValueError):
        SC.ObstacleCloudSource("gt_seg", gt_obstacle_set="some")
