"""AG3S-TASK-DYNAMIC-MAP — the target-only TSDF (① `esdf.target_layer`) and the region-of-interest carve (② `esdf.roi_carve`).

CPU (always):
  - `target_owned`: within `tol` of a target point, strictly nearer the target than any non-target point (a tie
    stays an obstacle), at or above `z_floor`; per-point `tol`.
  - `target_pixel_mask`: the target box's pixels, not the neighbour's, not the table under it, not masked pixels.
  - `roi_boxes` / `in_boxes`; `TargetLayerRequest` defaults; config defaults off and validation; the CLI flags map
    to the esdf keys, `--target-layer` needs `--kept-geometry-follow`, both need AG3S; the builder refuses a request
    whose switch is off (before torch).
GPU (cuRobo + CUDA, `.venv-openpi-live`; skipped elsewhere) — synthetic table + target box + neighbour box + far box,
analytic ray-cast depth from three cameras:
  - ①: the hand's (target-free) layer has no target but keeps the neighbour and the table under the target; the
    main layer keeps the target for the whole body.
  - ①-b: the target moved 40 mm → its old place is free in both layers after one frame (the plain TSDF keeps it).
  - ①: what the static TSDF took of the target before it was routed is freed once the request is active.
  - ②: a neighbour removed inside the region of interest is cleared; one removed outside it is not.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, EsdfConfig
from AG3S.ag3s.fields.curobo_builder import (CuroboFieldBuilder, TargetLayerRequest, in_boxes, roi_boxes,
                                             target_owned, target_pixel_mask)


# -- CPU ------------------------------------------------------------------------------------------------------
def test_target_owned_rules():
    tp = np.array([[0.0, 0.0, 1.0]])
    nt = np.array([[0.02, 0.0, 1.0]])
    pts = np.array([[0.005, 0.0, 1.0],    # 5 mm from the target, 15 mm from the neighbour: target
                    [0.010, 0.0, 1.0],    # a tie (10 mm each): stays an obstacle
                    [0.012, 0.0, 1.0],    # nearer the neighbour
                    [0.0, 0.0, 0.985],    # 15 mm from the target: beyond tol
                    [0.0, 0.0, 0.993]])   # within tol but below the floor
    own = target_owned(pts, tp, nt, 0.012, z_floor=0.995)
    assert own.tolist() == [True, False, False, False, False]
    assert target_owned(pts[[3]], tp, None, np.array([0.02]), None).tolist() == [True]   # per-point tol
    assert not target_owned(pts, None, nt, 0.01, None).any()


K = np.array([[200.0, 0.0, 32.0], [0.0, 200.0, 24.0], [0.0, 0.0, 1.0]])


def test_target_pixel_mask_picks_the_target_pixels_only():
    # camera at the origin looking along +z: left half = target at 1.00 m, right half = neighbour at 1.00 m,
    # bottom rows = the table plane at 1.20 m (below the floor once mapped to base z — here z is depth).
    depth = np.full((48, 64), 1.0)
    depth[40:, :] = 1.2
    v, u = np.mgrid[0:48, 0:64]
    pts = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1) * depth[..., None]
    tp = pts[:40, :30].reshape(-1, 3)
    nt = pts[:40, 34:].reshape(-1, 3)
    req = TargetLayerRequest(target_points=tp, nontarget_points=nt, tol=0.01, z_floor=None)
    m = target_pixel_mask(depth, K, np.eye(4), req)
    assert m[:40, :30].all() and not m[:40, 34:].any() and not m[40:, :].any()
    masked = depth.copy()
    masked[:10, :10] = 0.0                                    # robot-masked pixels decide nothing
    assert not target_pixel_mask(masked, K, np.eye(4), req)[:10, :10].any()


def test_roi_boxes_and_request_defaults():
    b = roi_boxes([np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), None, np.zeros((0, 3))], 0.05,
                  boxes=[(np.array([1.0, 1.0, 1.0]), np.array([1.1, 1.1, 1.1]))])
    assert len(b) == 2 and np.allclose(b[0][0], [-0.05, -0.05, -0.05]) and np.allclose(b[1][1], [1.15] * 3)
    assert in_boxes(np.array([[0.12, 0.0, 0.0], [0.2, 0.0, 0.0], [1.14, 1.0, 1.0]]), b).tolist() == [True, False, True]
    r = TargetLayerRequest(target_points=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]))
    assert r.active and np.allclose(r.centre, [0.1, 0.0, 0.0])
    assert not TargetLayerRequest(held=True).active and not TargetLayerRequest().active
    with pytest.raises(ValueError):
        TargetLayerRequest(tol=0.0)


def test_config_defaults_and_validation():
    e = EsdfConfig()
    assert e.target_layer is False and e.roi_carve is False
    assert e.target_layer_pixel_tol == 0.010 and e.target_layer_reset_m == 0.010 and e.roi_pad == 0.10
    for bad in ({"target_layer": 1}, {"roi_carve": "yes"}, {"roi_pad": 0.0}, {"target_layer_pixel_tol": -1.0},
                {"target_layer_reset_m": float("nan")}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": {"backend": "curobo", **bad}})
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"esdf": {"backend": "legacy", "target_layer": True}})


def test_builder_refuses_a_request_whose_switch_is_off():
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo"}}).esdf
    b = CuroboFieldBuilder(cfg, bounds=([0, 0, 0], [1, 1, 1]))
    cam = types.SimpleNamespace(name="c", depth=np.ones((4, 4)), camera_intrinsics=K, T_base_cam=np.eye(4))
    with pytest.raises(ValueError, match="target_layer"):
        b.update([cam], target_layer=TargetLayerRequest())
    with pytest.raises(ValueError, match="roi_carve"):
        b.update([cam], roi_points=[None])


def test_seen_through_footprint_keeps_silhouette_edges():
    from AG3S.ag3s.fields.curobo_builder import seen_through

    depth = np.full((48, 64), 1.0)       # background 1.0 m
    depth[:, :32] = 0.5                  # an object fills columns 0..31 at 0.5 m
    cam = types.SimpleNamespace(name="c", depth=depth, camera_intrinsics=K, T_base_cam=np.eye(4), robot_mask=None)
    z = 0.8
    edge = np.array([[(32 - K[0, 2]) * z / K[0, 0], 0.0, z]])     # projects onto column 32, the first background pixel
    far = np.array([[(50 - K[0, 2]) * z / K[0, 0], 0.0, z]])      # deep in the background
    assert seen_through(edge, [cam], 0.03)[0][0]                    # nearest pixel: seen through
    assert not seen_through(edge, [cam], 0.03, footprint_voxel=0.005)[0][0]   # footprint touches the object
    assert seen_through(far, [cam], 0.03, footprint_voxel=0.005)[0][0]
    mask = np.zeros((48, 64), bool)
    mask[24, 51] = True                                             # one robot pixel inside the footprint decides nothing
    cam.robot_mask = mask
    assert not seen_through(far, [cam], 0.03, footprint_voxel=0.005)[0][0]
    near = np.array([[(50 - K[0, 2]) * 0.05 / K[0, 0], 0.0, 0.05]])   # footprint radius > FOOTPRINT_MAX_PX: left alone
    cam.robot_mask = None
    assert not seen_through(near, [cam], 0.03, footprint_voxel=0.005)[0][0]


def test_seen_through_footprint_robot_ignore_keeps_the_centre_rule():
    from AG3S.ag3s.fields.curobo_builder import seen_through

    depth = np.full((48, 64), 1.0)
    z = 0.8
    far = np.array([[(50 - K[0, 2]) * z / K[0, 0], 0.0, z]])     # projects onto column 50, row 24
    mask = np.zeros((48, 64), bool)
    mask[24, 51] = True                                           # a robot pixel in the footprint, not the centre
    cam = types.SimpleNamespace(name="c", depth=depth, camera_intrinsics=K, T_base_cam=np.eye(4), robot_mask=mask)
    kw = dict(footprint_voxel=0.005)
    assert not seen_through(far, [cam], 0.03, **kw)[0][0]                                    # block (default)
    assert seen_through(far, [cam], 0.03, footprint_robot="ignore", **kw)[0][0]               # ignore: carved
    mask[24, 50] = True                                                                       # the centre is robot
    assert not seen_through(far, [cam], 0.03, footprint_robot="ignore", **kw)[0][0]           # hidden by the hand: kept
    depth[22:27, 52] = 0.5                                                                    # a near object in the footprint
    mask[:] = False
    assert not seen_through(far, [cam], 0.03, footprint_robot="ignore", **kw)[0][0]
    # pad 0: radius ceil(0.5·200·0.005/0.8) = 1 px — columns 49..51, the object at column 52 is outside
    assert seen_through(far, [cam], 0.03, footprint_robot="ignore", footprint_voxel=0.005, footprint_pad_px=0)[0][0]


def test_footprint_config_and_cli():
    from AG3S.trajopt import serve_safe

    assert EsdfConfig().roi_carve_footprint is False and EsdfConfig().roi_carve_footprint_pad_px == 1
    for bad in ({"roi_carve_footprint": True}, {"roi_carve": True, "roi_carve_footprint_pad_px": -1},
                {"roi_carve": True, "roi_carve_footprint": 1}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": {"backend": "curobo", **bad}})
    sec = serve_safe.dynamic_field_section(serve_safe.build_parser().parse_args(
        [*BASE_ARGV, "--roi-carve", "--roi-carve-footprint"]))
    assert sec == {"roi_carve": True, "roi_carve_footprint": True}
    sec = serve_safe.dynamic_field_section(serve_safe.build_parser().parse_args(
        [*BASE_ARGV, "--roi-carve", "--roi-carve-footprint", "--roi-carve-footprint-pad-px", "0",
         "--roi-carve-footprint-robot", "ignore"]))
    assert sec == {"roi_carve": True, "roi_carve_footprint": True, "roi_carve_footprint_pad_px": 0,
                   "roi_carve_footprint_robot": "ignore"}
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", **sec}})
    assert cfg.esdf.roi_carve_footprint_robot == "ignore" and cfg.esdf.roi_carve_footprint_pad_px == 0
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"esdf": {"backend": "curobo", "roi_carve": True, "roi_carve_footprint": True,
                                       "roi_carve_footprint_robot": "maybe"}})


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_cli_flags_map_to_esdf_keys():
    from AG3S.trajopt import serve_safe

    parse = lambda *a: serve_safe.build_parser().parse_args([*BASE_ARGV, *a])   # noqa: E731
    assert serve_safe.dynamic_field_section(parse()) == {}
    sec = serve_safe.dynamic_field_section(parse("--target-layer", "--target-layer-tol-mm", "8",
                                                 "--target-layer-reset-mm", "15", "--roi-carve",
                                                 "--roi-pad-mm", "120"))
    assert sec == {"target_layer": True, "target_layer_pixel_tol": 0.008, "target_layer_reset_m": 0.015,
                   "roi_carve": True, "roi_pad": 0.12}
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", **sec}})
    assert cfg.esdf.target_layer and cfg.esdf.roi_carve


def test_roi_carve_serving_default():
    """E4 (task §10): a cuRobo AG3S server with no carve flag runs the adopted carve; `--no-roi-carve`, the legacy field,
    `--no-safe` / `--no-perception` and any explicit carve flag keep the earlier meaning."""
    from AG3S.trajopt import serve_safe

    sec = lambda *a: serve_safe.dynamic_field_section(serve_safe.build_parser().parse_args([*BASE_ARGV, *a]))  # noqa: E731
    assert sec("--esdf-backend", "curobo") == serve_safe.ROI_CARVE_SERVING_DEFAULT == {
        "roi_carve": True, "roi_carve_footprint": True, "roi_carve_footprint_pad_px": 0, "roi_carve_footprint_robot": "ignore",
        "roi_carve_device": "cuda"}                                         # AG3S-ROI-CARVE-GPU
    assert sec("--esdf-backend", "curobo", "--no-roi-carve") == {}          # = the esdf config before the default
    assert sec() == {}                                                      # legacy field
    assert sec("--esdf-backend", "curobo", "--no-safe") == {}
    assert sec("--esdf-backend", "curobo", "--no-perception") == {}
    assert sec("--esdf-backend", "curobo", "--roi-carve") == {"roi_carve": True}     # the E2 kits are unchanged
    assert sec("--esdf-backend", "curobo", "--target-layer", "--kept-geometry-follow") == {
        "target_layer": True, **serve_safe.ROI_CARVE_SERVING_DEFAULT}                 # other flags do not switch it off
    assert sec("--esdf-backend", "curobo", "--target-layer", "--kept-geometry-follow", "--no-roi-carve") == {"target_layer": True}
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", **serve_safe.ROI_CARVE_SERVING_DEFAULT}})
    assert cfg.esdf.roi_carve and cfg.esdf.roi_carve_footprint and cfg.esdf.roi_carve_footprint_pad_px == 0
    assert EsdfConfig().roi_carve is False                                  # the config default stays off


@pytest.mark.parametrize("argv, needle", [
    (["--esdf-backend", "curobo", "--no-roi-carve", "--roi-carve"], "--no-roi-carve"),
    (["--esdf-backend", "curobo", "--target-layer"], "--kept-geometry-follow"),
    (["--target-layer", "--kept-geometry-follow"], "--esdf-backend curobo"),
    (["--esdf-backend", "curobo", "--roi-carve", "--roi-pad-mm", "-5"], "--roi-pad-mm"),
    (["--no-safe", "--roi-carve"], "--roi-carve"),
    (["--esdf-backend", "curobo", "--roi-carve-footprint"], "--roi-carve-footprint"),
    (["--esdf-backend", "curobo", "--roi-carve", "--roi-carve-footprint-robot", "ignore"], "--roi-carve-footprint"),
])
def test_cli_refusals(monkeypatch, capsys, argv, needle):
    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *argv])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2
    assert needle in capsys.readouterr().err


# -- GPU ------------------------------------------------------------------------------------------------------
def _gpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():  # pragma: no cover
        pytest.skip("cuRobo needs CUDA")


KG = np.array([[220.0, 0.0, 160.0], [0.0, 220.0, 120.0], [0.0, 0.0, 1.0]])
TABLE = (np.array([0.30, -0.60, 0.780]), np.array([1.00, 0.60, 0.820]))
ZT = 0.820


def _box(cx, cy, half, height):
    return (np.array([cx - half, cy - half, ZT]), np.array([cx + half, cy + half, ZT + height]))


TARGET = _box(0.60, 0.00, 0.030, 0.060)
NEIGHBOUR = _box(0.60, 0.10, 0.025, 0.050)   # 45 mm gap; wholly inside the target's box ± 0.10 m (the ROI)
FAR = _box(0.60, -0.40, 0.025, 0.050)


def _look_at(pos, target, up=(0.0, 0.0, 1.0)):
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = pos
    return T


VIEWS = [_look_at(np.array(p), np.array([0.60, 0.0, 0.85]))
         for p in ([0.0, 0.0, 1.35], [0.60, -0.75, 1.30], [0.60, 0.75, 1.30])]


def _raycast(Tbc, boxes, h=240, w=320, centre=0.0):
    u, v = np.meshgrid(np.arange(w) + centre, np.arange(h) + centre)
    dc = np.stack([(u - KG[0, 2]) / KG[0, 0], (v - KG[1, 2]) / KG[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    dw = dc @ Tbc[:3, :3].T
    o = Tbc[:3, 3]
    best = np.full(len(dw), np.inf)
    who = np.full(len(dw), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (lo, hi) in enumerate(boxes):
            t1, t2 = (lo - o) / dw, (hi - o) / dw
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= tmin) & (tmax > 0)
            t = np.where(tmin > 0, tmin, tmax)
            better = hit & (t < best)
            best = np.where(better, t, best)
            who = np.where(better, k, who)
    depth = np.where(np.isfinite(best), best * (dw @ Tbc[:3, 2]), 0.0).reshape(h, w)
    return np.rint(depth * 1000.0) / 1000.0, who.reshape(h, w)


def _frame(boxes, target_index):
    """Cameras + the target / non-target points as the pipeline would hand them (back-projected pixels)."""
    from AG3S.ag3s.fields.esdf import CameraDepth

    cams, tp, nt = [], [], []
    n_target_px = 0
    v, u = np.mgrid[0:240, 0:320]
    for i, T in enumerate(VIEWS):
        d, who = _raycast(T, boxes)
        cams.append(CameraDepth(name=f"c{i}", depth=d, camera_intrinsics=KG, T_base_cam=T))
        pc = np.stack([(u - KG[0, 2]) * d / KG[0, 0], (v - KG[1, 2]) * d / KG[1, 1], d], -1)
        pb = pc.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
        w = who.reshape(-1)
        n_target_px += int((w == target_index).sum())
        # every 2nd pixel ≈ the server's ~5 mm point spacing at this range
        tp.append(pb[w == target_index][::2])
        nt.append(pb[(w >= 0) & (w != target_index) & (w != 0)][::2])   # 0 = the table (support: not evidence)
    _frame.n_target_px = n_target_px
    return cams, np.vstack(tp), np.vstack(nt)


def _builder(**esdf):
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {
        "voxel_size": 0.020, "max_distance": 0.4, "exclude_support_surfaces": False, "backend": "curobo",
        "fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005,
        "bounds_lower": [0.0, -0.8, 0.5], "bounds_upper": [1.4, 0.8, 1.6], **esdf}}).esdf
    return CuroboFieldBuilder(cfg)


def _update(builder, boxes, *, layer=True, roi=False, request=None, target_index=1, window_at=None):
    from AG3S.ag3s.fields.curobo_builder import FineWindowRequest

    cams, tp, nt = _frame(boxes, target_index)
    if request is None and layer:
        request = TargetLayerRequest(target_points=tp, nontarget_points=nt, tol=0.010, z_floor=ZT + 0.005)
    window = FineWindowRequest(manipulated_points=tp if window_at is None else window_at,
                               placement="target_centroid")
    return builder.update(cams, fine_window=window, target_free_points=tp, target_points=tp,
                          **({"target_layer": request} if request is not None else {}),
                          **({"roi_points": [tp]} if roi else {}))


def _top(box, dz=0.004):
    lo, hi = box
    return np.array([[0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), hi[2] + dz]])


def test_target_layer_takes_the_target_out_of_the_hands_layer_only():
    _gpu()
    b = _builder(target_layer=True)
    boxes = [TABLE, TARGET, NEIGHBOUR, FAR]
    for _ in range(4):
        f = _update(b, boxes)
    st = f.stats["target_layer"]
    assert st["active"] and st["n_target_px"] >= 0.9 * _frame.n_target_px, (st["n_target_px"], _frame.n_target_px)
    assert st["n_target_px"] <= _frame.n_target_px                     # nothing else (neighbour, table) routed
    assert f.stats["target_free"]["mode"] == "target_layer"
    assert float(f.distance(_top(TARGET))[0]) < 0.010                 # the whole body still sees the target
    assert float(f.target_free_distance(_top(TARGET))[0]) > 0.030      # the hand does not
    assert float(f.target_free_distance(_top(NEIGHBOUR))[0]) < 0.010   # the neighbour stays for the hand
    under = np.array([[0.60 + 0.045, 0.0, ZT + 0.004]])                # the table 15 mm beside the target
    assert float(f.target_free_distance(under)[0]) < 0.010


def test_target_moved_leaves_no_trace():
    _gpu()
    moved = _box(0.64, 0.00, 0.030, 0.060)
    old_top = _top(TARGET) + np.array([[-0.020, 0.0, 0.0]])            # on the old top, 6 mm off the new one
    results = {}
    for layer in (False, True):
        b = _builder(target_layer=layer)
        for _ in range(6):
            _update(b, [TABLE, TARGET, NEIGHBOUR], layer=layer)
        f = _update(b, [TABLE, moved, NEIGHBOUR], layer=layer)
        results[layer] = (float(f.distance(old_top)[0]), f.stats.get("target_layer"))
    plain, (with_layer, st) = results[False][0], results[True]
    assert st["reset"] == "moved" and st["centre_shift_mm"] > 30.0
    assert with_layer > 0.010, (plain, with_layer)
    assert plain < with_layer, (plain, with_layer)


def test_target_traces_from_before_routing_are_freed():
    _gpu()
    b = _builder(target_layer=True)
    boxes = [TABLE, TARGET, NEIGHBOUR]
    for _ in range(4):                                                 # no target yet: all pixels → static
        f = _update(b, boxes, request=TargetLayerRequest())
    assert float(f.distance(_top(TARGET))[0]) < 0.010
    f = _update(b, boxes)
    assert sum(f.stats["target_layer"]["n_static_freed"].values()) > 0
    assert float(f.target_free_distance(_top(TARGET))[0]) > 0.030


POST = _box(0.60, 0.075, 0.015, 0.150)   # a 30 mm post inside the region (the place-obstacle size)


def _post_missing(builder, frames, centre):
    """Fraction of the post's side-surface samples the field puts > 10 mm away, after `frames` updates."""
    from AG3S.ag3s.fields.curobo_builder import FineWindowRequest
    from AG3S.ag3s.fields.esdf import CameraDepth

    boxes = [TABLE, TARGET, POST]
    tp = None
    for _ in range(frames):
        cams = []
        tps = []
        v, u = np.mgrid[0:240, 0:320]
        for i, T in enumerate(VIEWS):
            d, who = _raycast(T, boxes, centre=centre)
            cams.append(CameraDepth(name=f"c{i}", depth=d, camera_intrinsics=KG, T_base_cam=T))
            pc = np.stack([(u - KG[0, 2]) * d / KG[0, 0], (v - KG[1, 2]) * d / KG[1, 1], d], -1)
            pb = pc.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
            tps.append(pb[who.reshape(-1) == 1][::2])
        tp = np.vstack(tps)
        f = builder.update(cams, fine_window=FineWindowRequest(manipulated_points=tp, placement="target_centroid"),
                           target_points=tp, roi_points=[tp])
    lo, hi = POST
    zs = np.linspace(ZT + 0.02, hi[2] - 0.01, 12)
    s = np.linspace(lo[1], hi[1], 7)
    pts = np.array([[x, y, z] for z in zs for (x, y) in
                    [(lo[0], yy) for yy in s] + [(hi[0], yy) for yy in s]])
    return float(np.mean(f.distance(pts) > 0.010))


def test_roi_carve_footprint_keeps_a_thin_post():
    """Half-pixel centred depth (as a renderer gives) against the nearest-pixel projection: the plain carve eats the
    post's silhouette edges, the footprint carve keeps them."""
    _gpu()
    plain = _post_missing(_builder(roi_carve=True), 6, centre=0.5)
    fp = _post_missing(_builder(roi_carve=True, roi_carve_footprint=True), 6, centre=0.5)
    assert fp <= plain, (plain, fp)
    assert fp <= 0.05, (plain, fp)


def test_roi_carve_clears_inside_the_region_only():
    _gpu()
    out = {}
    for roi in (False, True):
        b = _builder(roi_carve=True) if roi else _builder()
        for _ in range(6):
            _update(b, [TABLE, TARGET, NEIGHBOUR, FAR], layer=False, roi=roi)
        f = _update(b, [TABLE, TARGET], layer=False, roi=roi)          # neighbour and far box taken away
        out[roi] = (float(f.distance(_top(NEIGHBOUR))[0]), float(f.distance(_top(FAR))[0]), f.stats.get("roi_carve"))
    (n_plain, far_plain, _), (n_roi, far_roi, st) = out[False], out[True]
    assert st["coarse_tsdf"]["n_freed"] + st["fine_tsdf"]["n_freed"] > 0
    assert n_roi > n_plain and n_roi > 0.010, (n_plain, n_roi)        # inside: the old neighbour is gone
    assert abs(far_roi - far_plain) < 1e-6, (far_plain, far_roi)       # outside: untouched
