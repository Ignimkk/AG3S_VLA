"""IFG-CL1 — attention-surface target extent (`esdf.target_extent = "attention_surface"`) and the rendered-robot mask
(`esdf.robot_mask = "render"`), both default off. Off must be bit-identical.

| # | property |
|---|---|
| 1 | config defaults · round trip · validation; CLI (`--target-extent attention-surface`, `--robot-mask render`) → esdf keys; start-up refusals (no / wrong `--attention-source`, relax policy, `--no-safe`); record meta and server metadata only when on |
| 2 | **parity** (storage): on ≥ 5 dev frames the server-shaped `select_target_surface` (dummy GT, per-camera 16×16 maps, depth / K / T and the R6 render mask) picks the same node set as the offline locked chain (`ifg_round_ns.g_and_top1`, real GT) — IoU 1, same seeds, node count = the stored R6 row's `pred_voxels`; the real GT gives the same selection as the dummy |
| 3 | code image: the code-1 pixels re-lifted (`backproject_labels`) fall in exactly the selected nodes' voxels; code 2 never does; robot pixels are 0 |
| 4 | abstain: missing camera · no attention · zero attention |
| 5 | pipeline (fake builder): the code images feed the oracle memory and evidence; abstain → estimated with the reason; records · debug codes; default = no call, no key |
| 6 | render mask (storage + GL): server path (robot_state + finger opening + K/T) vs the stored offline R6 masks, and vs the R6 procedure (MJCF camera, full qpos) on a synthetic state; pipeline union with the sphere mask |
| 7 | constraint record: code images and JSON summaries only when on |
"""

from __future__ import annotations

import json
import os
import pathlib
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.runtime.attention_surface import (
    CAMERA_ORDER, SurfaceView, attention_vector, select_target_surface)
from AG3S.ag3s.runtime.oracle_segmentation import CODE_NONE, CODE_STATIC, CODE_TARGET, backproject_labels

STORE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
RUN = STORE / "results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba"
REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL_XML = REPO / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
needs_run = pytest.mark.skipif(not (RUN / "s0/replay").exists() or not (RUN / "rounds/R6/robot_mask").exists(),
                               reason="IFG run (shared storage) not mounted")
#: dev frames over the moments (approach · pre-grasp · grasp · carry · pre-place · post-place · retreat · F-series)
PARITY_FRAMES = ("T53_ep1804_f011", "T54c_ep1817_f016", "T53_ep1853_f007", "T54c_ep1806_f031",
                 "T53_ep1839_f036", "T54c_ep1831_f004")


# ============================================================================ 1 config · CLI · metadata


def test_config_defaults_round_trip_and_validation():
    d = AG3SConfig().esdf
    assert (d.target_extent, d.robot_mask) == ("estimated", "spheres")
    cfg = AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "robot_mask": "render"}})
    again = AG3SConfig.from_dict(cfg.to_dict())
    assert again.esdf == cfg.esdf and (again.esdf.target_extent, again.esdf.robot_mask) == ("attention_surface",
                                                                                            "render")
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"esdf": {"robot_mask": "mesh"}})
    for f in ({"nontarget_evidence": "surface"}, {"owned_ambiguous": "remove"}):
        with pytest.raises(AG3SConfigError, match="attention_surface"):
            AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", "target_extent": "attention_surface",
                                           **f}})


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])


def test_cli_flags_map_to_esdf_keys_only_when_given():
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    for argv in ((), ("--robot-mask", "spheres"), ("--target-extent", "estimated")):
        args = _parse(*argv)
        assert target_exclusion_section(args) == {} and ifg_cl1_record_meta(args) == {}
    args = _parse("--target-exclusion", "ball-owned", "--target-extent", "attention-surface", "--robot-mask", "render",
                  "--oracle-extent-identity", "gt")
    sec = target_exclusion_section(args)
    assert sec == {"target_exclusion": "ball_owned", "target_extent": "attention_surface", "robot_mask": "render",
                   "oracle_extent_identity": "gt"}
    cfg = AG3SConfig.from_dict({"esdf": sec}).esdf
    assert {k: getattr(cfg, k) for k in sec} == sec
    assert ifg_cl1_record_meta(args) == {"target_extent": "attention_surface", "robot_mask": "render"}


def _reject(*argv):
    from AG3S.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])
    reject_bad_flag_combinations(ap, args)
    return args


def test_cli_refuses_attention_surface_without_the_locked_source(capsys):
    on = ("--esdf-backend", "curobo", "--target-exclusion", "ball-owned", "--target-extent", "attention-surface",
          "--target-field-policy", "exclude-authorized")
    for extra in ((), ("--attention-source", "action:last:L8H2"), ("--attention-source", "noun_target:L10H6")):
        with pytest.raises(SystemExit) as e:
            _reject(*on, *extra)
        assert e.value.code == 2 and "noun_target:L10H7" in capsys.readouterr().err
    with pytest.raises(SystemExit):                                    # relax: no target-free tier to change
        _reject(*on[:6], "--attention-source", "noun_target:L10H7")
    assert "--target-field-policy" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _reject(*on, "--attention-source", "noun_target:L10H7", "--nontarget-evidence", "surface")
    assert "F1" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _reject("--robot-mask", "render", "--no-safe")
    assert "--robot-mask render" in capsys.readouterr().err
    # accepted: the locked source, an oracle flag (read by the same path) and the render mask
    args = _reject(*on, "--attention-source", "noun_target:L10H7", "--robot-mask", "render",
                   "--oracle-max-dist-voxels", "5")
    assert args.target_extent == "attention-surface"
    # the oracle flags are still refused without an oracle-path extent
    with pytest.raises(SystemExit):
        _reject("--oracle-max-dist-voxels", "5")
    assert "attention-surface" in capsys.readouterr().err


def test_ag3s_render_mask_needs_a_renderer():
    from AG3S.ag3s.runtime.pipeline import AG3S

    with pytest.raises(ValueError, match="robot renderer"):
        AG3S(AG3SConfig.from_dict({"esdf": {"robot_mask": "render"}}))
    a = AG3S(AG3SConfig.from_dict({"esdf": {"robot_mask": "render"}}), robot_render=object())
    assert a.robot_render is not None


def test_server_metadata_only_when_on():
    from AG3S.trajopt import wire

    t46e = pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")
    base = t46e._safe_policy_with().metadata
    assert "robot_mask" not in base and "target_extent" not in base
    assert t46e._safe_policy_with(target_extent="estimated", robot_mask="spheres").metadata == base
    on = t46e._safe_policy_with(target_extent="attention_surface").metadata
    assert wire.ORACLE_SEG_REQUIRED not in on, "the client sends no segmentation for attention_surface"
    assert on.pop("target_extent") == "attention_surface" and on.pop("oracle_extent_identity") == "pipeline"
    assert on == base


# ============================================================================ 2 · 3 parity on dev frames (storage)


def _frame_inputs(fid):
    with np.load(RUN / "s0/replay" / f"{fid}.npz") as z:
        depth, K, T = z["depth"], z["K"], z["T_base_cam"]
        cams = [str(c) for c in z["cameras"]]
    with np.load(RUN / "rounds/R6/robot_mask" / f"{fid}.npz") as z:
        rm = z["robot_mask"]
    with np.load(RUN / "s0/labels" / f"{fid}.npz") as z:
        cm, cl = z["class_map"], [str(c) for c in z["classes"]]
    return depth, K, T, rm, cm, cl, cams


def _offline(fid):
    """The locked chain exactly as `ifg_round_ns.frame` runs it for `top1-only-ns+rm:N` (real GT), node ids found by
    position (independent of the module's `node_eval` mapping)."""
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_reader as R
    from AG3S.ag3s.experiments.studies import ifg_round_ns as RNS

    ref = next(r for r in R.list_frames() if r.frame_id == fid)
    fin = R.load_frame(ref, features=())
    depth, K, T, rm, cm, cl, _ = _frame_inputs(fid)
    cfg = G.GraphConfig()
    geo = G.build_geometry(depth, K, T, rm, cm, cl, cfg)
    res = RNS.g_and_top1(geo, cfg, fin.maps["N"], RNS.eligible(geo, cfg, True))
    if res is None:
        return fin.maps["N"], set(), None, geo
    sub, six, m, mt, info = res
    sel = np.isfinite(mt) & (mt >= 0.5)
    lut = {tuple(p): i for i, p in enumerate(geo.pos)}
    return fin.maps["N"], {lut[tuple(p)] for p in sub.pos[sel]}, [int(s) for s in six], geo


def _server_views(fid, nmap):
    depth, K, T, rm, *_ = _frame_inputs(fid)
    return [SurfaceView(c, depth[i], K[i], T[i], rm[i], np.asarray(nmap)[256 * i:256 * (i + 1)].reshape(16, 16))
            for i, c in enumerate(CAMERA_ORDER)]


@needs_run
@pytest.mark.parametrize("fid", PARITY_FRAMES)
def test_parity_with_the_offline_locked_arm(fid):
    nmap, off, seeds, geo_off = _offline(fid)
    views = _server_views(fid, nmap)
    res = select_target_surface(views, keep_debug=True)
    new = set(res.selected_nodes.tolist())
    iou = len(off & new) / max(len(off | new), 1)
    assert iou >= 0.99 and iou == 1.0, (fid, len(off), len(new))
    assert res.record["seed_patches"] == seeds
    np.testing.assert_array_equal(res.geometry.pos, geo_off.pos)
    row = next(json.loads(line) for line in open(RUN / "rounds/R6/rows" / f"{fid}.jsonl")
               if '"top1-only-ns+rm:N"' in line)
    assert len(new) == row["pred_voxels"], "each node is one eval voxel: the stored R6 row's prediction count"
    # the dummy GT does not reach the selection: the real GT gives the same nodes
    depth, K, T, rm, cm, cl, _ = _frame_inputs(fid)
    real = select_target_surface(views, class_map=cm, classes=cl)
    assert np.array_equal(real.selected_nodes, res.selected_nodes) and real.record["seed_patches"] == seeds


@needs_run
@pytest.mark.parametrize("fid", PARITY_FRAMES[:3])
def test_code_one_pixels_lift_into_the_selected_voxels(fid):
    from AG3S.ag3s.experiments.studies import ifg_graph as G

    nmap = _offline(fid)[0]
    views = _server_views(fid, nmap)
    res = select_target_surface(views, keep_debug=True)
    assert res.codes is not None and res.record["applied"]
    cfg = G.GraphConfig()
    allp = np.concatenate([G._cam_pixels(v.depth_m, v.K, v.T_base_cam, v.robot_mask, i, cfg.range_max)[0]
                           for i, v in enumerate(views)])
    kmin = np.floor(allp / cfg.voxel).astype(np.int64).min(0)

    def keys(p):
        k = np.floor(p / cfg.voxel).astype(np.int64) - kmin
        return {tuple(x) for x in k.tolist()}

    sel_keys = keys(res.geometry.pos[res.selected_nodes])
    lifted = {c: [backproject_labels(v.depth_m, v.K, v.T_base_cam, (res.codes[v.name] == c).astype(np.uint8))[0]
                  for v in views] for c in (CODE_TARGET, CODE_STATIC)}
    k1 = keys(np.vstack(lifted[CODE_TARGET]))
    k2 = keys(np.vstack(lifted[CODE_STATIC]))
    assert k1 == sel_keys, "code-1 pixels = exactly the selected nodes' voxels"
    assert not (k2 & sel_keys), "no code-2 pixel lies in a selected voxel"
    for v in views:
        assert np.all(res.codes[v.name][v.robot_mask] == CODE_NONE), "robot pixels carry no code"
        assert res.codes[v.name].shape == v.depth_m.shape and res.codes[v.name].dtype == np.uint8
    assert sum(res.record["n_target_px"].values()) == sum(int((res.codes[v.name] == 1).sum()) for v in views)
    assert res.record["pixel_node_keys"] in ("centroid_keys", "unique")


# ============================================================================ 4 abstain


def _tiny_views(att=True, drop=None):
    rng = np.random.default_rng(0)
    out = []
    for i, n in enumerate(CAMERA_ORDER):
        if n == drop:
            continue
        T = np.eye(4)
        T[:3, 3] = [0.0, 0.1 * i, 1.0]
        d = 0.8 + 0.01 * rng.random((32, 32))
        K = np.array([[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]])
        out.append(SurfaceView(n, d, K, T, np.zeros((32, 32), bool), np.ones((16, 16)) if att else None))
    return out


def test_abstain_reasons():
    assert select_target_surface(_tiny_views(drop="left_wrist")).record["why"] == "missing_camera:left_wrist"
    r = select_target_surface(_tiny_views(att=False))
    assert r.codes is None and r.record["why"] == "no_attention:head" and not r.record["applied"]
    v = _tiny_views()
    v = [SurfaceView(x.name, x.depth_m, x.K, x.T_base_cam, x.robot_mask, np.zeros((16, 16))) for x in v]
    r = select_target_surface(v)
    assert r.codes is None and r.record["why"] == "no_seed", r.record   # 32x32 px: no patch has 60 px, map 0
    assert attention_vector(_tiny_views())[0].shape == (768,)
    assert attention_vector(_tiny_views(att=False))[0] is None


# ============================================================================ 5 pipeline wiring (fake builder)


def _look_at(eye, target):
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def _raycast_rig(fw, hw=(240, 320), f=260.0):
    """Three cameras (head, left wrist, right wrist) over the fine-window rig's apple on the table and a neighbour
    83 mm away. Returns observations (depth, K, T, attention on the apple's patches) and the GT code images."""
    from AG3S.ag3s.types import CameraID

    t46e = pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")
    nb = fw.APPLE + np.array([0.0, 0.083, 0.0])
    sph = ((fw.APPLE, fw.APPLE_R, CODE_TARGET), (nb, fw.APPLE_R, 3))
    eyes = (fw.APPLE + [0.0, -0.05, 0.50], fw.APPLE + [-0.40, -0.15, 0.30], fw.APPLE + [-0.35, 0.20, 0.35])
    ids = (CameraID.HEAD, CameraID.LEFT_WRIST, CameraID.RIGHT_WRIST)
    obs, gt = [], {}
    for cid, eye in zip(ids, eyes):
        v = t46e._raycast(_look_at(eye, fw.APPLE), hw=hw, f=f, spheres=sph, table_z=fw.TABLE_Z)
        h, w = hw
        lab = v.labels
        cov = (lab == CODE_TARGET).reshape(16, h // 16, 16, w // 16).mean(axis=(1, 3))
        obs.append(types.SimpleNamespace(camera_id=cid, depth=v.depth_m, camera_intrinsics=v.K,
                                         T_base_cam=v.T_base_cam, attention_map=cov.astype(np.float32),
                                         robot_state=np.zeros(20)))
        gt[cid.value] = lab
    return obs, gt, nb


def _cameras(obs):
    from AG3S.ag3s.fields.esdf import CameraDepth

    return [CameraDepth(name=o.camera_id.value, depth=o.depth, camera_intrinsics=o.camera_intrinsics,
                        T_base_cam=o.T_base_cam, robot_mask=np.zeros(o.depth.shape, bool)) for o in obs]


def _wired(monkeypatch, **esdf):
    t46e = pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")
    return t46e._wired(monkeypatch, **esdf)


def test_pipeline_attention_surface_feeds_the_oracle_path(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface")
    obs, gt, nb = _raycast_rig(fw)
    cams = _cameras(obs)
    rec = ag3s._attention_surface_update(obs, cams, {c.name: c.robot_mask for c in cams}, {})
    ag3s._oracle_frame = rec
    surf = ag3s._surface_frame
    assert rec["segmentation"] and rec["source"] == "attention_surface" and rec["memory"]["updated"], (rec, surf)
    assert surf["applied"] and surf["arm"] == "top1-only-ns:N" and set(surf["robot_mask_source"].values()) == {
        "spheres"}
    # the selected surface is the apple (GT code 1), not the neighbour or the table
    for name, lab in gt.items():
        c = ag3s._surface_codes[name]
        t = c == CODE_TARGET
        if t.any():
            assert (lab[t] == CODE_TARGET).mean() > 0.95, name
    assert sum(surf["n_target_px"].values()) > 500
    field, _ = fw._build(ag3s)
    kw = ag3s._esdf_builder.calls[-1]
    ev = kw["oracle_evidence"]
    d = np.abs(np.linalg.norm(ev.target_points - fw.APPLE, axis=1) - fw.APPLE_R)
    assert np.median(d) < 0.003 and len(ev.target_points) > 100
    assert np.any(np.abs(np.linalg.norm(ev.other_points - nb, axis=1) - fw.APPLE_R) < 1e-3), "neighbour = code 2"
    tf = field.stats["target_free"]
    assert tf["oracle"]["applied"] and tf["oracle"]["object"] == "attention_surface"
    assert tf["attention_surface"]["applied"] and tf["attention_surface"]["n_selected_nodes"] > 0
    ag3s.reset()
    assert ag3s._surface_frame is None and len(ag3s._oracle_memory) == 0


def test_pipeline_attention_surface_abstain_builds_the_estimated_extent(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface")
    obs, _, _ = _raycast_rig(fw)
    for o in obs:
        o.attention_map = None
    cams = _cameras(obs)
    ag3s._oracle_frame = ag3s._attention_surface_update(obs, cams, {c.name: c.robot_mask for c in cams}, {})
    assert ag3s._oracle_frame["segmentation"] is False and ag3s._oracle_frame["memory"]["updated"] is False
    field, _ = fw._build(ag3s)
    assert "oracle_evidence" not in ag3s._esdf_builder.calls[-1]
    tf = field.stats["target_free"]
    assert tf["oracle"]["applied"] is False and tf["oracle"]["why"].startswith("attention_surface_abstained")
    assert tf["attention_surface"]["why"] == "no_attention:head"
    # a previous frame's memory is not used on an abstaining frame
    obs2, _, _ = _raycast_rig(fw)
    ag3s._oracle_frame = ag3s._attention_surface_update(obs2, _cameras(obs2), {c.name: None for c in cams}, {})
    assert len(ag3s._oracle_memory) > 0
    ag3s._oracle_frame = ag3s._attention_surface_update(obs, cams, {c.name: None for c in cams}, {})
    fw._build(ag3s)
    assert "oracle_evidence" not in ag3s._esdf_builder.calls[-1]


def test_pipeline_default_never_calls_the_new_steps(monkeypatch):
    from AG3S.ag3s.runtime import attention_surface as AS

    def boom(*a, **k):
        raise AssertionError("called with the default config")

    monkeypatch.setattr(AS, "select_target_surface", boom)
    for esdf in ({"target_exclusion": "ball_owned"},
                 {"target_exclusion": "ball_owned", "target_extent": "estimated", "robot_mask": "spheres"}):
        fw, ag3s = _wired(monkeypatch, **esdf)
        field, _ = fw._build(ag3s)
        assert "oracle_evidence" not in ag3s._esdf_builder.calls[-1]
        assert "attention_surface" not in field.stats["target_free"] and "robot_mask_render" not in field.stats
        assert ag3s._surface_frame is None and ag3s._oracle_memory is None and ag3s.robot_render is None
    fw, a = _wired(monkeypatch, target_exclusion="ball_owned")
    fw2, b = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="estimated", robot_mask="spheres")
    fw._build(a)
    fw2._build(b)
    ka, kb = a._esdf_builder.calls[-1], b._esdf_builder.calls[-1]
    assert ka.keys() == kb.keys()


class _FakeRender:
    """Marks a fixed pixel block as robot; records the call."""

    def __init__(self, block=(slice(0, 10), slice(0, 10))):
        self.block = block
        self.calls = []

    def masks(self, views, openings=None):
        self.calls.append((list(views), openings))
        out = []
        for v in views:
            m = np.zeros(np.shape(v["depth_m"]), bool)
            m[self.block] = True
            out.append(m)
        return out, {"cameras": {v["name"]: {"n_robot_px": 100} for v in views}, "ms": 1.0, "pose": {}}


def test_pipeline_render_mask_is_unioned_with_the_sphere_mask(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned")
    ag3s.config = ag3s.config.with_overrides({"esdf": {"robot_mask": "render"}})
    ag3s.robot_render = _FakeRender()
    ag3s._finger_record = {"opening_norm": {"left": 0.5, "right": 1.0}}
    obs, _, _ = _raycast_rig(fw)
    cams = _cameras(obs)
    cams[0] = type(cams[0])(**{**cams[0].__dict__, "robot_mask": np.eye(*cams[0].depth.shape, dtype=bool),
                               "device_depth": "dev"})
    out, spheres, render = ag3s._apply_render_mask(obs, cams)
    assert ag3s.robot_render.calls[-1][1] == {"left": 0.5, "right": 1.0}, "the measured opening is passed"
    u0 = out[0].robot_mask
    assert u0[:10, :10].all() and np.array_equal(u0 | np.eye(*u0.shape, dtype=bool), u0), "sphere ∪ render"
    assert u0.sum() == int((np.eye(*u0.shape, dtype=bool) | render["head"]).sum())
    assert out[0].device_depth is None, "the device depth was masked with the spheres only"
    assert np.array_equal(spheres["head"], np.eye(*u0.shape, dtype=bool))
    rec = ag3s._render_frame
    assert rec["applied"] and rec["cameras"]["head"]["n_union_px"] == int(u0.sum())
    assert rec["finger_source"] == "measured"
    # a failing renderer keeps the sphere masks and says so
    class _Bad:
        def masks(self, *a, **k):
            raise RuntimeError("no GL")

    ag3s.robot_render = _Bad()
    out, _, render = ag3s._apply_render_mask(obs, cams)
    assert out is cams and render == {} and ag3s._render_frame["why"] == "render_failed"


# ============================================================================ 6 render mask vs offline R6 (storage + GL)


def _renderer_or_skip():
    pytest.importorskip("mujoco")
    if not MODEL_XML.exists():
        pytest.skip("model XML not in this checkout")
    from AG3S.ag3s.runtime.robot_render_mask import RobotRenderMask

    try:
        r = RobotRenderMask(str(MODEL_XML))
        r.probe()
    except Exception as exc:  # noqa: BLE001 — no GL backend here
        pytest.skip(f"no MuJoCo renderer: {exc}")
    return r


def _scene_state(r, qpos):
    """`(robot_state (20,), openings)` from a full qpos, the way the client sends them."""
    import mujoco

    from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    from AG3S.ag3s.robot_models.gripper_state import openings_from_mujoco

    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in DEFAULT_RBY1_JOINTS]
    return np.asarray(qpos)[adr], openings_from_mujoco(m, d, r.convention), (m, d)


@needs_run
def test_render_mask_matches_the_offline_r6_masks():
    r = _renderer_or_skip()
    agree = []
    for fid in PARITY_FRAMES:
        with np.load(RUN / "s0/replay" / f"{fid}.npz") as z:
            depth, K, T, qpos = z["depth"], z["K"], z["T_base_cam"], z["qpos"]
        rs, op, _ = _scene_state(r, qpos)
        views = [{"name": n, "depth_m": depth[i], "K": K[i], "T_base_cam": T[i], "robot_state": rs}
                 for i, n in enumerate(CAMERA_ORDER)]
        masks, rec = r.masks(views, op)
        with np.load(RUN / "rounds/R6/robot_mask" / f"{fid}.npz") as z:
            r6 = z["robot_mask"]
        for i in range(3):
            agree.append(float((masks[i] == r6[i]).mean()))
        assert rec["pose"]["fingers"] == "measured"
    r.close()
    assert min(agree) >= 0.9999, agree


def test_render_mask_on_a_synthetic_state_matches_the_r6_procedure():
    """A synthetic arm pose: the R6 procedure (MJCF camera, full qpos, robot geoms only) on the full-scene depth vs the
    server path (robot_state + opening + K/T of the same camera)."""
    import mujoco

    r = _renderer_or_skip()
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    sc = TransportScene(str(MODEL_XML), height=480, width=640, settle_steps=0)
    sc.reset(settle_steps=0)
    rng = np.random.default_rng(7)
    for j in ("right_arm_1", "right_arm_3", "left_arm_1", "left_arm_3", "torso_1"):
        sc.data.qpos[sc._qadr[j]] += rng.uniform(-0.3, 0.3)
    sc.data.qpos[sc._qadr["gripper_finger_l1"]] = -0.03
    sc.data.qpos[sc._qadr["gripper_finger_l2"]] = 0.03
    mujoco.mj_forward(sc.model, sc.data)
    rs, op, _ = _scene_state(r, np.asarray(sc.data.qpos).copy())
    # the R6 procedure on its own model copy (ifg_render_mask.episode, per camera)
    from AG3S.ag3s.experiments.sources.mujoco_source import is_robot_body
    from AG3S.ag3s.experiments.studies.ifg_render_mask import TOL
    from AG3S.ag3s.experiments.studies.t47_replay import gripper_body_ids

    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(m)
    d.qpos[:] = sc.data.qpos
    mujoco.mj_forward(m, d)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
    grip = gripper_body_ids(m)
    rb = np.array([(i in grip or is_robot_body(names[i])) and m.body_mocapid[i] < 0 for i in range(m.nbody)])
    m.geom_group[:] = np.where(rb[m.geom_bodyid], m.geom_group, 5)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[:3] = 1
    ren = mujoco.Renderer(m, 480, 640)
    ren.enable_depth_rendering()
    agree = []
    try:
        for i, cam in enumerate(("zed_left", "wrist_cam_l", "wrist_cam_r")):
            frame = sc.capture(cam)
            ren.update_scene(d, camera=cam, scene_option=opt)
            rd = np.asarray(ren.render(), np.float64)
            off = (rd < 0.99 * m.stat.extent * m.vis.map.zfar) & (np.abs(frame.depth - rd) <= TOL)
            (mk,), _ = r.masks([{"name": cam, "depth_m": frame.depth, "K": frame.camera_intrinsics,
                                 "T_base_cam": frame.T_base_cam, "robot_state": rs}], op)
            agree.append(float((mk == off).mean()))
            assert off.sum() > 1000
    finally:
        ren.close()
        sc.close()
        r.close()
    assert min(agree) >= 0.999, agree


def test_render_intrinsics_guard():
    from AG3S.ag3s.runtime.robot_render_mask import RobotRenderMask

    K = np.array([[240.0, 0, 320], [0, 240.0, 240], [0, 0, 1]])
    assert RobotRenderMask.intrinsics_ok(K, (480, 640)) is None
    assert "principal" in RobotRenderMask.intrinsics_ok(K + np.array([[0, 0, 3.0], [0, 0, 0], [0, 0, 0]]), (480, 640))
    assert "fx" in RobotRenderMask.intrinsics_ok(np.diag([240.0, 250.0, 1.0]) + K * 0, (480, 640))


# ============================================================================ 7 constraint record


def test_record_carries_the_code_images_and_json_only_when_on(tmp_path):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    w = ConstraintRecordWriter(tmp_path / "rec", esdf_mode="none")
    cs = types.SimpleNamespace(status=None, validity=None, has_target=False, candidates=(), notes=(), profile={},
                               metrics={}, target=None, esdf=None)
    codes = {"head": np.full((4, 5), 2, np.uint8)}
    p = w.record(t_step=0, chunk_index=0, constraint_set=cs, debug={"attention_surface_codes": codes})
    p0 = w.record(t_step=1, chunk_index=1, constraint_set=cs, debug={})
    with np.load(p, allow_pickle=True) as z:
        assert np.array_equal(z["attention_surface_codes_head"], codes["head"])
    with np.load(p0, allow_pickle=True) as z:
        assert not any(k.startswith("attention_surface") for k in z.files)
    tf = {"mode": "oracle_visible", "oracle": {"applied": True}, "attention_surface": {"applied": True, "ms": {}}}
    info = ConstraintRecordWriter._write_exclusion_trace(types.SimpleNamespace(stats={"target_free": tf},
                                                                               exclusion_trace=None), {})
    assert info["target_exclusion"]["attention_surface"]["applied"] and info["target_exclusion"]["oracle"]["applied"]
    tf.pop("attention_surface")
    info = ConstraintRecordWriter._write_exclusion_trace(types.SimpleNamespace(stats={"target_free": tf},
                                                                               exclusion_trace=None), {})
    assert "oracle" not in info["target_exclusion"], "oracle_visible records are as before"
    w.close()
