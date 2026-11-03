"""T78 B — the attention-surface target extent on the GPU (`runtime.attention_surface_gpu`), default off.

| # | property |
|---|---|
| 1 | off = unchanged: `config.ATTENTION_SURFACE_DEVICE == "cpu"`, the default call is the explicit `device="cpu"` call (same codes, nodes, record — no new key) and never imports the GPU module; a bad device name is refused |
| 2 | `device="cuda"` with `keep_debug` / `class_map` stays on the CPU chain (same selection), and says so in the record; the pipeline passes `config.ATTENTION_SURFACE_DEVICE` |
| 3 | serve_safe `--attention-surface-device`: absent = no config change, no record-meta key; `cuda` needs `--target-extent attention-surface` (refused otherwise / with `--no-safe`); applying it sets the config |
| 4 | GPU (skipped without CUDA): synthetic table + target + touching-free neighbour — same selected nodes (IoU 1), code images ≥ 0.99 equal, every CPU record key present with the same counts; `codes=False`; abstain reasons; the support planes = `fit_support_surfaces` |
| 5 | GPU + storage (skipped otherwise): T76 SURF requests — selected-node IoU ≥ 0.99, code agreement ≥ 0.99 |
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from AG3S.ag3s import config as ag3s_config
from AG3S.ag3s.runtime.attention_surface import (
    ARM_SPHERES, CAMERA_ORDER, SurfaceView, graph_config, select_target_surface)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:   # noqa: BLE001 — torch missing in the CPU venv
        return False


needs_cuda = pytest.mark.skipif(not _cuda(), reason="no CUDA (torch) in this interpreter")


# ------------------------------------------------------------------------------------------- synthetic scene


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


TARGET = np.array([0.55, 0.05, 0.035])
NEIGHBOUR = np.array([0.55, 0.15, 0.035])
R = 0.035


def _raycast(T, hw=(240, 320), f=260.0):
    """Depth (z in the camera frame) of a table z = 0 and two spheres; labels 1 = target, 3 = neighbour, 2 = table."""
    h, w = hw
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    v, u = np.mgrid[0:h, 0:w].astype(float)
    dc = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u)], -1)   # z = 1 rays (camera)
    db = dc @ T[:3, :3].T
    o = T[:3, 3]
    tbest = np.full((h, w), np.inf)
    lab = np.zeros((h, w), np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = -o[2] / db[..., 2]
    ok = np.isfinite(tp) & (tp > 0)
    tbest[ok] = tp[ok]
    lab[ok] = 2
    for c, code in ((TARGET, 1), (NEIGHBOUR, 3)):
        oc = o - c
        a = np.sum(db * db, -1)
        b = 2 * np.sum(db * oc, -1)
        cc = np.sum(oc * oc) - R * R
        disc = b * b - 4 * a * cc
        hit = disc >= 0
        t = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
        better = hit & (t > 0) & (t < tbest)
        tbest[better] = t[better]
        lab[better] = code
    depth = np.where(np.isfinite(tbest), tbest, 0.0)   # z-depth: rays have unit z in the camera frame
    return depth, K, lab


def _scene(att=True, hw=(240, 320)):
    eyes = (TARGET + [0.0, -0.05, 0.50], TARGET + [-0.40, -0.15, 0.30], TARGET + [-0.35, 0.20, 0.35])
    views, labels = [], {}
    h, w = hw
    for n, eye in zip(CAMERA_ORDER, eyes):
        T = _look_at(eye, TARGET)
        d, K, lab = _raycast(T, hw)
        cov = (lab == 1).reshape(16, h // 16, 16, w // 16).mean(axis=(1, 3))
        rm = np.zeros(hw, bool)
        rm[:12, :] = True   # a band of "robot" pixels
        views.append(SurfaceView(n, d, K, T, rm, cov if att else None))
        labels[n] = lab
    return views, labels


def _iou(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return 1.0 if not (a or b) else len(a & b) / len(a | b)


# ============================================================================ 1 off = unchanged


def test_default_device_is_cpu_and_the_default_call_is_unchanged():
    assert ag3s_config.ATTENTION_SURFACE_DEVICE == "cpu"
    views, _ = _scene()
    a = select_target_surface(views, arm=ARM_SPHERES)
    b = select_target_surface(views, arm=ARM_SPHERES, device="cpu")
    assert a.record["applied"], a.record
    assert "device" not in a.record and "ms_detail" not in a.record
    ka = {k: v for k, v in a.record.items() if k != "ms"}
    kb = {k: v for k, v in b.record.items() if k != "ms"}
    assert ka == kb and set(a.record["ms"]) == {"build", "solve", "codes", "total"}
    assert np.array_equal(a.selected_nodes, b.selected_nodes)
    assert all(np.array_equal(a.codes[n], b.codes[n]) for n in CAMERA_ORDER)


def test_default_path_never_imports_the_gpu_module():
    code = ("import sys; from tests.ag3s.test_t78b_attention_surface_gpu import _scene; "
            "from AG3S.ag3s.runtime.attention_surface import select_target_surface; "
            "r = select_target_surface(_scene()[0]); "
            "print(r.record['applied'], 'AG3S.ag3s.runtime.attention_surface_gpu' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.stdout.split()[-2:] == ["True", "False"], out.stdout + out.stderr


def test_bad_device_is_refused():
    views, _ = _scene()
    with pytest.raises(ValueError, match="device"):
        select_target_surface(views, device="tpu")


# ============================================================================ 2 CPU fallbacks under "cuda"


def test_cuda_with_keep_debug_or_class_map_stays_on_the_cpu_chain():
    views, labels = _scene()
    ref = select_target_surface(views, arm=ARM_SPHERES, keep_debug=True)
    r = select_target_surface(views, arm=ARM_SPHERES, keep_debug=True, device="cuda")
    assert r.record["device"] == "cpu" and r.record["device_fallback"] == "keep_debug"
    assert r.geometry is not None and np.array_equal(r.selected_nodes, ref.selected_nodes)
    cm = [labels[n].astype(np.int64) for n in CAMERA_ORDER]
    r = select_target_surface(views, arm=ARM_SPHERES, class_map=cm, classes=("none", "target", "support", "other"),
                              device="cuda")
    assert r.record["device_fallback"] == "class_map" and np.array_equal(r.selected_nodes, ref.selected_nodes)


@pytest.mark.parametrize("configured", ["cpu", "cuda"])
def test_pipeline_passes_the_configured_device(monkeypatch, configured):
    """`pipeline._attention_surface_update` hands `config.ATTENTION_SURFACE_DEVICE` to the selection (the call is
    re-run on the CPU here so the test needs no GPU)."""
    from AG3S.ag3s.runtime import attention_surface as AS

    cl1 = pytest.importorskip("tests.ag3s.test_ifg_cl1_attention_surface")
    fw, ag3s = cl1._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface")
    obs, _, _ = cl1._raycast_rig(fw)
    cams = cl1._cameras(obs)
    seen, orig = [], AS.select_target_surface

    def spy(views, **kw):
        seen.append(kw.get("device"))
        return orig(views, **{**kw, "device": "cpu"})

    monkeypatch.setattr(AS, "select_target_surface", spy)
    monkeypatch.setattr(ag3s_config, "ATTENTION_SURFACE_DEVICE", configured)
    rec = ag3s._attention_surface_update(obs, cams, {c.name: c.robot_mask for c in cams}, {})
    assert seen == [configured] and rec["segmentation"]


# ============================================================================ 3 serve_safe flag


def _args(*argv):
    from AG3S.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])
    reject_bad_flag_combinations(ap, args)
    return args


SURF_ON = ("--esdf-backend", "curobo", "--target-exclusion", "ball-owned", "--target-extent", "attention-surface",
           "--target-field-policy", "exclude-authorized", "--attention-source", "noun_target:L10H7")


def test_cli_flag_absent_changes_nothing():
    from AG3S.trajopt.serve_safe import apply_attention_surface_device, ifg_cl1_record_meta

    for argv in ((), SURF_ON):
        args = _args(*argv)
        assert args.attention_surface_device is None
        assert "attention_surface_device" not in ifg_cl1_record_meta(args)
        assert apply_attention_surface_device(args, warm=False) is None
        assert ag3s_config.ATTENTION_SURFACE_DEVICE == "cpu"


def test_cli_flag_cuda_needs_attention_surface(capsys):
    for argv in (("--attention-surface-device", "cuda"),
                 ("--esdf-backend", "curobo", "--target-exclusion", "ball-owned", "--target-extent",
                  "attention-surface-hybrid", "--attention-source", "noun_target:L10H7", "--attention-surface-device",
                  "cuda")):
        with pytest.raises(SystemExit) as e:
            _args(*argv)
        assert e.value.code == 2 and "--attention-surface-device" in capsys.readouterr().err
    _args("--attention-surface-device", "cpu")   # the default, explicitly: allowed anywhere


def test_cli_flag_cuda_sets_the_config_and_the_record_meta(monkeypatch):
    from AG3S.trajopt.serve_safe import apply_attention_surface_device, ifg_cl1_record_meta

    monkeypatch.setattr(ag3s_config, "ATTENTION_SURFACE_DEVICE", "cpu")
    args = _args(*SURF_ON, "--attention-surface-device", "cuda")
    assert ifg_cl1_record_meta(args)["attention_surface_device"] == "cuda"
    assert apply_attention_surface_device(args, warm=False) == "cuda"
    assert ag3s_config.ATTENTION_SURFACE_DEVICE == "cuda"


# ============================================================================ 4 GPU on the synthetic scene


@needs_cuda
def test_gpu_selects_the_same_nodes_and_codes():
    views, labels = _scene()
    c = select_target_surface(views, arm=ARM_SPHERES, device="cpu")
    g = select_target_surface(views, arm=ARM_SPHERES, device="cuda")
    assert c.record["applied"] and g.record["applied"], (c.record, g.record)
    assert g.record["device"] == "cuda" and set(g.record["ms"]) == {"build", "solve", "codes", "total"}
    assert set(c.record) - {"pixel_node_keys"} <= set(g.record)
    for k in ("n_nodes", "n_support_nodes", "n_robot_px", "n_eligible_patches", "seed_patches", "n_roi",
              "n_unknown_nodes", "n_region_nodes_m05", "n_selected_nodes", "n_target_px", "n_labelled_px"):
        assert c.record[k] == g.record[k], k
    np.testing.assert_allclose(g.record["selected_centroid"], c.record["selected_centroid"], atol=1e-4)
    assert _iou(c.selected_nodes, g.selected_nodes) >= 0.99
    agree = np.mean([(c.codes[n] == g.codes[n]).mean() for n in CAMERA_ORDER])
    assert agree >= 0.99
    # and the selection is the target (sanity of the scene, not of the port)
    tgt = sum(int(((g.codes[n] == 1) & (labels[n] == 1)).sum()) for n in CAMERA_ORDER)
    assert tgt / sum(int((g.codes[n] == 1).sum()) for n in CAMERA_ORDER) > 0.9


@needs_cuda
def test_gpu_codes_false_and_abstain():
    views, _ = _scene()
    c = select_target_surface(views, arm=ARM_SPHERES, device="cpu", codes=False)
    g = select_target_surface(views, arm=ARM_SPHERES, device="cuda", codes=False)
    assert g.codes is None and g.record["applied"] and g.record["codes_skipped"]
    assert np.array_equal(c.selected_nodes, g.selected_nodes)
    zero = [SurfaceView(v.name, v.depth_m, v.K, v.T_base_cam, v.robot_mask, np.zeros((16, 16))) for v in views]
    for vv in (zero,):
        rc = select_target_surface(vv, device="cpu")
        rg = select_target_surface(vv, device="cuda")
        assert rc.record["why"] == rg.record["why"] == "no_seed"
    empty = [SurfaceView(v.name, np.zeros_like(v.depth_m), v.K, v.T_base_cam, v.robot_mask, v.attention)
             for v in views]
    rg = select_target_surface(empty, device="cuda")
    assert rg.codes is None and rg.record["why"].startswith("no_depth_points")
    nocam = select_target_surface(views[:2], device="cuda")
    assert nocam.record["why"] == "missing_camera:right_wrist"


@needs_cuda
def test_gpu_support_planes_match_fit_support_surfaces():
    import torch

    from AG3S.ag3s.runtime import attention_surface_gpu as GPU
    from AG3S.ag3s.stages.support_surface import fit_support_surfaces
    from AG3S.ag3s.types import PointCloud

    rng = np.random.default_rng(3)
    table = np.c_[rng.uniform(0, 1, (20000, 2)), rng.normal(0, 0.001, 20000)]
    shelf = np.c_[rng.uniform(0, 0.5, (6000, 2)), 0.4 + rng.normal(0, 0.001, 6000)]
    clutter = rng.uniform([0.2, 0.2, 0.02], [0.4, 0.4, 0.2], (3000, 3))
    pts = np.concatenate([table, clutter, shelf])
    _, ref = fit_support_surfaces(PointCloud(pts))
    dev = GPU.cuda_device()
    got = GPU.support_mask(torch.as_tensor(pts, device=dev), dev).cpu().numpy()
    assert ref.sum() > 20000 and (got == ref).mean() >= 0.9999, (int(ref.sum()), int(got.sum()))


# ============================================================================ 5 GPU on T76 requests (storage)


T76 = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage")) / (
    "results/development/20261030T170909+0900_8ab3c7ef7757_9e05cdff58b9/verify/T76/SURF")
needs_t76 = pytest.mark.skipif(not (T76 / "REG_k0/replay.json").is_file(), reason="T76 records not mounted")
CAMS = (("head", "zed_left"), ("left_wrist", "wrist_cam_l"), ("right_wrist", "wrist_cam_r"))


def _t76_views(group: str, k: int):
    rep = json.loads((T76 / group / "replay.json").read_text())[0]
    steps = sorted((pathlib.Path(rep["client_dir"]) / "ag3s/run_0000").glob("step_*.npz"),
                   key=lambda p: int(np.load(p)["t_step"]))
    z = np.load(sorted((T76 / group / "server_constraints/run_0000").glob("chunk_*.npz"))[k], allow_pickle=True)
    s = np.load(steps[k], allow_pickle=True)
    views = []
    for name, mj in CAMS:
        codes = z[f"attention_surface_codes_{name}"]
        views.append(SurfaceView(name, np.asarray(s[f"depth_{mj}"], np.float64) / 1000.0, s[f"K_{mj}"],
                                 s[f"T_base_cam_{mj}"], np.asarray(codes) == 0,
                                 np.asarray(z[f"attention_{name}"], np.float64)))
    return views


@needs_cuda
@needs_t76
@pytest.mark.parametrize("group,k", [("REG_k0", 5), ("REG_k0", 30), ("OCC", 10)])
def test_gpu_matches_cpu_on_t76_requests(group, k):
    views = _t76_views(group, k)
    c = select_target_surface(views, arm=ARM_SPHERES, device="cpu")
    g = select_target_surface(views, arm=ARM_SPHERES, device="cuda")
    assert c.record["applied"] == g.record["applied"]
    assert _iou(c.selected_nodes, g.selected_nodes) >= 0.99
    if c.codes is not None:
        assert np.mean([(c.codes[n] == g.codes[n]).mean() for n in CAMERA_ORDER]) >= 0.99
