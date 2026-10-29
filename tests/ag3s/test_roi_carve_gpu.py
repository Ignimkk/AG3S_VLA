"""AG3S-ROI-CARVE-GPU — the region-of-interest carve judged on the GPU (`esdf.roi_carve_device = "cuda"`).

CPU (always):
  - config default `cpu`, validation (unknown device, `cuda` without `roi_carve`); `--roi-carve-device` maps to the esdf
    key without switching the serving default off; `--no-roi-carve` with `cuda` is refused.
GPU (torch + CUDA; the builder test also needs cuRobo):
  - `_min_filter_gpu` = `scipy.ndimage.minimum_filter(mode="constant", cval=0)` for every footprint radius.
  - `seen_through_gpu` = `seen_through` voxel for voxel on random scenes (nearest pixel, footprint block / ignore, pad 0 / 1,
    robot masks, invalid and out-of-range depth, points behind and beside the cameras).
  - the synthetic builder scene of `test_task_dynamic_map`: the same freed counts and a bit-identical field on both devices.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, EsdfConfig

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


# -- CPU ------------------------------------------------------------------------------------------------------
def test_config_default_and_validation():
    assert EsdfConfig().roi_carve_device == "cpu"
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "roi_carve": True, "roi_carve_device": "cuda"}})
    assert cfg.esdf.roi_carve_device == "cuda"
    for bad in ({"roi_carve": True, "roi_carve_device": "gpu"}, {"roi_carve_device": "cuda"}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": {"backend": "curobo", **bad}})


def test_cli_device_flag_keeps_the_serving_default():
    from AG3S.trajopt import serve_safe

    sec = lambda *a: serve_safe.dynamic_field_section(serve_safe.build_parser().parse_args([*BASE_ARGV, *a]))  # noqa: E731
    assert serve_safe.ROI_CARVE_SERVING_DEFAULT["roi_carve_device"] == "cuda"     # G1–G3 passed (task §4)
    assert sec("--esdf-backend", "curobo") == serve_safe.ROI_CARVE_SERVING_DEFAULT
    assert sec("--esdf-backend", "curobo", "--roi-carve-device", "cpu") == {
        **serve_safe.ROI_CARVE_SERVING_DEFAULT, "roi_carve_device": "cpu"}          # back to the numpy judgement
    assert sec("--esdf-backend", "curobo", "--roi-carve", "--roi-carve-device", "cuda") == {
        "roi_carve": True, "roi_carve_device": "cuda"}
    assert "roi_carve_device" not in sec("--esdf-backend", "curobo", "--no-roi-carve")


def test_cli_refuses_a_device_without_the_carve(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, "--esdf-backend", "curobo", "--no-roi-carve",
                                      "--roi-carve-device", "cuda"])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2 and "--roi-carve-device" in capsys.readouterr().err


# -- GPU ------------------------------------------------------------------------------------------------------
def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():  # pragma: no cover
        pytest.skip("needs CUDA")
    return torch


def test_min_filter_matches_scipy():
    torch = _cuda()
    from scipy.ndimage import minimum_filter
    from AG3S.ag3s.fields.curobo_builder import FOOTPRINT_MAX_PX, _min_filter_gpu

    rng = np.random.default_rng(0)
    img = rng.uniform(0.2, 2.0, (37, 53))
    img[rng.uniform(size=img.shape) < 0.1] = 0.0
    img[rng.uniform(size=img.shape) < 0.05] = np.inf
    t = torch.as_tensor(img, device="cuda")
    for r in range(0, FOOTPRINT_MAX_PX + 1):
        ref = minimum_filter(img, size=2 * r + 1, mode="constant", cval=0.0)
        assert np.array_equal(_min_filter_gpu(t, r).cpu().numpy(), ref), r


def _random_cameras(rng, n=3, h=60, w=80):
    K = np.array([[90.0, 0.0, 40.0], [0.0, 90.0, 30.0], [0.0, 0.0, 1.0]])
    cams = []
    for i in range(n):
        a = rng.uniform(-0.3, 0.3, 3)
        cx, sx = np.cos(a), np.sin(a)
        Rx = np.array([[1, 0, 0], [0, cx[0], -sx[0]], [0, sx[0], cx[0]]])
        Ry = np.array([[cx[1], 0, sx[1]], [0, 1, 0], [-sx[1], 0, cx[1]]])
        Rz = np.array([[cx[2], -sx[2], 0], [sx[2], cx[2], 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = rng.uniform(-0.05, 0.05, 3)
        # a far wall with a slope, near boxes, and clustered holes (scattered ones would block every footprint)
        v, u = np.mgrid[0:h, 0:w]
        d = 1.3 + 0.002 * u + 0.001 * v
        for _ in range(4):
            r0, c0 = rng.integers(0, h - 12), rng.integers(0, w - 12)
            d[r0:r0 + rng.integers(4, 12), c0:c0 + rng.integers(4, 12)] = rng.uniform(0.4, 0.9)
        d[2:6, 60:66] = 0.0
        d[40:44, 5:9] = np.nan
        d[50:54, 70:76] = 3.0                                       # beyond depth_max
        d[rng.integers(0, h, 6), rng.integers(0, w, 6)] = 0.0       # a few single holes
        mask = np.zeros((h, w), bool)
        mask[rng.integers(0, h, 8), rng.integers(0, w, 8)] = True
        mask[10:20, 30:45] = True
        cams.append(types.SimpleNamespace(name=f"c{i}", depth=d, camera_intrinsics=K, T_base_cam=T,
                                          robot_mask=None if i == 2 else mask))
    return cams


def _random_points(rng, cams, n=4000):
    T = cams[0].T_base_cam
    pc = np.stack([rng.uniform(-0.8, 0.8, n), rng.uniform(-0.6, 0.6, n), rng.uniform(-0.2, 1.6, n)], 1)
    return pc @ T[:3, :3].T + T[:3, 3]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seen_through_gpu_matches_the_numpy_judgement(seed):
    torch = _cuda()
    from AG3S.ag3s.fields.curobo_builder import gpu_cameras, seen_through, seen_through_gpu

    rng = np.random.default_rng(seed)
    cams = _random_cameras(rng)
    p = _random_points(rng, cams)
    g = gpu_cameras(cams, "cuda", 0.1, 2.0)
    pt = torch.as_tensor(p, device="cuda")
    for kw in ({}, {"footprint_voxel": 0.005}, {"footprint_voxel": 0.02, "footprint_pad_px": 0},
               {"footprint_voxel": 0.005, "footprint_robot": "ignore"},
               {"footprint_voxel": 0.005, "footprint_pad_px": 0, "footprint_robot": "ignore"}):
        ref, per_ref = seen_through(p, cams, 0.03, depth_min=0.1, depth_max=2.0, **kw)
        got, per = seen_through_gpu(pt, g, 0.03, **kw)
        assert ref.any() and (~ref).any(), kw
        assert np.array_equal(got.cpu().numpy(), ref), (kw, int((got.cpu().numpy() != ref).sum()))
        assert {k: int(v) for k, v in per.items()} == per_ref, kw


def test_builder_cpu_and_cuda_carve_the_same_field():
    _cuda()
    pytest.importorskip("curobo")
    from tests.ag3s import test_task_dynamic_map as tdm

    out = {}
    for device in ("cpu", "cuda"):
        b = tdm._builder(roi_carve=True, roi_carve_footprint=True, roi_carve_footprint_pad_px=0,
                         roi_carve_footprint_robot="ignore", roi_carve_device=device)
        for _ in range(6):
            tdm._update(b, [tdm.TABLE, tdm.TARGET, tdm.NEIGHBOUR, tdm.FAR], layer=False, roi=True)
        f = tdm._update(b, [tdm.TABLE, tdm.TARGET], layer=False, roi=True)
        st = f.stats["roi_carve"]
        probe = np.vstack([tdm._top(tdm.NEIGHBOUR), tdm._top(tdm.FAR), tdm._top(tdm.TARGET)])
        out[device] = ({k: (st[k]["n_roi_occupied"], st[k]["n_freed"], st[k]["per_camera"]) for k in ("coarse_tsdf", "fine_tsdf")},
                       np.asarray(f.distance(probe)), st["coarse_tsdf"].get("device"))
    assert out["cpu"][0] == out["cuda"][0]
    assert out["cpu"][0]["coarse_tsdf"][1] + out["cpu"][0]["fine_tsdf"][1] > 0
    assert np.array_equal(out["cpu"][1], out["cuda"][1])
    assert out["cpu"][2] is None and out["cuda"][2] == "cuda"
