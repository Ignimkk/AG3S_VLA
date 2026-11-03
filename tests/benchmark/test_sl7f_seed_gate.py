"""SL7f — the ATS gate of the attention-surface seed map (`--surface-seed-ats-gate`, `esdf.surface_seed_ats_gate`,
`attention_surface_post.ats_gate_views`), default off; Panda versions `AG3S-final-panda-v3-e0to-pc-surface-gate` and
`AG3S-final-panda-p0m-gate` (task `AG3S/ag3s/docs/handoff/SL7f.task.md`)."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)


def _view(name="head", attention=None):
    from AG3S.ag3s.runtime.attention_surface import SurfaceView

    K = np.array([[100.0, 0, 128.0], [0, 100.0, 128.0], [0, 0, 1]])
    return SurfaceView(name=name, depth_m=np.ones((256, 256)), K=K, T_base_cam=np.eye(4),
                       robot_mask=np.zeros((256, 256), bool),
                       attention=np.ones((16, 16), np.float32) if attention is None else attention)


def test_gate_keeps_only_the_patches_the_points_hit():
    from AG3S.ag3s.runtime.attention_surface_post import ats_gate_views

    # camera frame = base frame; a point at (0, 0, 1) projects to pixel (128, 128) -> patch (8, 8); (0.5, 0, 1) -> u 178 -> col 11
    pts = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [0.0, 0.0, -1.0]])   # the last one is behind the camera
    (v,), rec = ats_gate_views([_view()], pts)
    a = np.asarray(v.attention)
    assert rec["applied"] and rec["cameras"]["head"]["n_patches"] == 2 and rec["cameras"]["head"]["n_points_in_view"] == 2
    assert set(zip(*np.nonzero(a))) == {(8, 8), (8, 11)} and a[8, 8] == 1.0
    assert rec["cameras"]["head"]["attention_mass_kept"] == round(2 / 256, 4)


def test_gate_leaves_views_alone_without_points_or_attention():
    from AG3S.ag3s.runtime.attention_surface_post import ats_gate_views

    views = [_view(), _view("right_wrist", attention=np.zeros((16, 16), np.float32))]
    out, rec = ats_gate_views(views, None)
    assert out == views and rec == {"applied": False, "n_ats_points": 0, "cameras": {}, "why": "no_ats_points"}
    out, rec = ats_gate_views(views, np.array([[0.0, 0.0, 1.0]]))
    assert out[1] is views[1] and "right_wrist" not in rec["cameras"]          # zero map (the Panda dummy) untouched
    out, rec = ats_gate_views([_view()], np.array([[50.0, 0.0, 1.0]]))         # projects outside the image
    assert out[0].attention is not None and np.all(np.asarray(out[0].attention) == 1.0) and not rec["applied"]


def test_flag_is_refused_without_the_surface_extent_and_off_by_default():
    from AG3S.ag3s.config import AG3SConfigError, EsdfConfig
    from AG3S.trajopt import serve_safe as ss

    assert EsdfConfig().surface_seed_ats_gate is False
    with pytest.raises(AG3SConfigError):
        EsdfConfig(surface_seed_ats_gate=True).validate()
    ap = ss.build_parser()
    base = ["--config", "c", "--checkpoint", "k", "--model-xml", "m"]
    args = ap.parse_args(base + ["--surface-seed-ats-gate"])
    assert args.surface_seed_ats_gate is True and ap.parse_args(base).surface_seed_ats_gate is False
    assert ss.target_exclusion_section(ap.parse_args(base)).get("surface_seed_ats_gate") is None
    assert ss.target_exclusion_section(args)["surface_seed_ats_gate"] is True


def test_pc_surface_gate_builds_with_the_gate(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    if not pathlib.Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")
    v = "AG3S-final-panda-v3-e0to-pc-surface-gate"
    assert list(pf.AG3S_VERSIONS[v]) == list(pf.AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-surface"]) + ["--surface-seed-ats-gate"]
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(v)
    argv[argv.index("--config") + 1] = "pi05_libero"
    args = pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(v))
    ps.check_unported_flags(args, None)
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    e = served._policy.ag3s.config.esdf
    assert e.target_extent == "attention_surface" and e.surface_seed_ats_gate is True
    # the earlier versions keep the gate off
    for old in ("AG3S-final-panda-v3-e0to-pc-surface", "AG3S-final-panda-p0m"):
        assert "--surface-seed-ats-gate" not in pf.AG3S_VERSIONS[old]
    assert list(pf.AG3S_VERSIONS["AG3S-final-panda-p0m-gate"]) == list(pf.AG3S_VERSIONS["AG3S-final-panda-p0m"]) + [
        "--surface-seed-ats-gate"]
