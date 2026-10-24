"""Filter arms through the ``.venv-baselines`` sidecar: pipe round trip, identity path, intervention, wire level.

Run from ``.venv-openpi-live`` (the server side); the sidecar is spawned with ``.venv-baselines/bin/python``::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-openpi-live/bin/python -m pytest -q \
        Baseline/paper_baselines/closed_loop/tests/test_sidecar_filters.py
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.closed_loop.replay import readonly

from Baseline.paper_baselines.closed_loop.wire_obs import obs_to_dict
from Baseline.paper_baselines.methods.interface import SceneObservation

ROOT = pathlib.Path(__file__).resolve().parents[4]
PY = ROOT / ".venv-baselines/bin/python"
STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", ROOT.parent / "AG3S_VLA-storage"))
SL5C_LI_VLA = STORAGE / "results/benchmark/20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only"
pytestmark = pytest.mark.skipif(not PY.exists(), reason=".venv-baselines missing")

EE_OFFSET_LOCAL = np.array([0.0, 0.0, -0.08])  # = methods/aegis/adapter.py (AEGIS main_aegis.py:149-164)
Q1_DEFAULT = np.array([0.06, 0.12, 0.11])


def synth_obs(obstacles, extras=None):
    ee = np.eye(4)
    ee[:3, 3] = [0.45, 0.0, 0.30]
    ex = {"obstacle_ellipsoids": obstacles, **(extras or {})}
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7), ee_pose=ee, gripper=0.08,
                            instruction="pick up the bowl", extras=ex)


def sphere_ahead(gap_x):  # = methods/aegis/tests/test_aegis_filter.py
    p1 = np.array([0.45, 0.0, 0.30]) + EE_OFFSET_LOCAL
    return [{"name": "ball", "center": p1 + [Q1_DEFAULT[0] + 0.05 + gap_x, 0, 0], "semi_axes": [0.05] * 3}]


def straight(direction, T=10):
    A = np.zeros((T, 7))
    A[:, :3] = direction
    A[:, 6] = -1.0
    return A


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory):
    from Baseline.paper_baselines.closed_loop.sidecar import SidecarClient

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    sc = SidecarClient(PY, log_path=tmp_path_factory.mktemp("sc") / "sidecar.log", env=env)
    yield sc
    assert sc.close() == 0


def test_sidecar_is_the_baselines_interpreter(sidecar):
    v = sidecar.call("ping")["versions"]
    assert v["numpy"].startswith("2.") and v["cvxpy"] is not None and ".venv-baselines" in v["executable"]


def test_aegis_through_the_pipe(sidecar):
    sidecar.call("init", method="aegis", kwargs={"obstacle_source": "ellipsoids"})
    A = straight([1.0, 0, 0])
    r = sidecar.call("filter", obs=obs_to_dict(synth_obs(sphere_ahead(0.05))), actions=A)
    assert r["actions"].shape == A.shape and np.asarray(r["intervened"]).all()
    assert np.all(r["actions"][:, 0] < 1.0) and np.array_equal(r["actions"][:, 6], A[:, 6])
    sidecar.call("reset")
    S = straight([-1.0, 0, 0])
    r = sidecar.call("filter", obs=obs_to_dict(synth_obs(sphere_ahead(0.05))), actions=S)
    assert r["actions"].tobytes() == S.tobytes() and not np.asarray(r["intervened"]).any()  # identity, bit for bit
    sidecar.call("reset")
    r = sidecar.call("filter", obs=obs_to_dict(synth_obs([])), actions=A)  # no obstacle -> identity
    assert r["actions"].tobytes() == A.tobytes() and r["info"]["flag_safety_control"] is False


@pytest.mark.parametrize("method", ["knows", "multilink_cbf"])
def test_other_filters_identity_without_obstacles_and_run_with_one(sidecar, method):
    sidecar.call("init", method=method, kwargs={"obstacle_source": "given"})
    A = straight([1.0, 0, 0])
    r = sidecar.call("filter", obs=obs_to_dict(synth_obs([], {"obstacle_points": {}})), actions=A)
    assert r["actions"].tobytes() == A.tobytes() and not np.asarray(r["intervened"]).any()
    sidecar.call("reset")
    q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])  # Panda ready pose; obstacle 12 cm in +x of the hand
    from AG3S.adapters.panda import model as pm

    C, _R = pm.build_panda_bare_model().sphere_centers_numeric(q)
    tip = np.asarray(C)[np.argmax(np.asarray(C)[:, 0])]
    obs = synth_obs([], {"obstacle_points": {"ball": None}})
    obs.q, obs.ee_pose[:3, 3] = q, tip
    rng = np.random.default_rng(0)
    ball = rng.normal(size=(300, 3))
    obs.extras["obstacle_points"]["ball"] = tip + [0.14, 0, 0] + 0.04 * ball / np.linalg.norm(ball, axis=1, keepdims=True)
    obs.extras["robot_state"] = np.concatenate([q, [0.04, -0.04]])
    r = sidecar.call("filter", obs=obs_to_dict(obs), actions=A)
    assert np.all(np.isfinite(r["actions"])) and np.asarray(r["intervened"]).any()


def test_sidecar_error_is_reported_not_fatal(sidecar):
    from Baseline.paper_baselines.closed_loop.sidecar import SidecarError

    sidecar.call("init", method="aegis", kwargs={"obstacle_source": "ellipsoids"})
    with pytest.raises(SidecarError):
        sidecar.call("filter", obs=obs_to_dict(synth_obs(sphere_ahead(0.05))), actions=np.zeros((10, 5)))
    assert sidecar.call("ping")["versions"]  # still alive


@pytest.mark.skipif(not SL5C_LI_VLA.is_dir(), reason="SL5c VLA-only run not available")
def test_wire_filter_arm_on_recorded_calls(tmp_path):
    import json

    pytest.importorskip("openpi_client")
    pytest.importorskip("jax")
    from Baseline.paper_baselines.closed_loop import serve_method as S
    from Baseline.paper_baselines.closed_loop.replay import RecordedEpisode

    ep = RecordedEpisode(SL5C_LI_VLA, "safelibero_spatial_t0_e0")
    args = S.build_parser().parse_args(["--method", "aegis", "--obstacles", "depth", "--target", "none", "--port", "0",
                                        "--fake-policy", "recorded", "--fake-source-run", str(SL5C_LI_VLA),
                                        "--sidecar-python", str(PY)])
    srv, info = S.build_server(args, record_root=tmp_path)
    try:
        for k in range(3):
            out = srv.infer(readonly(ep.request(k)))
            assert np.asarray(out["actions"]).shape == (10, 7) and out["panda/mode"] == "method:aegis"
    finally:
        srv.close()
    rows = [json.loads(l) for l in (tmp_path / "method_calls/calls.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["fallback"] is None for r in rows), [r.get("error") for r in rows]
    assert {"intervened", "da_xyz_l2_max", "h_min", "h_steps", "qp_status", "perception"} <= set(rows[0])
    assert info["provenance"]["obstacle"]["obstacle_source"] == "given" and not info["arm"]["privileged"]
