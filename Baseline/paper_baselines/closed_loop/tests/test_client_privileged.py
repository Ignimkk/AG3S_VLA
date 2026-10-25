"""``libero_safe_client.py --emit-privileged-obstacles`` (BASELINE-BENCH-READY): off = byte-identical client, on = only
``ag3s/privileged/*`` added; the privileged keys round-trip wire -> server extras -> sidecar -> methods; ``--gt`` arms.

Run from ``.venv-openpi-live`` (openpi msgpack, the server side). The client itself runs in ``.venv-libero`` (Python 3.8)
through ``client_harness.py`` with a fake env / geometry / websocket::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD:$PWD/third_party/openpi/src \
        .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_client_privileged.py
"""
from __future__ import annotations

import json
import os
import pathlib
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
PY38 = ROOT / ".venv-libero/bin/python"
HARNESS = pathlib.Path(__file__).with_name("client_harness.py")
#: the client before BASELINE-BENCH-READY (HEAD 3412861c = CLOSED-LOOP-FIX V5)
PRE = "3412861ca8399fc147df037a51db60bbecca455b"
STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", ROOT.parent / "AG3S_VLA-storage"))
SL5C_LI_VLA = STORAGE / "results/benchmark/20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only"
T_WORLD_BASE = np.array([[1, 0, 0, -0.66], [0, 1, 0, 0.0], [0, 0, 1, 0.912], [0, 0, 0, 1.0]])  # = client_harness


def harness(tmp, source, mode="vla_only", emit=False, seg=False) -> dict:
    if not PY38.exists():
        pytest.skip(".venv-libero missing")
    if source != "worktree" and subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", source],
                                               capture_output=True).returncode:
        pytest.skip(f"commit {source} not in this checkout")
    sys.path.insert(0, str(ROOT))
    from benchmark.adapters.libero_panda.client_env import client_environment

    env = client_environment(dict(os.environ), ROOT)
    env["OMP_NUM_THREADS"] = "1"
    argv = [str(PY38), str(HARNESS), "--source", source, "--mode", mode, "--out", str(tmp)]
    argv += ["--emit"] if emit else []
    argv += ["--seg"] if seg else []
    out = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def read_requests(path: pathlib.Path) -> list[dict]:
    from openpi_client import msgpack_numpy

    data, out, i = path.read_bytes(), [], 0
    while i < len(data):
        (n,) = struct.unpack("<Q", data[i:i + 8])
        out.append(msgpack_numpy.unpackb(data[i + 8:i + 8 + n]))
        i += 8 + n
    return out


# --- 1. flag off: byte-identical to the pre-change client --------------------------------------------------------
@pytest.mark.parametrize("mode,seg", [("vla_only", False), ("vla_v11ab", False), ("vla_v11ab", True)])
def test_flag_off_is_byte_identical_to_the_pre_change_client(tmp_path, mode, seg):
    pre = harness(tmp_path / "pre", PRE, mode, seg=seg)
    new = harness(tmp_path / "new", "worktree", mode, seg=seg)
    assert pre["python"].startswith("3.8")
    # every request: same keys in the same order, same msgpack bytes per key and for the whole request
    assert new["requests"] == pre["requests"]
    assert len(new["requests"]) == 6 and all(not k.startswith("ag3s/privileged/") for r in new["requests"]
                                             for k in r["keys"])
    assert new["save"] == pre["save"]
    assert new["calls_jsonl_sha"] == pre["calls_jsonl_sha"]  # the per-call log, byte for byte
    assert new["calls_npz"] == pre["calls_npz"]              # every array of _calls.npz (dtype, shape, bytes)
    a, b = pre["parsed_sl5c_argv"], dict(new["parsed_sl5c_argv"])
    assert b.pop("emit_privileged_obstacles") is False and b == a  # the SL5c argv parses to the same arguments


def test_flag_on_only_adds_privileged_keys(tmp_path):
    off = harness(tmp_path / "off", "worktree", "vla_v11ab", seg=True)
    on = harness(tmp_path / "on", "worktree", "vla_v11ab", emit=True, seg=True)
    for r0, r1 in zip(off["requests"], on["requests"]):
        added = [k for k in r1["keys"] if k not in r0["keys"]]
        assert [k for k in r1["keys"] if k in r0["keys"]] == r0["keys"]  # order of the existing keys unchanged
        assert added == ["ag3s/privileged/frame", "ag3s/privileged/obstacles", "ag3s/privileged/monitor_obstacle",
                         "ag3s/privileged/segmentation_geom/head", "ag3s/privileged/segmentation_geom/left_wrist"]
        assert all(r1["sha"][k] == r0["sha"][k] for k in r0["keys"])
    assert on["calls_jsonl_sha"] == off["calls_jsonl_sha"] and on["calls_npz"] == off["calls_npz"]


# --- 2. round trip: wire -> server extras -> sidecar framing -> methods ---------------------------------------------
@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    pytest.importorskip("openpi_client")
    tmp = tmp_path_factory.mktemp("emit")
    harness(tmp, "worktree", "vla_only", emit=True, seg=True)
    return read_requests(tmp / "requests.msgpack"), json.loads((tmp / "truth.json").read_text())


def test_privileged_keys_round_trip_to_the_extras(emitted):
    from Baseline.paper_baselines.closed_loop import privileged as P

    reqs, truth = emitted
    req = reqs[0]
    assert req["ag3s/privileged/monitor_obstacle"] == "moka_pot_obstacle_1"
    ex = P.extras(req, req["panda/T_world_base"], shape="collision")
    T_bw = np.linalg.inv(T_WORLD_BASE)
    assert ex["privileged/obstacle_names"] == ["moka_pot_1"]
    assert [o["name"] for o in ex["privileged/obstacles"] if o["active"]] == truth["active"]
    assert ex["privileged/instances_geom_ids"] == {"moka_pot_1": truth["moka_geom_ids"]}
    prims = ex["privileged/obstacle_geoms"]
    assert [p["name"] for p in prims] == [f"moka_pot_obstacle_1_g{k}" for k in (1, 2)]  # collision boxes only
    for p, gid in zip(prims, truth["moka_collision_ids"]):
        assert p["type"] == "box" and np.allclose(p["size"], truth["geom_size"][gid])
        assert np.allclose(p["pos"], T_bw[:3, :3] @ np.array(truth["geom_xpos"][gid]) + T_bw[:3, 3], atol=1e-12)
        assert np.allclose(p["rot"], np.array(truth["geom_xmat"][gid]).reshape(3, 3), atol=1e-12)
    seg = ex["privileged/segmentation_geom"]
    assert sorted(seg) == ["head", "left_wrist"] and seg["head"].dtype == np.int16 and (seg["head"] == 2).sum() == 64
    # shapes: the repo rule adds the visual mesh as a box; the sphere encloses the collision boxes
    repo = P.obstacle_geoms(ex["privileged/obstacles"], "repo")
    assert [p["name"] for p in repo] == [f"moka_pot_obstacle_1_g{k}" for k in (0, 1, 2)]
    (sph,) = P.obstacle_geoms(ex["privileged/obstacles"], "sphere")
    corners = np.concatenate([P._box_corners(g) for g in ex["privileged/obstacles"][0]["geoms"] if g["collision"]])
    assert np.linalg.norm(corners - sph["pos"], axis=1).max() <= sph["size"][0] + 1e-12
    # sidecar framing (numpy 1 <-> 2 pipe): bit-identical
    from Baseline.paper_baselines.closed_loop.framing import pack, unpack

    back = unpack(pack(ex))
    assert back["privileged/obstacle_names"] == ex["privileged/obstacle_names"]
    assert all(np.array_equal(back["privileged/segmentation_geom"][c], seg[c]) and
               back["privileged/segmentation_geom"][c].dtype == seg[c].dtype for c in seg)
    for p, q in zip(prims, back["privileged/obstacle_geoms"]):
        assert np.asarray(p["pos"]).tobytes() == np.asarray(q["pos"]).tobytes()
    with pytest.raises(KeyError, match="emit-privileged-obstacles"):
        P.extras({k: v for k, v in req.items() if not k.startswith("ag3s/privileged/")}, T_WORLD_BASE)


def test_policy_input_is_unchanged_by_the_privileged_keys(emitted):
    pytest.importorskip("jax")
    from AG3S.adapters.panda.server import PandaVLAOnly

    seen = []

    class Rec:
        def infer(self, obs, **kw):
            seen.append(obs)
            return {"actions": np.zeros((10, 7))}

    req = emitted[0][0]
    PandaVLAOnly(Rec()).infer(req)
    PandaVLAOnly(Rec()).infer({k: v for k, v in req.items() if not k.startswith("ag3s/privileged/")})
    assert sorted(seen[0]) == sorted(seen[1]) and not any(k.startswith("ag3s/") for k in seen[0])


def test_methods_read_the_privileged_extras(emitted):
    from Baseline.paper_baselines.closed_loop import privileged as P
    from Baseline.paper_baselines.methods.guidance_common.scene import build_scene
    from Baseline.paper_baselines.methods.interface import SceneObservation
    from Baseline.paper_baselines.methods.safety_common import ObstacleCloudSource

    req = emitted[0][0]
    H, W = 24, 32
    K = np.array([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]])
    ee = np.eye(4)
    ee[:3, 3] = [0.6, 0.05, 0.1]
    ex = {"T_world_base": T_WORLD_BASE, **P.extras(req, T_WORLD_BASE, shape="collision")}
    obs = SceneObservation(rgb={}, depth={c: np.full((H, W), 0.8) for c in ("head", "left_wrist")},
                           intrinsics={c: K for c in ("head", "left_wrist")},
                           T_base_cam={c: np.eye(4) for c in ("head", "left_wrist")}, q=np.zeros(7), ee_pose=ee,
                           gripper=0.08, instruction="pick up the black bowl", extras=ex)
    sc = build_scene(obs, obstacle_source="privileged")
    assert sc.info["obstacle_source"] == "privileged/obstacle_geoms"
    assert "privileged/obstacle_geoms" in sc.info["privileged_inputs"] and len(sc.obstacles.box_c) == 2
    clouds = ObstacleCloudSource("gt_seg", robot_filter=False, min_points=10)(obs)
    assert clouds.privileged and sorted(clouds.points) == ["moka_pot_1"]  # the bowl (geom 6) is not an instance
    assert obs.privileged("obstacle_names") == ["moka_pot_1"]


# --- 3. server: --gt arms read them, the none arm ignores them -----------------------------------------------------
@pytest.fixture(scope="module")
def recorded():
    if not SL5C_LI_VLA.is_dir():
        pytest.skip("SL5c VLA-only run not available")
    pytest.importorskip("openpi_client")
    pytest.importorskip("jax")
    from Baseline.paper_baselines.closed_loop.replay import RecordedEpisode

    return RecordedEpisode(SL5C_LI_VLA, "safelibero_spatial_t0_e0")


def _server(method, *extra, tmp=None):
    from Baseline.paper_baselines.closed_loop import serve_method as S

    args = S.build_parser().parse_args(["--method", method, "--port", "0", "--fake-policy", "recorded",
                                        "--fake-source-run", str(SL5C_LI_VLA), *extra])
    return S.build_server(args, record_root=tmp)


def with_obstacle(req: dict, offset=(0.05, 0.0, 0.0)) -> dict:
    """``req`` + client-format privileged keys: one active box 5 cm beside the recorded EE (world frame)."""
    p = np.asarray(req["observation/state"], np.float64)[:3] + offset
    box = {"id": 206, "name": "moka_pot_obstacle_1_g2", "type": 6, "size": [0.04, 0.04, 0.1], "pos": p.tolist(),
           "rot": np.eye(3).reshape(9).tolist(), "collision": True, "group": 0}
    vis = {**box, "id": 205, "name": "moka_pot_obstacle_1_g0", "type": 7, "collision": False, "group": 1}
    parked = {"name": "milk_obstacle_1", "active": False, "pos": [0.0, -10.0, 0.2], "quat_wxyz": [1.0, 0, 0, 0],
              "geoms": [{**box, "id": 228, "name": "milk_obstacle_1_g1", "pos": [0.0, -10.0, 0.2]}]}
    out = dict(req)
    out.update({"ag3s/privileged/frame": "world", "ag3s/privileged/monitor_obstacle": "moka_pot_obstacle_1",
                "ag3s/privileged/obstacles": [{"name": "moka_pot_obstacle_1", "active": True, "pos": p.tolist(),
                                               "quat_wxyz": [1.0, 0, 0, 0], "geoms": [vis, box]}, parked]})
    return out


def test_none_arm_serves_the_same_chunk_with_privileged_keys(recorded, tmp_path):
    from Baseline.paper_baselines.closed_loop.replay import readonly

    srv, _ = _server("none", tmp=tmp_path)
    for k in range(3):
        out = srv.infer(readonly(with_obstacle(recorded.request(k))))
        assert np.asarray(out["actions"]).tobytes() == recorded.calls["actions_response"][k].tobytes()
    srv.close()


@pytest.mark.parametrize("method,gt,n_prim", [("nudge", None, 1), ("constrained_flow_matching", None, 1),
                                              ("constrained_flow_matching", "repo", 2)])
def test_gt_guidance_arm_uses_the_privileged_obstacles(recorded, tmp_path, method, gt, n_prim):
    from Baseline.paper_baselines.closed_loop.replay import readonly

    srv, info = _server(method, "--gt", *([gt] if gt else []), tmp=tmp_path)
    assert info["arm"]["privileged"] is True and info["arm"]["target"] == "none" and "sidecar" not in info
    for k in range(2):
        out = srv.infer(readonly(with_obstacle(recorded.request(k))))
        assert np.asarray(out["actions"]).shape == (10, 7)
    srv.close()
    rows = [json.loads(l) for l in (tmp_path / "method_calls/calls.jsonl").read_text().splitlines()]
    assert all(r["fallback"] is None for r in rows), [r.get("error") for r in rows]
    for r in rows:
        sc = r["perception"]["scene"]
        assert sc["obstacle_source"] == "privileged/obstacle_geoms" and sc["privileged_inputs"] == ["privileged/obstacle_geoms"]
        assert r["privileged"]["active"] == ["moka_pot_obstacle_1"] and r["privileged"]["n_geoms"] == n_prim
        assert r["guidance"]["guided"]


def test_gt_arm_without_client_flag_falls_back_and_says_why(recorded, tmp_path):
    from Baseline.paper_baselines.closed_loop.replay import readonly

    srv, _ = _server("nudge", "--gt", tmp=tmp_path)
    out = srv.infer(readonly(recorded.request(0)))
    srv.close()
    row = json.loads((tmp_path / "method_calls/calls.jsonl").read_text().splitlines()[0])
    assert np.asarray(out["actions"]).shape == (10, 7) and row["fallback"] is not None
    assert "emit-privileged-obstacles" in row["error"]


# --- 4. arms / launcher records ------------------------------------------------------------------------------------
def test_gt_arm_records():
    from Baseline.paper_baselines.closed_loop import arms as A

    for name, src in A.GT_DEFAULT.items():
        r = A.resolve(name, gt="default")
        assert r["privileged"] is True and r["gt_source"] == src and r["variant"] == f"{name}:gt-{src}"
        assert r["privileged_inputs"] and r["client_flags"][0] == "--emit-privileged-obstacles"
        assert ("--capture-segmentation" in r["client_flags"]) == (src == "seg")
        assert any(d.startswith("PRIVILEGED: ") for d in r["deviation_from_paper"])
    assert A.resolve("aegis", gt="seg")["privileged_inputs"] == ["segmentation_geom", "instances_geom_ids"]
    assert A.resolve("omniguide", gt="default")["target"] == "detector"  # OmniGuide's attractor reads the target
    for bad in (("none", "default"), ("aegis", "spheres"), ("nudge", "seg")):
        with pytest.raises(ValueError):
            A.resolve(bad[0], gt=bad[1])
    assert all(A.resolve(n)["privileged"] is False for n in A.ARMS)
    assert A.arm_spec("constrained_flow_matching:gt")["rec"]["variant"] == "constrained_flow_matching:gt-spheres"
