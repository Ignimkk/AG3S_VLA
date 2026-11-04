"""SL10 J1 — geom snapshot 만들기 (`panda_geom_snapshot.py`) 와 그것을 읽는 voxel class 집계. 합성 MuJoCo 모델, CPU, LIBERO 불필요."""
from __future__ import annotations

import json
import pathlib
import types

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from AG3S.trajopt.experiments import panda_fixed_input_ablation as tool  # noqa: E402
from AG3S.trajopt.experiments import panda_geom_snapshot as snapmod  # noqa: E402

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="table" pos="0 0 0"><geom name="table_top" type="box" size="1 1 0.01" pos="0 0 -0.01"/></body>
    <body name="cup_main" pos="0.2 0 0.05"><freejoint name="cup_j"/><geom name="cup_g" type="box" size="0.03 0.03 0.05"/></body>
    <body name="cup_aux" pos="0.2 0.0 0.12"><geom name="cup_aux_g" type="box" size="0.02 0.02 0.02"/></body>
    <body name="plate_main" pos="0.5 0 0.01"><freejoint name="plate_j"/><geom name="plate_g" type="box" size="0.1 0.1 0.01"/></body>
    <body name="box_main" pos="0 0.4 0.05"><freejoint name="box_j"/><geom name="box_g" type="box" size="0.04 0.04 0.05"/></body>
    <body name="robot0_link1" pos="-0.5 0 0.3"><geom name="link1_g" type="sphere" size="0.05"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(XML)


def _state(model, shift=0.0):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    q = d.qpos.copy()
    q[0] += shift                                           # cup x
    return np.concatenate([[0.0], q, d.qvel])


def _record(model):
    t = snapmod.model_tables(model)
    return {"model_body_names": np.asarray(t["body_names"]), "model_geom_names": np.asarray(t["geom_names"]),
            "model_geom_bodyid": t["geom_bodyid"]}


def test_tables_match_and_mismatch_is_refused(model):
    rec = _record(model)
    snapmod.check_tables_against_record(snapmod.model_tables(model), rec)           # same -> silent
    bad = dict(rec, model_geom_names=np.asarray(list(rec["model_geom_names"])[:-1] + ["other"]))
    with pytest.raises(ValueError, match="geom name table"):
        snapmod.check_tables_against_record(snapmod.model_tables(model), bad)
    bad = dict(rec, model_body_names=np.asarray(list(rec["model_body_names"])[::-1]))
    with pytest.raises(ValueError, match="body table"):
        snapmod.check_tables_against_record(snapmod.model_tables(model), bad)
    bad = dict(rec, model_geom_bodyid=np.asarray(rec["model_geom_bodyid"])[::-1].copy())
    with pytest.raises(ValueError, match="geom->body"):
        snapmod.check_tables_against_record(snapmod.model_tables(model), bad)
    with pytest.raises(ValueError, match="no model_body_names"):
        snapmod.check_tables_against_record(snapmod.model_tables(model), {})


def test_snapshot_roundtrip_restores_the_recorded_pose(model, tmp_path):
    s0, s1 = _state(model, 0.0), _state(model, 0.25)
    arrays = snapmod.snapshot_arrays(model, np.stack([s0, s1]), [10, 15])
    path = snapmod.write_snapshot(snapmod.snapshot_path(tmp_path, "ep"), arrays, target_bodies=["cup_main", "cup_aux"],
                                  destination_bodies=["plate_main"], provenance={"run": "x"}, exclusive=True)
    assert path.name == "ep_geomsnap.npz"
    with pytest.raises(FileExistsError):
        snapmod.write_snapshot(path, arrays, exclusive=True)                    # never overwrites
    a = tool.load_geom_snapshot(path, 15)
    b = tool.load_geom_snapshot(path, 10)
    gid = list(a["geom_names"]).index("cup_g")
    assert a["geom_xpos"][gid][0] == pytest.approx(0.45) and b["geom_xpos"][gid][0] == pytest.approx(0.2)
    assert a["target_bodies"] == ["cup_main", "cup_aux"] and a["destination_bodies"] == ["plate_main"]
    assert tool.load_geom_snapshot(path, 99) is None
    prov = json.loads(str(np.load(path)["provenance_json"]))
    assert prov["schema"] == snapmod.SCHEMA and prov["run"] == "x"
    # the old entry point writes the same arrays (no target tables)
    old = tmp_path / "old.npz"
    tool.geom_snapshot_from_mujoco(model, np.stack([s0, s1]), [10, 15], old)
    assert "target_bodies" not in tool.load_geom_snapshot(old, 10)


def test_wrong_state_length_is_refused(model):
    with pytest.raises(ValueError, match="sim_state length"):
        snapmod.snapshot_arrays(model, np.zeros((1, 5)), [0])


def test_classify_accepts_a_set_of_target_bodies(model, tmp_path):
    arrays = snapmod.snapshot_arrays(model, np.stack([_state(model)]), [0])
    path = snapmod.write_snapshot(snapmod.snapshot_path(tmp_path, "ep"), arrays, target_bodies=["cup_main", "cup_aux"],
                                  destination_bodies=["plate_main"])
    snap = tool.load_geom_snapshot(path, 0)
    pts = np.array([[0.2, 0.0, 0.05],        # cup
                    [0.2, 0.0, 0.12],        # the cup's second body
                    [0.0, 0.0, 0.0],         # table top
                    [0.5, 0.05, 0.02],       # plate
                    [0.0, 0.4, 0.06],        # box
                    [-0.5, 0.0, 0.3],        # robot
                    [0.3, 0.3, 0.9]])        # ghost
    out = tool.classify_points(pts, snap, target_body=snap["target_bodies"], destination_body=snap["destination_bodies"])
    assert [c for c, _ in out["geom_of_point"]] == ["target", "target", "support", "destination", "other", "robot", "ghost"]
    # a single name (the old calling convention) still works
    one = tool.classify_points(pts, snap, target_body="cup_main", destination_body="plate_main")
    assert one["counts"]["target"] == 1 and one["counts"]["other"] == 2         # cup_aux is no longer the target


def _layer(origin, voxel, shape, occupied_ijk):
    d = np.full(shape, 0.1, np.float32)
    for ijk in occupied_ijk:
        d[tuple(ijk)] = -0.005
    return types.SimpleNamespace(grid=types.SimpleNamespace(origin=np.asarray(origin, float), voxel_size=voxel), distance_grid=d)


def test_near_voxel_owner_is_counted_per_class(model, tmp_path):
    arrays = snapmod.snapshot_arrays(model, np.stack([_state(model)]), [0])
    path = snapmod.write_snapshot(snapmod.snapshot_path(tmp_path, "ep"), arrays, target_bodies=["cup_main", "cup_aux"],
                                  destination_bodies=["plate_main"])
    snap = tool.load_geom_snapshot(path, 0)
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.0, 0.0]                       # base frame = world shifted by +0.1 in x
    origin, v = [-0.4, -0.2, -0.1], 0.01
    # occupied voxels (base frame): two inside the cup, one on the table top; sphere centre at the cup's side
    def ijk_of(p_world):
        return tuple(np.round((np.asarray(p_world) - T[:3, 3] - origin) / v).astype(int))
    occ = [ijk_of([0.2, 0.0, 0.05]), ijk_of([0.21, 0.0, 0.05]), ijk_of([0.2, 0.0, 0.0])]
    lay = _layer(origin, v, (100, 100, 60), occ)
    centre_base = np.asarray([0.2 - 0.1 + 0.04, 0.0, 0.05])
    res = tool._classify_near(lay, centre_base, 0.08, T, snap, snap["target_bodies"], snap["destination_bodies"])
    assert res["counts"]["target"] == 2 and res["counts"]["support"] == 1
    assert res["nearest_class"] == "target" and res["nearest_body"] == "cup_main"
    # nothing occupied within reach -> None (not a class)
    assert tool._classify_near(lay, np.asarray([0.3, 0.3, 0.3]), 0.02, T, snap, [], []) is None


def test_verdict_collision_rows_keep_attached_sphere_violations():
    row = {"verdict_reasons": [
        {"kind": "occluded_target", "action": "execute", "detail": "x", "evidence": {}},
        {"kind": "collision", "action": "hold", "detail": "8 row(s)", "evidence": {
            "row": {"block": "esdf", "step": 4, "link": "attached:panda_hand[0]", "clearance_m": -0.027, "tier": "fine",
                    "why": "attached_point", "point_m": [0.5, 0.3, 0.1], "object_distance_m": -0.006},
            "n_violating": 8, "residual_violation_m": 0.027}}]}
    (r,) = tool.verdict_collision_rows(row)
    assert r["is_attached"] and r["why"] == "attached_point" and r["clearance_mm"] == pytest.approx(-27.0)
    assert r["object_distance_mm"] == pytest.approx(-6.0) and r["n_violating"] == 8 and r["action"] == "hold"
    assert tool.verdict_collision_rows({}) == []


def test_snapshot_dir_path_and_cli_flags():
    assert tool.geom_snapshot_dir_path("/a", "libero_spatial_t1_e0") == pathlib.Path("/a/libero_spatial_t1_e0_geomsnap.npz")
    assert tool.geom_snapshot_dir_path("/a", "s") == snapmod.snapshot_path("/a", "s")
    ns = tool.build_parser().parse_args(["--run-dir", "r", "--out", "o", "--geom-snapshot-dir", "d", "--j1-max-rows", "9"])
    assert ns.geom_snapshot_dir == "d" and ns.j1_max_rows == 9
    ns = snapmod.build_parser().parse_args(["--run-dir", "r", "--out-dir", "o"])
    assert ns.frames == "calls" and ns.rgb_check is True


def test_thin_surface_without_negative_voxels_is_found_as_surface_then_coarse(model, tmp_path):
    arrays = snapmod.snapshot_arrays(model, np.stack([_state(model)]), [0])
    path = snapmod.write_snapshot(snapmod.snapshot_path(tmp_path, "ep"), arrays)
    snap = tool.load_geom_snapshot(path, 0)
    T = np.eye(4)
    origin, v = [-0.2, -0.2, -0.1], 0.01
    d = np.full((80, 80, 40), 0.1, np.float32)
    ijk = tuple(np.round((np.asarray([0.0, 0.0, 0.0]) - origin) / v).astype(int))
    d[ijk] = 0.004                                    # a TSDF-thin table surface: positive at every voxel centre, none < 0
    fine = types.SimpleNamespace(grid=types.SimpleNamespace(origin=np.asarray(origin), voxel_size=v), distance_grid=d)
    centre = np.asarray([0.0, 0.0, 0.03])
    res = tool._classify_near(fine, centre, 0.05, T, snap, [], [])
    assert res["layer"] == "surface" and res["nearest_class"] == "support"
    # a coarse layer that has a negative voxel is tried before the fine layer's surface voxels
    dc = np.full((40, 40, 20), 0.1, np.float32)
    dc[tuple(np.round((np.asarray([0.0, 0.0, 0.0]) - origin) / 0.02).astype(int))] = -0.01
    coarse = types.SimpleNamespace(grid=types.SimpleNamespace(origin=np.asarray(origin), voxel_size=0.02), distance_grid=dc)
    assert tool._classify_near(fine, centre, 0.05, T, snap, [], [], fallback=coarse)["layer"] == "coarse_occupied"
    assert tool._classify_near(fine, np.asarray([0.15, 0.15, 0.3]), 0.02, T, snap, [], []) is None
