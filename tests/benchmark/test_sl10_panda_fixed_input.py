"""SL10 §2.1 — Panda 고정 입력 재풀이 도구 (CPU, 합성 입력).

* 변형 스위치가 의도한 항만 끄는지 (cost 가중치 · scene 필드 · target 을 뺀 거리장)
* 충실도 실패 → `"valid": false` 이고 변형을 하나도 안 도는지
* 변환 C 뒤 포화 · 계획 − 실현 가능 지표 계산
* 합성 run 디렉터리 → 짝짓기 · q_now · 충실도 A/B/C/D · 변형 실행 (자기 일관성)
* 서버 동작 불변 (도구는 server / action / recorder 를 바꾸지 않는다)
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

from AG3S.trajopt.experiments import panda_fixed_input_ablation as tool  # noqa: E402

Q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])


# --------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def kin():
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import model as pm

    return pa.PandaKinematics(pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper")))


def _sphere_field(centre, radius, *, voxel_fine=0.01, voxel_coarse=0.04, n_fine=40, n_coarse=24):
    """구 장애물의 SDF 를 거친 · 미세 계층으로 쌓은 `CuroboEsdfField` (base 좌표)."""
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

    layers = []
    for voxel, n in ((voxel_coarse, n_coarse), (voxel_fine, n_fine)):
        origin = np.asarray(centre) - 0.5 * n * voxel
        ax = [origin[i] + voxel * np.arange(n) for i in range(3)]
        g = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
        d = (np.linalg.norm(g - np.asarray(centre), axis=-1) - radius).astype(np.float32)
        layers.append(layer_from_arrays(np.clip(d, -0.05, 0.4), origin, voxel))
    return CuroboEsdfField(tuple(layers), outside_distance=0.4)


# --------------------------------------------------------------------------------------- 변형 스위치
def test_variant_table_matches_the_task():
    assert set(tool.VARIANTS) == {"replay", "no_target", "no_rate_terms", "no_attached",
                                  "no_target_no_rate_terms_no_attached", "action_box"}
    v = tool.VARIANTS
    assert not any([v["replay"].target_removed, v["replay"].rate_terms_off, v["replay"].attached_off, v["replay"].action_box])
    only = {"no_target": "target_removed", "no_rate_terms": "rate_terms_off", "no_attached": "attached_off",
            "action_box": "action_box"}
    flags = ("target_removed", "rate_terms_off", "attached_off", "action_box")
    for name, on in only.items():
        assert [f for f in flags if getattr(v[name], f)] == [on], name
    abc = v["no_target_no_rate_terms_no_attached"]
    assert (abc.target_removed, abc.rate_terms_off, abc.attached_off, abc.action_box) == (True, True, True, False)


def test_rate_terms_switch_changes_only_the_two_weights():
    from AG3S.trajopt.config import TrajOptConfig

    base = TrajOptConfig.from_dict({"cost": {"w_continuity": 0.5, "w_deflection_rate": 20.0, "w_smooth": 3.0},
                                    "collision": {"backend": "esdf", "esdf_margin": 0.0, "use_support_planes": False}})
    assert tool.variant_cost_overrides(tool.VARIANTS["replay"]) == {}
    assert tool.variant_cost_overrides(tool.VARIANTS["no_target"]) == {}
    assert tool.variant_cost_overrides(tool.VARIANTS["no_attached"]) == {}
    ov = tool.variant_cost_overrides(tool.VARIANTS["no_rate_terms"])
    assert ov == {"cost": {"w_continuity": 0.0, "w_deflection_rate": 0.0}}
    new = base.with_overrides(ov)
    a, b = dataclasses.asdict(base), dataclasses.asdict(new)
    changed = {(s, k) for s in a for k in a[s] if a[s][k] != b[s][k]}
    assert changed == {("cost", "w_continuity"), ("cost", "w_deflection_rate")}


def _scene(field, S=5):
    from AG3S.trajopt.linearize import SceneSnapshot

    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
        d_safe=np.zeros((S, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=np.full(S, 0.02), candidate_ids=np.zeros(0, np.int64), esdf=field, esdf_margin=0.0,
        manipulated_points=np.zeros((3, 3)), manipulated_link_margin=np.full(S, 0.01),
        target_free_mask=np.zeros(S, bool), attached_points=np.zeros((2, 3)), attached_parent_link="panda_hand",
        attached_radii=np.full(2, 0.05))


def test_scene_switches_touch_only_their_fields():
    field = _sphere_field([0.5, 0.0, 0.4], 0.05)
    scene = _scene(field)
    wo = tool.field_without_target(field, np.array([[0.5, 0.0, 0.4]]), radius_m=0.06)
    same = tool.apply_variant_scene(scene, tool.VARIANTS["replay"])
    assert same is scene
    names = [f.name for f in dataclasses.fields(scene)]

    def diff(a, b):
        out = set()
        for n in names:
            x, y = getattr(a, n), getattr(b, n)
            if n == "esdf":
                eq = x is y
            elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                eq = (x is None) == (y is None) and (x is None or np.array_equal(x, y))
            else:
                eq = x is y or x == y
            if not eq:
                out.add(n)
        return out

    att = tool.apply_variant_scene(scene, tool.VARIANTS["no_attached"])
    assert diff(scene, att) == set(tool.ATTACHED_FIELDS)
    tgt = tool.apply_variant_scene(scene, tool.VARIANTS["no_target"], field_without_target=wo)
    assert diff(scene, tgt) == {"esdf", "manipulated_points", "manipulated_link_margin", "target_free_mask"}
    assert tgt.attached_points is scene.attached_points            # (a) does not touch the attached spheres
    abc = tool.apply_variant_scene(scene, tool.VARIANTS["no_target_no_rate_terms_no_attached"], field_without_target=wo)
    assert diff(scene, abc) == {"esdf", "manipulated_points", "manipulated_link_margin", "target_free_mask", *tool.ATTACHED_FIELDS}
    with pytest.raises(ValueError):
        tool.apply_variant_scene(scene, tool.VARIANTS["no_target"])    # (a) without a target-free field must not no-op


def test_field_without_target_removes_only_the_target():
    field = _sphere_field([0.5, 0.0, 0.4], 0.05)
    # a second, far obstacle merged into the same grid: keep it
    layer = field.layers[-1]
    d = np.asarray(layer.distance_grid).copy()
    g = layer.grid
    ax = [g.origin[i] + g.voxel_size * np.arange(d.shape[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
    d2 = np.linalg.norm(pts - np.array([0.5 + 0.15, 0.0, 0.4]), axis=-1) - 0.03
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

    both = CuroboEsdfField((field.layers[0], layer_from_arrays(np.minimum(d, d2).astype(np.float32), g.origin, g.voxel_size)),
                           outside_distance=0.4)
    wo = tool.field_without_target(both, np.array([[0.5, 0.0, 0.4]]), radius_m=0.07)
    p_target = np.array([[0.5, 0.0, 0.4], [0.5, 0.0, 0.46]])
    assert (both.distance(p_target) < 0.0).any() or both.distance(p_target)[1] < 0.02
    assert (wo.distance(p_target) > 0.05).all()                    # the target is gone
    p_other = np.array([[0.65, 0.0, 0.4], [0.65, 0.0, 0.45]])
    assert np.allclose(wo.distance(p_other), both.distance(p_other), atol=1e-6)   # the other obstacle is untouched
    p_far = np.array([[0.55, 0.12, 0.4]])
    assert wo.distance(p_far)[0] >= both.distance(p_far)[0] - 1e-6            # removal never makes a point closer


def test_action_box_hook_is_unavailable_without_the_option():
    from AG3S.trajopt.config import TrajOptConfig

    assert tool.detect_action_box_overrides(TrajOptConfig()) is None


# --------------------------------------------------------------------------------------- 충실도 · valid=false
@dataclasses.dataclass
class _FakeCall:
    id: str
    fidelity: dict


def _good():
    return {"reference_mm": 0.0, "record_clearance_mm": 0.1, "replay_mm": 0.2, "conversion_reproduced": True}


def test_fidelity_failure_marks_invalid_and_runs_no_variant():
    calls = [_FakeCall("a#1", _good()), _FakeCall("a#2", {**_good(), "replay_mm": 3.5})]
    ran = []

    def solve(call, variant):
        ran.append((call.id, variant.name))
        return {"deviation": {"exceeds": True}}

    out = tool.run_calls(calls, solve, list(tool.VARIANTS), tol_mm=1.0, strict=True)
    assert out["valid"] is False and out["invalid_calls"] == ["a#2"] and "fidelity" in out["invalid_reason"]
    assert out["fidelity"]["a#2"]["failed"] == ["replay_mm"]
    assert ran == [] and out["results"] == {} and out["variants_run"] == []


def test_missing_measurement_and_conversion_mismatch_are_invalid():
    for bad in ({"replay_mm": None}, {"conversion_reproduced": False}, {"reference_mm": float("nan")},
                {"record_clearance_mm": None}):
        v = tool.fidelity_verdict({**_good(), **bad}, 1.0)
        assert v["within_tolerance"] is False, bad


def test_no_strict_skips_only_the_invalid_call_and_keeps_valid_false():
    calls = [_FakeCall("a#1", _good()), _FakeCall("a#2", {**_good(), "reference_mm": 9.0})]
    seen = []
    out = tool.run_calls(calls, lambda c, v: seen.append((c.id, v.name)) or {"x": 1}, ["replay", "no_target"], tol_mm=1.0,
                         strict=False)
    assert out["valid"] is False and set(out["results"]) == {"a#1"}
    assert seen == [("a#1", "replay"), ("a#1", "no_target")]


def test_valid_run_runs_every_requested_variant():
    calls = [_FakeCall("a#1", _good())]
    out = tool.run_calls(calls, lambda c, v: {"v": v.name}, list(tool.VARIANTS), tol_mm=1.0)
    assert out["valid"] is True and out["invalid_reason"] is None
    assert list(out["results"]["a#1"]) == list(tool.VARIANTS)


def test_instrument_summary_applies_the_half_rule():
    def r(dev):
        return {"deviation": {"exceeds": dev > 0.1, "max_rad": dev, "p90_rad": dev / 2},
                "conversion": {"n_saturated_rows_exec": 1, "plan_minus_realisable_mm_p90": 5.0}}

    results = {f"c#{i}": {"replay": r(0.4), "no_target": r(0.05 if i < 3 else 0.4),   # 1/4 exceed -> below half
                          "no_rate_terms": r(0.05 if i < 2 else 0.4),               # 2/4 exceed -> exactly half: not "below"
                          "no_attached": {"unavailable": "x"}} for i in range(4)}
    s = tool.instrument_summary(results, variants=["replay", "no_target", "no_rate_terms", "no_attached"])
    assert s["no_target"]["fraction_dev_gt_0p1"] == 0.25 and s["no_target"]["reduces_below_half"] is True
    assert s["no_rate_terms"]["fraction_dev_gt_0p1"] == 0.5 and s["no_rate_terms"]["reduces_below_half"] is False
    assert s["no_attached"] == {"n": 0, "n_unavailable": 4}
    assert s["replay"]["fraction_dev_gt_0p1"] == 1.0


# --------------------------------------------------------------------------------------- 변환 C 지표
def test_conversion_metrics_count_saturation_and_unrealisable_plan(kin):
    from AG3S.adapters.panda import action as pa

    rng = np.random.default_rng(0)
    A = np.clip(rng.normal(0, 0.3, (10, 7)), -1, 1).astype(np.float32)
    conv = pa.delta_chunk_to_joint_chunk(kin, A, Q0, np.eye(4))
    ref = conv.joint_chunk.astype(np.float32)
    # no change: the policy's bits, nothing saturated, nothing to realise
    m0 = tool.conversion_metrics(kin, conv, ref, ref, execution_length=5)
    assert m0["n_saturated_rows"] == 0 and m0["saturated_components"] == 0 and m0["corrected_rows"] == []
    assert m0["plan_minus_realisable_mm_max"] < 0.05      # IK tolerance only
    # a 6 cm sideways nudge of rows 1-4 is far beyond what +-1 can carry (1/beta = 14x): saturates
    big = ref.copy()
    big[1:5, 0] += 0.25
    m1 = tool.conversion_metrics(kin, conv, big, ref, execution_length=5)
    assert m1["n_saturated_rows"] > 0 and m1["saturated_components"] > 0
    # the metric agrees with action.py itself
    _, rec = pa.joint_chunk_to_delta_chunk(kin, conv, big, ref, mode="C")
    assert m1["saturated_rows"] == rec["saturated_rows"] and m1["corrected_rows"] == rec["corrected_rows"]
    assert m1["plan_minus_realisable_mm_max"] > 5.0
    # a small nudge is realisable
    small = ref.copy()
    small[1:5, 0] += 0.002
    m2 = tool.conversion_metrics(kin, conv, small, ref, execution_length=5)
    assert m2["n_saturated_rows"] == 0 and m2["plan_minus_realisable_mm_max"] < m1["plan_minus_realisable_mm_max"]


def test_conversion_metrics_do_not_count_the_policys_own_saturation(kin):
    from AG3S.adapters.panda import action as pa

    A = np.zeros((10, 7), np.float32)
    A[:, 0] = 1.0                                       # the VLA itself is at the +1 limit
    conv = pa.delta_chunk_to_joint_chunk(kin, A, Q0, np.eye(4))
    ref = conv.joint_chunk.astype(np.float32)
    m = tool.conversion_metrics(kin, conv, ref, ref, execution_length=5)
    assert m["saturated_components"] == 0


def test_deviation_metrics():
    ref = np.zeros((10, 8))
    ch = ref.copy()
    ch[2, 3] = 0.3
    ch[7, 1] = -0.2
    d = tool.deviation_metrics(ch, ref, execution_length=5)
    assert d["max_rad"] == pytest.approx(0.3) and d["exec_max_rad"] == pytest.approx(0.3) and d["exceeds"] is True
    assert d["p90_rad"] < 0.3
    d2 = tool.deviation_metrics(np.where(ref == 0, 0.05, 0), ref, execution_length=5)
    assert d2["exceeds"] is False


# --------------------------------------------------------------------------------------- J1
def test_sphere_kinds_and_row_selection():
    assert [tool.sphere_kind(n) for n in ("panda_leftfinger", "panda_rightfinger", "panda_hand", "panda_link5", "held:panda_hand")] \
        == ["finger", "finger", "palm", "arm", "attached"]
    names = ["panda_leftfinger", "panda_hand", "panda_link4", "panda_link4"]
    c = np.array([[0.05, 0.001, 0.30, 0.30],
                  [-0.004, 0.015, 0.30, 0.01]])
    r = tool.select_rows(c, names, band_m=0.02)
    assert r["violating"] == {"finger": 1, "palm": 0, "arm": 0, "attached": 0, "other": 0}
    assert r["within_band"] == {"finger": 1, "palm": 2, "arm": 1, "attached": 0, "other": 0}
    assert {(x["step"], x["sphere"]) for x in r["rows"]} == {(0, 1), (1, 0), (1, 1), (1, 3)}


def test_voxel_geom_classes_against_a_synthetic_snapshot():
    # bodies: 0 world, 1 table, 2 cup (target), 3 plate (destination), 4 box (other), 5 robot0_link1
    boxes = [(0, [0, 0, -0.5], 0.5), (1, [0, 0, 0.0], 0.0)]
    names = ["world", "table", "cup_main", "plate_main", "box_main", "robot0_link1"]
    pos = np.array([[0, 0, -10.0], [0, 0, 0.0], [0.2, 0, 0.05], [0.5, 0, 0.01], [0, 0.4, 0.05], [-0.5, 0, 0.3]])
    G = len(names)
    snap = {"geom_xpos": pos, "geom_xmat": np.tile(np.eye(3).reshape(-1), (G, 1)),
            "geom_size": np.array([[1, 1, 1], [1, 1, 0.0], [0.03, 0.03, 0.05], [0.1, 0.1, 0.01], [0.04, 0.04, 0.05], [0.03, 0.03, 0.03]]),
            "geom_type": np.array([6, 6, 6, 6, 6, 6]), "geom_bodyid": np.arange(G),
            "geom_contype": np.ones(G, int), "geom_conaffinity": np.ones(G, int), "body_names": np.array(names)}
    snap["geom_xpos"][0] = [0, 0, -50.0]
    pts = np.array([[0.2, 0.0, 0.05],        # inside the cup
                    [0.0, 0.0, 0.0],         # on the table top
                    [0.5, 0.05, 0.02],       # in the plate
                    [0.0, 0.4, 0.06],        # in the box
                    [-0.5, 0.0, 0.3],        # in the robot
                    [0.3, 0.3, 0.5]])        # 40 cm from anything: a ghost
    out = tool.classify_points(pts, snap, target_body="cup_main", destination_body="plate_main")
    assert [c for c, _ in out["geom_of_point"]] == ["target", "support", "destination", "other", "robot", "ghost"]
    assert out["counts"] == {"target": 1, "support": 1, "destination": 1, "other": 1, "robot": 1, "ghost": 1}


# --------------------------------------------------------------------------------------- 합성 run (자기 일관성)
SEQ = 3


def _write_run(tmp, rt, field, target_points, *, refined=None, rec_conv=None, grasp_state="closing", clearance=None):
    """한 episode · call 하나짜리 run 디렉터리. `refined` 가 None 이면 reference 와 같다 (placeholder)."""
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import model as pm

    run = pathlib.Path(tmp)
    (run / "episodes").mkdir(parents=True, exist_ok=True)
    cdir = run / "server_constraints" / "run_0000"
    cdir.mkdir(parents=True, exist_ok=True)
    T = np.eye(4)
    T[:3, 3] = [-0.66, 0.0, 0.912]
    rng = np.random.default_rng(3)
    A = np.zeros((10, 7), np.float32)
    A[:, 0] = 0.6
    A[:, 2] = -0.1
    A += rng.normal(0, 0.01, A.shape).astype(np.float32)
    prev = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.001])
    # eef history: obs_index = SEQ*5; step p1 - p0 = prev[:3]
    n_obs = 40
    obs = SEQ * 5
    pos = np.zeros((n_obs, 3))
    pos[obs] = [0.1, 0.0, 0.0]
    pos[obs - 1] = pos[obs] - prev[:3]
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_obs, 1))
    conv = pa.delta_chunk_to_joint_chunk(rt.kin, A, Q0, T, prev_step_world=(prev[:3], prev[3:6]))
    ref = conv.joint_chunk.astype(np.float32)
    refined = ref if refined is None else np.asarray(refined, np.float32)
    rs = np.zeros((n_obs, 9))
    rs[obs, :7] = Q0
    rs[obs, 7:] = [0.04, -0.04]
    np.savez(run / "episodes" / "ep0.npz", obs_robot0_eef_pos=pos, obs_robot0_eef_quat=quat)
    np.savez(run / "episodes" / "ep0_geom.npz", **{"robot_state/left_wrist": rs, "robot_state/head": rs})
    row = {"seq": SEQ, "obs_index": obs, "grasp": {"state": grasp_state}, "to": {"reference_deviation": 0.0},
           "conversion": rec_conv or {"mode": "C", "saturated_rows": [], "corrected_rows": []}}
    (run / "episodes" / "ep0_calls.jsonl").write_text(json.dumps(row) + "\n")
    np.savez(run / "episodes" / "ep0_calls.npz", actions_policy=A[None].astype(np.float64), joint_reference=ref[None],
             joint_refined=refined[None], actions_executed=A[None, :5].astype(np.float64), T_world_base=T[None])
    names = [str(n) for n in rt.robot.sphere_link_names]
    layers = {}
    for prefix, layer in (("esdf_coarse", field.layers[0]), ("esdf", field.layers[1])):
        layers[f"{prefix}_distance"] = np.asarray(layer.distance_grid, np.float16)
        layers[f"{prefix}_origin"] = np.asarray(layer.grid.origin, np.float64)
        layers[f"{prefix}_voxel_size"] = np.float64(layer.grid.voxel_size)
    n_s = len(names)
    summary = {"t_step": SEQ, "status": "ok", "has_target": True, "esdf_stats": {"n_attached_points": 0},
               "sphere_link_names": names, "target_permission": {"policy": "exclude_authorized", "authorized_links":
                                                                 ["panda_hand", "panda_leftfinger", "panda_rightfinger"],
                                                                 "target_field_exclude": None, "held": None},
               "to": {"sqp_iterations": 2, "iterations": 2, "metrics": {"geometry_certified": True}},
               "exec_continuity": {"mode": "no cached plan (first chunk after reset or server restart)"},
               "exec_feedback": {"available": False}}
    np.savez_compressed(cdir / "chunk_00000.npz", t_step=np.int64(SEQ), chunk_index=np.int64(1), reference_chunk=ref,
                        refined_chunk=refined, target_points=np.asarray(target_points, np.float32),
                        target_centroid=np.asarray(target_points, np.float64).mean(0),
                        clearance=np.zeros((10, n_s), np.float32) if clearance is None else clearance,
                        summary_json=np.asarray(json.dumps(summary)), **layers)
    return run


@pytest.fixture(scope="module")
def runtime():
    from AG3S.adapters.panda import profile
    from AG3S.trajopt import serve_safe as ss

    v = "AG3S-final-panda-v3-e0to-pc"
    args = ss.build_parser().parse_args(profile.version_server_argv(v) + profile.version_server_flags(v) + ["--robot", "panda"])
    return tool.build_runtime(args)


def _synthetic(tmp_path, rt):
    """장애물을 손 진행 방향 바로 앞에 둔 장. placeholder 기록 → prepare → 풀어서 'recorded' 를 채운 run."""
    p0 = rt.kin.fk(Q0)[0]
    obstacle = p0 + np.array([0.05, 0.0, 0.0])          # right in front of the grip site: the SQP pushes (~0.15 rad)
    field = _sphere_field(obstacle, 0.04, voxel_fine=0.01, voxel_coarse=0.04, n_fine=44, n_coarse=24)
    target_points = obstacle + 0.04 * np.concatenate([np.eye(3), -np.eye(3)])     # six surface points of the ball
    run = _write_run(tmp_path, rt, field, target_points)
    calls = tool.load_episode_calls(run / "episodes", "ep0")
    tool.attach_chunks(calls, run / "server_constraints" / "run_0000")
    prep, _ = tool.prepare_call(rt, calls[0], tol_mm=1.0)
    result = tool.solve(rt, prep, tool.variant_config(rt, tool.VARIANTS["replay"], prep.sqp_iterations), prep.scene)
    refined = np.asarray(result.chunk, np.float32)
    cm = tool.conversion_metrics(rt.kin, prep.conv, refined, prep.recorded_reference(), execution_length=rt.execution_length)
    traj = rt.layout.chunk_to_trajectory(refined.astype(np.float64))[:, :10]
    centres = rt.lin.sphere_states(traj, Q0)[0]
    clearance = (np.asarray(field.distance(centres.reshape(-1, 3))).reshape(centres.shape[:2]) - rt.lin.robot_radii[None, :]
                 - rt.esdf_margin).astype(np.float32)
    run = _write_run(tmp_path, rt, field, target_points, refined=refined, clearance=clearance,
                     rec_conv={"mode": "C", "saturated_rows": cm["saturated_rows"], "corrected_rows": cm["corrected_rows"]})
    return run, field, refined, target_points


def test_synthetic_run_roundtrip_fidelity_and_variants(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(tool, "TARGET_REMOVAL_RADIUS_M", 0.06)   # six points cannot cover the ball's centre at 2 cm
    rt = runtime
    run, field, refined, target_points = _synthetic(tmp_path, rt)
    calls = tool.load_episode_calls(run / "episodes", "ep0")
    pairing = tool.attach_chunks(calls, run / "server_constraints" / "run_0000")
    assert pairing["matched"] == 1 and pairing["calls_without_chunk"] == 0
    c = calls[0]
    assert c.id == f"ep0#{SEQ}" and c.phase == "closing"
    assert np.allclose(c.q_now, Q0) and np.allclose(c.prev_step[:3], [0.001, 0.0, 0.0], atol=1e-12)
    # the TrajOpt did move something in this scene (otherwise the test proves nothing)
    assert np.abs(refined[:, :7] - c.joint_reference[:, :7]).max() > 1e-4
    prep, why = tool.prepare_call(rt, c, tol_mm=1.0)
    assert prep is not None
    f = prep.fidelity
    assert f["reference_mm"] < 1e-3 and f["record_clearance_mm"] < 0.5 and f["replay_mm"] < 0.05
    assert f["conversion_reproduced"] is True
    assert tool.fidelity_verdict(f, 1.0)["within_tolerance"]
    # variants: replay reproduces the record; (a) / (b) run; (c) has no attached spheres here -> unavailable
    rep = tool.run_variant(rt, prep, tool.VARIANTS["replay"])
    assert rep["vs_recorded_refined_mm"] < 0.05
    no_t = tool.run_variant(rt, prep, tool.VARIANTS["no_target"])
    assert rep["deviation"]["max_rad"] > 0.1 and rep["conversion"]["n_saturated_rows"] > 0   # the mechanism of SL10 in one call
    assert no_t["deviation"]["max_rad"] < 0.5 * rep["deviation"]["max_rad"]      # the obstacle is gone: it pushes far less
    assert tool.run_variant(rt, prep, tool.VARIANTS["no_rate_terms"])["deviation"]["max_rad"] >= 0.0
    assert "unavailable" in tool.run_variant(rt, prep, tool.VARIANTS["no_attached"])
    assert "unavailable" in tool.run_variant(rt, prep, tool.VARIANTS["action_box"])
    # the whole driver, strict
    out = tool.run_calls([prep], lambda p, v: tool.run_variant(rt, p, v), list(tool.VARIANTS), tol_mm=1.0)
    assert out["valid"] is True and set(out["results"][prep.id]) == set(tool.VARIANTS)
    s = tool.instrument_summary(out["results"], variants=out["variants_run"])
    assert s["no_target"]["n"] == 1 and s["action_box"]["n"] == 0


def test_synthetic_run_with_a_wrong_record_is_invalid(tmp_path, runtime):
    rt = runtime
    run, field, refined, target_points = _synthetic(tmp_path, rt)
    # record a refined chunk that the solver cannot reproduce (shifted 2 cm worth of joint angle): replay_mm must fail
    bad = refined.copy()
    bad[:, 0] += 0.2
    _write_run(tmp_path, rt, field, target_points, refined=bad)
    calls = tool.load_episode_calls(run / "episodes", "ep0")
    tool.attach_chunks(calls, run / "server_constraints" / "run_0000")
    prep, _ = tool.prepare_call(rt, calls[0], tol_mm=1.0)
    verdict = tool.fidelity_verdict(prep.fidelity, 1.0)
    assert not verdict["within_tolerance"] and "replay_mm" in verdict["failed"]
    ran = []
    out = tool.run_calls([prep], lambda p, v: ran.append(v.name) or {}, list(tool.VARIANTS), tol_mm=1.0)
    assert out["valid"] is False and ran == []


VERSION = "AG3S-final-panda-v3-e0to-pc"


def test_cli_end_to_end_valid_and_invalid(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(tool, "TARGET_REMOVAL_RADIUS_M", 0.06)
    rt = runtime
    run, field, refined, target_points = _synthetic(tmp_path / "ok", rt)
    out = tmp_path / "ok" / "j2.json"
    code = tool.main(["--run-dir", str(run), "--ag3s-version", VERSION, "--out", str(out), "--min-dev-rad", "0.05"])
    data = json.loads(out.read_text())
    assert code == 0 and data["valid"] is True
    cid = f"ep0#{SEQ}"
    assert data["targets"] == [cid] and data["target_phases"][cid] == "closing"
    assert set(data["results"][cid]) == set(tool.VARIANTS)
    assert data["results"][cid]["action_box"]["unavailable"]
    assert "rows" in data["j1"][cid] and data["j1"][cid]["voxel_classes"]["unavailable"]
    assert data["instrument_summary"]["no_target"]["reduces_below_half"] is True
    # J1 only: no solve, no fidelity gate
    out_j1 = tmp_path / "ok" / "j1.json"
    assert tool.main(["--run-dir", str(run), "--ag3s-version", VERSION, "--out", str(out_j1), "--j1-only",
                      "--min-dev-rad", "0.05"]) == 0
    assert json.loads(out_j1.read_text())["j2"].startswith("skipped")
    # a record the solver cannot reproduce: valid=false, exit 2, no variant results
    bad_run, field, refined, target_points = _synthetic(tmp_path / "bad", rt)
    bad = refined.copy()
    bad[:, 0] += 0.2
    _write_run(tmp_path / "bad", rt, field, target_points, refined=bad)
    out_bad = tmp_path / "bad" / "j2.json"
    code = tool.main(["--run-dir", str(bad_run), "--ag3s-version", VERSION, "--out", str(out_bad), "--all-calls"])
    data = json.loads(out_bad.read_text())
    assert code == 2 and data["valid"] is False and data["results"] == {} and data["invalid_calls"] == [cid]
    assert data["invalid_reason"]


def test_selection_by_phase_and_deviation(tmp_path, runtime):
    rt = runtime
    run, field, refined, target_points = _synthetic(tmp_path, rt)
    calls = tool.load_episode_calls(run / "episodes", "ep0")
    tool.attach_chunks(calls, run / "server_constraints" / "run_0000")
    c = calls[0]
    dev = float(np.abs(c.joint_refined[:, :7] - c.joint_reference[:, :7]).max())
    assert tool.select_calls(calls, ["closing"], min_dev_rad=dev * 0.5) == [c]
    assert tool.select_calls(calls, ["closing"], min_dev_rad=dev * 2) == []
    assert tool.select_calls(calls, ["held"], min_dev_rad=0.0) == []
    assert tool.select_calls(calls, ["held"], only_deviating=False) == []
    assert tool.select_calls(calls, ["closing"], only_deviating=False) == [c]


# --------------------------------------------------------------------------------------- 서버 불변
def test_tool_does_not_modify_server_adapter_or_recorder():
    """도구는 읽기만 한다: 이 branch 가 base 대비 바꾼 파일 중 서버 · adapter · 기록기 · 변환이 없다."""
    base = subprocess.run(["git", "merge-base", "HEAD", "62c0076f"], cwd=ROOT, capture_output=True, text=True)
    if base.returncode != 0:
        pytest.skip("base commit not available")
    changed = subprocess.run(["git", "diff", "--name-only", base.stdout.strip()], cwd=ROOT, capture_output=True,
                             text=True).stdout.split()
    protected = ("AG3S/adapters/panda/", "AG3S/trajopt/serve_safe.py", "AG3S/trajopt/safe_policy.py",
                 "AG3S/ag3s/experiments/sources/constraint_record.py", "AG3S/trajopt/refiner.py", "AG3S/trajopt/sqp.py",
                 "AG3S/trajopt/linearize.py", "robots/panda/")
    assert [f for f in changed if f.startswith(protected)] == []
