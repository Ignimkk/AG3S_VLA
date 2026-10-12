"""SL4: V11AB AG3S profile for the Panda (config derived from serve_safe, declared overrides only), robot_state
split / finger application, attention attachment direction, ESDF bounds, scene GT geometry, harness dry-run and
one CPU frame (CPU; no GPU)."""
import ast
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from AG3S.adapters.panda import attention as pa
from AG3S.adapters.panda import model as pm
from AG3S.adapters.panda import profile as pf
from benchmark.adapters.libero_panda import client_env
from benchmark.evaluators import libero_scene_gt as G
from benchmark.runners import perception_replay as pr
from common.workspace import PROJECT_ROOT, shared_storage_root

LIBERO_PYTHON = PROJECT_ROOT / ".venv-libero/bin/python"
SCENE_EXPORT = PROJECT_ROOT / "benchmark/adapters/libero_panda/scene_export.py"
DEV_RUN = "20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0"
AUDIT_RUN = "20261012T202459_309e4c6da477_sl3-attention-audit-dev"


def _has(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


def _storage_runs():
    base = shared_storage_root() / "results/benchmark"
    return (base / DEV_RUN).is_dir() and (base / AUDIT_RUN).is_dir()


needs_runs = pytest.mark.skipif(not _storage_runs(), reason="SL3 dev / audit runs not in this storage root")


# --- profile: flags -> serve_safe -> config ----------------------------------------------------------------
def test_v11ab_flags_are_read_from_the_launcher_scripts():
    flags = pf.v11ab_server_flags()
    text = " ".join(flags)
    for expected in ("--esdf-backend curobo", "--voxel 0.020", "--fine-voxel 0.005", "--tsdf-voxel 0.005", "--links gripper",
                     "--target-field-policy exclude-authorized", "--no-self-collision", "--held-body-cover",
                     "--closing-keep-geometry", "--latched-identity-hold", "--supported-max-bottom-mm 60",
                     "--held-slip-detach", "--held-cover-cap-mm 10", "--hold-follow-to 3", "--finger-cover both",
                     "--plan-horizon 16", "--w-deflection-rate 20"):
        assert expected in text, expected
    assert "--servo-accel" not in text and "--record-constraints" not in text  # V11ABC only / $REC
    assert "--servo-accel relaxed" in " ".join(pf.v11ab_server_flags("V11ABC"))
    with pytest.raises(ValueError):
        pf.v11ab_server_flags("V99")
    req = pf.v11ab_required_args()
    assert req["config"] == "pi05_rby1_randomized_pick_place_16d_lora" and req["model_xml"].endswith("model_transport.xml")
    assert pf.v11ab_client_phase() == "approach"
    args = pf.parse_v11ab_args()
    assert args.subtask_gate is True and args.esdf_backend == "curobo" and args.links == "gripper"


def test_build_kwargs_follow_the_serve_safe_main_call():
    src = (PROJECT_ROOT / "AG3S/trajopt/serve_safe.py").read_text()
    main = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    call = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "build_ag3s")
    names = [k.arg for k in call.keywords]
    kw = pf.v11ab_build_kwargs(pf.parse_v11ab_args())
    assert list(kw) == [n for n in names if n not in pf.LOG_ONLY_BUILD_KEYWORDS]
    assert set(pf.LOG_ONLY_BUILD_KEYWORDS) <= set(names)
    # the two log-only keywords reach nothing but the start-up log line in build_ag3s
    build = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "build_ag3s")
    loads = [n for n in ast.walk(build) if isinstance(n, ast.Name) and n.id in pf.LOG_ONLY_BUILD_KEYWORDS]
    in_log = [a.value for n in ast.walk(build) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "announce_row_budget"
              for a in n.keywords if isinstance(a.value, ast.Name)]
    assert len(loads) == 2 and all(any(x is y for y in in_log) for x in loads)
    assert kw["backend"] == "curobo" and kw["voxel"] == 0.02 and kw["fine_voxel"] == 0.005 and kw["links"] == "gripper"
    assert kw["target_field_policy"] == "exclude-authorized" and kw["self_collision"] is False and kw["held_body_cover"] is True
    assert kw["identity"] == {"closing_keep_geometry": True, "latched_identity_hold": True, "supported_max_bottom": 0.06,
                              "held_slip_detach": True, "held_cover_cap": 0.01, "hold_follow_to": 3}
    assert kw["constraint_sphere_options"] == {"capsule_radius_scale": 0.05, "sphere_spacing": 0.4,
                                               "max_spheres_per_capsule": 32}
    assert kw["subtask_gate"] is True and kw["held_free_observed_pad"] is None


@pytest.mark.skipif(not _has("mujoco"), reason="needs mujoco (RB-Y1 model for the real build_ag3s)")
def test_profile_config_equals_the_real_build_ag3s_config(monkeypatch):
    """Run serve_safe.build_ag3s (RB-Y1 models, real code) with the V11AB keywords; capture what it hands AG3S."""
    import AG3S.ag3s.runtime.pipeline as pipeline
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import grasp_parent_links

    captured = {}

    class Capture:
        held_body_slip_m = 0.0
        held_lift_hysteresis_m = 0.0
        held_free_observed_pad = None
        held_body_cover = False

        def __init__(self, config, **kwargs):
            captured.update(config=config, **kwargs)
            self.config = config

        def self_filter_settings(self):
            return {}

    monkeypatch.setattr(pipeline, "AG3S", Capture)
    args = pf.parse_v11ab_args()
    kw = pf.v11ab_build_kwargs(args)
    model_xml = pf.v11ab_required_args()["model_xml"]
    if not Path(model_xml).exists():
        pytest.skip(f"RB-Y1 model {model_xml} not in this checkout")
    real = ss.build_ag3s(model_xml, **kw)
    mine = pf.v11ab_ag3s_config(kw)
    a, b = pf.effective_config_dict(captured["config"]), pf.effective_config_dict(mine)
    assert set(a) == set(b)
    assert {k: (a[k], b[k]) for k in a if a[k] != b[k]} == {}
    assert captured["config"].to_dict() == mine.to_dict()
    assert captured["attached_parent_links"] == grasp_parent_links() == ("ee_left", "ee_right")
    assert real.held_body_cover is True and real.held_free_observed_pad is None


def test_diff_against_v11ab_is_exactly_the_declared_overrides():
    v11ab = pf.v11ab_ag3s_config()
    rows = pf.diff_against_v11ab(pf.panda_ag3s_config(v11ab), v11ab)
    assert [r["key"] for r in rows] == sorted(pf.OVERRIDE_REASONS)
    assert all(r["reason"] != "UNDECLARED" for r in rows)
    bounds = next(r for r in rows if r["key"] == "esdf.bounds_lower")
    lo, hi = pf.default_esdf_bounds()
    assert bounds["axes_changed"] == ["z"] and bounds["v11ab"][2] == lo[2] == 0.0 and bounds["panda"][2] == -0.10
    assert bounds["panda"][:2] == list(lo[:2])
    panda = pf.effective_config_dict(pf.panda_ag3s_config(v11ab))
    assert panda["esdf.bounds_upper"] == list(hi) == pf.effective_config_dict(v11ab)["esdf.bounds_upper"]
    links = next(r for r in rows if r["key"] == "contact.contact_links")
    assert links["panda"] == {"left": list(pm.PANDA_GRIPPER_LINKS)} and set(links["v11ab"]) == {"left", "right"}
    # the diagnostic V11AB box: only the contact links differ
    assert [r["key"] for r in pf.diff_against_v11ab(pf.panda_ag3s_config(v11ab, esdf_bounds="v11ab"), v11ab)] == [
        "contact.contact_links"]
    # anything else shows up as UNDECLARED
    extra = pf.panda_ag3s_config(v11ab, extra_overrides={"esdf": {"max_distance": 0.5}})
    undeclared = [r for r in pf.diff_against_v11ab(extra, v11ab) if r["reason"] == "UNDECLARED"]
    assert [r["key"] for r in undeclared] == ["esdf.max_distance"]
    with pytest.raises(ValueError):
        pf.panda_overrides("bogus")


def test_build_panda_ag3s_injects_only_the_panda_geometry():
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    ag3s, info = pf.build_panda_ag3s()
    cfg = ag3s.config
    assert cfg.collision_backend == "esdf" and cfg.esdf.backend == "curobo" and cfg.esdf.voxel_size == 0.02
    assert cfg.constraint.target_field_policy == "exclude_authorized" and cfg.constraint.self_collision is False
    assert cfg.contact.links_for("left") == pm.PANDA_GRIPPER_LINKS and cfg.contact.links_for("right") == ()
    assert cfg.clustering.subtask_gate is True and cfg.clustering.hold_follow_to == 3
    lower, upper = CuroboFieldBuilder._resolve_bounds(cfg.esdf, None)  # what the cuRobo builder will use
    lo, hi = pf.default_esdf_bounds()
    assert lower.tolist() == [lo[0], lo[1], -0.1] and upper.tolist() == list(hi) and np.allclose(lower, [-0.3, -0.9, -0.1])
    lower_v, _ = CuroboFieldBuilder._resolve_bounds(pf.panda_ag3s_config(esdf_bounds="v11ab").esdf, None)
    assert lower_v.tolist() == list(lo) and lo[2] == 0.0
    assert ag3s.robot_model.n_spheres == 4271 and ag3s.constraint_robot_model.n_spheres == 22
    assert set(ag3s.constraint_robot_model.sphere_link_names) == set(pm.PANDA_GRIPPER_LINKS)
    assert info["attached_parent_links"] == ["panda_hand"] == list(pf.panda_attached_parent_links())
    assert ag3s.held_body_cover is True and ag3s.held_free_observed_pad is None
    assert ag3s.max_opening.known and abs(ag3s.max_opening.value_m - 0.078) < 1e-9
    entry = pf.manifest_entry()
    assert [r["key"] for r in entry["config_diff_against_v11ab"]] == sorted(pf.OVERRIDE_REASONS)
    assert {r["what"] for r in entry["constructor_differences"]} >= {"robot_model (self-filter)", "constraint_robot_model",
                                                                    "attached_parent_links"}
    assert entry["esdf_bounds_effective"]["lower"] == [lo[0], lo[1], -0.1]


# --- observations: robot_state split, fingers, attention direction --------------------------------------------
def _synthetic_geom(rs9, n=1):
    K = np.array([[200.0, 0, 127.5], [0, 200.0, 127.5], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 1.0]
    geom = {"cameras": np.array(["head", "left_wrist"]), "stamp_mode": np.array("sim_frozen"),
            "obs_index": np.arange(n)}
    for cam in ("head", "left_wrist"):
        geom[f"depth_mm/{cam}"] = np.full((n, 32, 32), 900, np.uint16)
        geom[f"K/{cam}"] = np.repeat(K[None], n, 0)
        geom[f"T_base_cam/{cam}"] = np.repeat(T[None], n, 0)
        geom[f"robot_state/{cam}"] = np.repeat(np.asarray(rs9, np.float64)[None], n, 0)
        geom[f"stamp/{cam}"] = np.arange(n, dtype=np.float64)
    return geom


def test_robot_state_split_and_finger_parameters_reach_both_models():
    rs9 = np.array([0.1, -0.2, 0.3, -2.0, 0.05, 2.1, 0.7, 0.031, -0.027])
    grid = np.zeros((2, 16, 16), np.float32)
    ag3s, _ = pf.build_panda_ag3s()
    obs, fingers, got, scene = pr.build_observations(_synthetic_geom(rs9), 0, grid, "unmirror", ag3s.robot_model)
    assert [o.camera_id.value for o in obs] == ["head", "left_wrist"] and np.array_equal(got, rs9)
    for o in obs:
        assert o.robot_state.shape == (7,) and np.array_equal(o.robot_state, rs9[:7])
        assert np.allclose(o.depth, 0.9)  # wire mm -> m
    assert fingers == {"panda_finger_joint1": 0.031, "panda_finger_joint2": 0.027}  # JOINT_MAP: joint2 negated
    record = pr.apply_frame_state(ag3s, fingers, rs9)
    for model in (ag3s.robot_model, ag3s.constraint_robot_model):
        assert model.joint_parameters == fingers
    assert record["set_by_caller"] is True and record["applied"]["self_filter"] == fingers
    assert record["applied"]["constraint"] == fingers
    # same instant required across cameras
    bad = _synthetic_geom(rs9)
    bad["robot_state/left_wrist"] = bad["robot_state/left_wrist"] + 1e-3
    with pytest.raises(ValueError, match="one instant"):
        pr.build_observations(bad, 0, grid, "unmirror", ag3s.robot_model)


@pytest.mark.parametrize("mapping", ["unmirror", "mirror"])
def test_attention_is_attached_per_camera_in_the_chosen_orientation(mapping):
    rs9 = np.zeros(9)
    grids = np.zeros((2, 16, 16), np.float32)
    grids[0, 3, 2] = 1.0  # head: policy patch (row 3, col 2)
    grids[1, 9, 14] = 1.0  # left_wrist: policy patch (row 9, col 14)
    obs, *_ = pr.build_observations(_synthetic_geom(rs9), 0, grids, mapping, None)
    head, wrist = (np.asarray(o.attention_map) for o in obs)
    if mapping == "unmirror":  # OpenCV column = 15 - policy column (SL3 mapping)
        assert np.argwhere(head).tolist() == [[3, 13]] and np.argwhere(wrist).tolist() == [[9, 1]]
        assert np.array_equal(np.stack([head, wrist]), pa.unmirror_grid(grids))
    else:
        assert np.argwhere(head).tolist() == [[3, 2]] and np.argwhere(wrist).tolist() == [[9, 14]]
    with pytest.raises(ValueError):
        pr.build_observations(_synthetic_geom(rs9), 0, grids, "flip", None)


# --- scene GT geometry ------------------------------------------------------------------------------------
def test_point_triangle_distance_is_exact():
    rng = np.random.default_rng(3)
    tri = rng.normal(size=(40, 3, 3))
    p = rng.normal(size=(25, 3)) * 2.0
    d = G.point_triangles_distance(p, tri, chunk=97)
    # brute force over a dense barycentric sampling: never smaller than the exact value, and close
    u = rng.random((40000, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    dense = (tri[:, None, 0] * (1 - u.sum(1))[None, :, None] + tri[:, None, 1] * u[None, :, 0, None]
             + tri[:, None, 2] * u[None, :, 1, None]).reshape(-1, 3)
    brute = np.sqrt(((p[:, None] - dense[None]) ** 2).sum(-1)).min(1)
    assert np.all(d <= brute + 1e-12) and np.max(brute - d) < 0.03
    # vertex / edge / face regions of one triangle
    t = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    q = np.array([[0.2, 0.2, 0.5], [-1, -1, 0], [0.5, -1, 0], [2, 2, 0], [0.5, 0.5, 1]])
    expect = [0.5, np.sqrt(2), 1.0, np.sqrt(2) * 1.5, 1.0]
    assert np.allclose(G.point_triangles_distance(q, t), expect)


def test_primitive_meshes_samples_and_containment():
    v, f = G.primitive_mesh(G.BOX, [0.1, 0.2, 0.3])
    area = 0.5 * np.linalg.norm(np.cross(v[f][:, 1] - v[f][:, 0], v[f][:, 2] - v[f][:, 0]), axis=1).sum()
    assert np.isclose(area, 8 * (0.1 * 0.2 + 0.2 * 0.3 + 0.1 * 0.3))
    v, f = G.primitive_mesh(G.CYLINDER, [0.05, 0.1])
    r = np.hypot(v[:-2, 0], v[:-2, 1])
    assert np.allclose(r, 0.05)
    pts = G.sample_triangles(*G.primitive_mesh(G.SPHERE, [0.04]), 0.004, np.random.default_rng(0))
    assert len(pts) > 100 and np.all(np.abs(np.linalg.norm(pts, axis=1) - 0.04) < 0.04 * (1 - np.cos(np.pi / G.SEGMENTS)) * 2)
    a = G.sample_triangles(*G.primitive_mesh(G.BOX, [0.1, 0.1, 0.1]), 0.01, np.random.default_rng(5))
    b = G.sample_triangles(*G.primitive_mesh(G.BOX, [0.1, 0.1, 0.1]), 0.01, np.random.default_rng(5))
    assert np.array_equal(a, b)  # deterministic


def test_depth_residual_sign_and_bounds():
    K = np.array([[100.0, 0, 15.5], [0, 100.0, 15.5], [0, 0, 1]])
    T = np.eye(4)  # camera = base
    depth = np.full((32, 32), 1.0)
    depth[0:8, 0:8] = 0.0
    pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.2], [0.0, 0.0, 0.8], [5.0, 0.0, 1.0], [-0.15, -0.15, 1.0]])
    res = G.depth_residual(pts, depth, K, T)
    assert np.isclose(res[0], 0.0) and np.isclose(res[1], -0.2) and np.isclose(res[2], 0.2)
    assert np.isnan(res[3]) and np.isnan(res[4])  # off-image / no depth


def test_scene_export_is_python38():
    ast.parse(SCENE_EXPORT.read_text(), feature_version=(3, 8))


# --- harness on the SL3 dev data (storage) -----------------------------------------------------------------
@needs_runs
def test_harness_dry_run_output(capsys, monkeypatch):
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(PROJECT_ROOT))
    pr.main(["--source-run", DEV_RUN, "--attention-run", AUDIT_RUN, "--no-gt", "--task-ids", "0,3", "--limit-calls", "2",
             "--dry-run"])
    text = capsys.readouterr().out
    out = json.loads(text[text.index("{"):])
    assert out["calls"] == 4 and [c["stem"] for c in out["checks"]] == ["libero_spatial_t0_e0", "libero_spatial_t3_e0"]
    for check in out["checks"]:
        assert check["cameras"] == ["head", "left_wrist"] and len(check["q7"]) == 7
        assert check["attention_equals_audit_opencv"] is True
    s = out["settings"]
    assert s["step"] == "SL4" and s["options"]["phase"] == "approach" and s["options"]["attention_mapping"] == "unmirror"
    assert [r["key"] for r in s["profile"]["config_diff_against_v11ab"]] == sorted(pf.OVERRIDE_REASONS)
    assert s["sources"]["dev"]["run_id"] == DEV_RUN and s["sources"]["attention"]["run_id"] == AUDIT_RUN
    ep = s["sources"]["episodes"][0]
    assert len(ep["attn_npz_sha256"]) == 64 and len(ep["geom_npz_sha256"]) == 64
    with pytest.raises(ValueError, match="no scene GT"):
        pr.index_scene_gt(str(PROJECT_ROOT / "tests"), pr.index_sources(DEV_RUN, AUDIT_RUN, task_ids=[0]))


@needs_runs
@pytest.mark.skipif(not LIBERO_PYTHON.exists(), reason="needs .venv-libero")
def test_scene_export_and_one_cpu_frame(tmp_path, monkeypatch):
    """py3.8 scene export (1 frame) -> one harness frame on CPU (legacy ESDF backend; test-only override)."""
    import AG3S.ag3s.config as ag3s_config

    gt_dir = tmp_path / "scene"
    run = shared_storage_root() / "results/benchmark" / DEV_RUN
    proc = subprocess.run([str(LIBERO_PYTHON), str(SCENE_EXPORT), "--run", str(run), "--out", str(gt_dir), "--task-ids", "0",
                           "--limit-frames", "1"], capture_output=True, text=True, timeout=600,
                          env=client_env.client_environment())
    assert proc.returncode == 0, proc.stderr[-2000:]
    summary = json.loads((gt_dir / "scene_export.json").read_text())
    ep = summary["episodes"][0]
    assert ep["frames"] == 1 and ep["checks"]["T_world_base_vs_record_max_abs"] == 0.0
    assert ep["checks"]["robot_qpos_vs_robot_state_max_abs"] == 0.0 and ep["target"] == "akita_black_bowl_1"
    with pytest.raises(subprocess.CalledProcessError):  # never overwrites
        subprocess.run([str(LIBERO_PYTHON), str(SCENE_EXPORT), "--run", str(run), "--out", str(gt_dir), "--task-ids", "0",
                        "--limit-frames", "1"], capture_output=True, text=True, timeout=600, check=True,
                       env=client_env.client_environment())

    gt = G.SceneGT(gt_dir / "scene_libero_spatial_t0_e0.npz")
    roles = {x["name"]: x["role"] for x in gt.instance_table()}
    assert roles["akita_black_bowl_1"] == "target" and roles["plate_1"] == "destination" and roles["mount"] == "mount"
    assert roles["main_table"] == "fixture" and roles["akita_black_bowl_2"] == "object"
    samples = gt.samples(spacing=0.006, cap=800)
    P = gt.points_base(0, samples)
    own = gt.exact_distance(P[::40], 0)[np.arange(len(P[::40])), samples.instance[::40]]
    assert np.max(own) < 1e-9  # samples lie on their own surface
    table = P[samples.instance == gt.instance_index("main_table")]
    assert abs(np.max(table[:, 2]) - (-0.012)) < 2e-3  # table top at base z -0.012 (SL1)

    # one frame through V11AB's perception on CPU: legacy ESDF + relax (the cuRobo target-free layer needs the GPU)
    for switch in ("RECON_DEVICE", "SELF_FILTER_DEVICE", "SUPPORT_RANSAC_DEVICE"):  # CPU suite: never the GPU
        monkeypatch.setattr(ag3s_config, switch, "cpu")
    monkeypatch.setattr(pr, "_torch", lambda: None)
    ag3s, info = pf.build_panda_ag3s(extra_overrides={"esdf": {"backend": "legacy"},
                                                      "constraint": {"target_field_policy": "relax"}})
    ag3s.strict_invariants = False  # production setting (AG3S_STRICT_INVARIANTS unset in the server)
    src = pr.index_sources(DEV_RUN, AUDIT_RUN, task_ids=[0])
    out_dir = tmp_path / "run"
    (out_dir / "perception").mkdir(parents=True)
    frames = io.StringIO()
    res = pr.run_episode(ag3s, src["episodes"][0], Path(src["dev"]["run_dir"]), mapping="unmirror", opts=pr.FrameOptions(),
                         gt_path=str(gt_dir / "scene_libero_spatial_t0_e0.npz"),
                         samples_cfg={"spacing": 0.006, "cap": 800, "seed": 0}, out_dir=out_dir, frames_out=frames,
                         limit_calls=1, spheres_all=pm.build_panda_constraint_model(None))
    assert res["calls"] == 1
    row = json.loads(frames.getvalue().splitlines()[0])
    assert row["obs_index"] == 10 and row["finger_joints"]["set_by_caller"] is True
    assert row["fusion"]["cameras_with_attention"] == ["head", "left_wrist"]
    z = np.load(out_dir / res["perception_npz"])
    S, I = 22, len(gt.instances())
    assert z["sph_center"].shape == (1, S, 3) and z["sph_exact"].shape == (1, S, I) and z["sph_all_center"].shape == (1, 61, 3)
    assert z["gt_esdf_main"].shape == (1, len(samples.points_local)) and np.isfinite(z["gt_esdf_main"]).all()
    assert z["slice_xy_main"].shape == (1, 101, 101) and z["sph_authorized"].all()
    assert np.array_equal(z["q7"][0], z["robot_state9"][0, :7])
    assert z["cloud_points_offsets"][-1] == len(z["cloud_points"]) == row["n_filtered_points"]
    # the field of this frame starts at the Panda floor (z -0.10; V11AB box: 0.0)
    assert abs(ag3s._last_esdf_field.grid.origin[2] - (-0.10)) < 0.02 + 1e-9  # noqa: SLF001
