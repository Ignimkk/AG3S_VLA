"""T80 — `pi05_infer --extra-apple / --move-target-*` (default off), the apple_2 model, the scene generator checks and
the evaluation's apple / apple_2 distinction. CPU only (stub policy for `main()`)."""
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
INFER = ROOT / "integration/rby1/bringup/pi05_infer.py"
XML = ROOT / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
sys.path.insert(0, str(INFER.parent))
import t80_scene_ops as ops  # noqa: E402

mujoco = pytest.importorskip("mujoco")


def _load(name, source=None):
    spec = importlib.util.spec_from_file_location(name, INFER)
    mod = importlib.util.module_from_spec(spec)
    if source is None:
        spec.loader.exec_module(mod)
    else:   # HEAD source run as if it were this file (REPO_ROOT etc. from __file__)
        mod.__file__ = str(INFER)
        exec(compile(source, str(INFER), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def mod():
    m = _load("_t80_pi05_infer")
    assert pathlib.Path(m.__file__).resolve() == INFER.resolve()
    return m


@pytest.fixture(scope="module")
def models():
    return mujoco.MjModel.from_xml_path(str(XML)), ops.build_model_with_extra_apple(str(XML))


# --------------------------------------------------------------------------------------------- the apple_2 model
def test_clone_is_appended_and_shipped_part_unchanged(models):
    m0, m = models
    assert (m.nq, m.nv, m.nbody, m.ngeom) == (m0.nq + 7, m0.nv + 6, m0.nbody + 1, m0.ngeom + 4)
    assert (m.nmesh, m.nmat) == (m0.nmesh, m0.nmat)          # assets shared, not copied
    assert m.body("apple_2").id == m0.nbody and m.jnt_qposadr[m.joint("apple_2_free").id] == m0.nq
    for f in ("qpos0", "body_pos", "body_quat", "body_mass", "body_inertia", "geom_size", "geom_pos", "geom_friction",
              "geom_type", "geom_dataid", "jnt_qposadr", "jnt_dofadr", "actuator_gainprm", "actuator_trnid",
              "cam_pos", "cam_quat", "cam_fovy"):
        a0, a = getattr(m0, f), getattr(m, f)
        assert np.array_equal(a0, a[:len(a0)]), f
    for i in range(m0.nkey):
        assert np.array_equal(m.key_qpos[i, :m0.nq], m0.key_qpos[i])
    teleop = m.key("teleop").id
    assert np.array_equal(m.key_qpos[teleop, m0.nq:], np.asarray(ops.PARK_POSE))
    assert m.opt.timestep == m0.opt.timestep


def test_clone_physics_identical(models):
    _, m = models
    a, b = m.body("apple").id, m.body("apple_2").id
    ga = [g for g in range(m.ngeom) if m.geom_bodyid[g] == a]
    gb = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    assert len(ga) == len(gb) == 4
    for f in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_dataid", "geom_matid", "geom_rgba", "geom_friction",
              "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority", "geom_solmix", "geom_solref",
              "geom_solimp", "geom_margin", "geom_gap", "geom_group", "geom_rbound", "geom_aabb"):
        assert np.array_equal(getattr(m, f)[ga], getattr(m, f)[gb]), f
    for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        assert np.array_equal(getattr(m, f)[a], getattr(m, f)[b]), f
    ja, jb = m.joint("apple_free").id, m.joint("apple_2_free").id
    da, db = m.jnt_dofadr[ja], m.jnt_dofadr[jb]
    for f in ("dof_damping", "dof_armature", "dof_frictionloss"):
        assert np.array_equal(getattr(m, f)[da:da + 6], getattr(m, f)[db:db + 6]), f
    assert m.jnt_type[ja] == m.jnt_type[jb] == mujoco.mjtJoint.mjJNT_FREE


def test_clone_settles_like_the_apple(models):
    """As the client does it: apple_2 gets the settled apple's z and orientation (shifted in x) and stays at rest."""
    _, m = models
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("teleop").id)
    for i in range(m.nu):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    qa, _ = ops.free_qpos_adr(m, "apple_free")
    ops.place_free_body(m, d, "apple_free", [0.55, 0.20, 0.86, 1.0, 0.0, 0.0, 0.0])
    for _ in range(750):
        mujoco.mj_step(m, d)
    q = d.qpos[qa:qa + 7].copy()
    ops.place_free_body(m, d, "apple_free", q)
    ops.place_free_body(m, d, "apple_2_free", [q[0] + 0.20, *q[1:]])
    mujoco.mj_forward(m, d)
    p0 = {b: d.xpos[m.body(b).id].copy() for b in ("apple", "apple_2")}
    for _ in range(1000):
        mujoco.mj_step(m, d)
    for b in ("apple", "apple_2"):
        assert np.linalg.norm(d.xpos[m.body(b).id] - p0[b]) < 1e-3, b
    assert abs(d.xpos[m.body("apple").id][2] - d.xpos[m.body("apple_2").id][2]) < 2e-4


# --------------------------------------------------------------------------------------------- mover / options
def test_target_mover_fires_once(models):
    _, m = models
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("teleop").id)
    mujoco.mj_forward(m, d)
    site = d.site_xpos[m.site("left_ee").id].copy()
    mv = ops.TargetMover(arm="left", body="apple", joint="apple_free", threshold_m=0.15, dxy=(0.06, 0.0))
    ops.place_free_body(m, d, "apple_free", [site[0], site[1], site[2] - 0.30, 1, 0, 0, 0])
    mujoco.mj_forward(m, d)
    assert mv.step(m, d, 0) is None and not mv.moved                       # 300 mm away
    qa, va = ops.free_qpos_adr(m, "apple_free")
    d.qvel[va:va + 6] = 0.3
    ops.place_free_body(m, d, "apple_free", [site[0], site[1], site[2] - 0.10, 0.9, 0.1, 0.3, 0.2])
    d.qvel[va:va + 6] = 0.3
    mujoco.mj_forward(m, d)
    before = d.qpos[qa:qa + 7].copy()
    rec = mv.step(m, d, 7)
    assert rec is not None and rec["t_step"] == 7 and abs(rec["palm_distance_m"] - 0.10) < 1e-9
    after = d.qpos[qa:qa + 7]
    assert np.allclose(after[:2] - before[:2], [0.06, 0.0], atol=1e-12) and np.array_equal(after[2:], before[2:])
    assert np.all(d.qvel[va:va + 6] == 0.0)
    assert rec["qpos_before"] == before.tolist() and rec["qpos_after"] == after.tolist()
    assert mv.step(m, d, 8) is None and mv.record["t_step"] == 7               # once


def test_option_validation(mod):
    import argparse
    ns = lambda **kw: argparse.Namespace(**{"model": "rby1_randomized_pick_place_16d", "extra_apple": None,  # noqa: E731
                                            "move_target_when_palm_mm": None, "move_target_dxy": None, **kw})
    assert mod.t80_option_error(ns()) is None
    assert mod.t80_option_error(ns(model="rby1")) is None                      # off: nothing checked
    assert "require --model" in mod.t80_option_error(ns(model="rby1", extra_apple=[0.5, 0.2]))
    assert mod.t80_option_error(ns(extra_apple=[0.5, 0.2])) is None
    assert "finite" in mod.t80_option_error(ns(extra_apple=[float("nan"), 0.2]))
    assert "go together" in mod.t80_option_error(ns(move_target_when_palm_mm=150.0))
    assert "go together" in mod.t80_option_error(ns(move_target_dxy=[0.06, 0.0]))
    assert mod.t80_option_error(ns(move_target_when_palm_mm=150.0, move_target_dxy=[0.06, 0.0])) is None
    assert "> 0" in mod.t80_option_error(ns(move_target_when_palm_mm=0.0, move_target_dxy=[0.06, 0.0]))
    assert "0.10" in mod.t80_option_error(ns(move_target_when_palm_mm=150.0, move_target_dxy=[0.2, 0.0]))


def test_off_path_does_not_import_scene_ops():
    src = INFER.read_text()
    assert "import t80_scene_ops" in src and src.count("t80_scene_ops()") >= 3
    assert 'ap.add_argument(\n        "--extra-apple", type=float, nargs=2, default=None' in src
    assert '"--move-target-when-palm-mm", type=float, default=None' in src
    assert '"--move-target-dxy", type=float, nargs=2, default=None' in src


# --------------------------------------------------------------------------------------------- main() with a stub policy
class _FakePolicy:
    def infer(self, obs):
        s = np.asarray(obs["state"], dtype=np.float64)
        rows = np.tile(s, (10, 1))
        ramp = 0.002 * (np.arange(10)[:, None] + 1)
        rows[:, 0:7] += ramp
        rows[:, 8:15] -= ramp
        return {"actions": rows}


EPISODE = 56


def _run_main(m, monkeypatch, tmp_path, name, extra=()):
    out = tmp_path / name
    monkeypatch.setattr(m, "prepare_run_outputs", lambda args, names: setattr(args, "run_dir", tmp_path))
    monkeypatch.setattr(m, "configure_run_logging", lambda name: None)
    monkeypatch.setattr(m, "load_remote_policy", lambda remote: _FakePolicy())
    monkeypatch.setattr("sys.argv", ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d",
                                     "--remote", "localhost:1", "--episode-index", str(EPISODE), "--headless",
                                     "--warmup-steps", "50", "--max-steps", "20", "--trajectory-out", str(out / "traj.npz"),
                                     "--record-frames", str(out), *extra])
    m.main()
    frames = [json.loads(x) for x in (out / "frames.jsonl").read_text().splitlines()]
    return np.load(out / "traj.npz"), frames, json.loads((out / "manifest.json").read_text())


_VOLATILE = {"now", "camera_stamps", "newest_stamp", "camera_render_stamps", "render_spread_sec", "timing_ms",
             "field", "last_field_update", "camera_skew_sec", "stamps", "render_stamps"}


def _stable(frames):
    return [{k: v for k, v in r.items() if k not in _VOLATILE} for r in frames]


def _head_source():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:integration/rby1/bringup/pi05_infer.py"],
                              check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@pytest.fixture(scope="module")
def episode_ok(mod):
    try:
        mod.load_randomized_episode(EPISODE)
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"16-D episode metadata not available: {exc}")


def test_main_off_is_bit_identical_to_head(mod, monkeypatch, tmp_path, episode_ok):
    src = _head_source()
    if src is None or "t80_option_error" in src:
        pytest.skip("no git HEAD before T80")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    head = _load("_t80_head_pi05_infer", src)
    t_new, f_new, man_new = _run_main(mod, monkeypatch, tmp_path, "new")
    t_old, f_old, man_old = _run_main(head, monkeypatch, tmp_path, "old")
    assert sorted(t_new.files) == sorted(t_old.files)
    for k in t_new.files:
        if k != "inference_ms":
            assert np.array_equal(t_new[k], t_old[k]), k
    assert _stable(f_new) == _stable(f_old)
    assert len([r for r in f_new if r.get("qpos")][0]["qpos"]) == 66
    for key in ("scene", "policy"):
        assert man_new[key] == man_old[key]
    assert man_new["extra"]["object_truth"] == man_old["extra"]["object_truth"]
    assert "t80" not in man_new["scene"] and "t80_result" not in man_new
    assert not (tmp_path / "new" / "t80.json").exists()


def test_main_on_extra_apple_and_move(mod, monkeypatch, tmp_path, episode_ok):
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    off, _, _ = _run_main(mod, monkeypatch, tmp_path, "off")
    rec = mod.load_randomized_episode(EPISODE)
    ax, ay = rec["initial_fruit_poses"]["apple"][:2]
    xy = (ax, ay + 0.11)
    # 1 km threshold: the move fires after the first control step
    t, frames, man = _run_main(mod, monkeypatch, tmp_path, "on",
                               ("--remove-objects", "pear", "--extra-apple", f"{xy[0]:.4f}", f"{xy[1]:.4f}",
                                "--move-target-when-palm-mm", "1e6", "--move-target-dxy", "0.06", "0.0"))
    extra = {"t80_json", "t80_apple_qpos", "t80_apple_2_qpos", "t80_move_t", "t80_move_qpos_before", "t80_move_qpos_after"}
    assert sorted(t.files) == sorted(set(off.files) | extra)
    assert t["t80_apple_qpos"].shape == (20, 7) and t["t80_apple_2_qpos"].shape == (20, 7)
    # apple_2 sits where it was put, on the table, at rest after the warm-up
    a2 = t["t80_apple_2_qpos"][0]
    assert abs(a2[0] - float(f"{xy[0]:.4f}")) < 3e-3 and abs(a2[1] - float(f"{xy[1]:.4f}")) < 3e-3
    assert abs(a2[2] - t["t80_apple_qpos"][0][2]) < 3e-3
    assert int(t["t80_move_t"]) == 0
    b, a = t["t80_move_qpos_before"], t["t80_move_qpos_after"]
    assert np.allclose(a[:2] - b[:2], [0.06, 0.0]) and np.array_equal(a[2:], b[2:])
    assert abs(t["t80_apple_qpos"][0][0] - a[0]) < 1e-9                     # recorded after the move
    moved = [r for r in frames if "t80_target_move" in r]
    assert len(moved) == 1 and moved[0]["kind"] == "control" and moved[0]["t_step"] == 1
    ctrl = [r for r in frames if r["kind"] == "control"]
    assert len(ctrl[0]["qpos"]) == 73
    plan = [r for r in frames if r["kind"] == "planning"]
    assert "apple_2" in plan[0]["object_poses"]
    assert man["scene"]["t80"]["extra_apple_world_xy"] == [float(f"{xy[0]:.4f}"), float(f"{xy[1]:.4f}")]
    assert man["t80_result"]["target_move"]["moved"] is True
    side = json.loads((tmp_path / "on" / "t80.json").read_text())
    assert side["target_move"]["record"]["t_step"] == 0


# --------------------------------------------------------------------------------------------- scenes / eval
def test_training_range_and_outside():
    from AG3S.ag3s.experiments.studies import t80_scenes as S
    recs = []
    for i in range(101):
        for arm, sy in (("left", 1.0), ("right", -1.0)):
            recs.append({"split": "train", "target_fruit": "apple", "used_arm": arm, "basket_pose": [0.5, 0.0, 0.88, 1, 0, 0, 0],
                         "initial_fruit_poses": {"apple": [0.5 + (i - 50) / 1000.0, sy * (0.3 + (i - 50) / 2000.0), 0.85]}})
    recs.append({"split": "test", "target_fruit": "apple", "used_arm": "left", "basket_pose": [0, 0, 0],
                 "initial_fruit_poses": {"apple": [9.0, 9.0, 0.85]}})
    rng = S.training_range(recs)
    assert rng["left"]["n"] == 101 and np.allclose(rng["left"]["dx"], [-0.045, 0.045])
    assert np.allclose(rng["left"]["dy"], [0.2775, 0.3225]) and np.allclose(rng["right"]["dy"], [-0.3225, -0.2775])
    assert S.in_range([0.0, 0.30], rng["left"]) and not S.in_range([0.0, -0.30], rng["left"])
    assert S.outside_mm([0.0, 0.3325], rng["left"]) == pytest.approx(10.0)
    assert S.outside_mm([0.055, 0.3325], rng["left"]) == pytest.approx(np.hypot(10, 10))


def test_surface_gap_and_feasibility(models):
    from AG3S.ag3s.experiments.studies import t80_scenes as S
    _, m = models
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("teleop").id)
    ops.place_free_body(m, d, "apple_free", [0.55, 0.30, 0.85, 1, 0, 0, 0])
    r = S.apple_radius(m)
    for gap in (0.030, 0.050):
        ops.place_free_body(m, d, "apple_2_free", [0.55, 0.30 + 2 * r + gap, 0.85, 1, 0, 0, 0])
        mujoco.mj_forward(m, d)
        assert S.surface_gap(m, d, "apple", "apple_2") == pytest.approx(gap, abs=2e-3)
    # far from the crate's far wall: never a bogus 0 (mj_geomDistance mesh-box failure is caught)
    assert S.surface_gap(m, d, "apple", "crate") > 0.05
    assert S.on_table(m, d, [0.55, 0.30], r) and not S.on_table(m, d, [0.41, 0.30], r)
    rep = {"settle_displacement_mm": {"apple_2": 0.5}, "objects": {"apple_2": {"on_table": True, "gaps_mm": {"pear": 29.0, "apple": 31}}}}
    assert S.feasible("A", rep) == ["apple_2-pear gap 29.0 mm"]
    rep["objects"]["apple_2"]["gaps_mm"]["pear"] = 30.0
    assert S.feasible("A", rep) == []
    rep["settle_displacement_mm"]["apple_2"] = 12.0
    assert S.feasible("A", rep) == ["apple_2 settles 12.00 mm"]


def test_eval_distinguishes_the_two_apples(models):
    from AG3S.ag3s.experiments.studies import t80_eval as E
    gt = E.build_truth(str(XML))
    m = gt.m
    q = np.array(m.key_qpos[m.key("teleop").id])
    qa, qb = m.jnt_qposadr[m.joint("apple_free").id], m.jnt_qposadr[m.joint("apple_2_free").id]
    q[qa:qa + 7] = [0.55, 0.30, 0.85, 1, 0, 0, 0]
    q[qb:qb + 7] = [0.55, 0.15, 0.85, 1, 0, 0, 0]
    samples, _ = gt.at(q)
    names = [k[1] for k in gt.kind]
    assert "apple_2" in names and ("fruit", "apple_2") in [(k[0], k[1]) for k in gt.kind]
    s1 = np.vstack([samples[i] for i, n in enumerate(names) if n == "apple"])
    s2 = np.vstack([samples[i] for i, n in enumerate(names) if n == "apple_2"])
    rng = np.random.default_rng(0)
    p1, p2 = s1[rng.choice(len(s1), 40)], s2[rng.choice(len(s2), 40)]
    assert E.apple_counts(p1, samples, gt.kind) == (40, 0)
    assert E.apple_counts(p2, samples, gt.kind) == (0, 40)
    assert E.target_class(*E.apple_counts(p1, samples, gt.kind)) == "apple"
    assert E.target_class(*E.apple_counts(p2, samples, gt.kind)) == "apple_2"
    assert E.target_class(*E.apple_counts(np.vstack([p1, p2[:15]]), samples, gt.kind)) == "both"
    assert E.target_class(*E.apple_counts(np.vstack([p1, p2[:5]]), samples, gt.kind)) == "apple"
    assert E.target_class(0, 0) == "none"
    lab = E.class_labels(np.vstack([p1[:3], p2[:3], [[5.0, 5.0, 5.0]]]), samples, gt.kind, np.zeros((0, 3)))
    assert lab.tolist() == ["apple"] * 3 + ["apple_2"] * 3 + ["none"]
    assert E.hand_apple({"apple": 0.30, "apple_2": 0.30}, {"apple": 0.28, "apple_2": 0.299}) == "apple"
    assert E.hand_apple({"apple": 0.30, "apple_2": 0.30}, {"apple": 0.298, "apple_2": 0.31}) == "none"


def test_split_server_chunks(tmp_path):
    from AG3S.ag3s.experiments.studies import t80_eval as E
    srv = tmp_path / "server_ag3s_8298_x"
    (srv / "server_constraints" / "run_0000").mkdir(parents=True)
    seeds = [563, None, None, 563, None, 563, None, None, None]
    for i, s in enumerate(seeds):
        np.savez(srv / "server_constraints" / "run_0000" / f"chunk_{i:05d}.npz",
                 summary_json=np.asarray(json.dumps({"policy_seed": {"applied": s, "episode": 563}})))
    (srv / "order.txt").write_text("A_l\nA_r\nB_l\n")
    out = E.split_server_chunks(srv)
    assert [len(out[k]) for k in ("A_l", "A_r", "B_l")] == [3, 2, 4]
    (srv / "order.txt").write_text("A_l\nA_r\n")
    with pytest.raises(ValueError):
        E.split_server_chunks(srv)


def test_eval_c_follow_and_clear(mod, monkeypatch, tmp_path, episode_ok):
    """C metrics on a stub-policy client run (move fired after step 0) with hand-made server chunks: request 0 and 1
    target the old position, request 2 the new one -> followed after 1 request, old region cleared after 1."""
    from AG3S.ag3s.experiments.studies import t80_eval as E
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    out = tmp_path / "C"
    monkeypatch.setattr(mod, "prepare_run_outputs", lambda args, names: setattr(args, "run_dir", tmp_path))
    monkeypatch.setattr(mod, "configure_run_logging", lambda name: None)
    monkeypatch.setattr(mod, "load_remote_policy", lambda remote: _FakePolicy())
    monkeypatch.setattr("sys.argv", ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1",
                                     "--episode-index", str(EPISODE), "--headless", "--warmup-steps", "50", "--max-steps", "24",
                                     "--record-frames", str(out), "--record-ag3s", str(out / "ag3s"), "--record-depth",
                                     "--remove-objects", "banana", "orange", "pear",
                                     "--move-target-when-palm-mm", "1e6", "--move-target-dxy", "0.06", "0.0"])
    mod.main()
    gt = E.build_truth(str(XML), extra=False)
    steps = sorted((out / "ag3s" / "run_0000").glob("step_*.npz"))
    assert len(steps) == 3
    ai = [i for i, k in enumerate(gt.kind) if k[1] == "apple"]
    pts = []
    for st in steps:
        samples, _ = gt.at(np.load(st)["qpos"])
        pts.append(np.vstack([samples[i] for i in ai]))
    chunks = []
    for j, tp in enumerate([pts[0], pts[0], pts[2]]):
        f = tmp_path / f"chunk_{j:05d}.npz"
        np.savez(f, summary_json=np.asarray(json.dumps({"has_target": True, "policy_seed": {"applied": None}})),
                 target_points=tp, target_free_removed_ijk=np.zeros((0, 3), int),
                 target_free_trace_origin=np.zeros(3), target_free_trace_voxel_size=np.asarray(0.005))
        chunks.append(f)
    r = E.evaluate_run({"name": "C", "experiment": "C", "requested_body": "apple"}, out, chunks, gt)
    s = r["summary"]
    assert s["move_request"] == 1
    assert s["requests_to_follow"] == 1 and s["requests_to_clear_old"] == 1
    c1, c2 = r["rows"][1]["c"], r["rows"][2]["c"]
    assert c1["old_target_points"] > 0 and c2["old_target_points"] == 0 and c2["target_to_new_mm"] <= E.FOLLOW_MM
    assert c1["target_to_new_mm"] > 40.0
