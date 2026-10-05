"""T42-2 — black obstacles for the pick and the place phase, in pi05_TO_hybrid.

What this file pins:

1. **clear = the bd6a002 scene, bit for bit.** `--obstacle-profile clear` runs `pi05_infer.main()` on
   `model_transport.xml` with no manager; every numeric `MjModel` array and the reset state equal those of the
   bd6a002 client run on the bd6a002 asset tree.  The recolouring changes visual `geom_rgba` only: every
   other array of the obstacle XML is the bd6a002 one for every geom that existed there.
2. **Slots and config.** `post_0`, `fence_0`, `overbar_0` (pick) and `crate_post_0` (place): capsule/cylinder
   only, one collision geom per mocap body + a visual twin, black by default.  Size keys per slot with ranges and
   defaults, wrong-slot keys rejected, `rgba` per entry.
3. **Placement and sizes.** Position / yaw / sizes land where the config says (post top, bar centre and axis,
   legs from the table to the bar).  A runtime size gives the *same model* as compiling that size: every numeric
   `MjModel` array is bit-identical to an `MjSpec` compile (and the solver's `m.stat` values; the drawing-only
   `extent` / `center` / `meansize` keep their XML-compile values), and MuJoCo's own contacts are the same.  `park_all`
   restores the compiled model.  Each size key moves the contact boundary: a value just below / above it gives
   separation / a MuJoCo contact with the robot or a fruit (so broad-phase and BVH see the new size).
4. **Physical collision** with the robot (contact, `robot_collision`, `unsafe`, contact force) and with fruits
   (dropped apple / orange: contact force, no pass-through).
5. **Distances** for every new slot = an independent GJK against robot meshes, fruits and the crate; the reset
   clearance (`object_clearance`) and the run-time stop (`robot_clearance`, `unsafe`) go through it.
6. **Two slots at once** (a pick and a place obstacle): union of geoms, clearance = min of the singles = GJK, and
   `main()` reaches the policy call with both placed.

CPU only: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
mujoco = pytest.importorskip("mujoco")

from tests.sim._gjk import gjk_dist, make_support  # noqa: E402

HYBRID = pathlib.Path("/mnt/dev/work/pi05_TO_hybrid")
MJCF_REL = pathlib.Path("rby1_description") / "models" / "rby1a" / "mujoco"
MJCF = HYBRID / MJCF_REL
TRANSPORT_XML = MJCF / "model_transport.xml"
TRANSPORT_OBSTACLE_XML = MJCF / "model_transport_pick_place_obstacles.xml"
PI05_INFER = HYBRID / "rby1_bringup" / "pi05_infer.py"
#: pi05_TO_hybrid HEAD before T42-2 (T40 H)
BASE_COMMIT = "bd6a002"
TABLE_Z = 0.82
NEW_SLOTS = ("post_0", "fence_0", "overbar_0", "crate_post_0")
CAP = 0.25
TOL = 5e-5
#: the numpy reference GJK can return ~1e-10 instead of 0.0 for a deep overlap; below this it is "touching"
TOUCH = 1e-6

if not TRANSPORT_OBSTACLE_XML.exists():  # pragma: no cover - asset-dependent
    pytest.skip("pi05_TO_hybrid scene assets not present", allow_module_level=True)
if str(HYBRID / "rby1_manipulation" / "src") not in sys.path:
    sys.path.insert(0, str(HYBRID / "rby1_manipulation" / "src"))

from rby1_manipulation.simulation import pick_place_obstacles as ppo  # noqa: E402

G = mujoco.mjtGeom


def _config(*entries, name="p") -> dict:
    return {
        "version": 1,
        "table_surface_z": TABLE_Z,
        "profiles": {"clear": {"scene": "any", "obstacles": []},
                     name: {"scene": "fruit", "obstacles": [dict(e) for e in entries]}},
    }


def _entry(slot, position=(0.62, 0.0, TABLE_Z), yaw=0.0, **sizes) -> dict:
    return {"slot": slot, "position": list(position), "yaw": yaw, **sizes}


def _gid(m, name):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert g >= 0, name
    return g


def _bid(m, name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    assert b >= 0, name
    return b


def _qadr(m, joint):
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)])


def _teleop_scene(xml=TRANSPORT_OBSTACLE_XML, model=None):
    m = model if model is not None else mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop"))
    for a in range(m.nu):
        d.ctrl[a] = d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]]
    mujoco.mj_forward(m, d)
    return m, d


def _manager(m, d, *entries):
    return ppo.PickPlaceObstacleManager(m, d, _config(*entries))


def _gjk(m, d, g1, g2) -> float:
    return gjk_dist(make_support(m, d, g1), make_support(m, d, g2), d.geom_xpos[g2] - d.geom_xpos[g1] + 1e-9)


def _gjk_min(m, d, obstacles, others, cap=CAP) -> float:
    best = cap
    for h in obstacles:
        for o in others:
            if np.linalg.norm(d.geom_xpos[h] - d.geom_xpos[o]) - m.geom_rbound[h] - m.geom_rbound[o] > cap:
                continue
            best = min(best, _gjk(m, d, h, int(o)))
    return best


def _numeric_arrays(m) -> dict:
    out = {}
    for name in dir(m):
        if name.startswith("_"):
            continue
        try:
            value = getattr(m, name)
        except Exception:  # pragma: no cover - property that needs arguments
            continue
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            out[name] = value
    return out


def _model_diff(a, b, ignore_paths=False) -> list[str]:
    """Names of numeric MjModel arrays that differ (shape or any bit).  `ignore_paths`: skip `*_pathadr`, the
    offsets of asset file names in `m.paths` (a model loaded from another directory stores longer names)."""
    fa, fb = _numeric_arrays(a), _numeric_arrays(b)
    assert fa.keys() == fb.keys()
    return sorted(k for k in fa if not (ignore_paths and k.endswith("_pathadr"))
                  and (fa[k].shape != fb[k].shape or not np.array_equal(fa[k], fb[k])))


def _contacts(d) -> list[tuple]:
    return sorted(
        (int(c.geom1), int(c.geom2), float(c.dist), *np.round(c.pos, 12).tolist())
        for c in d.contact[: d.ncon]
    )


# ----------------------------------------------------------------------------- pi05_infer.main() harness


class _Reached(Exception):
    pass


def _load_client(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:  # the client needs its own venv (mink, PIL): .venv-openpi-live has them
        pytest.skip(f"pi05_infer.py dependencies missing in this interpreter: {exc}")
    return mod


def _run_until_policy(client, episode, argv) -> dict:
    """Run `client.main()` up to `load_remote_policy` (no server) and return main()'s locals there."""
    captured: dict = {}

    def stub(remote):
        captured.update(inspect.currentframe().f_back.f_locals)
        raise _Reached

    original = client.load_remote_policy
    client.load_remote_policy = stub
    old_argv = sys.argv
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1",
                "--episode-index", str(episode), "--headless", *argv]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            client.main()
    except _Reached:
        pass
    except SystemExit as exc:
        raise AssertionError(f"main() exited ({exc.code}):\n{out.getvalue()[-2000:]}") from exc
    finally:
        sys.argv = old_argv
        client.load_remote_policy = original
    assert captured, "main() never reached load_remote_policy"
    captured["_stdout"] = out.getvalue()
    return captured


@pytest.fixture(scope="module")
def client():
    try:
        import PIL  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("PIL missing")
    return _load_client(PI05_INFER, "_t42_pi05_infer")


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(HYBRID), *args], check=True, capture_output=True, text=True).stdout


def _mirror(src: pathlib.Path, dst: pathlib.Path, rel: pathlib.Path, changed: set[str]) -> None:
    """dst = src with every path in `changed` (repo-relative) taken from BASE_COMMIT; the rest symlinked."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        entry_rel = (rel / entry.name).as_posix()
        hits = [c for c in changed if c == entry_rel or c.startswith(entry_rel + "/")]
        if not hits:
            (dst / entry.name).symlink_to(entry)
        elif entry.is_dir():
            _mirror(entry, dst / entry.name, rel / entry.name, changed)
        else:
            try:
                (dst / entry.name).write_bytes(subprocess.run(
                    ["git", "-C", str(HYBRID), "show", f"{BASE_COMMIT}:{entry_rel}"],
                    check=True, capture_output=True).stdout)
            except subprocess.CalledProcessError:  # new in the working tree: absent at BASE_COMMIT
                pass


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory):
    """A repo root that is pi05_TO_hybrid at BASE_COMMIT for rby1_description and rby1_bringup (files changed
    since are restored from git, the rest symlinked).  rby1_manipulation is the working tree: one process can
    import one copy, and `clear` never constructs the obstacle manager (the only module T42-2 changed there)."""
    try:
        changed = set(_git("diff", "--name-only", BASE_COMMIT, "--", "rby1_description", "rby1_bringup").split())
        changed |= set(_git("ls-files", "--others", "--exclude-standard", "--", "rby1_description",
                            "rby1_bringup").split())
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        pytest.skip(f"cannot read {BASE_COMMIT} from git")
    root = tmp_path_factory.mktemp("bd6a002")
    for sub in ("rby1_description", "rby1_bringup"):
        _mirror(HYBRID / sub, root / sub, pathlib.Path(sub), changed)
    (root / "rby1_manipulation").symlink_to(HYBRID / "rby1_manipulation")
    return root, changed


@pytest.fixture(scope="module")
def base_client(base_tree):
    root, _ = base_tree
    path = root / "rby1_bringup" / "pi05_infer.py"
    alias = path.with_name("pi05_infer_bd6a002.py")
    alias.write_bytes(path.read_bytes())
    return _load_client(alias, "_t42_pi05_infer_base")


# ----------------------------------------------------------------------------- 1. clear = the bd6a002 scene


def test_clear_profile_is_the_bd6a002_scene_bit_for_bit(client, base_client, base_tree):
    root, changed = base_tree
    assert (MJCF_REL / "scenes" / "pick_place_obstacles.xml").as_posix() in changed  # the mirror is not a no-op
    new = _run_until_policy(client, 1828, ["--obstacle-profile", "clear"])
    old = _run_until_policy(base_client, 1828, ["--obstacle-profile", "clear"])
    assert pathlib.Path(new["model_xml"]).resolve() == TRANSPORT_XML.resolve()
    assert pathlib.Path(old["model_xml"]).parent == root / MJCF_REL
    assert new["obstacle_manager"] is None and old["obstacle_manager"] is None
    m_new, d_new, m_old, d_old = new["m"], new["d"], old["m"], old["d"]
    for slot in NEW_SLOTS:
        assert mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_BODY, f"pick_obstacle_{slot}") == -1
    assert _model_diff(m_new, m_old, ignore_paths=True) == []
    for field in ("qpos", "qvel", "qacc", "ctrl", "act", "mocap_pos", "mocap_quat", "time", "geom_xpos", "xpos"):
        assert np.array_equal(getattr(d_new, field), getattr(d_old, field)), field


def test_recolouring_changes_visual_rgba_only(base_tree):
    """Obstacle XML vs its bd6a002 version: every geom that existed has the same arrays except visual rgba;
    collision geometry of bollard / divider / hurdle is unchanged; all obstacle visuals are black now."""
    root, _ = base_tree
    new = mujoco.MjModel.from_xml_path(str(TRANSPORT_OBSTACLE_XML))
    old = mujoco.MjModel.from_xml_path(str(root / MJCF_REL / TRANSPORT_OBSTACLE_XML.name))
    assert (new.nq, new.nv, new.nu) == (old.nq, old.nv, old.nu)
    names_old = [mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(old.ngeom)]
    for g_old, name in enumerate(names_old):
        g_new = g_old if name is None else _gid(new, name)
        assert g_new == g_old  # new geoms are appended: no index moves
        for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                      "geom_group", "geom_friction", "geom_solimp", "geom_solref", "geom_margin", "geom_gap",
                      "geom_condim", "geom_rbound", "geom_aabb"):
            assert np.array_equal(getattr(new, field)[g_new], getattr(old, field)[g_old]), (name, field)
        recoloured = name is not None and name.startswith("pick_obstacle_") and not (
            old.geom_contype[g_old] or old.geom_conaffinity[g_old])
        if recoloured:
            assert np.array_equal(new.geom_rgba[g_new], np.float32(ppo.BLACK_RGBA)), name
        else:
            assert np.array_equal(new.geom_rgba[g_new], old.geom_rgba[g_old]), name
    for b in range(old.nbody):
        name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_BODY, b)
        if name:
            assert _bid(new, name) == b
        assert np.array_equal(new.body_pos[b], old.body_pos[b]) and new.body_mocapid[b] == old.body_mocapid[b]


def test_obstacle_xml_with_new_slots_parked_steps_like_clear():
    m0, d0 = _teleop_scene(TRANSPORT_XML)
    m1, d1 = _teleop_scene()
    ppo.PickPlaceObstacleManager(m1, d1, _config()).activate("clear")
    assert np.array_equal(d0.qpos, d1.qpos)
    for _ in range(200):
        mujoco.mj_step(m0, d0)
        mujoco.mj_step(m1, d1)
    np.testing.assert_allclose(d1.qpos, d0.qpos, atol=1e-9)


# ----------------------------------------------------------------------------- 2. slots and config


@pytest.mark.parametrize("slot", NEW_SLOTS)
def test_new_slot_is_capsules_and_cylinders_one_collision_geom_per_mocap_body(slot):
    m = mujoco.MjModel.from_xml_path(str(TRANSPORT_OBSTACLE_XML))
    for body_name in ppo.SLOT_BODIES[slot]:
        body = _bid(m, body_name)
        assert m.body_mocapid[body] >= 0
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
        collision = [g for g in geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
        visual = [g for g in geoms if g not in collision]
        assert len(collision) == 1 and len(visual) == 1, body_name
        (c,), (v,) = collision, visual
        assert int(m.geom_type[c]) in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER)
        assert (m.geom_group[c], m.geom_contype[c], m.geom_conaffinity[c]) == (3, 1, 1)
        assert (m.geom_group[v], m.geom_contype[v], m.geom_conaffinity[v]) == (2, 0, 0)
        assert m.geom_type[v] == m.geom_type[c] and np.array_equal(m.geom_size[v], m.geom_size[c])
        assert np.array_equal(m.geom_quat[v], m.geom_quat[c]) and not m.geom_pos[c].any()
        assert np.array_equal(m.geom_rgba[v], np.float32(ppo.BLACK_RGBA))
        assert m.body_bvhnum[body] == 1 and not m.body_ipos[body].any()
    assert ppo.SLOT_PHASE[slot] == ("place" if slot == "crate_post_0" else "pick")


def test_every_obstacle_visual_is_black_by_default():
    m = mujoco.MjModel.from_xml_path(str(TRANSPORT_OBSTACLE_XML))
    d = mujoco.MjData(m)
    mgr = _manager(m, d)
    assert set(mgr._visual_geoms) == set(ppo.SLOTS)
    for slot, geoms in mgr._visual_geoms.items():
        assert geoms, slot
        for g in geoms:
            assert np.array_equal(m.geom_rgba[g], np.float32(ppo.BLACK_RGBA)), (slot, g)


@pytest.mark.parametrize("slot", NEW_SLOTS)
def test_size_keys_are_validated_and_default(slot, tmp_path):
    params = ppo.SLOT_PARAMS[slot]
    entry = ppo.validate_pick_place_obstacle_config(_config(_entry(slot)))["profiles"]["p"]["obstacles"][0]
    assert {k: entry[k] for k in params} == {k: v[2] for k, v in params.items()}
    for key, (low, high, default) in params.items():
        for ok in (low, 0.5 * (low + high), high):
            e = ppo.validate_pick_place_obstacle_config(_config(_entry(slot, **{key: ok})))
            assert e["profiles"]["p"]["obstacles"][0][key] == pytest.approx(ok)
        for bad in (low - 1e-3, high + 1e-3, float("nan"), float("inf")):
            with pytest.raises(ValueError, match=key):
                ppo.validate_pick_place_obstacle_config(_config(_entry(slot, **{key: bad})))
    for key in set(ppo.SIZE_KEYS) - set(params):
        with pytest.raises(ValueError, match="only valid for slots"):
            ppo.validate_pick_place_obstacle_config(_config(_entry(slot, **{key: 0.1})))
    sizes = {k: v[0] for k, v in params.items()}
    raw = _config(_entry(slot, position=[0.6, 0.21, 0.82], yaw=0.3, rgba=[0.1, 0.2, 0.3, 1.0], **sizes))
    path = tmp_path / "t42.json"
    path.write_text(json.dumps(raw))
    assert ppo.load_pick_place_obstacle_config(path)["profiles"]["p"]["obstacles"][0] == {
        "slot": slot, "position": [0.6, 0.21, 0.82], "yaw": 0.3, "rgba": [0.1, 0.2, 0.3, 1.0], **sizes}


def test_old_slots_keep_their_keys_and_reject_the_new_ones():
    for slot in ("bollard_0", "divider_0"):
        for key in ppo.SIZE_KEYS:
            with pytest.raises(ValueError, match="only valid for slots"):
                ppo.validate_pick_place_obstacle_config(_config(_entry(slot, **{key: 0.1})))
    for key in ("height", "radius", "length"):
        with pytest.raises(ValueError, match="only valid for slots"):
            ppo.validate_pick_place_obstacle_config(_config(_entry("hurdle_0", **{key: 0.1})))
    e = ppo.validate_pick_place_obstacle_config(_config(_entry("hurdle_0")))["profiles"]["p"]["obstacles"][0]
    assert e == {"slot": "hurdle_0", "position": [0.62, 0.0, TABLE_Z], "yaw": 0.0, "bar_height": 0.26}


@pytest.mark.parametrize("rgba", [[0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 1.2, 1], [0, 0, -0.1, 1], [0, float("nan"), 0, 1]])
def test_rgba_is_validated(rgba):
    with pytest.raises(ValueError, match="rgba"):
        ppo.validate_pick_place_obstacle_config(_config(_entry("post_0", rgba=rgba)))


def test_rgba_recolours_the_visual_geoms_only_and_park_restores_black():
    m, d = _teleop_scene()
    red = [0.9, 0.1, 0.1, 1.0]
    mgr = _manager(m, d, _entry("post_0", position=(0.62, 0.0, TABLE_Z), rgba=red),
                   _entry("fence_0", position=(0.70, 0.2, TABLE_Z)))
    before = m.geom_rgba.copy()
    mgr.activate("p")
    changed = {int(g) for g in np.flatnonzero(np.any(m.geom_rgba != before, axis=1))}
    assert changed == set(mgr._visual_geoms["post_0"])
    for g in changed:
        assert np.array_equal(m.geom_rgba[g], np.float32(red))
    mgr.activate("clear")
    assert np.array_equal(m.geom_rgba, before)


# ----------------------------------------------------------------------------- 3. placement and sizes


def _size_grid(slot):
    params = ppo.SLOT_PARAMS[slot]
    yield {k: v[2] for k, v in params.items()}
    yield {k: v[0] for k, v in params.items()}
    yield {k: v[1] for k, v in params.items()}


def _slot_geom(m, slot, part):
    suffix = {"post": "_collision", "bar": "_collision_bar", "leg_left": "_collision_leg_left",
              "leg_right": "_collision_leg_right"}[part]
    return _gid(m, f"pick_obstacle_{slot}{suffix}")


def _segment(m, d, g):
    """End points of a capsule/cylinder axis (world)."""
    axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
    return d.geom_xpos[g] - m.geom_size[g][1] * axis, d.geom_xpos[g] + m.geom_size[g][1] * axis


@pytest.mark.parametrize("slot", NEW_SLOTS)
@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0])
def test_activate_places_and_sizes_the_slot(slot, yaw):
    m, d = _teleop_scene()
    pos = np.array([0.60, 0.18, TABLE_Z])
    grid = list(_size_grid(slot))
    cfg = _config()
    for i, sizes in enumerate(grid):
        cfg["profiles"][f"s{i}"] = {"scene": "fruit", "obstacles": [_entry(slot, position=pos, yaw=yaw, **sizes)]}
    mgr = ppo.PickPlaceObstacleManager(m, d, cfg)
    for i, sizes in enumerate(grid):
        mgr.activate(f"s{i}")
        assert mgr.active_slots == (slot,)
        np.testing.assert_allclose(mgr.positions()[slot], pos)
        direction = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        if slot in ppo.POST_SLOTS:
            g = _slot_geom(m, slot, "post")
            low, high = _segment(m, d, g)
            np.testing.assert_allclose(sorted([low[2], high[2]]), [TABLE_Z, TABLE_Z + sizes["height"]], atol=1e-12)
            np.testing.assert_allclose(d.geom_xpos[g][:2], pos[:2], atol=1e-12)
            assert m.geom_size[g][0] == sizes["radius"]
        else:
            bar = _slot_geom(m, slot, "bar")
            np.testing.assert_allclose(d.geom_xpos[bar], pos + [0, 0, sizes["bar_height"]], atol=1e-12)
            axis = d.geom_xmat[bar].reshape(3, 3)[:, 2]
            assert abs(axis @ direction) == pytest.approx(1.0, abs=1e-12)
            assert tuple(m.geom_size[bar][:2]) == (sizes["radius"], sizes["length"] / 2)
            for part, sign in (("leg_left", -1.0), ("leg_right", 1.0)):
                leg = _slot_geom(m, slot, part)
                low, high = sorted(_segment(m, d, leg), key=lambda p: p[2])
                foot = pos + sign * sizes["length"] / 2 * direction
                np.testing.assert_allclose(low, foot, atol=1e-12)  # stands on the table at the bar end
                np.testing.assert_allclose(high, foot + [0, 0, sizes["bar_height"]], atol=1e-12)  # up to the bar axis
                assert m.geom_size[leg][0] == ppo.FRAME_LEG_RADIUS[slot]
        for body in mgr._slot_body_ids[slot]:  # the visual twin is where the collision geom is, same size
            c, v = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
            assert np.array_equal(d.geom_xpos[c], d.geom_xpos[v]) and np.array_equal(m.geom_size[c], m.geom_size[v])


def _compiled_with_sizes(slot, sizes):
    spec = mujoco.MjSpec.from_file(str(TRANSPORT_OBSTACLE_XML))
    for body_name, (_, size) in zip(ppo.SLOT_BODIES[slot], ppo.slot_layout(slot, sizes)):
        for geom in spec.body(body_name).geoms:
            geom.size = [size[0], size[1], 0.0]
    return spec.compile()


def test_mjspec_compile_equals_from_xml_path():
    """The reference below is an MjSpec compile; with no edit it is the XML model bit for bit."""
    assert _model_diff(mujoco.MjSpec.from_file(str(TRANSPORT_OBSTACLE_XML)).compile(),
                       mujoco.MjModel.from_xml_path(str(TRANSPORT_OBSTACLE_XML))) == []


@pytest.mark.parametrize("slot", NEW_SLOTS)
def test_runtime_sizes_equal_a_model_compiled_with_those_sizes(slot):
    """Every numeric MjModel array after activate() = an MjSpec compile with the same sizes; MuJoCo's contacts
    with the slot overlapping the robot and the payload are the same in both; park_all restores the XML model."""
    rng = np.random.default_rng(42)
    pristine = mujoco.MjModel.from_xml_path(str(TRANSPORT_OBSTACLE_XML))
    m, d = _teleop_scene()
    grids = list(_size_grid(slot)) + [
        {k: float(rng.uniform(lo, hi)) for k, (lo, hi, _) in ppo.SLOT_PARAMS[slot].items()} for _ in range(2)]
    cfg = _config()
    for i, sizes in enumerate(grids):
        cfg["profiles"][f"s{i}"] = {"scene": "fruit", "obstacles": [_entry(slot, position=(0.62, 0.0, TABLE_Z), **sizes)]}
    mgr = ppo.PickPlaceObstacleManager(m, d, cfg)
    for i, sizes in enumerate(grids):
        mgr.activate(f"s{i}")
        ref = _compiled_with_sizes(slot, sizes)
        assert _model_diff(m, ref) == [], sizes
        # m.stat: the solver's meaninertia and meanmass are the same.  extent / center / meansize (scene bounding
        # box and mean geom size the compiler keeps for drawing: free-camera default, frame / contact marker
        # scale; the parked slots at x = 12-15 m set the box) are not recomputed at run time and keep the values
        # of the XML compile, as in every obstacle run.
        for field in ("meaninertia", "meanmass"):
            assert getattr(m.stat, field) == getattr(ref.stat, field), field
        # same contacts: the slot dropped onto the left gripper and onto the apple, both models
        m_ref, d_ref = _teleop_scene(model=ref)
        for target in (_gid(m, "apple_geom"), max(
                (g for g in mgr.robot_geom_ids if m.geom_bodyid[g] == _bid(m, "ee_finger_l1")),
                key=lambda g: m.geom_rbound[g])):
            mgr._place(slot, np.zeros(3), 0.4, sizes=sizes)
            mujoco.mj_kinematics(m, d)
            offset = d.geom_xpos[mgr._collision_geoms[slot][0]].copy()  # the post / the bar, footprint at 0
            mgr._place(slot, d.geom_xpos[target] - offset + [0.004, 0.0, 0.0], 0.4, sizes=sizes)
            mujoco.mj_forward(m, d)
            d_ref.qpos[:] = d.qpos
            d_ref.mocap_pos[:] = d.mocap_pos
            d_ref.mocap_quat[:] = d.mocap_quat
            mujoco.mj_forward(m_ref, d_ref)
            ours = [c for c in _contacts(d) if {c[0], c[1]} & set(mgr._collision_geoms[slot])]
            assert ours, (slot, sizes, "no contact with the slot")
            assert _contacts(d) == _contacts(d_ref)
    mgr.activate("clear")
    assert _model_diff(m, pristine) == []


# --- each size key moves MuJoCo's contact boundary


def _gripper_geoms(m, mgr, side="l"):
    bodies = {_bid(m, f"ee_finger_{side}1"), _bid(m, f"ee_finger_{side}2"), _bid(m, f"EE_BODY_{side.upper()}")}
    return [g for g in mgr.robot_geom_ids if int(m.geom_bodyid[g]) in bodies]


def _lowest_gripper_point(m, d, mgr):
    best = None
    for g in _gripper_geoms(m, mgr):
        p = make_support(m, d, g)(np.array([0.0, 0.0, -1.0]))
        if best is None or p[2] < best[2]:
            best = p
    return best


def _boundary_case(m, d, mgr, slot, key):
    """(probe geoms, footprint, yaw, base sizes, search interval of `key`) such that the slot touches the probe
    at one end of the interval and is clear of it at the other, and the contact set grows monotonically in
    between (the interval is the slot's range unless noted)."""
    low, high, _ = ppo.SLOT_PARAMS[slot][key]
    sizes = {k: v[2] for k, v in ppo.SLOT_PARAMS[slot].items()}
    apple = [_gid(m, "apple_geom")]
    a = d.geom_xpos[apple[0]].copy()
    apple_top = make_support(m, d, apple[0])(np.array([0.0, 0.0, 1.0]))[2] - TABLE_Z
    tip = _lowest_gripper_point(m, d, mgr)
    gripper = _gripper_geoms(m, mgr)
    under_tip = [tip[0], tip[1], TABLE_Z]
    if slot in ppo.POST_SLOTS and key == "height":  # post right under the gripper's lowest point
        return gripper, under_tip, 0.0, sizes, (low, high)
    if slot in ppo.POST_SLOTS and key == "radius":  # post beside the apple, axis 0.07 m out along +x
        return apple, [a[0] + 0.07, a[1], TABLE_Z], 0.0, sizes, (low, high)
    if slot == "fence_0" and key == "bar_height":  # rail right across the apple: through it .. above it
        return apple, [a[0], a[1], TABLE_Z], 0.0, sizes, (low, high)
    if slot == "fence_0" and key == "radius":  # rail 12 mm above the apple top: touches at r ~ 12 mm
        sizes["bar_height"] = apple_top + 0.012
        return apple, [a[0], a[1], TABLE_Z], 0.0, sizes, (low, high)
    if key == "length":  # bar high above the apple, apple on the bar axis `s` from the centre:
        # the leg at length/2 reaches the apple; search up to length/2 = s (leg through the apple centre)
        s = 0.14 if slot == "fence_0" else 0.17
        sizes["bar_height"] = ppo.SLOT_PARAMS[slot]["bar_height"][1] if slot == "fence_0" else 0.30
        assert sizes["bar_height"] > apple_top + 0.03
        return apple, [a[0] - s, a[1], TABLE_Z], 0.0, sizes, (low, 2 * s)
    if slot == "overbar_0" and key == "bar_height":  # bar across, right under the gripper: below .. into it
        return gripper, under_tip, math.pi / 2, sizes, (low, tip[2] - TABLE_Z + 0.02)
    if slot == "overbar_0" and key == "radius":  # bar axis 19 mm under the gripper tip
        sizes["bar_height"] = tip[2] - TABLE_Z - 0.019
        return gripper, under_tip, math.pi / 2, sizes, (low, high)
    raise AssertionError((slot, key))


def _touches(m, d, mgr, slot, probe) -> float:
    mujoco.mj_kinematics(m, d)
    return _gjk_min(m, d, mgr._collision_geoms[slot], probe)


def _mj_contact(m, d, mgr, slot, probe) -> bool:
    mujoco.mj_forward(m, d)
    ours, theirs = set(mgr._collision_geoms[slot]), set(probe)
    return any({int(c.geom1), int(c.geom2)} & ours and {int(c.geom1), int(c.geom2)} & theirs
               for c in d.contact[: d.ncon])


SIZE_CASES = [(slot, key) for slot in NEW_SLOTS for key in ppo.SLOT_PARAMS[slot]]


@pytest.mark.parametrize("slot,key", SIZE_CASES)
def test_each_size_key_moves_the_contact_boundary(slot, key):
    """Bisect `key` (independent GJK) for the value where the slot starts touching a probe (apple or gripper);
    4 mm on either side: GJK > 0 and no MuJoCo contact / GJK = 0 and a MuJoCo contact; the manager's
    clearance equals the GJK on both sides."""
    m, d = _teleop_scene()
    mgr = _manager(m, d, _entry(slot))
    mgr.activate("p")
    probe, footprint, yaw, sizes, (low, high) = _boundary_case(m, d, mgr, slot, key)
    assert ppo.SLOT_PARAMS[slot][key][0] <= low < high <= ppo.SLOT_PARAMS[slot][key][1]

    def gap(value):
        mgr._place(slot, footprint, yaw, sizes={**sizes, key: value})
        return _touches(m, d, mgr, slot, probe)

    g_low, g_high = gap(low), gap(high)
    assert (g_low > TOUCH) != (g_high > TOUCH), (slot, key, g_low, g_high)  # the boundary is inside the range
    sep_end = low if g_low > TOUCH else high
    a, b = sep_end, (high if sep_end == low else low)
    for _ in range(40):
        mid = 0.5 * (a + b)
        a, b = (mid, b) if gap(mid) > TOUCH else (a, mid)
    boundary = 0.5 * (a + b)
    step = 0.004 if sep_end == low else -0.004
    clear_value, touch_value = boundary - step, boundary + step
    assert low <= min(clear_value, touch_value) and max(clear_value, touch_value) <= high, (slot, key, boundary)
    others = sorted(probe)
    reference = gap(clear_value)
    assert reference > 1e-3
    assert not _mj_contact(m, d, mgr, slot, probe)
    assert mgr._minimum_distance(others, CAP) == pytest.approx(reference, abs=TOL)
    assert gap(touch_value) <= TOUCH
    assert _mj_contact(m, d, mgr, slot, probe), (slot, key, "MuJoCo misses the contact at the new size")
    assert mgr._minimum_distance(others, CAP) <= 0.0


# ----------------------------------------------------------------------------- 4. physical collision


def _contact_force(m, d, ours, theirs) -> float:
    total, f = 0.0, np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if pair & ours and pair & theirs:
            mujoco.mj_contactForce(m, d, i, f)
            total += f[0]
    return total


def _top_of(m, d, slot):
    """World point on top of the slot where a fruit can land (post top centre / bar centre top)."""
    if slot in ppo.POST_SLOTS:
        g = _slot_geom(m, slot, "post")
        return d.geom_xpos[g] + [0, 0, m.geom_size[g][1]]
    g = _slot_geom(m, slot, "bar")
    return d.geom_xpos[g] + [0, 0, m.geom_size[g][0]]


@pytest.mark.parametrize("slot", NEW_SLOTS)
@pytest.mark.parametrize("fruit", ["apple", "orange"])
def test_dropped_fruit_hits_the_slot_and_does_not_pass_through(slot, fruit):
    m, d = _teleop_scene()
    pos = (0.75, 0.0, TABLE_Z)  # in front of the crate, away from both teleop grippers
    sizes = {"height": 0.15} if slot in ppo.POST_SLOTS else {"bar_height": ppo.SLOT_PARAMS[slot]["bar_height"][1]}
    mgr = _manager(m, d, _entry(slot, position=pos, yaw=math.pi / 2, **sizes))
    mgr.activate("p")
    assert mgr.robot_clearance() > 0.05
    body_geoms = {g for g in mgr.payload_geom_ids if m.geom_bodyid[g] == _bid(m, fruit)}
    adr = _qadr(m, f"{fruit}_free")
    top = _top_of(m, d, slot)
    d.qpos[adr:adr + 7] = [top[0], top[1], top[2] + 0.07, 1, 0, 0, 0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    ours = set(mgr._collision_geoms[slot])
    peak, deepest, first_contact, centre_after_contact = 0.0, 0.0, None, []
    for step in range(int(0.4 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        mgr.observe_contacts()
        peak = max(peak, _contact_force(m, d, ours, body_geoms))
        for c in d.contact[: d.ncon]:
            if {int(c.geom1), int(c.geom2)} & ours and {int(c.geom1), int(c.geom2)} & body_geoms:
                deepest = min(deepest, float(c.dist))
                first_contact = step if first_contact is None else first_contact
        if first_contact is not None and step - first_contact < int(0.05 / m.opt.timestep):
            centre_after_contact.append(float(d.xpos[_bid(m, fruit)][2]))
    assert peak > 0.5, peak  # N: the obstacle pushes back
    assert mgr.payload_collision and not mgr.robot_collision
    assert any(f"{fruit}_geom" in " ".join(pair) and any(n.startswith(f"pick_obstacle_{slot}_collision") for n in pair)
               for pair in mgr.contact_pairs)
    # no pass-through: for 50 ms after the impact the fruit centre stays above the obstacle top, and the
    # soft-contact penetration stays under 1 cm (MuJoCo's impact depth here is 5-6 mm for every slot)
    assert min(centre_after_contact) > top[2], (min(centre_after_contact), top[2])
    assert deepest > -0.01, deepest


@pytest.mark.parametrize("slot", NEW_SLOTS)
def test_slot_overlapping_the_gripper_is_a_contact_and_an_obstacle_stop(slot):
    m, d = _teleop_scene()
    mgr = _manager(m, d, _entry(slot))
    mgr.activate("p")
    left = _gripper_geoms(m, mgr)
    finger = max((g for g in left if m.geom_bodyid[g] == _bid(m, "ee_finger_l1")), key=lambda g: m.geom_rbound[g])
    part = mgr._collision_geoms[slot][0]  # the post, or the bar
    mgr._place(slot, np.zeros(3), 0.0)
    mujoco.mj_kinematics(m, d)
    offset = d.geom_xpos[part].copy()
    mgr._place(slot, d.geom_xpos[finger] - offset, 0.0)  # bypasses the table-workspace validation
    mujoco.mj_forward(m, d)
    assert _gjk(m, d, part, finger) <= TOUCH
    assert mgr.pair_distance(part, finger, CAP) <= 0.0
    assert mgr.unsafe(0.02) and mgr.robot_collision
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom#{g}" for g in left}
    assert any(set(pair) & names for pair in mgr.contact_pairs)
    mujoco.mj_step(m, d)
    assert _contact_force(m, d, set(mgr._collision_geoms[slot]), set(left)) > 0.0


# ----------------------------------------------------------------------------- 5. distances = GJK


@pytest.mark.parametrize("slot", NEW_SLOTS)
def test_every_pair_and_both_clearances_equal_gjk(slot):
    """Random arm poses and sizes; the slot with one of its geoms at a random robot / payload geom (+-3 cm),
    random yaw: every pair within the cap = GJK (separated) or <= 0 (overlap); robot_clearance (run-time stop)
    and object_clearance (reset check) = the GJK minimum."""
    m, d = _teleop_scene()
    mgr = _manager(m, d, _entry(slot))
    mgr.activate("p")
    rng = np.random.default_rng(4202)
    arm = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_arm_{i}") for s in ("left", "right") for i in range(7)]
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    geoms = sorted(mgr._collision_geoms[slot])
    fruits = [g for g in mgr.payload_geom_ids if m.geom_type[g] == G.mjGEOM_MESH]
    seen: dict = {}
    n_overlap = 0
    for trial in range(10):
        mujoco.mj_resetDataKeyframe(m, d, key)
        for j in arm:
            lo, hi = m.jnt_range[j]
            d.qpos[m.jnt_qposadr[j]] = 0.4 * d.qpos[m.jnt_qposadr[j]] + 0.6 * rng.uniform(lo, hi)
        mujoco.mj_kinematics(m, d)
        pool = (mgr.robot_geom_ids, fruits, mgr.payload_geom_ids)[trial % 3]
        target = int(rng.choice(pool))
        part = int(rng.choice(geoms))
        yaw = rng.uniform(-np.pi, np.pi)
        sizes = {k: float(rng.uniform(lo, hi)) for k, (lo, hi, _) in ppo.SLOT_PARAMS[slot].items()}
        mgr._place(slot, np.zeros(3), yaw, sizes=sizes)
        mujoco.mj_kinematics(m, d)
        offset = d.geom_xpos[part].copy()
        mgr._place(slot, d.geom_xpos[target] - offset + rng.normal(0.0, 0.03, 3), yaw, sizes=sizes)
        mujoco.mj_kinematics(m, d)
        for others, clearance in ((mgr.robot_geom_ids, mgr.robot_clearance),
                                  (mgr.payload_geom_ids, mgr.object_clearance)):
            best = CAP
            for h in geoms:
                for o in others:
                    if np.linalg.norm(d.geom_xpos[h] - d.geom_xpos[o]) - m.geom_rbound[h] - m.geom_rbound[o] > CAP:
                        continue
                    reference, ours = _gjk(m, d, h, int(o)), mgr.pair_distance(h, int(o), CAP)
                    k = (int(m.geom_type[h]), int(m.geom_type[o]))
                    seen[k] = seen.get(k, 0) + 1
                    if reference <= TOUCH:
                        n_overlap += 1
                        assert ours <= TOUCH, (h, o, ours, reference)
                    else:
                        assert ours == pytest.approx(min(reference, CAP), abs=TOL), (h, o, ours, reference)
                    best = min(best, reference)
            value = clearance(CAP)
            if best > TOUCH:
                assert value == pytest.approx(best, abs=TOL)
            else:
                assert value <= 0.0
    assert n_overlap > 0
    kinds = {G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE} & {int(m.geom_type[g]) for g in geoms}
    for kind in kinds:
        for other in (G.mjGEOM_MESH, G.mjGEOM_BOX):
            assert seen.get((int(kind), int(other)), 0) > 0, (slot, kind, other)


@pytest.mark.parametrize("slot", ["post_0", "overbar_0", "crate_post_0"])
def test_stop_rule_threshold_on_the_new_slots(slot):
    """The slot 15 mm (GJK) below the gripper stops the run (<= 0.02 m); 30 mm does not."""
    m, d = _teleop_scene()
    mgr = _manager(m, d, _entry(slot))
    mgr.activate("p")
    tip = _lowest_gripper_point(m, d, mgr)
    left = _gripper_geoms(m, mgr)
    key = "height" if slot in ppo.POST_SLOTS else "bar_height"
    low, high, _ = ppo.SLOT_PARAMS[slot][key]
    for gap, stops in ((0.015, True), (0.030, False)):
        lo, hi = low, high
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            mgr._place(slot, [tip[0], tip[1], TABLE_Z], 0.3, sizes={key: mid})
            mujoco.mj_kinematics(m, d)
            lo, hi = (mid, hi) if _gjk_min(m, d, mgr._collision_geoms[slot], left) > gap else (lo, mid)
        mujoco.mj_forward(m, d)
        robot_gjk = _gjk_min(m, d, mgr._collision_geoms[slot], mgr.robot_geom_ids)
        assert robot_gjk == pytest.approx(gap, abs=1e-4)
        mgr.reset_metrics()
        assert mgr.unsafe(0.02) is stops, gap
        assert mgr.min_robot_clearance == pytest.approx(gap, abs=1e-4)


# ----------------------------------------------------------------------------- 6. two slots at once


@pytest.mark.parametrize("pair", [("post_0", "crate_post_0"), ("fence_0", "hurdle_0"), ("overbar_0", "crate_post_0")])
def test_a_pick_and_a_place_obstacle_together(pair):
    m, d = _teleop_scene()
    entries = {
        "post_0": _entry("post_0", position=(0.70, 0.22, TABLE_Z)),
        "fence_0": _entry("fence_0", position=(0.70, 0.25, TABLE_Z), yaw=0.2),
        "overbar_0": _entry("overbar_0", position=(0.72, 0.20, TABLE_Z), yaw=math.pi / 2, bar_height=0.3),
        "crate_post_0": _entry("crate_post_0", position=(0.66, -0.17, TABLE_Z)),
        "hurdle_0": _entry("hurdle_0", position=(0.75, -0.05, TABLE_Z), yaw=math.pi / 2),
    }
    cfg = _config(*(entries[s] for s in pair))
    for s in pair:
        cfg["profiles"][s] = {"scene": "fruit", "obstacles": [entries[s]]}
    mgr = ppo.PickPlaceObstacleManager(m, d, cfg)
    singles = {}
    for s in pair:
        mgr.activate(s)
        singles[s] = (mgr.robot_clearance(), mgr.object_clearance(), set(mgr.collision_geom_ids))
    mgr.activate("p")
    assert mgr.active_slots == pair
    assert mgr.collision_geom_ids == singles[pair[0]][2] | singles[pair[1]][2]
    assert singles[pair[0]][2].isdisjoint(singles[pair[1]][2])
    robot, payload = mgr.robot_clearance(), mgr.object_clearance()
    assert robot == pytest.approx(min(singles[s][0] for s in pair), abs=1e-9)
    assert payload == pytest.approx(min(singles[s][1] for s in pair), abs=1e-9)
    obstacles = sorted(mgr.collision_geom_ids)
    assert robot == pytest.approx(_gjk_min(m, d, obstacles, mgr.robot_geom_ids), abs=TOL)
    assert payload == pytest.approx(_gjk_min(m, d, obstacles, mgr.payload_geom_ids), abs=TOL)
    assert robot > 0.02 and payload > 0.0
    for s in pair:
        np.testing.assert_allclose(mgr.positions()[s], entries[s]["position"])


def test_main_reaches_the_policy_with_a_pick_and_a_place_obstacle(client, tmp_path):
    """Episode 1828 (apple, left arm): a black post beside the apple + a black post beside the crate rim
    (the T42-2 smoke profile), one of them recoloured; main()'s reset checks pass and both are placed."""
    post = _entry("post_0", position=(0.625, 0.30, TABLE_Z), height=0.20, radius=0.02)
    crate_post = _entry("crate_post_0", position=(0.64, 0.17, TABLE_Z), height=0.26, radius=0.02,
                        rgba=[0.2, 0.2, 0.2, 1.0])
    path = tmp_path / "two.json"
    path.write_text(json.dumps(_config(post, crate_post, name="two")))
    run = _run_until_policy(client, 1828, ["--obstacle-config", str(path), "--obstacle-profile", "two"])
    assert pathlib.Path(run["model_xml"]).resolve() == TRANSPORT_OBSTACLE_XML.resolve()
    m, d, mgr = run["m"], run["d"], run["obstacle_manager"]
    assert mgr.active_slots == ("post_0", "crate_post_0")
    assert "obstacles  : two ('post_0', 'crate_post_0')" in run["_stdout"]
    obstacles = sorted(mgr.collision_geom_ids)
    robot, payload = mgr.robot_clearance(), mgr.object_clearance()
    assert robot > 0.02 and payload > 0.0
    assert robot == pytest.approx(_gjk_min(m, d, obstacles, mgr.robot_geom_ids), abs=TOL)
    assert payload == pytest.approx(_gjk_min(m, d, obstacles, mgr.payload_geom_ids), abs=TOL)
    g = _slot_geom(m, "crate_post_0", "post")
    np.testing.assert_allclose(d.geom_xpos[g], [0.64, 0.17, TABLE_Z + 0.13], atol=1e-12)
    assert m.geom_size[_slot_geom(m, "post_0", "post")][1] == pytest.approx(0.10)
    for v in mgr._visual_geoms["crate_post_0"]:
        assert np.array_equal(m.geom_rgba[v], np.float32([0.2, 0.2, 0.2, 1.0]))
    for v in mgr._visual_geoms["post_0"]:
        assert np.array_equal(m.geom_rgba[v], np.float32(ppo.BLACK_RGBA))
    apple = _bid(m, "apple")
    np.testing.assert_allclose(d.xpos[apple][:2], [0.558, 0.332], atol=3e-3)
