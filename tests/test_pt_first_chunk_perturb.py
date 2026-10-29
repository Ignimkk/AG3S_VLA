"""AG3S-PRE-CORE-TO PT — `pi05_infer --first-chunk-perturb-rad R --perturb-seed K` (default off).

Arg validation, vector determinism per seed, only the used arm's 7 joints of steps 0..7 change,
the off path returns the action untouched, and the loop / record wiring is in place.
"""
import importlib.util
import pathlib
import re

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
INFER = ROOT / "integration/rby1/bringup/pi05_infer.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_pt_pi05_infer", INFER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert pathlib.Path(m.__file__).resolve() == INFER.resolve()
    return m


def _action():
    return np.arange(16, dtype=np.float64) * 0.1 + 0.05


def test_arg_validation(mod):
    err = mod.first_chunk_perturb_error
    assert err(None, None) is None                      # off
    assert err(1e-6, 1) is None
    assert err(1e-6, 0) is None
    assert "requires --perturb-seed" in err(1e-6, None)
    assert "--perturb-seed only" in err(None, 3)
    for bad in (0.0, -1e-6, float("nan"), float("inf")):
        assert "> 0" in err(bad, 1)
    assert ">= 0" in err(1e-6, -1)


def test_vector_deterministic_per_seed_and_differs_between_seeds(mod):
    v1 = mod.first_chunk_perturb_vector(1e-6, 1)
    assert v1.shape == (7,) and v1.dtype == np.float64
    assert np.array_equal(v1, mod.first_chunk_perturb_vector(1e-6, 1))
    assert np.all(np.abs(v1) <= 1e-6)
    vs = [mod.first_chunk_perturb_vector(1e-6, k) for k in (1, 2, 3, 4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(vs[i], vs[j])


def test_vector_does_not_touch_global_rng(mod):
    np.random.seed(123)
    expected = np.random.random(5)
    np.random.seed(123)
    mod.make_first_chunk_perturb(1e-6, 2, "left")
    assert np.array_equal(np.random.random(5), expected)


@pytest.mark.parametrize("arm,sl", [("left", slice(0, 7)), ("right", slice(8, 15))])
def test_only_used_arm_and_first_eight_steps_change(mod, arm, sl):
    p = mod.make_first_chunk_perturb(1e-3, 7, arm)
    assert p["arm"] == arm
    mask = np.zeros(16, dtype=bool)
    mask[sl] = True
    for t in range(20):
        a = _action()
        before = a.copy()
        out = mod.perturb_first_chunk_action(a, t, p)
        assert np.array_equal(a, before)                # input (a chunk row view) never written
        if t < 8:
            assert out is not a
            assert np.array_equal(out[~mask], before[~mask])   # grippers 7 / 15 and the other arm
            assert np.array_equal(out[mask], before[mask] + p["vector"])
        else:
            assert out is a


def test_chunk_row_view_not_mutated(mod):
    chunk = np.tile(_action(), (8, 1))
    ref = chunk.copy()
    p = mod.make_first_chunk_perturb(1e-3, 1, "right")
    for t in range(8):
        row = np.asarray(chunk[t], dtype=np.float64)    # same expression the loop uses (a view)
        mod.perturb_first_chunk_action(row, t, p)
    assert np.array_equal(chunk, ref)


def test_off_path_returns_action_untouched(mod):
    assert mod.make_first_chunk_perturb(None, None, None) is None
    for t in range(10):
        a = _action()
        assert mod.perturb_first_chunk_action(a, t, None) is a


def test_bad_arm_refused(mod):
    with pytest.raises(ValueError):
        mod.make_first_chunk_perturb(1e-6, 1, None)


def test_loop_and_record_wiring():
    src = INFER.read_text()
    # the perturbation is applied immediately before the action goes to ctrl
    assert re.search(r"action = perturb_first_chunk_action\(action, t_step, first_chunk_perturb\)\n"
                     r"\s*apply_action\(mcfg\[\"action_format\"\], action, d, idx, act\)", src)
    # the record appends the (perturbed) `action` that was applied
    assert "executed_actions.append(action.copy())" in src
    # traj.npz keys only when on (off = same key set as before)
    for key in ("perturb_rad", "perturb_seed", "perturb_vector", "perturb_arm"):
        assert f'"{key}"' in src
    assert "if first_chunk_perturb is not None else {})" in src
    # parser defaults: off
    assert re.search(r'"--first-chunk-perturb-rad", type=float, default=None', src)
    assert re.search(r'"--perturb-seed", type=int, default=None', src)


# --- stub-policy CPU run of main() (no server, no GPU): the flag end to end --------------------------------

#: traj.npz keys of this stub run with the flag off (= the client before PT, checked bit for bit 2026-10-26).
OFF_KEYS = sorted(['chunk_indices', 'chunk_start_steps', 'condition', 'control_hz', 'executed_actions',
                   'execution_length', 'fruit_basket_offset_m', 'fruit_layout_index', 'fruit_preloaded',
                   'fruit_slot_order', 'inference_ms', 'inference_states', 'measured_qpos', 'obstacle_profile',
                   'obstacle_safety_json', 'predicted_chunks', 'prompt', 'used_vls', 'warmup_steps'])
PERTURB_KEYS = ["perturb_arm", "perturb_rad", "perturb_seed", "perturb_vector"]


class _FakePolicy:
    """Deterministic in the observed state: chunk rows = state + a small ramp on both arms."""

    def infer(self, obs):
        s = np.asarray(obs["state"], dtype=np.float64)
        rows = np.tile(s, (10, 1))
        ramp = 0.002 * (np.arange(10)[:, None] + 1)
        rows[:, 0:7] += ramp
        rows[:, 8:15] -= ramp
        return {"actions": rows}


def _run_main(mod, monkeypatch, tmp_path, name, episode, extra=()):
    out = tmp_path / f"{name}.npz"
    monkeypatch.setattr(mod, "prepare_run_outputs", lambda args, names: setattr(args, "run_dir", tmp_path))
    monkeypatch.setattr(mod, "configure_run_logging", lambda name: None)
    monkeypatch.setattr(mod, "load_remote_policy", lambda remote: _FakePolicy())
    monkeypatch.setattr("sys.argv", ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d",
                                     "--remote", "localhost:1", "--episode-index", str(episode), "--headless",
                                     "--warmup-steps", "50", "--max-steps", "20", "--trajectory-out", str(out),
                                     *extra])
    mod.main()
    return np.load(out)


@pytest.mark.parametrize("episode,arm", [(1800, "left"), (1801, "right")])
def test_main_off_vs_on(mod, monkeypatch, tmp_path, episode, arm):
    try:
        mod.load_randomized_episode(episode)
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"16-D episode metadata not available: {exc}")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    off = _run_main(mod, monkeypatch, tmp_path, "off", episode)
    off2 = _run_main(mod, monkeypatch, tmp_path, "off2", episode)
    on = _run_main(mod, monkeypatch, tmp_path, "on", episode, ("--first-chunk-perturb-rad", "1e-6",
                                                                "--perturb-seed", "3"))
    assert sorted(off.files) == OFF_KEYS
    for k in off.files:                                  # off is deterministic (wall-clock inference_ms aside)
        if k != "inference_ms":
            assert np.array_equal(off[k], off2[k]), k
    assert sorted(on.files) == sorted(OFF_KEYS + PERTURB_KEYS)
    assert str(on["perturb_arm"]) == arm and int(on["perturb_seed"]) == 3 and float(on["perturb_rad"]) == 1e-6
    v = on["perturb_vector"]
    assert np.array_equal(v, mod.first_chunk_perturb_vector(1e-6, 3))
    sl = mod.PERTURB_ARM_SLICES_16D[arm]
    mask = np.zeros(16, dtype=bool)
    mask[sl] = True
    # first chunk: same observation -> same planned rows; executed = planned + v on the used arm only
    assert np.array_equal(on["predicted_chunks"][0], off["predicted_chunks"][0])
    for t in range(8):
        assert np.array_equal(on["executed_actions"][t][~mask], off["executed_actions"][t][~mask])
        assert np.array_equal(on["executed_actions"][t][mask], off["executed_actions"][t][mask] + v)
    # steps >= 8 are the (unperturbed) rows of chunk 2, planned from the now slightly different state
    assert np.array_equal(on["executed_actions"][8], on["predicted_chunks"][1][0])
    assert not np.array_equal(on["measured_qpos"][8], off["measured_qpos"][8])
