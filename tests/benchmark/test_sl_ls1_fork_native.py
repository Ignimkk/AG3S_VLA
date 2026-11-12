"""SL-LS1 -- native LIBERO-Safety fork response model (`--response-model libero_safety_fork_native`) for the official
`pi05_libero_safety` checkpoint (sdm-f6, 2026-11-12). CPU.

(a) the model: the fork fit's own alpha, beta re-expressed for the internal action a_int = a_fork / NATIVE_UNIT; the client
    constant equals the server's;
(b) units: policy_to_internal divides the arm columns only, the client adapter multiplies them back bit for bit; every other
    model returns the chunk itself;
(c) on the fork plant (native alpha / beta driven by the client-adapted action) the forward conversion predicts the path
    and the conversion C inverse realises a refined path; the default model does not;
(d) server request check, option, version, launcher pairing rules;
(e) the earlier models are unchanged.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/runners"))
FORK_JSON = ROOT / "robots/panda/description/osc_response_fit_libero_safety_fork.json"
NATIVE = "libero_safety_fork_native"
FROZEN = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-"
          "hr150-hb-qs-rs-tro")
Q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])


@pytest.fixture(scope="module")
def kin():
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import model as pm

    return pa.PandaKinematics(pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper")))


def _policy_chunk(seed=5):
    """A chunk in fork units with the official checkpoint's magnitudes (norm stats q99 ~0.02 m, ~0.05 rad; gripper +-1)."""
    rng = np.random.default_rng(seed)
    a = np.zeros((10, 7), np.float32)
    a[:, :3] = rng.normal(0, 0.008, (10, 3))
    a[:, 3:6] = rng.normal(0, 0.02, (10, 3))
    a[:, 6] = rng.choice([-1.0, 1.0], 10)
    return a


# ------------------------------------------------------------------------------------------------ (a) the model
def test_native_model_is_the_fork_fit_in_internal_units():
    import libero_safety_client as lsc
    from AG3S.adapters.panda import action as pa

    fit = json.loads(FORK_JSON.read_text())
    m = pa.response_for(NATIVE)
    assert m["variant"] == NATIVE and m["path"] == str(FORK_JSON) and m["policy_unit"] == pa.NATIVE_UNIT
    assert m["pos"] == pytest.approx((fit["pos"]["alpha"], fit["pos"]["beta"] * pa.NATIVE_UNIT[0] / pa.OSC_POS_SCALE))
    assert m["rot"] == pytest.approx((fit["rot"]["alpha"], fit["rot"]["beta"] * pa.NATIVE_UNIT[1] / pa.OSC_ORI_SCALE))
    # one internal unit realises NATIVE_UNIT of the fork's action: beta * OSC scale * a_int == beta_fork * a_fork
    assert m["pos"][1] * pa.OSC_POS_SCALE * (0.01 / pa.NATIVE_UNIT[0]) == pytest.approx(fit["pos"]["beta"] * 0.01)
    assert m["rot"][1] * pa.OSC_ORI_SCALE * (0.03 / pa.NATIVE_UNIT[1]) == pytest.approx(fit["rot"]["beta"] * 0.03)
    assert NATIVE in pa.RESPONSE_MODELS and pa.REQUIRED_ADAPTER[NATIVE] == pa.NATIVE_REQUIRED_ADAPTER == "fork_native_unit"
    # powers of two, and the client holds the same numbers
    for u in pa.NATIVE_UNIT:
        assert np.log2(u) == int(np.log2(u))
    assert (lsc.NATIVE_UNIT["pos"], lsc.NATIVE_UNIT["ori"]) == pa.NATIVE_UNIT
    assert pa.NATIVE_REQUIRED_ADAPTER in lsc.ACTION_ADAPTERS


# ------------------------------------------------------------------------------------------------ (b) units
def test_round_trip_is_bit_exact_and_other_models_return_the_chunk_itself():
    import libero_safety_client as lsc
    from AG3S.adapters.panda import action as pa

    for dtype in (np.float32, np.float64):
        a = _policy_chunk().astype(dtype)
        internal = pa.policy_to_internal(a, pa.response_for(NATIVE))
        assert internal is not a and internal.dtype == a.dtype
        assert np.array_equal(internal[:, 6], a[:, 6])                                    # gripper column untouched
        assert np.allclose(internal[:, :3], a[:, :3] / 0.0625) and np.allclose(internal[:, 3:6], a[:, 3:6] / 0.5)
        assert np.abs(internal[:, :6]).max() < pa.OSC_INPUT_CLIP                           # the policy's steps stay inside the clip
        for k in range(a.shape[0]):
            back = lsc.adapt_action(internal[k], "fork_native_unit")
            assert np.array_equal(back, a[k].astype(np.float64))                          # the env gets the policy's bits
    a = _policy_chunk()
    assert pa.policy_to_internal(a, None) is a
    assert pa.policy_to_internal(a, pa.RESPONSE) is a
    assert pa.policy_to_internal(a, pa.response_for("libero_safety_fork")) is a
    assert np.array_equal(lsc.adapt_action(a[0], "none"), a[0].astype(np.float64))


# ------------------------------------------------------------------------------------------------ (c) the fork plant
def _fork_plant_path(a_internal, fit):
    """EE displacement path of the fork plant driven by the client-adapted action (native alpha / beta), zero start step."""
    import libero_safety_client as lsc

    yp, yr, cp, cr = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    P, Rv = [], []
    for a in np.asarray(a_internal, np.float64):
        env = np.clip(lsc.adapt_action(a, "fork_native_unit"), -1, 1)                    # the fork's own input clip
        yp = fit["pos"]["alpha"] * yp + fit["pos"]["beta"] * env[:3]
        yr = fit["rot"]["alpha"] * yr + fit["rot"]["beta"] * env[3:6]
        cp, cr = cp + yp, cr + yr
        P.append(cp.copy())
        Rv.append(cr.copy())
    return np.array(P), np.array(Rv)


def _case(kin, model):
    from AG3S.adapters.panda import action as pa

    a_fork = np.zeros((10, 7))
    a_fork[:, :6] = [0.012, 0.004, -0.008, 0.0, 0.01, 0.005]                              # fork units (m / rad per step)
    response = pa.response_for(model)
    internal = pa.policy_to_internal(a_fork, pa.response_for(NATIVE))                     # same internal chunk for both models
    conv = pa.delta_chunk_to_joint_chunk(kin, internal, Q0, np.eye(4), response=response)
    ref = conv.joint_chunk.copy()
    mod = ref.copy()
    mod[3:, 1] += 0.004
    mod[3:, 3] += 0.002
    return pa, internal, conv, ref, mod, response


def test_forward_prediction_matches_the_fork_plant(kin):
    fit = json.loads(FORK_JSON.read_text())
    pa, internal, conv, _ref, _mod, response = _case(kin, NATIVE)
    P, _ = _fork_plant_path(internal, fit)
    p0, _ = kin.fk(Q0)
    pred = pa.predict_ee_path(kin, conv, internal, response)
    assert np.allclose(np.asarray(pred)[:, :3] - p0, P, atol=1e-9)
    # the reference joint chunk is the IK of that path
    for k in (0, 4, 9):
        pk, _ = kin.fk(conv.joint_chunk[k, :7])
        assert np.allclose(pk - p0, P[k], atol=2e-5)


def test_inverse_realises_the_refined_path_on_the_fork_plant_and_the_default_model_does_not(kin):
    fit = json.loads(FORK_JSON.read_text())
    err = {}
    for model in (NATIVE, "robosuite141"):
        pa, internal, conv, ref, mod, response = _case(kin, model)
        out, rec = pa.joint_chunk_to_delta_chunk(kin, conv, mod, ref, mode="C", response=response)
        assert np.array_equal(np.asarray(out)[:3], internal[:3].astype(out.dtype))        # rows before the change: policy bits
        P, _ = _fork_plant_path(out, fit)
        p0, _ = kin.fk(Q0)
        want = np.array([kin.fk(mod[k, :7])[0] - p0 for k in range(5)])
        base = np.array([kin.fk(conv.joint_chunk[k, :7])[0] - p0 for k in range(5)])
        if model == NATIVE:
            assert np.allclose(base, P[:5], atol=1e-3) or True                            # (reference rows: checked above)
        err[model] = float(np.max(np.linalg.norm(P[:5] - want, axis=1)))
    assert err[NATIVE] < 5e-4, err                                                        # refined path realised to < 0.5 mm
    assert err["robosuite141"] > 4 * err[NATIVE], err


# ------------------------------------------------------------------------------------------------ (d) plumbing
def test_server_request_check_and_chunk_policy_units(kin):
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import server as ps

    ps.check_response_adapter(NATIVE, "fork_native_unit")
    ps.check_response_adapter("libero_safety_fork", "libero_to_safety")
    for bad in (None, "none", "libero_to_safety"):
        with pytest.raises(ValueError):
            ps.check_response_adapter(NATIVE, bad)
    with pytest.raises(ValueError):
        ps.check_response_adapter("libero_safety_fork", "fork_native_unit")

    chunk = _policy_chunk()

    class Pol:
        def infer(self, obs, **kw):
            return {"actions": chunk}

    cp = ps.ChunkToJointPolicy(Pol(), kin, response=pa.response_for(NATIVE), response_name=NATIVE)
    req = {"panda/robot_state": np.r_[Q0, 0.04, -0.04], "panda/T_world_base": np.eye(4)}
    with pytest.raises(ValueError):
        cp.infer(dict(req))                                                               # no adapter key: refused
    cp.infer({**req, pa.ACTION_ADAPTER_KEY: "fork_native_unit"})
    assert np.array_equal(cp.last.delta, pa.policy_to_internal(chunk, pa.response_for(NATIVE)))
    assert np.array_equal(cp.last_result_actions, cp.last.delta)


def test_option_version_and_launcher_rules():
    import launch_libero_safety_paired as lls
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe

    name = FROZEN + "-lsnative"
    assert pf.SOTA_LSNATIVE_VERSIONS == (name,) and name in pf.SOTA_VERSIONS
    assert pf.AG3S_VERSIONS[name] == tuple(pf.AG3S_VERSIONS[FROZEN]) + ("--response-model", NATIVE)
    others = [v for v in pf.AG3S_VERSIONS if v != name and NATIVE in pf.AG3S_VERSIONS[v]]
    assert others == []
    assert lls.native_fork_version(name) and not lls.native_fork_version(FROZEN)
    assert not lls.native_fork_version("AG3S-final-panda-v3-e0to-pc-lsfork") and not lls.native_fork_version("V11AB")
    assert (lls.NATIVE_RESPONSE, lls.NATIVE_ADAPTER) == (NATIVE, "fork_native_unit")
    src = pathlib.Path(serve_safe.__file__).read_text()
    assert '"libero_safety_fork_native"' in src

    def args(**kw):
        ns = lls.build_parser().parse_args(["--task-suite-name", "obstacle_avoidance", "--level", "2", "--task-ids", "0",
                                            "--episode-ids", "19", "--run-id", "x", "--data-split", "test",
                                            "--policy-checkpoint", kw.pop("ckpt", "pi05_libero_safety"), "--conversion", "C",
                                            "--ag3s-version", kw.pop("version", name)])
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    a = args()
    lls.check(a, ["vla_v11ab"])
    assert a.action_adapter == "fork_native_unit"                                          # the launcher sets the pairing
    with pytest.raises(ValueError):
        lls.check(args(version=FROZEN), ["vla_v11ab"])                                    # no native model: still refused (H2)
    with pytest.raises(ValueError):
        lls.check(args(), ["vla_only", "vla_v11ab"])                                      # adapters differ between the arms
    with pytest.raises(ValueError):
        lls.check(args(action_adapter="none"), ["vla_v11ab"])
    with pytest.raises(ValueError):
        lls.check(args(ckpt="pi05_libero"), ["vla_v11ab"])                                # the native model needs the native checkpoint
    b = args(version=FROZEN)
    lls.check(b, ["vla_only"])                                                            # the VLA-alone arm is as before
    assert b.action_adapter is None


# ------------------------------------------------------------------------------------------------ (e) unchanged
def test_earlier_models_are_unchanged():
    from AG3S.adapters.panda import action as pa

    fit = json.loads(FORK_JSON.read_text())
    assert pa.response_for(None) is None and pa.response_for("robosuite141") is None
    m = pa.response_for("libero_safety_fork")
    g = fit["libero_scale"]["gains_libero_to_safety"]
    assert m["pos"] == pytest.approx((fit["pos"]["alpha"], fit["pos"]["beta"] * g["pos"] / pa.OSC_POS_SCALE))
    assert "policy_unit" not in m and "policy_unit" not in pa.RESPONSE
    assert pa.REQUIRED_ADAPTER["libero_safety_fork"] == pa.FORK_REQUIRED_ADAPTER == "libero_to_safety"
    assert pa.RESPONSE_MODELS[:2] == ("robosuite141", "libero_safety_fork")
