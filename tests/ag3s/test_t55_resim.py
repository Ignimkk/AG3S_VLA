"""T55 re-simulation driver (`t55_resim`) — synthetic latch / intervention logic, a tiny MuJoCo model for the
collision edits, and (storage present) a truncated real T53 record.

Pinned:
  - the dev record list excludes T47 ``long`` / ``ep1800`` (lead, user decision 2026-10-13);
  - interventions: parameter validation, tags, the closing frame from executed used-hand steps only, the hold
    onset, the intervention frame (and why it is not applicable);
  - offline strong rule: evidence-mode `GraspLatch` on wire-shaped feedback — a blocked, settled closure attaches,
    an empty closure stays ``closing``, a reach rejection reverts to ``closing``, the target lock happens at the
    end of the lock request (observe before update, `SafePolicy` order); the lock request of a server record is
    the first one whose end-of-request state is not ``searching``; feedback values are float32-rounded (wire);
  - ``park`` removes every contact of the object and moves it; ``ghost`` switches off object-hand contacts only
    (floor kept) and restores the bits;
  - outputs refuse without ``AG3S_RUN_DIR`` and inside the read-only T53 run;
  - real (skipped without storage): a 3-frame prefix of T53 ``ep1804`` re-simulates bit-exactly (qpos, images,
    measured, applied_gripper, feedback, recorded latch state), and a ``park`` at frame 1 leaves frames <= 1
    identical and the apple outside the scene afterwards.
"""

from __future__ import annotations

import pathlib
import types

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t55_resim as rs

T53_RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/"
                       "20261013T100210+0900_4a540f1a18d7_cb037c425d2f")


def test_dev_keys_exclude_prefix_records():
    assert "long" not in rs.DEV_KEYS and "ep1800" not in rs.DEV_KEYS
    assert set(rs.DEV_EXCLUDED) == {"long", "ep1800"}
    assert rs.DEV_KEYS == ("t2", "headfix", "ep1807", "ep1850", "ep1900", "ep0", "ep500", "ep1200")
    with pytest.raises(SystemExit):
        rs.dev_source("long")


def test_intervention_validation_and_tags():
    assert rs.Intervention("empty", "park").tag == "empty_L0O0_park"
    assert rs.Intervention("slip", "ghost", after_frames=3, delay_frames=1, offset_steps=2).tag == "slip_A3D1O2_ghost"
    with pytest.raises(ValueError):
        rs.Intervention("empty", "ghost")
    with pytest.raises(ValueError):
        rs.Intervention("drop", "park")
    with pytest.raises(ValueError):
        rs.Intervention("slip", "park", after_frames=0)
    p = rs.Intervention("empty", "park").params()
    assert p["tag"] == "empty_L0O0_park" and tuple(p["park_pos"]) == rs.PARK_POS


def _src(actions, executed, used="right"):
    return types.SimpleNamespace(actions=np.asarray(actions, float), executed=np.asarray(executed, bool),
                                 used_arm=used)


def test_first_closing_frame_executed_used_hand_only():
    a = np.ones((32, 16))
    a[10, 7] = 0.0          # left hand closes: not the used (right) hand
    a[12, 15] = 0.2         # right closes but the step was a HOLD (not executed)
    a[20, 15] = 0.5
    ex = np.ones(32, bool)
    ex[12] = False
    assert rs.first_closing_frame(_src(a, ex), 8, 0.85) == 2
    assert rs.first_closing_frame(_src(np.ones((16, 16)), np.ones(16, bool)), 8, 0.85) is None
    a2 = np.ones((16, 16))
    a2[3, 15] = -0.5         # apply_action clips to [0, 1]
    assert rs.first_closing_frame(_src(a2, np.ones(16, bool)), 8, 0.85) == 0


def test_hold_onset_and_intervention_frame():
    assert rs.hold_onset_frame([0, 1, 0, 1, 1, 1], 2) == 4
    assert rs.hold_onset_frame([0, 1, 0], 2) is None
    a = np.ones((32, 16))
    a[17, 15] = 0.0
    s = _src(a, np.ones(32, bool))
    assert rs.intervention_frame(rs.Intervention("empty", "park"), s, None, horizon=8, closed_below=0.85) == (2, "f_close=2")
    f, why = rs.intervention_frame(rs.Intervention("empty", "park", lead_frames=3), s, None, horizon=8, closed_below=0.85)
    assert f is None and "lead" in why
    iv = rs.Intervention("slip", "park", after_frames=2, delay_frames=1)
    assert rs.intervention_frame(iv, s, [0, 1, 1, 1], horizon=8, closed_below=0.85) == (3, "f_hold=2")
    assert rs.intervention_frame(iv, s, None, horizon=8, closed_below=0.85)[0] is None
    assert rs.intervention_frame(iv, s, [0, 0, 1, 1], horizon=8, closed_below=0.85)[0] is None  # beyond the end


def _fb(seq, applied, measured, executed=None):
    n = 8
    ex = [True] * n if executed is None else executed
    return {"available": True, "seq": seq, "t_step_start": (seq - 1) * n, "n_steps": n, "n_exec": sum(ex),
            "executed": ex, "applied_gripper": [[1.0, applied]] * n, "measured_gripper": [1.0, measured]}


def _per(n, reject_at=None):
    return [{"reach_m": 0.05, "manipulated_id": 0, "manipulated_state": "visible", "placed": False,
             "held_observation": None, "reject_reason": ("reach" if i == reject_at else None)} for i in range(n)]


NOFB = {"available": False, "reason": "no chunk has run yet"}


def test_offline_latch_blocked_closure_attaches():
    cfg = rs.latch_config(False)
    fb = [NOFB, _fb(1, 1.0, 1.0), _fb(2, 0.0, 0.70), _fb(3, 0.0, 0.70), _fb(4, 0.0, 0.70)]
    out = rs.offline_latch(fb, _per(5), lock_request=0, used_arm="right", cfg=cfg)
    assert out["phase"].tolist() == ["latched", "latched", "closing", "held", "held"]
    assert out["attach"].tolist() == [False, False, False, True, False]
    # empty hand: fingers close to the command (not blocked, nearly fully closed)
    fb_e = [NOFB, _fb(1, 1.0, 1.0), _fb(2, 0.0, 0.02), _fb(3, 0.0, 0.02), _fb(4, 0.0, 0.02)]
    out_e = rs.offline_latch(fb_e, _per(5), lock_request=0, used_arm="right", cfg=cfg)
    assert out_e["phase"].tolist() == ["latched", "latched", "closing", "closing", "closing"]
    assert not out_e["attach"].any()
    # AG3S.attach rejection -> back to closing (revert_attach)
    out_r = rs.offline_latch(fb, _per(5, reject_at=3), lock_request=0, used_arm="right", cfg=cfg)
    assert out_r["phase"][3] == "closing" and not out_r["attach"][3]


def test_offline_latch_lock_after_observe():
    cfg = rs.latch_config(False)
    # closing feedback in the lock request itself is ignored (still searching while observe_grasp runs)
    fb = [NOFB, _fb(1, 0.0, 0.70), _fb(2, 0.0, 0.70), _fb(3, 0.0, 0.70)]
    out = rs.offline_latch(fb, _per(4), lock_request=1, used_arm="right", cfg=cfg)
    assert out["phase_before"].tolist()[:3] == ["searching", "searching", "latched"]
    assert out["phase"].tolist()[:3] == ["searching", "latched", "closing"]
    with pytest.raises(ValueError):
        rs.offline_latch(fb, _per(3), lock_request=1, used_arm="right", cfg=cfg)


def test_perception_from_record():
    cfg = rs.latch_config(False)
    grasp = [{"state": "searching"}, {"state": "searching", "evidence": {"reach_mm": 80.0, "manipulated_id": 2,
                                                                         "manipulated_state": "visible"}},
             {"state": "latched", "evidence": {"reach_mm": 50.0, "manipulated_id": 2, "manipulated_state": "visible",
                                               "hold_checks": {"observed_distance_mm": 10.0, "observed_limit_mm": 70.0,
                                                               "hand_rise_mm": 1.0, "object_rise_mm": 1.0}}},
             {"state": "closing", "rejected": {"reason": "reach 150 mm"}}]
    per, lock = rs.perception_from_record(grasp, cfg)
    assert lock == 2
    assert per[1]["reach_m"] == pytest.approx(0.08) and per[0]["reach_m"] is None
    assert per[2]["held_observation"]["held_radius_m"] == pytest.approx(0.07 - cfg.revoke_centre_slack)
    assert per[3]["reject_reason"] == "reach 150 mm"


def test_wire32_and_feedback_equality():
    x = np.array([0.1, 1.0 / 3.0])
    w = rs.wire32(x)
    assert w.dtype == np.float64 and np.array_equal(w, x.astype(np.float32).astype(np.float64))
    a = {"available": True, "executed": [True], "applied_gripper": [[1.0, 0.5]], "measured_gripper": [1.0, 0.7]}
    b = dict(a, measured_gripper=[1.0, 0.7000001])
    assert rs._feedback_equal(a, a) and not rs._feedback_equal(a, b)  # noqa: SLF001
    assert rs._feedback_equal(NOFB, {"available": False})  # noqa: SLF001


MJCF = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="hand" pos="0 0 0.05"><geom name="hand_g" type="box" size="0.03 0.03 0.03"/></body>
    <body name="apple" pos="0 0 0.05"><freejoint name="apple_free"/><geom name="apple_g" type="sphere" size="0.06"/></body>
  </worldbody>
</mujoco>
"""


def _contacts(m, d):
    import mujoco

    mujoco.mj_forward(m, d)
    names = set()
    for i in range(d.ncon):
        c = d.contact[i]
        names.add(tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)))))
    return names


def test_ghost_and_park_collision_edits():
    import mujoco

    m = mujoco.MjModel.from_xml_string(MJCF)
    d = mujoco.MjData(m)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    sidx = types.SimpleNamespace(fruit_body={"apple": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")})
    before = _contacts(m, d)
    assert ("apple_g", "hand_g") in before and ("apple_g", "floor") in before
    mods = rs._Mods(m, sidx, "apple")  # noqa: SLF001
    info = mods.ghost([gid("hand_g")])
    after = _contacts(m, d)
    assert ("apple_g", "hand_g") not in after and ("apple_g", "floor") in after
    assert info["object_geoms"] == [gid("apple_g")]
    assert mods.hand_distance(d, cap=0.05) < 0  # still overlapping the hand
    mods.restore()
    assert _contacts(m, d) == before
    mods2 = rs._Mods(m, sidx, "apple")  # noqa: SLF001
    mods2.park(d, (-20.0, 0.0, -5.0))
    mujoco.mj_forward(m, d)
    assert not any("apple_g" in c for c in _contacts(m, d))
    assert d.xpos[sidx.fruit_body["apple"]][0] == pytest.approx(-20.0)
    d.qvel[:] = 1.0
    mods2.pin(d)
    assert np.all(d.qvel[mods2.vadr:mods2.vadr + 6] == 0)


def test_out_root_refusals(monkeypatch):
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit):
        rs.out_root("resim/dev")
    if not T53_RUN.is_dir():
        pytest.skip("T53 run not present")
    monkeypatch.setenv("AG3S_RUN_DIR", str(T53_RUN))
    with pytest.raises(SystemExit):
        rs.out_root("resim/test", forbid=[T53_RUN])


def _truncated(src, n_frames):
    H = 8
    n = n_frames * H
    return rs.Source(**{**{f.name: getattr(src, f.name) for f in rs.dataclasses.fields(src)},
                        "step_files": src.step_files[:n_frames], "t_steps": src.t_steps[:n_frames],
                        "actions": src.actions[:n], "executed": src.executed[:n],
                        "ref_measured": src.ref_measured[:n], "ref_applied_gripper": src.ref_applied_gripper[:n],
                        "ref_feedback": src.ref_feedback[:n_frames], "ref_grasp": src.ref_grasp[:n_frames]})


@pytest.mark.skipif(not (T53_RUN / "verify" / "T53" / "runs" / "C1a" / "R").is_dir(), reason="T53 records not present")
def test_real_t53_prefix_parity_and_park():
    servers = sorted((T53_RUN / "verify/T53/runs/C1a").glob("server_829*_*"))
    src = rs.t53_source("ep1804", t53_run_dir=T53_RUN, episodes_root=T53_RUN / "verify/T53/runs/C1a/R",
                        server_roots=servers)
    assert src.warmup_steps == 1000 and src.ref_grasp is not None
    short = _truncated(src, 3)
    cl = rs.load_client(rs.default_client(T53_RUN))
    res = rs.simulate(cl, short, keep_images=True)
    lat = rs.run_latches(short, res)
    par = rs.e0_parity(short, res, lat)
    assert par["exact"], {k: v for k, v in par.items() if k != "per_frame"}
    assert par["qpos_max_abs"] == 0 and par["image_max_abs"] == 0 and par["applied_gripper_max_abs"] == 0
    assert par["feedback_requests_equal"] == 3 and par["latch_record_equal"] == 3
    iv = rs.Intervention("empty", "park")
    res2 = rs.simulate(cl, short, intervention=iv, frame=1, keep_images=True)
    assert res2["intervention"]["applied"] and res2["intervention"]["step"] == 8
    assert np.array_equal(res2["arrays"]["qpos"][:2], res["arrays"]["qpos"][:2])
    j = [str(f) for f in res2["arrays"]["fruits"]].index("apple")
    assert res2["arrays"]["fruit_pos"][2, j, 0] < -10
    assert not np.array_equal(res2["arrays"]["image_cam_high"][2], res["arrays"]["image_cam_high"][2])


# ------------------------------------------------------------------ user decisions 2026-10-13 (추가 사전 고정 1)


def test_baseline_of_set_and_slip_baseline():
    assert rs.BASELINE_OF_SET == {"dev": "unintervened_resim", "test": "record"}
    sb = rs.slip_baseline([0, 1, 1, 0], 2)
    assert sb == {"after_frames": 2, "e0_hold_frame": 2, "excluded": False}
    sb2 = rs.slip_baseline([0, 1, 0, 1], 2)
    assert sb2["excluded"] and sb2["e0_hold_frame"] is None and sb2["reason"] == rs.SLIP_EXCLUSION_REASON


def _fake_record(monkeypatch, tmp_path, *, set_name, exact, p_task):
    """process_record with the physics stubbed: E0 parity `exact`, raw P_task `p_task` per frame."""
    F = len(p_task)
    calls = []
    monkeypatch.setattr(rs, "load_client", lambda p: types.SimpleNamespace(OPEN_LOOP_HORIZON=8))
    monkeypatch.setattr(rs, "md5_of", lambda p: "md5")

    def sim(cl, src, intervention=None, frame=None, keep_images=True, observer=None):
        calls.append(None if intervention is None else intervention.tag)
        return {"arrays": {"qpos": np.zeros((F, 3)), "penetration": np.zeros(F)}, "feedback": [], "seconds": 0.0,
                "intervention": {"applied": intervention is not None}}

    monkeypatch.setattr(rs, "simulate", sim)
    monkeypatch.setattr(rs, "run_latches", lambda src, res: {})
    monkeypatch.setattr(rs, "e0_parity", lambda src, res, lat: {"exact": exact, "qpos_max_abs": 0.0 if exact else 0.3})
    monkeypatch.setattr(rs, "raw_preds", lambda a, t, u: {"p_task": np.asarray(p_task, bool),
                                                          "p_grasp": np.asarray(p_task, bool)})
    monkeypatch.setattr(rs, "variant_payload", lambda *a, **k: {"x": np.zeros(1)})
    a = np.ones((F * 8, 16))
    a[16:, 15] = 0.0                                    # closing from frame 2
    src = types.SimpleNamespace(key="k", set_name=set_name, target="apple", used_arm="right", actions=a,
                                executed=np.ones(F * 8, bool), describe=lambda: {"key": "k"})
    ivs = [rs.Intervention("empty", "park"), rs.Intervention("slip", "ghost", after_frames=2)]
    out = rs.process_record(src, ivs, client_path=tmp_path / "c.py", out_dir=tmp_path, force=True,
                            keep_images=False, run_meta={})
    return out, calls


def test_dev_runs_interventions_on_a_nonexact_e0(monkeypatch, tmp_path):
    out, calls = _fake_record(monkeypatch, tmp_path, set_name="dev", exact=False, p_task=[0, 0, 0, 1, 1, 1])
    assert out["baseline"] == "unintervened_resim" and calls == [None, "empty_L0O0_park", "slip_A2D0O0_ghost"]
    assert out["variants"]["slip_A2D0O0_ghost"]["frame"] == 4 and out["slip_excluded"] == {}
    import json

    meta = json.loads((tmp_path / "k__E0.json").read_text())
    assert meta["baseline"] == "unintervened_resim" and meta["slip_baseline"]["2"]["e0_hold_frame"] == 4


def test_test_refuses_interventions_on_a_nonexact_e0(monkeypatch, tmp_path):
    out, calls = _fake_record(monkeypatch, tmp_path, set_name="test", exact=False, p_task=[0, 0, 0, 1, 1, 1])
    assert calls == [None] and all("STOP" in out["variants"][t]["refused"]
                                   for t in ("empty_L0O0_park", "slip_A2D0O0_ghost"))


def test_slip_excluded_without_an_e0_hold(monkeypatch, tmp_path):
    import json

    out, calls = _fake_record(monkeypatch, tmp_path, set_name="dev", exact=False, p_task=[0, 0, 1, 0, 1, 0])
    assert calls == [None, "empty_L0O0_park"]
    assert out["slip_excluded"] == {"slip_A2D0O0_ghost": rs.SLIP_EXCLUSION_REASON}
    j = json.loads((tmp_path / "k__slip_A2D0O0_ghost.json").read_text())
    assert j["excluded_from_slip_evaluation"] and j["not_applicable"] == rs.SLIP_EXCLUSION_REASON
