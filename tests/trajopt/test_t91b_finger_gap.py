"""T91b — wrist-camera "target between the fingers" evidence in the phase machine, the finger-gap geometry, and the SafeLIBERO
adapter. Fixed here: off = the T91 outputs (stage and evidence keys), an empty closed hand is not carry, a surface between the
fingers + closed + moved = carry, the hysteresis, the fallback without wrist evidence, robot settings of the Panda adapter, the
SafeLIBERO ground-truth rule, and that gap-threshold selection cannot see test scenes.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t91_phase_eval as E
from AG3S.ag3s.experiments.studies import t91b_safelibero as S
from AG3S.trajopt.finger_gap import (FingerGapConfig, panda_eye_in_hand, project_points, projection_fraction, rby1_wrist,
                                     surface_fraction)
from AG3S.trajopt.phase_machine import GripperScale, PhaseConfig, PhaseConfigError, PhaseMachine, PhaseObservation

TGT = np.array([0.50, 0.30, 0.85])


def stages(frames, cfg):
    m = PhaseMachine(cfg)
    out, evs = [], []
    for f in frames:
        st, ev = m.step(PhaseObservation(**f))
        out.append(st)
        evs.append(ev)
    return out, evs


def hold_frames(gap, n_close=2, n_lift=6, g_hold=0.6, step=0.01, tgt_fresh=False):
    """Open hand near the target, closes (opening g_hold), then the hand rises ``step`` per request. ``gap`` = wrist evidence."""
    fr = [dict(gripper_raw=1.0, ee_pos=TGT + [0, 0, 0.02], target_pos=TGT, target_fresh=True, chunk_gripper=[1.0] * 8,
               finger_gap=0.0) for _ in range(3)]
    fr += [dict(gripper_raw=g_hold, ee_pos=TGT + [0, 0, 0.02], target_pos=TGT, target_fresh=True, chunk_gripper=[0.5] * 8,
                finger_gap=gap) for _ in range(n_close)]
    for i in range(1, n_lift + 1):
        fr.append(dict(gripper_raw=g_hold, ee_pos=TGT + [0, 0, 0.02 + step * i], target_pos=TGT, target_fresh=tgt_fresh,
                       chunk_gripper=[0.5] * 8, finger_gap=gap))
    return fr


ON = dict(finger_gap_mode="on", gap_tau=0.2, gap_m=2)


# --------------------------------------------------------------------------------- off = T91

def test_off_ignores_gap_fields_and_adds_no_evidence_keys():
    """Mode off: stages and the evidence dict are the same whether or not the wrist numbers are passed."""
    rng = np.random.default_rng(3)
    cfg = PhaseConfig()
    assert cfg.finger_gap_mode == "off"
    m1, m2 = PhaseMachine(cfg), PhaseMachine(cfg)
    ee = TGT + [0, 0, 0.3]
    for i in range(60):
        ee = ee + rng.normal(0, 0.01, 3)
        g = float(np.clip(1.0 - 0.02 * max(0, i - 20), 0.4, 1.0))
        kw = dict(gripper_raw=g, ee_pos=ee, target_pos=TGT, target_fresh=bool(rng.random() < 0.5), chunk_gripper=[g] * 8)
        s1, e1 = m1.step(PhaseObservation(**kw))
        s2, e2 = m2.step(PhaseObservation(**kw, finger_gap=float(rng.random()), finger_gap_proj=float(rng.random())))
        assert s1 == s2 and e1 == e2
        assert not any(k.startswith("gap") for k in e2)


def test_defaults_keep_t91_thresholds_and_gap_off():
    c = PhaseConfig()
    for k, v in E.TRAIN_SELECTED.items():
        assert getattr(c, k) == v
    assert c.open_ref_value is None and c.finger_gap_mode == "off"


def test_gap_defaults_are_the_rby1_train_selection():
    from AG3S.trajopt.finger_gap import rby1_wrist

    d, c = E.GAP_SELECTED["depth"], PhaseConfig()
    assert (c.gap_tau, c.gap_m) == (d["gap_tau"], d["gap_m"]) and c.gap_tau_proj == E.GAP_SELECTED["projection"]["gap_tau_proj"]
    cfg = rby1_wrist()
    assert (cfg.roi_u[0], cfg.roi_u[1], cfg.roi_v[0], cfg.roi_v[1], cfg.d_min, cfg.d_max) == (
        d["roi"][0], d["roi"][1], d["roi"][2], d["roi"][3], d["d_min"], d["d_max"])


# --------------------------------------------------------------------------------- rules

def test_empty_hand_close_is_not_carry_even_when_the_hand_moves_away():
    """Closed on nothing: the wrist ROI shows no surface -> never carry (the T91 occlusion rule would call it carry)."""
    fr = hold_frames(gap=0.0)
    on, _ = stages(fr, PhaseConfig(**ON))
    assert "carry" not in on and "place" not in on
    off, _ = stages([{**f, "finger_gap": None} for f in fr], PhaseConfig())      # T91 reads the same scene as carry
    assert "carry" in off


def test_surface_between_fingers_closed_and_moved_is_carry():
    on, ev = stages(hold_frames(gap=0.8), PhaseConfig(**ON))
    assert "carry" in on
    first = on.index("carry")
    # not before the hand moved >= bootstrap_hand_mm (20 mm) from where it closed
    assert ev[first]["moved_since_close_mm"] >= 20.0 and ev[first - 1]["moved_since_close_mm"] < 20.0


def test_closed_on_a_surface_but_not_moved_is_grasp_not_carry():
    fr = hold_frames(gap=0.8, n_close=6, n_lift=0)
    on, _ = stages(fr, PhaseConfig(**ON))
    assert "carry" not in on and on[-1] == "grasp"


def test_enter_needs_m_consecutive_hits_and_leave_needs_m_misses():
    base = hold_frames(gap=0.8, n_lift=8)
    # one hit after misses is not enough with gap_m = 2 ...
    seq = [dict(f, finger_gap=(0.8 if i == 9 else 0.0) if i >= 5 else f["finger_gap"]) for i, f in enumerate(base)]
    assert "carry" not in stages(seq, PhaseConfig(**ON))[0]
    # ... and 2 consecutive hits are
    seq2 = [dict(f, finger_gap=0.8 if i >= 5 else 0.0) for i, f in enumerate(base)]
    on, _ = stages(seq2, PhaseConfig(**ON))
    assert "carry" in on
    # once carrying: one miss keeps carry, two misses in a row end it (retreat)
    one_miss = [dict(f, finger_gap=0.0 if i == 10 else 0.8) for i, f in enumerate(base)]
    two_miss = [dict(f, finger_gap=0.0 if i in (10, 11) else 0.8) for i, f in enumerate(base)]
    a, _ = stages(one_miss, PhaseConfig(**ON))
    b, _ = stages(two_miss, PhaseConfig(**ON))
    assert a[10] == "carry" and a[-1] == "carry"
    assert b[11] == "retreat"
    # gap_m = 1 enters on the first hit
    c, _ = stages(seq, PhaseConfig(**{**ON, "gap_m": 1}))
    assert "carry" in c


def test_fully_closed_hand_shows_its_own_fingers_and_is_not_carry():
    """Opening below ``blocked_min_open``: the fingers themselves fill the ROI; a surface there is not an object."""
    fr = hold_frames(gap=0.9, g_hold=0.03)
    on, _ = stages(fr, PhaseConfig(**ON))
    assert "carry" not in on


def test_no_wrist_evidence_falls_back_to_t91():
    fr = [{**f, "finger_gap": None} for f in hold_frames(gap=0.8)]
    on, _ = stages(fr, PhaseConfig(**ON))
    off, _ = stages(fr, PhaseConfig())
    assert on == off


def test_place_after_carry_by_destination_and_open_command():
    fr = hold_frames(gap=0.8, n_lift=8)
    dst = TGT + [0.0, 0.0, 0.0]
    fr = [dict(f, destination_pos=dst) for f in fr]
    on, _ = stages(fr, PhaseConfig(**ON))
    assert on[-1] == "place"


def test_projection_and_combined_sources():
    base = hold_frames(gap=0.0)
    proj = [dict(f, finger_gap=0.0, finger_gap_proj=0.9 if i >= 5 else 0.0) for i, f in enumerate(base)]
    assert "carry" in stages(proj, PhaseConfig(**{**ON, "finger_gap_source": "projection", "gap_tau_proj": 0.3}))[0]
    assert "carry" not in stages(proj, PhaseConfig(**{**ON, "finger_gap_source": "depth"}))[0]
    assert "carry" not in stages(proj, PhaseConfig(**{**ON, "finger_gap_source": "both", "gap_tau_proj": 0.3}))[0]
    assert "carry" in stages(proj, PhaseConfig(**{**ON, "finger_gap_source": "either", "gap_tau_proj": 0.3}))[0]


def test_config_validation():
    with pytest.raises(PhaseConfigError):
        PhaseConfig(finger_gap_mode="maybe")
    with pytest.raises(PhaseConfigError):
        PhaseConfig(finger_gap_source="sonar")
    with pytest.raises(PhaseConfigError):
        PhaseConfig(gap_m=0)
    with pytest.raises(PhaseConfigError):
        PhaseConfig(open_ref_value=0.0)


def test_machine_with_gap_is_causal():
    fr = hold_frames(gap=0.8, n_lift=8)
    cfg = PhaseConfig(**ON)
    full, _ = stages(fr, cfg)
    for k in (3, 7, 11):
        part, _ = stages(fr[:k], cfg)
        assert part == full[:k]


# --------------------------------------------------------------------------------- geometry

def test_surface_fraction_counts_roi_pixels_in_the_depth_window():
    cfg = FingerGapConfig(roi_u=(0.25, 0.75), roi_v=(0.5, 1.0), d_min=0.03, d_max=0.11)
    d = np.full((100, 200), 0.5)
    assert surface_fraction(d, cfg) == 0.0
    d[50:100, 50:100] = 0.07                                  # left half of the ROI (cols 50..149), all rows of the ROI
    assert surface_fraction(d, cfg) == pytest.approx(0.5)
    d[50:100, 100:150] = 0.02                                 # too close: not "within finger reach"
    assert surface_fraction(d, cfg) == pytest.approx(0.5)
    d[50:100, 100:150] = np.nan
    assert surface_fraction(d, cfg) == pytest.approx(0.5)
    assert surface_fraction(np.zeros(5), cfg) is None


def test_project_points_roundtrip_and_behind_camera():
    K = np.array([[200.0, 0, 100], [0, 200.0, 80], [0, 0, 1]])
    R = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])        # camera axes in the base frame (z forward = +x of base)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [1.0, 2.0, 3.0]
    pc = np.array([[0.1, -0.05, 0.4], [0.0, 0.0, -0.2]])
    P = pc @ R.T + T[:3, 3]
    uvz = project_points(P, K, T)
    assert uvz[0, 0] == pytest.approx(200 * 0.1 / 0.4 + 100) and uvz[0, 1] == pytest.approx(200 * -0.05 / 0.4 + 80)
    assert uvz[0, 2] == pytest.approx(0.4)
    assert np.isnan(uvz[1]).all()


def test_projection_fraction_splats_points_inside_the_roi_only():
    cfg = FingerGapConfig(roi_u=(0.4, 0.6), roi_v=(0.6, 0.9), d_min=0.03, d_max=0.11, splat_frac=0.02)
    inside = np.array([[100.0, 150.0, 0.08]])                 # ROI of a 200 x 200 image is cols 80..120, rows 120..180
    outside = np.array([[10.0, 10.0, 0.08]])
    assert projection_fraction(inside, (200, 200), cfg) > 0.0
    assert projection_fraction(outside, (200, 200), cfg) == 0.0
    assert projection_fraction(np.zeros((0, 3)), (200, 200), cfg) is None
    assert projection_fraction(np.full((2, 3), np.nan), (200, 200), cfg) is None


def test_robot_roi_settings_are_valid():
    for cfg in (rby1_wrist(), panda_eye_in_hand(), S.panda_gap_config(0.03, 0.11)):
        r0, r1, c0, c1 = cfg.box(120, 160)
        assert 0 <= r0 < r1 <= 120 and 0 <= c0 < c1 <= 160
    with pytest.raises(ValueError):
        FingerGapConfig(roi_u=(0.6, 0.4))
    with pytest.raises(ValueError):
        FingerGapConfig(d_min=0.2, d_max=0.1)


# --------------------------------------------------------------------------------- Panda / SafeLIBERO adapter

def test_panda_robot_settings_cannot_be_overridden_and_rest_comes_from_rby1_values():
    c = PhaseConfig.panda(blocked_min_open=0.5, chunk_gripper_column=7, d_near=0.05, **ON)
    assert c.chunk_gripper_column == 6 and c.blocked_min_open == pytest.approx(0.025325)
    assert (c.chunk_open_value, c.chunk_closed_value, c.open_ref_value) == (-1.0, 1.0, 0.04)
    assert c.d_near == 0.05 and c.finger_gap_mode == "on"


def test_panda_opening_is_finger_qpos_over_0_04_and_chunk_sign_is_closed_positive():
    sc = GripperScale(0.0, 3, 0.04)
    assert sc.update(0.04) == pytest.approx(1.0)
    assert sc.update(0.02) == pytest.approx(0.5)               # reference is the constant, not the running maximum
    assert sc.update(0.001) == pytest.approx(0.025)
    cmd = sc.normalise_chunk([-1.0, 1.0], -1.0, 1.0)
    assert list(cmd) == [1.0, 0.0]                             # -1 open -> 1, +1 close -> 0
    # close intent from a +1 chunk row
    m = PhaseMachine(PhaseConfig.panda())
    _, ev = m.step(PhaseObservation(gripper_raw=0.04, ee_pos=TGT + [0, 0, 0.02], target_pos=TGT, target_fresh=True,
                                    chunk_gripper=[-1.0, -1.0, 1.0]))
    assert ev["close_intent"] is True and ev["open_intent"] is True


def test_panda_empty_closure_is_not_carry_and_rim_hold_is_carry():
    """Empty closure settles at 0.0125 (normalised) and holds nothing; a thin-rim hold sits at >= 0.038 and can be carry."""
    cfg = PhaseConfig.panda(**ON)
    empty = hold_frames(gap=0.9, g_hold=0.0125)
    rim = hold_frames(gap=0.9, g_hold=0.05)
    for f in empty + rim:
        f["gripper_raw"] = f["gripper_raw"] * 0.04         # metres: 1.0 -> 0.04 m
    assert "carry" not in stages(empty, cfg)[0]
    assert "carry" in stages(rim, cfg)[0]


def test_gt_phases_follow_the_t77_definition():
    n = 12
    z = np.full((n, 3), 0.0)
    z[:, 2] = 0.9
    z[:, 0:2] = 0.5                                             # outside the destination box [0, 0.2]^2
    left = np.zeros(n, bool)
    right = np.zeros(n, bool)
    box = np.tile([0.0, 0.0, 0.2, 0.2], (n, 1)).astype(float)
    eef = np.tile([1.0, 1.0, 1.0], (n, 1)).astype(float)
    eef[2:, :] = [0.5, 0.5, 0.93]                               # 30 mm from the target: grasp window from frame 2
    left[4:9] = right[4:9] = True                               # both fingers touch frames 4..8
    z[6:9, 2] = 0.9 + 0.03                                      # rose 30 mm >= 20 mm from frame 6
    z[7:9, 0:2] = 0.1                                           # inside the destination box from frame 7
    gt = S.gt_phases(z, left, right, box, eef, list(range(n)))
    names = [E.STAGES[i] for i in gt]
    assert names[:2] == ["approach"] * 2
    assert names[2:6] == ["grasp"] * 4                          # touching but not lifted is still grasp
    assert names[6] == "carry" and names[7:9] == ["place", "place"]
    assert names[9:] == ["retreat"] * 3                         # not held after having been held
    # touching one finger only never counts as held
    gt2 = S.gt_phases(z, left, np.zeros(n, bool), box, eef, list(range(n)))
    assert "carry" not in [E.STAGES[i] for i in gt2]
    # the rim sensitivity: a touch outside the 60 mm window is grasp only with grasp_or_touch
    eef_far = np.tile([1.0, 1.0, 1.0], (n, 1)).astype(float)
    a = S.gt_phases(z, left, np.zeros(n, bool), box, eef_far, list(range(n)))
    b = S.gt_phases(z, left, np.zeros(n, bool), box, eef_far, list(range(n)), grasp_or_touch=True)
    assert E.STAGES[a[5]] == "approach" and E.STAGES[b[5]] == "grasp"


def _fake_episode(n=6):
    gap = {"depth": np.array([np.nan, 0.5, 0.5, 0.0, 0.5, 0.5]), "proj": np.full(n, np.nan)}
    ep = {"keep": np.array([1, 2, 3, 4, 5]), "grip_raw": np.full(n, 0.04), "ee": np.zeros((n, 3)),
          "centroid": np.tile([0.1, 0.2, 0.3], (n, 1)), "state": np.array(["visible"] * 3 + ["occluded"] * 3, dtype=object),
          "chunk_grip": np.tile([-1.0] * 10, (n, 1))}
    ep["centroid"][2] = np.nan
    return ep, gap


def test_observation_adapter_maps_columns_and_never_carries_ground_truth():
    ep, gap = _fake_episode()
    obs = list(S.observations(ep, gap, dest=None))
    assert len(obs) == 5                                        # only the kept calls
    assert obs[0].finger_gap == 0.5 and obs[2].finger_gap == 0.0
    assert obs[1].target_pos is None                            # NaN centroid -> no target
    assert obs[0].target_fresh is True and obs[3].target_fresh is False
    assert obs[0].destination_pos is None
    fields = set(PhaseObservation.__dataclass_fields__)
    assert not fields & {"gt", "phase", "held", "target_gt", "contact", "palm_dist"}
    ep2, _ = _fake_episode()
    assert S.observations(ep2, None).__next__().finger_gap is None


def test_merged_accuracy_treats_carry_and_place_as_one():
    run = {"label": "a", "scene": "t0", "split": "test", "y": np.array([E.CARRY, E.PLACE, E.APPROACH]),
           "t_step": np.arange(3), "request": np.arange(3)}
    p = np.array([E.PLACE, E.CARRY, E.APPROACH])
    assert S.merged_accuracy([run], [p]) == 1.0
    assert E.scene_mean([run], [p], E._acc) == pytest.approx(1 / 3)


def test_gap_selection_refuses_test_runs():
    with pytest.raises(AssertionError):
        E.select_gap([{"split": "test", "scene": "x"}], {}, "depth")
    with pytest.raises(AssertionError):
        E.select_gap([], {}, "depth")
