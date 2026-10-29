"""AG3S-CORE-EVAL-G1 — the geometric grasp correction G1 ("VLA 닫기 근처 보정") merged from `feature/geo-grasp`.

Task: `AG3S/ag3s/docs/handoff/AG3S-CORE-EVAL-G1.task.md`. Report: `AG3S-CORE-EVAL-G1.impl.md`.
G1 server flags: `--geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off` (default off).

What this file fixes:

1. **Off = the dump as before** — without `--geo-*` flags no `geo_*` key in `cost_overrides` nor in
   `TrajOptConfig.to_dict()` (the `--record-constraints` meta `trajopt_config`); the A8 command line's config is the same
   with and without the merge (no geo key, same values).
2. **G1 flags** — map to exactly the four cost keys; the dump carries exactly those; `from_dict(to_dict())` round trip.
3. **G7 not merged** — the approach guidance (`approach_guide.py`, `--approach-*`, `cost.approach_*`, `GraspAnchor.
   path_offset` / `track_from`, the refiner's `reference_trajectory_override` / `approach_seeds`) is absent.
4. **G1 with the grasp-goal term (A4, `--grasp-anchor W --grasp-anchor-path-weight 0`)** — user ruling 2026-10-29 option
   (a): accepted; on a chunk where G1 has a geometric goal it replaces the A4 anchor (the record says so:
   `to.metrics.grasp_anchor.replaced_by_geo` · `replaced_variant`, `grasp.anchor.replaced_by_geo`, `grasp.geo.anchor.slot`),
   on the others the A4 anchor stays as it is. Every other combination is refused as before (IFG-TOA path term, E1-B
   box, widened window, redirect mode). One flag alone: no new key anywhere.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import shlex

import pytest

from AG3S.trajopt import serve_safe
from AG3S.trajopt.config import CostConfig, TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_anchor import GraspAnchor

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
G1 = ["--geo-grasp", "--geo-goal-mode", "correct", "--geo-track-scale", "0.25", "--geo-continuity-off"]
#: ADOPTED_KO.md §A8 server flags (the TO-relevant ones; recorded `REG_k0/runs/server_*/server_proc.txt`)
A8 = ("--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper "
      "--target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 "
      "--sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover "
      "--closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach "
      "--held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles "
      "--lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --to-fast-eval --esdf-host-mirror "
      "--target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 "
      "--attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 "
      "--attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 "
      "--nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint "
      "--roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 "
      "--to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass --palm-shell on --palm-shell-margin 5")
A4 = ["--grasp-anchor", "2000", "--grasp-anchor-path-weight", "0"]
GEO_FIELDS = {f.name for f in dataclasses.fields(CostConfig) if f.name.startswith("geo_")}


def _parse(argv):
    return serve_safe.build_parser().parse_args(BASE_ARGV + list(argv))


# ------------------------------------------------------------------------------------------------ 1. off
def test_off_has_no_geo_key_anywhere():
    assert GEO_FIELDS, "the merged CostConfig carries the geo fields"
    c = TrajOptConfig()
    assert c.cost.geo_grasp is False
    assert not GEO_FIELDS & set(c.to_dict()["cost"])
    assert TrajOptConfig.from_dict(c.to_dict()) == c
    a = _parse(shlex.split(A8) + A4)
    w = serve_safe.cost_overrides(a)
    assert not any(k.startswith("geo_") for k in w)
    assert w["w_grasp_goal"] == 2000.0 and w["w_grasp_tcp"] == 0.0
    d = serve_safe.trajopt_config_from_args(a).to_dict()
    assert not GEO_FIELDS & set(d["cost"])
    assert d["cost"]["w_grasp_goal"] == 2000.0 and d["cost"]["w_grasp_tcp"] == 0.0


# ------------------------------------------------------------------------------------------------ 2. G1 flags
def test_g1_flags_map_to_four_cost_keys_and_round_trip():
    w = serve_safe.cost_overrides(_parse(G1))
    assert w == {"geo_grasp": True, "geo_goal_mode": "correct", "geo_track_scale": 0.25, "geo_continuity_off": True}
    cfg = serve_safe.trajopt_config_from_args(_parse(G1))
    d = cfg.to_dict()["cost"]
    assert {k for k in d if k.startswith("geo_")} == set(w)
    assert TrajOptConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.cost.geo_correct_pos == 0.040 and cfg.cost.geo_correct_rot_deg == 20.0     # decision doc §2: 40 mm / 20°
    assert cfg.cost.geo_time_budget_ms is None                                            # generator default (150 ms)
    # the A8 command line (A4 excluded, see 4.) + G1 builds
    cfg2 = serve_safe.trajopt_config_from_args(_parse(shlex.split(A8) + G1))
    assert cfg2.cost.geo_grasp and cfg2.cost.geo_goal_mode == "correct" and cfg2.collision.palm_shell_margin == 0.005


# ------------------------------------------------------------------------------------------------ 3. G7 not merged
def test_g7_approach_guidance_is_not_merged():
    assert importlib.util.find_spec("AG3S.trajopt.approach_guide") is None
    assert not any(f.name.startswith("approach_") for f in dataclasses.fields(CostConfig))
    assert not {"path_offset", "track_from"} & {f.name for f in dataclasses.fields(GraspAnchor)}
    with pytest.raises(SystemExit):
        _parse(["--approach-guide", "cost"])
    import inspect

    from AG3S.trajopt import refiner, sqp
    assert "reference_trajectory_override" not in inspect.getsource(refiner)
    assert "approach_seeds" not in inspect.getsource(refiner) and "approach_seeds" not in inspect.getsource(sqp)


# ------------------------------------------------------------------------------------------------ 4. G1 x A4
def test_g1_with_the_e1c_goal_term_builds_and_other_combinations_are_refused():
    cfg = serve_safe.trajopt_config_from_args(_parse(shlex.split(A8) + A4 + G1))
    assert cfg.cost.geo_grasp and cfg.cost.w_grasp_goal == 2000.0 and cfg.cost.w_grasp_tcp == 0.0
    assert TrajOptConfig.from_dict(cfg.to_dict()) == cfg
    refused = (["--grasp-anchor", "2000"] + G1,                                       # IFG-TOA (path term on)
               ["--grasp-goal-box", "10"] + G1,                                       # E1-B
               A4 + ["--grasp-goal-window", "widen"] + G1,                            # GGW widen (not adopted)
               A4 + ["--geo-grasp"])                                                  # redirect mode (G2)
    for extra in refused:
        with pytest.raises(TrajOptConfigError, match="geo_grasp replaces the IFG-TOA / E1 grasp anchor"):
            serve_safe.trajopt_config_from_args(_parse(extra))


def _anchor(**kw):
    import numpy as np

    base = dict(link="ee_left", goal=np.array([0.5, 0.1, 0.9]), close_row=4, w_tcp=0.0, w_goal=2000.0)
    base.update(kw)
    return GraspAnchor(**base)


@pytest.mark.parametrize("geo_has_goal", [True, False])
def test_both_on_the_slot_holds_g1_where_it_has_a_goal_and_a4_elsewhere(geo_has_goal):
    """`SafePolicy._geo_request` with an A4 anchor already in the context (as `_grasp_anchor_gate` leaves it): a chunk
    where the geometric controller returns a goal gets that goal (marked as replacing E1-C); a chunk where it returns
    none keeps the A4 anchor object untouched. The records say which term held the slot."""
    import types

    import numpy as np

    from AG3S.trajopt.grasp_anchor import anchor_record
    from tests.trajopt.test_safe_policy import DOF, H, _safe_policy

    cfg = TrajOptConfig().with_overrides({"cost": {"w_grasp_tcp": 0.0, "w_grasp_goal": 2000.0, "geo_grasp": True,
                                                   "geo_goal_mode": "correct", "geo_track_scale": 0.25,
                                                   "geo_continuity_off": True}})
    sp = _safe_policy(to_config=cfg)
    assert sp._geo is not None
    a4 = _anchor()
    assert a4.variant == "E1-C"
    g1 = _anchor(goal=np.array([0.52, 0.12, 0.88]), goal_R=np.eye(3), w_rot=5.0, goal_kind="correct")
    assert g1.variant == "GEO"
    geo_rec = {"active": geo_has_goal}

    def fake_anchor(q_now, chunk):
        sp._geo.last["anchor"] = geo_rec
        return g1 if geo_has_goal else None

    sp._geo.step = lambda **kw: {}
    sp._geo.anchor = fake_anchor
    sp._pending = {"chunk": np.zeros((H, DOF)), "timing": {}}
    sp._last_grasp_record = {}
    sp._last_grasp_anchor = {"active": True}            # the A4 gate record (`grasp.anchor`) of this chunk
    ctx = {"grasp_anchor": a4}
    sp._geo_request(np.zeros(sp.constraint_robot_model.nq), ["left"], ctx, None, {"filtered_cloud": np.zeros((0, 3))})
    got = ctx["grasp_anchor"]
    if geo_has_goal:
        assert got.variant == "GEO" and np.array_equal(got.goal, g1.goal) and got.replaced_variant == "E1-C"
        assert sp._last_grasp_anchor["replaced_by_geo"] is True and geo_rec["slot"] == "geo"
        p = np.zeros((H, 3))
        rec = anchor_record(got, p, p, np.tile(np.eye(3), (H, 1, 1)), np.tile(np.eye(3), (H, 1, 1)))
        assert rec["variant"] == "GEO" and rec["replaced_by_geo"] is True and rec["replaced_variant"] == "E1-C"
    else:
        assert got is a4 and np.array_equal(got.goal, a4.goal)
        assert sp._last_grasp_anchor["replaced_by_geo"] is False and geo_rec["slot"] == "E1-C"
        rec = anchor_record(got, np.zeros((H, 3)), np.zeros((H, 3)))
        assert rec["variant"] == "E1-C" and "replaced_by_geo" not in rec


def test_one_flag_alone_leaves_no_new_record_key():
    """G1 alone (no A4 anchor in the context): no slot / replaced keys; A4 alone: the anchor record as before."""
    import numpy as np

    from AG3S.trajopt.grasp_anchor import anchor_record
    from tests.trajopt.test_safe_policy import DOF, H, _safe_policy

    cfg = TrajOptConfig().with_overrides({"cost": {"geo_grasp": True, "geo_goal_mode": "correct"}})
    sp = _safe_policy(to_config=cfg)
    g1 = _anchor(goal_R=np.eye(3), w_rot=5.0, goal_kind="correct")
    rec_geo = {"active": True}

    def fake_anchor(q_now, chunk):
        sp._geo.last["anchor"] = rec_geo
        return g1

    sp._geo.step = lambda **kw: {}
    sp._geo.anchor = fake_anchor
    sp._pending = {"chunk": np.zeros((H, DOF)), "timing": {}}
    sp._last_grasp_record = {}
    sp._last_grasp_anchor = {}
    ctx: dict = {}
    sp._geo_request(np.zeros(sp.constraint_robot_model.nq), ["left"], ctx, None, {"filtered_cloud": np.zeros((0, 3))})
    assert ctx["grasp_anchor"] is g1 and g1.replaced_variant is None and "slot" not in rec_geo
    assert sp._last_grasp_anchor == {}
    p = np.zeros((H, 3))
    assert "replaced_by_geo" not in anchor_record(g1, p, p, np.tile(np.eye(3), (H, 1, 1)), np.tile(np.eye(3), (H, 1, 1)))
    assert "replaced_by_geo" not in anchor_record(_anchor(), p, p)
