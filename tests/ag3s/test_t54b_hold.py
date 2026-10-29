"""T54b ``hold_attach`` on synthetic frames (no MuJoCo): `t54b_hold.run_frames_attach` with the real AG3S.

* the recorded attach / detach reach every arm at the same frame with the same parameters, and each arm
  attaches **its own** manipulated geometry (`test_injected_attach_*`);
* arm histories stay separate (an arm's rows do not depend on the arms beside it);
* T54's ``hold`` is unchanged: `t54_replay` is not touched, and without an attach / detach event the
  ``hold_attach`` rows equal the ``hold`` rows field for field;
* event parsing (`grasp_event`), injection outcomes (`inject_grasp`) and fidelity (`fidelity_fields`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t54_arms as A
from AG3S.ag3s.experiments.studies import t54_replay as R
from AG3S.ag3s.experiments.studies import t54b_hold as H
from AG3S.ag3s.runtime.pipeline import AG3S
from tests.ag3s.test_t54_replay import (CFG, N_F, TIMING, _ground_fn, _strip, arm, frames,  # noqa: F401
                                        maps, scene)

PARENT = "link_1"


def _summary(*, attach=False, detach=False, state="latched", grasp_active=False, attached=None,
             mode="evidence", manip=None, has_target=True, rejected=None):
    return {
        "t_step": 8, "chunk_index": 1, "has_target": has_target, "target_n_points": 10,
        "exclusion": {"grasp_active": grasp_active, "attached_label": attached,
                      "authorized_links": [PARENT, "ee_left"]},
        "admissibility": {"decision": {"mode": "keep", "frozen": grasp_active, "latched": True}},
        "grasp": {"mode": mode, "hand": "left", "parent_link": PARENT, "hold_link": None,
                  "state_before": "latched", "state": state, "attach": attach, "detach": detach,
                  "rejected": rejected, "evidence": {"reach_limit_mm": 120.0}},
        "subtask": {"p": {"pick": 0.8, "place": 0.1, "home": 0.1}, "placed_seen": state == "placed"},
        "manipulated": manip or {"id": 0, "state": "visible", "centroid": [0.616, 0.002, 0.804]},
    }


def _events(attach_at=1, detach_at=2):
    out = []
    for i in range(N_F):
        out.append(H.grasp_event(_summary(attach=i == attach_at, detach=i == detach_at,
                                          state="held" if i == attach_at else "latched",
                                          grasp_active=i == attach_at,
                                          attached="manipulated" if i == attach_at else None)))
    for e in out:   # the synthetic arm has no ee_left palm slot: attach on the finger link itself
        e["hold_link"] = None
        e["allowed_contact_links"] = [PARENT]
    return out


def _prepare(scene):
    def prepare(ag, fr, att):
        by_id = R.camera_id_maps(att)
        return [dataclasses.replace(o, attention_map=by_id[o.camera_id]) for o in scene.observations], fr["phase"]
    return prepare


def _srv_arrays(scene):
    """The 'serving' chosen target of the synthetic frames = the target sphere's GT points."""
    from tests.ag3s.fixtures import LABEL_SPHERE

    pts = np.vstack([v.points_of(LABEL_SPHERE) for v in scene.views.values()])

    def f(ev):
        if not ev["serving"]["has_target"]:
            return {"target_points": np.zeros((0, 3)), "target_centroid": None}
        return {"target_points": pts, "target_centroid": pts.mean(axis=0)}
    return f


class LoggingAG3S(AG3S):
    """Logs attach / detach / latch calls (behaviour unchanged) and checks the attached geometry is
    this AG3S's own manipulated geometry."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = []

    def attach(self, geometry, **kw):
        own = self.manipulated
        self.calls.append(("attach", self.frame_index, {k: (np.asarray(v).round(9).tolist()
                                                            if isinstance(v, np.ndarray) else v)
                                                        for k, v in kw.items()},
                           own is not None and geometry is own.geometry))
        return super().attach(geometry, **kw)

    def detach(self):
        self.calls.append(("detach", self.frame_index))
        return super().detach()

    def set_grasp_active(self, active):
        self.calls.append(("grasp_active", bool(active)))
        super().set_grasp_active(active)


def _run(scene, frames, maps, arm_model, names, *, events, attach_reach="off", built=None):
    def build():
        ag = LoggingAG3S(CFG, robot_model=arm_model, attached_parent_links=(PARENT,))
        if built is not None:
            built.append(ag)
        return ag
    return H.run_frames_attach(frames(), n_f=N_F, maps=maps, arms=A.resolve_arms(names), build=build,
                               events=events, prepare=_prepare(scene), attach_reach=attach_reach,
                               serving_arrays=_srv_arrays(scene), log=lambda m: None)


# ------------------------------------------------------------------------------- injection


def test_injected_attach_is_identical_across_arms_and_uses_each_arms_own_geometry(scene, frames, maps, arm):
    built = []
    rows = _run(scene, frames, maps, arm, ["L8H2", "A", "N"], events=_events(), built=built)
    assert len(built) == 3 and len({id(a) for a in built}) == 3          # one AG3S per arm
    grasp_calls = [[c for c in ag.calls if c[0] in ("attach", "detach")] for ag in built]
    assert grasp_calls[0] == grasp_calls[1] == grasp_calls[2]            # same frame, same parameters
    att = [c for c in grasp_calls[0] if c[0] == "attach"]
    assert len(att) == 1 and att[0][1] == 1                              # before frame 1's call
    assert att[0][2]["parent_link"] == PARENT and att[0][2]["label"] == "manipulated"
    assert att[0][2]["reach"] is None                                    # attach_reach = off
    assert all(c[3] for ag in built for c in ag.calls if c[0] == "attach")   # own geometry
    assert [c[1] for c in grasp_calls[0] if c[0] == "detach"] == [2]
    for name in ("L8H2", "A", "N"):
        out = [r["attach_outcome"] for r in rows[name]]
        assert out[1]["attach_called"] and out[1]["attach_result"] == "attached"
        assert out[2]["detach_called"] and out[2]["released"] is True
        assert rows[name][1]["fidelity"]["rep_attached"] is True
        assert rows[name][2]["fidelity"]["rep_attached"] is False
        assert rows[name][1]["grasp_injected"]["attach"] is True
    # the latch flag reaches every arm in SafePolicy order: after the grasp event, before the call
    seq = [c[0] for c in built[0].calls]
    assert seq.index("attach") < [i for i, c in enumerate(seq) if c == "grasp_active"][1]


def test_serving_reach_gate_rejects_and_is_recorded_per_arm(scene, frames, maps, arm):
    built = []
    rows = _run(scene, frames, maps, arm, ["L8H2", "A"], events=_events(), attach_reach="serving",
                built=built)
    for name in ("L8H2", "A"):
        o = rows[name][1]["attach_outcome"]
        assert o["attach_called"] and o["attach_result"] == "rejected"
        assert o["attach_rejected"]["limit_mm"] == pytest.approx(120.0)
        assert rows[name][1]["fidelity"]["rep_attached"] is False
    assert all(c[2]["reach"] == pytest.approx(0.12) for ag in built for c in ag.calls if c[0] == "attach")


def test_unusable_manipulated_object_is_rejected_without_calling_attach():
    class Stub:
        manipulated = None
        called = False

        def attach(self, *a, **kw):
            Stub.called = True

    ev = _events()[1]
    obs = [type("O", (), {"timestamp": 1.0, "robot_state": np.zeros(2)})()]
    out = H.inject_grasp(Stub(), ev, obs)
    assert out["attach_result"] == "rejected" and not Stub.called
    assert "no usable geometry" in out["attach_rejected"]["reason"]
    assert H.inject_grasp(Stub(), None, obs) == {"attach_called": False, "detach_called": False,
                                                 "attach_result": None}
    with pytest.raises(ValueError):
        H.inject_grasp(Stub(), ev, obs, attach_reach="maybe")


def test_hold_attach_histories_are_separate_per_arm(scene, frames, maps, arm):
    joint = _run(scene, frames, maps, arm, ["L8H2", "A", "N"], events=_events())
    for name in ("L8H2", "A", "N"):
        alone = _run(scene, frames, maps, arm, [name], events=_events())
        assert _strip(joint[name]) == _strip(alone[name]), name
    assert joint["L8H2"][0]["decision_mode"] == "first"
    assert all(r["decision_mode"] != "first" for r in joint["L8H2"][1:])


# ------------------------------------------------------------------- T54 hold unchanged


#: md5 of the T54 files T54b must not touch (T54.impl.md state; the old hold mode lives there).
T54_MD5 = {"t54_replay.py": "21a6dd92dc0796f1790de03a5788490e",
           "t54_arms.py": "fabb3c47f357018148a4bee1de24bfe7"}


def test_t54_replay_and_arms_are_byte_identical_to_t54():
    d = pathlib.Path(R.__file__).parent
    for name, want in T54_MD5.items():
        assert hashlib.md5((d / name).read_bytes()).hexdigest() == want, name


SHARED = ("frame", "t_step", "arm", "identity_correct", "has_target", "chosen_cluster_purity",
          "coverage", "chosen__other_object", "decision_mode", "decision_reason", "decision_frozen",
          "pre_split_digest", "post_split_digest", "chosen_rank", "clusters", "freeze")


def test_without_grasp_events_hold_attach_equals_the_t54_hold_rows(scene, frames, maps, arm):
    ev = _events(attach_at=-1, detach_at=-1)
    assert not any(e["attach"] or e["detach"] for e in ev)
    new = _run(scene, frames, maps, arm, ["L8H2", "A"], events=ev)
    old = R.run_frames(frames(), n_f=N_F, maps=maps, arms=A.resolve_arms(["L8H2", "A"]),
                       condition="hold",
                       build=lambda: AG3S(CFG, robot_model=arm, attached_parent_links=(PARENT,)),
                       freeze=ev,
                       ground_fn=_ground_fn(scene), log=lambda m: None)
    for name in ("L8H2", "A"):
        a = [json.dumps({k: r.get(k) for k in SHARED}, sort_keys=True, default=str) for r in new[name]]
        b = [json.dumps({k: r.get(k) for k in SHARED}, sort_keys=True, default=str) for r in old[name]]
        assert a == b, name
        assert all(r["condition"] == "hold" for r in old[name])
        assert all(r["condition"] == "hold_attach" for r in new[name])
    # with an attach the two conditions part from the attach frame on (the held object is filtered)
    att = _run(scene, frames, maps, arm, ["L8H2"], events=_events(attach_at=1, detach_at=-1))
    assert att["L8H2"][0]["n_cloud"] == new["L8H2"][0]["n_cloud"]
    assert att["L8H2"][1]["fidelity"]["rep_attached"] and not new["L8H2"][1]["fidelity"]["rep_attached"]


# ------------------------------------------------------------------------- recorded events


def test_grasp_event_reads_latch_attach_detach_and_parameters():
    e = H.grasp_event(_summary(attach=True, state="held", grasp_active=True, attached="manipulated"))
    assert e["attach"] and not e["detach"] and e["latch_state"] == "held"
    assert e["parent_link"] == PARENT and e["reach_m"] == pytest.approx(0.12)
    assert e["grasp_active"] is True and e["grasp_latched"] is True          # T54 freeze keys kept
    assert e["serving"]["attached"] is True and e["serving"]["manip_id"] == 0
    assert e["allowed_contact_links"] == ["ee_finger_l1", "ee_finger_l2"]    # SafePolicy defaults
    r = H.grasp_event(_summary(attach=True, rejected={"reason": "reach"}))
    assert not r["attach"] and r["serving_attach_rejected"]                  # a rejected attach is no attach
    d = H.grasp_event(_summary(detach=True, state="placed"))
    assert d["detach"] and d["placed"] is True
    leg = H.grasp_event(_summary(attach=True, mode="legacy_gripper"))
    assert H.event_problems([leg]) and not H.event_problems([e, d])
    s = H.events_summary([e, r, d])
    assert s["attach_frames"] == [0] and s["detach_frames"] == [2] and s["serving_attach_rejected"] == 1
    assert s["latch_state_frames"] == {"held": 1, "latched": 1, "placed": 1}


def test_grasp_links_match_safe_policy_defaults():
    from AG3S.trajopt.safe_policy import DEFAULT_GRASP_LINKS, DEFAULT_HOLD_LINKS

    for hand in ("left", "right"):
        parent, allowed, hold = H.grasp_links_of(hand)
        assert (parent, allowed) == (DEFAULT_GRASP_LINKS[hand][0], tuple(DEFAULT_GRASP_LINKS[hand][1]))
        assert hold == DEFAULT_HOLD_LINKS[hand]
    assert H.grasp_links_of(None) == (None, (), None)


def test_events_from_chunks_reads_npz(tmp_path):
    ps = []
    for i in range(2):
        p = tmp_path / f"chunk_{i:05d}.npz"
        np.savez(p, t_step=np.int64(i + 1), summary_json=np.asarray(json.dumps(_summary(attach=i == 1))),
                 target_points=np.ones((4, 3)), target_centroid=np.ones(3))
        ps.append(p)
    ev = H.events_from_chunks(ps)
    assert [e["attach"] for e in ev] == [False, True] and ev[1]["chunk"] == str(ps[1])
    arr = H.load_serving_arrays(ev[0])
    assert arr["target_points"].shape == (4, 3) and arr["target_centroid"].tolist() == [1, 1, 1]
    q = tmp_path / "chunk_none.npz"
    np.savez(q, t_step=np.int64(3), summary_json=np.asarray(json.dumps(_summary(has_target=False))))
    assert H.load_serving_arrays({"chunk": str(q)})["target_points"].shape == (0, 3)


# ------------------------------------------------------------------------------- fidelity


def test_fidelity_fields_and_summary(scene, frames, maps, arm):
    rows = _run(scene, frames, maps, arm, ["L8H2"], events=_events())
    f0 = rows["L8H2"][0]["fidelity"]
    assert f0["srv_has_target"] and f0["rep_has_target"] and f0["has_target_agree"]
    assert f0["chosen_centroid_dist_m"] is not None
    assert f0["manip_same_object"] in (True, False) and f0["srv_manip_id"] == 0
    s = H.fidelity_summary({"ep0": rows["L8H2"], "ep1": rows["L8H2"]})
    assert s["pooled"]["has_target_agree"]["n"] == 2 * N_F
    assert s["attach_results"] == {"attached": 2}
    assert set(s["episode_mean"]) == set(H.AGREE_FIELDS)


def test_fidelity_same_object_rule():
    assert H._same(False, False, None, 0.03) and not H._same(True, False, None, 0.03)
    assert H._same(True, True, 0.02, 0.03) and not H._same(True, True, 0.05, 0.03)


# ------------------------------------------------------------------- entry table option


def _summary_for_entry(vals):
    out = {}
    for cond, mv in vals.items():
        out[cond] = {"all": {m: {"A": {"diff_vs_L8H2": {"mean": v, "ci95_episode_bootstrap": [v - 1, v + 1]}}}
                             for m, v in mv.items()}, "timing": {}}
    return out


def test_entry_criteria_default_is_the_t54_table_and_hold_condition_is_an_option():
    from AG3S.ag3s.experiments.studies import t54_stats as st

    cur = {"identity": 0.01, "coverage": 0.0, "purity": -0.01, "neighbour": 1.0, "no_target": 0.0}
    s = _summary_for_entry({"current": cur, "hold": {"identity": -0.10, "coverage": 0.0},
                            "hold_attach": {"identity": 0.0, "coverage": -0.01}})
    d = st.entry_criteria(s, arm="A")
    assert set(d) == {"arm", "base", "condition", "note", "rows", "all_numeric_rows_pass"}
    assert set(d["rows"]) == {"identity", "coverage", "improvement_observed", "hold", "timing"}
    assert d["rows"]["hold"]["identity"] == -0.10 and not d["rows"]["hold"]["pass"]
    assert d["all_numeric_rows_pass"] is False
    h = st.entry_criteria(s, arm="A", hold_condition="hold_attach")
    assert h["rows"]["hold"]["identity"] == 0.0 and h["rows"]["hold"]["pass"]
    assert h["all_numeric_rows_pass"] is True and h["hold_condition"] == "hold_attach"
    p = st.entry_criteria(s, arm="A", hold_condition="hold_attach", direction_positive=True)
    assert p["rows"]["direction"]["pass"] and p["all_numeric_rows_pass"]
    neg = _summary_for_entry({"current": {**cur, "identity": -0.01}, "hold_attach": {"identity": 0.0,
                                                                                    "coverage": 0.0}})
    n = st.entry_criteria(neg, arm="A", hold_condition="hold_attach", direction_positive=True)
    assert n["rows"]["identity"]["pass"] and not n["rows"]["direction"]["pass"]
    assert n["all_numeric_rows_pass"] is False


def test_find_rows_prefers_this_run_then_extra_dirs(tmp_path):
    from AG3S.ag3s.experiments.studies import t54_study as S

    root = tmp_path / "out"
    other = tmp_path / "t54" / "test" / "rows"
    other.mkdir(parents=True)
    (other / "ep1__current.jsonl").write_text("")
    (other / "ep1__current.meta.json").write_text("{}")
    assert S.find_rows(root, "test", "ep1", "current", [other]) == other / "ep1__current.jsonl"
    mine = S.rows_path(root, "test", "ep1", "current")
    mine.parent.mkdir(parents=True)
    mine.write_text("")
    mine.with_suffix(".meta.json").write_text("{}")
    assert S.find_rows(root, "test", "ep1", "current", [other]) == mine
    assert S.find_rows(root, "test", "ep2", "hold", [other]) == S.rows_path(root, "test", "ep2", "hold")
