"""T54 replay conditions on synthetic frames (no MuJoCo): `t54_replay.run_frames` with the real AG3S.

* ``hold``: one AG3S per arm — an arm's rows do not depend on which other arms ran beside it, and
  the recorded latch inputs reach every arm identically, in `SafePolicy` order;
* ``current``: one AG3S, reset before every (frame, arm) call — a row equals a fresh AG3S on that
  frame alone;
* the recorded-summary parser (`freeze_inputs`) and the c1a profile builder (`c1a_build_kwargs`).
"""

from __future__ import annotations

import dataclasses
import time
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.experiments.studies import t54_arms as A
from AG3S.ag3s.experiments.studies import t54_replay as R
from AG3S.ag3s.runtime.pipeline import AG3S
from tests.ag3s.fixtures import (LABEL_BOX, LABEL_CYLINDER, LABEL_ROBOT, LABEL_SPHERE, LABEL_TABLE,
                                 PlanarTwoLinkArm, make_attention_grid, make_three_camera_scene)

CFG = AG3SConfig().with_overrides({
    "timing": {"expected_cameras": ["head", "left_wrist", "right_wrist"]},
    "clustering": {"gripper_max_opening": 0.12},
})
MJ_CAMS = ["zed_left", "wrist_cam_l", "wrist_cam_r"]  # -> head / left_wrist / right_wrist
TIMING = ("ag3s_ms", "ms_profile", "ms_primary_lift_posthoc", "ms_lift_ground", "ms_secondary_lift")
N_F = 3


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def scene(arm):
    return make_three_camera_scene(robot_model=arm)


def _labeler(scene):
    """GT point labels (`t47b_metrics.POINT_LABELS` indices) by nearest labelled depth point."""
    from scipy.spatial import cKDTree

    to_pl = {LABEL_SPHERE: 0, LABEL_BOX: 1, LABEL_CYLINDER: 1, LABEL_TABLE: 2, LABEL_ROBOT: 4}
    pts, lab = [], []
    for v in scene.views.values():
        for k, pl in to_pl.items():
            p = v.points_of(k)
            pts.append(p)
            lab.append(np.full(len(p), pl, np.int64))
    P, L = np.vstack(pts), np.concatenate(lab)
    tree = cKDTree(P)

    def f(points):
        q = np.asarray(points, np.float64).reshape(-1, 3)
        if not len(q):
            return np.zeros(0, np.int64)
        d, j = tree.query(q, k=1)
        out = np.full(len(q), 7, np.int64)
        out[d <= 0.003] = L[j[d <= 0.003]]
        return out

    return f


@pytest.fixture(scope="module")
def maps(scene):
    """(F, 3, 16, 16) per source: l8h2 = the fixture's sphere map; noun = a sharp box map on frame 1,
    sphere map otherwise (so a choice-by-noun arm can move between frames)."""
    view = next(iter(scene.views.values()))
    box = {o.name: o.center for o in view.objects}["neighbour_box"]
    sph = np.stack([o.attention_map for o in scene.observations])
    boxm = np.stack([make_attention_grid(scene.views[o.camera_id].hw,
                                         scene.views[o.camera_id].camera_intrinsics,
                                         scene.views[o.camera_id].T_base_cam, box,
                                         sigma_patches=0.7, floor=0.0)
                     for o in scene.observations])
    l8h2 = np.stack([sph] * N_F).astype(np.float32)
    noun = np.stack([sph, boxm, sph]).astype(np.float32)
    return {"l8h2": l8h2, "noun": noun}


@pytest.fixture(scope="module")
def frames(scene):
    lab = _labeler(scene)
    centre = next(iter(scene.views.values())).target.center

    def gen():
        for i in range(N_F):
            yield {"i": i, "n_f": N_F, "step": types.SimpleNamespace(t_step=8 * i),
                   "phase": "approach", "mj_cams": MJ_CAMS, "label_of": lab, "gt_centre": centre}
    return gen


def _ground_fn(scene):
    def ground(ag, fr, att):
        by_id = R.camera_id_maps(att)
        obs = [dataclasses.replace(o, attention_map=by_id[o.camera_id]) for o in scene.observations]
        t0 = time.perf_counter()
        cs, dbg = ag.process_multi_debug(obs, phase=fr["phase"])
        return cs, dbg, (time.perf_counter() - t0) * 1000.0
    return ground


def _strip(rows):
    """Rows without wall-clock fields, as canonical JSON text (NaN == NaN)."""
    import json

    from AG3S.ag3s.experiments.studies.t47b_metrics import jsonable

    return [json.dumps(jsonable({k: v for k, v in r.items() if k not in TIMING}), sort_keys=True)
            for r in rows]


class RecordingAG3S(AG3S):
    """AG3S that logs the latch setters in call order (behaviour unchanged)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = []

    def set_grasp_active(self, active):
        self.calls.append(("grasp_active", bool(active)))
        super().set_grasp_active(active)

    def set_grasp_latched(self, latched):
        self.calls.append(("grasp_latched", bool(latched)))
        super().set_grasp_latched(latched)

    def set_subtask(self, p):
        self.calls.append(("subtask", None if p is None else dict(p)))
        super().set_subtask(p)

    def set_placed(self, placed):
        self.calls.append(("placed", bool(placed)))
        super().set_placed(placed)


def _run(scene, frames, maps, arm_model, names, condition, *, freeze=None, built=None):
    def build():
        ag = RecordingAG3S(CFG, robot_model=arm_model)
        if built is not None:
            built.append(ag)
        return ag
    return R.run_frames(frames(), n_f=N_F, maps=maps, arms=A.resolve_arms(names),
                        condition=condition, build=build, freeze=freeze,
                        ground_fn=_ground_fn(scene), log=lambda m: None)


FREEZE = [
    {"grasp_active": False, "grasp_latched": False, "subtask_p": None, "placed": False},
    {"grasp_active": True, "grasp_latched": True, "subtask_p": {"pick": 0.9, "place": 0.05, "home": 0.05},
     "placed": False},
    {"grasp_active": False, "grasp_latched": True, "subtask_p": {"pick": 0.1, "place": 0.1, "home": 0.8},
     "placed": True},
]


# ------------------------------------------------------------------------------ conditions


def test_hold_histories_are_separate_per_arm(scene, frames, maps, arm):
    built = []
    joint = _run(scene, frames, maps, arm, ["L8H2", "A", "N"], "hold", built=built)
    assert len(built) == 3 and len({id(a) for a in built}) == 3   # one AG3S per arm
    for name in ("L8H2", "A", "N"):
        alone = _run(scene, frames, maps, arm, [name], "hold")
        assert _strip(joint[name]) == _strip(alone[name]), name
    # the hold condition carries history: after the first frame the decision is not "first"
    assert joint["L8H2"][0]["decision_mode"] == "first"
    assert all(r["decision_mode"] != "first" for r in joint["L8H2"][1:])


def test_current_resets_before_every_call(scene, frames, maps, arm):
    built = []
    cur = _run(scene, frames, maps, arm, ["L8H2", "A"], "current", built=built)
    assert len(built) == 1                                          # one shared AG3S, reset per call
    assert all(r["decision_mode"] == "first" for rs in cur.values() for r in rs)
    # every current row equals a brand-new AG3S on that frame alone
    for i in range(N_F):
        def one_frame():
            for fr in frames():
                if fr["i"] == i:
                    yield fr
        for name in ("L8H2", "A"):
            fresh = R.run_frames(one_frame(), n_f=N_F, maps=maps, arms=[A.ARMS[name]],
                                 condition="current",
                                 build=lambda: AG3S(CFG, robot_model=arm),
                                 ground_fn=_ground_fn(scene), log=lambda m: None)
            assert _strip(fresh[name]) == _strip([cur[name][i]])
    # and A's candidate lists equal L8H2's frame by frame (bit identity check fields)
    for a, l in zip(cur["A"], cur["L8H2"]):
        assert a["pre_split_digest"] == l["pre_split_digest"] is not None
        assert a["post_split_digest"] == l["post_split_digest"]


def test_recorded_latch_inputs_reach_every_hold_arm_identically(scene, frames, maps, arm):
    built = []
    rows = _run(scene, frames, maps, arm, ["L8H2", "A"], "hold", freeze=FREEZE, built=built)
    want = []
    for f in FREEZE:
        want += [("grasp_active", f["grasp_active"]), ("grasp_latched", f["grasp_latched"]),
                 ("subtask", f["subtask_p"]), ("placed", f["placed"])]
    assert [ag.calls for ag in built] == [want, want]
    assert all(r["freeze_injected"] for rs in rows.values() for r in rs)
    assert rows["A"][1]["freeze"]["grasp_active"] is True
    # the current condition never injects (no history to freeze)
    built_c = []
    cur = _run(scene, frames, maps, arm, ["L8H2"], "current", freeze=FREEZE, built=built_c)
    assert built_c[0].calls == [] and not any(r["freeze_injected"] for r in cur["L8H2"])
    with pytest.raises(ValueError):
        _run(scene, frames, maps, arm, ["L8H2"], "hold", freeze=FREEZE[:2])


def test_row_fields(scene, frames, maps, arm):
    rows = _run(scene, frames, maps, arm, ["L8H2", "A", "B"], "current")
    r = rows["A"][1]
    for k in ("identity_correct", "chosen_cluster_purity", "coverage", "chosen__other_object",
              "chosen__destination", "chosen__table", "target_candidate_present",
              "target_candidate_present_admissible", "clusters", "ms_lift_ground",
              "ms_secondary_lift", "pre_split_digest", "on_target", "n_target_obs"):
        assert k in r
    assert r["secondary_sources"] == ["noun"] and rows["B"][0]["secondary_sources"] == ["l8h2"]
    assert rows["L8H2"][0]["secondary_sources"] == [] and rows["L8H2"][0]["ms_secondary_lift"] == 0.0
    assert rows["B"][0]["ratio_source"] in ("l8h2_raw", "l8h2_normalized")
    cl = r["clusters"]
    assert cl and all(c["choice_mass"] is not None for c in cl)
    assert all(c["choice_mass"] is None for c in rows["L8H2"][1]["clusters"])


# ---------------------------------------------------------------------- recorded inputs


def _summary(**kw):
    s = {"t_step": 5, "chunk_index": 4,
         "exclusion": {"grasp_active": False, "attached_label": None},
         "admissibility": {"decision": {"mode": "keep", "frozen": False, "latched": True}},
         "grasp": {"state_before": "latched", "state": "latched", "attach": False, "detach": False},
         "subtask": {"p": {"pick": 0.7, "place": 0.2, "home": 0.1}, "placed_seen": False}}
    for k, v in kw.items():
        s[k] = v
    return s


def test_freeze_inputs_read_the_values_the_serving_ag3s_had():
    f = R.freeze_inputs(_summary())
    assert (f["grasp_active"], f["grasp_latched"], f["placed"]) == (False, True, False)
    assert f["subtask_p"] == {"pick": 0.7, "place": 0.2, "home": 0.1}
    assert f["source_latched"] == "decision" and f["source_grasp_active"] == "exclusion"
    # no decision (no cluster that frame): latched inferred from the grasp record
    f2 = R.freeze_inputs(_summary(admissibility={}, exclusion={"grasp_active": True, "attached_label": None}))
    assert f2["source_latched"] == "fallback" and f2["grasp_latched"] is True        # closing
    f3 = R.freeze_inputs(_summary(admissibility={}, exclusion={"grasp_active": True,
                                                               "attached_label": "manipulated"}))
    assert f3["grasp_latched"] is False                                                # held
    f4 = R.freeze_inputs(_summary(admissibility={}, grasp={"state_before": "searching"}))
    assert f4["grasp_latched"] is False
    f5 = R.freeze_inputs(_summary(exclusion={}, subtask={"p": None, "placed_seen": True}))
    assert f5["source_grasp_active"] == "missing" and f5["grasp_active"] is False
    assert f5["subtask_p"] is None and f5["placed"] is True
    s = R.freeze_summary([f, f2, f3, f4, f5])
    assert s["n"] == 5 and s["latched_fallback"] == 3 and s["grasp_active_missing"] == 1
    assert s["attached_frames"] == 1


def test_freeze_from_chunks_reads_summary_json(tmp_path):
    import json

    paths = []
    for i in range(2):
        p = tmp_path / f"chunk_{i:05d}.npz"
        np.savez(p, t_step=np.int64(i + 1), summary_json=np.asarray(json.dumps(_summary())))
        paths.append(p)
    fz = R.freeze_from_chunks(paths)
    assert len(fz) == 2 and fz[1]["chunk"] == str(paths[1]) and fz[0]["grasp_latched"] is True


# ------------------------------------------------------------------------------ profiles


ARGV = ["--config", "pi05_x", "--checkpoint", "/c", "--model-xml", "/m/model_transport.xml",
        "--esdf-backend", "curobo", "--fine-voxel", "0.005", "--tsdf-voxel", "0.005",
        "--attention-split-radius-mm", "12", "--attention-split-ratio-source", "raw",
        "--latched-identity-hold", "--closing-keep-geometry",
        "--target-field-policy", "exclude-authorized", "--target-exclusion", "ball-owned",
        "--owned-max-dist-voxels", "3"]


def test_c1a_profile_keeps_grounding_flags_and_drops_only_field_only_settings():
    xml, kw, rec = R.c1a_build_kwargs(ARGV)
    assert xml == "/m/model_transport.xml" and kw["backend"] == "legacy"
    assert kw["identity"] == {"closing_keep_geometry": True, "latched_identity_hold": True,
                              "attention_split_radius": 0.012, "attention_split_ratio_source": "raw"}
    assert kw["subtask_gate"] is True                    # serve_safe default resolution (gate on)
    assert kw["target_field_policy"] == "relax" and kw["target_exclusion"] is None
    assert kw["fine_voxel"] is None and kw["tsdf_voxel"] is None
    assert rec["field_only_overrides"]["target_field_policy"] == ["exclude-authorized", "relax"]
    assert rec["placed_no_seed_certify"] is False and rec["server_esdf_backend"] == "curobo"
    _, kw_c, rec_c = R.c1a_build_kwargs(ARGV, backend="curobo")
    assert kw_c["target_field_policy"] == "exclude-authorized" and rec_c["field_only_overrides"] == {}
    _, _, rec_p = R.c1a_build_kwargs(ARGV + ["--placed-no-seed-certify"])
    assert rec_p["placed_no_seed_certify"] is True        # make_builder refuses this server


def test_server_argv_parses_the_proc_file(tmp_path):
    p = tmp_path / "server_proc.txt"
    p.write_text("cwd=/x\nPYTHONPATH=/y\n/venv/bin/python -m AG3S.trajopt.serve_safe "
                 + " ".join(ARGV) + " --port 8290\n")
    assert R.server_argv(p) == ARGV + ["--port", "8290"]
    p.write_text("nothing here\n")
    with pytest.raises(SystemExit):
        R.server_argv(p)


def test_t52_profile_is_the_t47b_build():
    assert R.t52_build_kwargs() == dict(voxel=0.020, range_max=2.0, links="gripper",
                                        backend="legacy", subtask_gate=False)
