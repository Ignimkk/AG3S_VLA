"""T31b on the recorded E3 / T14 runs — the merge is split, the single apple is not (asset-dependent).

The server records (`chunk_*.npz`: attention cloud, attention, support mask) are re-grounded with a
fresh `TargetConfirm` (max opening 71.46 mm), freezing where the server's `exclusion.grasp_active`
was set — the T30b replay. GT object centres come from the paired planning row.

  - E3a ep1808 r1 (apple + orange, 54·67·144 mm at t=0) and E3a ep1800 r1 (apple + banana,
    59·65·164 mm): before the grasp, the manipulated object is the apple split out of the merge —
    the apple's centre inside the exclusion ball, the neighbour's centre outside it, every chunk.
  - E3a ep1807 r1 and T14 (the apple alone, attention uneven as the hand closes in): the
    manipulated geometry is **identical** with and without the split.
Skipped when the records are not on this machine.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import dataclasses
import json
import pathlib

import numpy as np
import pytest

T28 = pathlib.Path(_ag3s_workspace_path('experiments/legacy/outputs/verify/T28'))
ANALYSIS = T28 / "p3/E3a_analysis.json"
T14 = pathlib.Path(_ag3s_workspace_path('legacy/outputs/live_test/20260928_t14'))
MODEL = pathlib.Path(
    _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml'))
OPENING = 0.07146
FINE_VOXEL = 0.005

pytestmark = pytest.mark.skipif(
    not (ANALYSIS.exists() and MODEL.exists() and (T28 / "E3a_rec/E3a_ep1808_r1").exists()),
    reason="E3 records / model not present")


def _e3_chunks(run, t_max):
    an = json.loads(ANALYSIS.read_text())["runs"][run]["chunks"]
    rows = [json.loads(line) for line in (T28 / "E3a_rec" / run / "frames.jsonl").read_text()
            .splitlines() if line]
    plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    return [(T28 / "E3a_rec" / c["server_chunk"], plan[c["seq"]]) for c in an
            if "server_chunk" in c and c["phase"] == "pre_grasp" and c["t"] <= t_max]


def _t14_chunks():
    rows = [json.loads(line) for line in (T14 / "execute_ep1807/frames.jsonl").read_text()
            .splitlines() if '"planning"' in line]
    plan = {int(r["seq"]): r for r in rows if r["kind"] == "planning"}
    out = []
    for f in sorted((T14 / "server_constraints/run_0000").glob("chunk_*.npz")):
        k = int(f.stem.split("_")[1])
        row = plan.get(k + 1)
        if row is None:
            continue
        if np.asarray(row["actions_reference"])[:8, 7].min() < 0.5:
            break                                      # the grasp attempt: pre-grasp ends here
        out.append((f, row))
    return out


def _in_base(model, data, qpos, world):
    import mujoco

    data.qpos[:len(qpos)] = qpos
    mujoco.mj_kinematics(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    R = data.xmat[bid].reshape(3, 3)
    return R.T @ (np.asarray(world, np.float64) - data.xpos[bid])


def _replay(chunks, *, split=True):
    import mujoco

    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target
    from AG3S.ag3s.types import AttentionPointCloud, PointCloud

    cfg = AG3SConfig()
    clustering = (cfg.clustering if split
                  else dataclasses.replace(cfg.clustering, attention_split_radius=None))
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    confirm = TargetConfirm(clustering, max_opening=OPENING, overlap_distance=FINE_VOXEL)
    out = []
    for npz, row in chunks:
        z = np.load(npz, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        confirm.freeze(bool((s.get("exclusion") or {}).get("grasp_active")))
        apc = AttentionPointCloud(PointCloud(z["attention_cloud"].astype(np.float64)),
                                  z["attention_values"], raw_attention=z["attention_raw"])
        g = ground_target(apc, clustering, seed_percentile=cfg.attention.seed_percentile,
                          seed_threshold=cfg.attention.seed_threshold,
                          exclude_mask=z["support_mask"], min_radius=cfg.geometry.min_radius,
                          timestamp=float(z["t_step"]), confirm=confirm)
        man = confirm.manipulated
        pts = man.points
        centre = pts.mean(axis=0)
        radius = float(np.quantile(np.linalg.norm(pts - centre, axis=1),
                                   cfg.esdf.target_ball_quantile)) \
            + cfg.esdf.target_ball_pad_voxels * FINE_VOXEL
        gt = {b: _in_base(model, data, row["qpos"], p["pos"])
              for b, p in row["object_poses"].items() if b in ("apple", "orange", "banana")}
        inside = {b: bool(np.linalg.norm(v - centre) <= radius) for b, v in gt.items()}
        out.append({"t": int(z["t_step"]), "man": man, "inside": inside, "grounding": g,
                    "decision": confirm.last})
    return out


@pytest.fixture(scope="module")
def ep1808():
    # t ≤ 96: before the policy's attention moves to the pear (a T26 switch at t=128).
    return _replay(_e3_chunks("E3a_ep1808_r1", 96))


@pytest.fixture(scope="module")
def ep1800():
    return _replay(_e3_chunks("E3a_ep1800_r1", 136))


def test_ep1808_t0_the_merge_is_split_and_the_apple_adopted(ep1808):
    first = ep1808[0]                                        # t=0
    assert first["decision"].mode == "first"
    rec = first["man"].record()
    assert rec["anchor"]["from_split"] is True and rec["split"]["manipulated_from_split"] is True
    # The attended merge (largest attention mass) is the 54·67·144 mm apple+orange; the
    # banana+pear merge beside it (157 mm, attention ~0) is split too, as an obstacle pair.
    merge = max(rec["split"]["decisions"], key=lambda d: d["before"]["attention_mass"])
    assert merge["accepted"] and 140.0 < merge["before"]["extents_mm"][-1] < 150.0
    assert merge["after"]["extents_mm"][-1] < 80.0 and merge["after"]["attention_share"] > 0.99
    assert first["man"].points.shape[0] == merge["after"]["n_points"]


def test_ep1808_before_the_grasp_the_ball_holds_the_apple_and_not_the_orange(ep1808):
    assert {r["man"].id for r in ep1808} == {0}
    assert [r["t"] for r in ep1808 if not r["inside"]["apple"]] == []
    assert [r["t"] for r in ep1808 if r["inside"]["orange"]] == []


def test_ep1800_before_the_grasp_the_ball_holds_the_apple_and_not_the_banana(ep1800):
    assert {r["man"].id for r in ep1800} == {0}
    assert ep1800[0]["man"].record()["anchor"]["from_split"] is True
    assert [r["t"] for r in ep1800 if not r["inside"]["apple"]] == []
    assert [r["t"] for r in ep1800 if r["inside"]["banana"]] == []


@pytest.mark.parametrize("which", ["ep1807", "t14"])
def test_the_single_apple_is_unchanged_by_the_split(which):
    if which == "t14" and not T14.exists():
        pytest.skip("T14 records not present")
    chunks = _e3_chunks("E3a_ep1807_r1", 64) if which == "ep1807" else _t14_chunks()
    new, old = _replay(chunks), _replay(chunks, split=False)
    assert len(new) == len(old) >= 8
    for a, b in zip(new, old):
        assert a["man"].id == b["man"].id and a["man"].state == b["man"].state, a["t"]
        np.testing.assert_array_equal(a["man"].points, b["man"].points, err_msg=str(a["t"]))
        # the apple itself was never split (other clusters in the scene may be)
        assert not a["man"].record()["anchor"]["from_split"], a["t"]
