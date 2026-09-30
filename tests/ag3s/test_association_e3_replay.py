"""T30 F3 on the recorded E3 run where the manipulated object left the apple (asset-dependent).

`outputs/verify/T28/E3a_rec/E3a_ep1808_r1`: the server records (`chunk_*.npz` — attention cloud,
attention, support mask) are re-grounded with a fresh `TargetConfirm`, as the server did. The
replay reproduces the recorded manipulated object on every chunk (T30b.impl, 900/900 over the
twelve E3 runs). The adopted cluster at t=0 is the apple merged with the orange (54·67·144 mm);
recorded, the exclusion sphere left the apple as the hand hid it. With F3 (a)+(b) the exclusion
geometry must contain the whole GT apple (centre + 34 mm) on every pre-grasp chunk.
Skipped when the records are not on this machine.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

RUN = pathlib.Path("/mnt/dev/work/outputs/verify/T28/E3a_rec/E3a_ep1808_r1")
ANALYSIS = pathlib.Path("/mnt/dev/work/outputs/verify/T28/p3/E3a_analysis.json")
MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")
APPLE_BOUND = 0.034           # the apple's bounding radius (61·69·69 mm) — lead's criterion
FINE_VOXEL = 0.005

pytestmark = pytest.mark.skipif(
    not (RUN.exists() and ANALYSIS.exists() and MODEL.exists()),
    reason="E3 records / model not present")


def _chunks():
    an = json.loads(ANALYSIS.read_text())["runs"][RUN.name]["chunks"]
    rows = [json.loads(line) for line in (RUN / "frames.jsonl").read_text().splitlines() if line]
    plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    # pre-grasp, until the grasp is in progress (the server's `exclusion.grasp_active`)
    out = []
    for c in an:
        if "server_chunk" not in c or c["phase"] != "pre_grasp" or c["t"] > 136:
            continue
        out.append((RUN.parent / c["server_chunk"], plan[c["seq"]]))
    return out


def _apple_in_base(model, data, qpos, apple_world):
    import mujoco

    data.qpos[:len(qpos)] = qpos
    mujoco.mj_kinematics(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    R = data.xmat[bid].reshape(3, 3)
    return R.T @ (np.asarray(apple_world, np.float64) - data.xpos[bid])


def _replay(**confirm_kw):
    import dataclasses

    import mujoco

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.stages.target_grounding import TargetConfirm, ground_target
    from benchmark.ag3s.types import AttentionPointCloud, PointCloud

    cfg = AG3SConfig()
    # T31b: this file pins T30 F3 on the grounding the server ran — clusters as grown, no attention
    # split (the record predates it). With the split, see `test_attention_split_e3_replay.py`.
    clustering = dataclasses.replace(cfg.clustering, attention_split_radius=None)
    cfg = dataclasses.replace(cfg, clustering=clustering)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    confirm = TargetConfirm(cfg.clustering, max_opening=0.07146, overlap_distance=0.005,
                            **confirm_kw)
    out = []
    for npz, row in _chunks():
        z = np.load(npz, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        confirm.freeze(bool((s.get("exclusion") or {}).get("grasp_active")))
        apc = AttentionPointCloud(PointCloud(z["attention_cloud"].astype(np.float64)),
                                  z["attention_values"], raw_attention=z["attention_raw"])
        ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                      seed_threshold=cfg.attention.seed_threshold, exclude_mask=z["support_mask"],
                      min_radius=cfg.geometry.min_radius, timestamp=float(z["t_step"]),
                      confirm=confirm)
        man = confirm.manipulated
        apple = _apple_in_base(model, data, row["qpos"], row["object_poses"]["apple"]["pos"])
        pts = man.points
        centre = pts.mean(axis=0)
        radius = float(np.quantile(np.linalg.norm(pts - centre, axis=1),
                                   cfg.esdf.target_ball_quantile)) \
            + cfg.esdf.target_ball_pad_voxels * FINE_VOXEL
        inside = float(np.linalg.norm(apple - centre)) + APPLE_BOUND <= radius
        out.append((int(z["t_step"]), s.get("manipulated"), man, inside, confirm.last))
    return out


@pytest.fixture(scope="module")
def replay_old():
    return _replay(extent_ratio=None, anchor_coverage=None)


@pytest.fixture(scope="module")
def replay_new():
    return _replay()


def test_the_replay_reproduces_the_recorded_manipulated_object(replay_old):
    for t, rec, man, _, _ in replay_old:
        assert rec is not None and man.id == rec["id"] and man.state == rec["state"], t
        assert np.allclose(man.centroid, rec["centroid"], atol=1e-6), t


def test_recorded_the_exclusion_left_the_apple(replay_old):
    """The E3 failure, as recorded: some pre-grasp chunks exclude a sphere without the apple in it."""
    assert not all(inside for *_, inside, _ in replay_old)


def test_with_f3_the_apple_stays_inside_the_exclusion_on_every_pre_grasp_chunk(replay_new):
    missing = [t for t, _, _, inside, _ in replay_new if not inside]
    assert missing == []
    assert {m.id for _, _, m, _, _ in replay_new} == {0}
    assert any(d.reason == "subset_kept_anchor" for *_, d in replay_new)
