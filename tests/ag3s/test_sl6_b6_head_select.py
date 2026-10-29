"""SL6 B6 — noun-query head re-selection (`sl6_head_select`), on synthetic inputs (CPU).

Pinned: the batched resampling is V11AB's `attention_pixel_map` bit for bit; the top-k mask is the stable argsort;
identity / mass / coverage on a hand-made segmentation; task averaging; ranking ties; the 2 %p keep-L10H7 rule;
an end-to-end `select` over a fake B5 run + B3 labels run (approach requests only, other labels ignored).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import sl6_head_select as HS


def test_resample_is_attention_pixel_map():
    from AG3S.adapters.panda import attention as pa

    g = np.random.default_rng(0).random((5, 16, 16)).astype(np.float32)
    maps = HS.pixel_maps(g)
    for i in range(5):
        assert np.array_equal(maps[i], pa.attention_pixel_map(g[i], (256, 256)))


def test_top_mask_is_stable_argsort():
    rng = np.random.default_rng(1)
    v = np.round(rng.random(1000), 2)                   # many ties
    for frac in (0.01, 0.1, 0.37):
        k = int(np.ceil(frac * v.size))
        ref = np.zeros(v.size, bool)
        ref[np.argsort(-v, kind="stable")[:k]] = True
        assert np.array_equal(HS.top_mask(v, frac), ref)


def _seg():
    seg = np.full((256, 256), 3, np.int64)              # table geom 3
    seg[:, :20] = -1
    seg[100:140, 30:70] = 5                             # target geom 5 (OpenCV coords)
    seg[100:140, 180:220] = 6                           # other bowl geom 6
    seg[200:256, 100:160] = 1                           # robot geom 1
    return seg


INST = {"akita_black_bowl_1": [5], "akita_black_bowl_2": [6], "main_table": [3]}


def _grid_on(r0, r1, c0, c1):
    """Policy-orientation grid peaked on an OpenCV pixel box (un-mirror: policy col = 15 - opencv col)."""
    g = np.full((16, 16), 1e-4, np.float32)
    for r in range(r0 // 16, (r1 - 1) // 16 + 1):
        for c in range(c0 // 16, (c1 - 1) // 16 + 1):
            g[r, 15 - c] = 1.0
    return g


def test_label_image_and_request_metrics():
    lab, names = HS.label_image(_seg(), INST, "akita_black_bowl_1", [1])
    assert names[0] == "akita_black_bowl_1" and names[-3:] == ["robot", "other", "none"]
    assert lab[120, 50] == 0 and lab[120, 200] == names.index("akita_black_bowl_2")
    assert lab[220, 120] == names.index("robot") and lab[0, 0] == names.index("none")
    grids = np.stack([_grid_on(100, 140, 30, 70), _grid_on(100, 140, 180, 220), np.ones((16, 16), np.float32)])
    met = HS.request_metrics(HS.pixel_maps(grids), lab, len(names))
    assert met["identity"].tolist() == [True, False, False]
    assert met["mass"][0] > 0.5 > met["mass"][1] and np.isclose(met["mass"][2], 1600 / 65536)
    assert met["coverage"][0] == 1.0 and met["coverage"][1] == 0.0
    # target invisible -> identity False, coverage nan
    lab2, names2 = HS.label_image(np.full((256, 256), 3), INST, "akita_black_bowl_1", [1])
    met2 = HS.request_metrics(HS.pixel_maps(grids[:1]), lab2, len(names2))
    assert not met2["identity"][0] and np.isnan(met2["coverage"][0]) and met2["mass"][0] == 0.0


def test_task_average_rank_and_decision():
    per = {0: [np.array([1.0, 0.0]), np.array([1.0, 1.0])], 1: [np.array([0.0, 1.0])]}
    avg, per_task = HS.task_average(per)
    assert np.allclose(avg, [0.5, 0.75]) and np.allclose(per_task[0], [1.0, 0.5])
    cells = [(l, h) for l in range(2, 18) for h in range(8)]
    ident = np.zeros(len(cells))
    mass = np.zeros(len(cells))
    cov = np.full(len(cells), np.nan)
    k = cells.index((10, 7))
    ident[k] = 0.50
    ident[cells.index((4, 1))] = 0.515                  # +1.5 %p -> keep L10H7
    order = HS.rank(ident, mass, cov, cells)
    assert cells[order[0]] == (4, 1)
    d = HS.decide(ident, order, cells)
    assert d["kept_L10H7"] and d["spec"] == "noun_target:L10H7" and d["top_cell"] == [4, 1]
    ident[cells.index((4, 1))] = 0.52                   # +2 %p -> switch
    d = HS.decide(ident, HS.rank(ident, mass, cov, cells), cells)
    assert not d["kept_L10H7"] and d["spec"] == "noun_target:L4H1" and np.isclose(d["diff_pp"], 2.0)
    # ties on identity -> mass, then coverage, then lower layer / head
    ident[:] = 0.3
    mass[cells.index((9, 0))] = 0.2
    assert cells[HS.rank(ident, mass, cov, cells)[0]] == (9, 0)
    mass[:] = 0
    assert cells[HS.rank(ident, mass, cov, cells)[0]] == (2, 0)


def test_select_end_to_end(tmp_path):
    src = tmp_path / "src"
    (src / "episodes").mkdir(parents=True)
    noun = tmp_path / "noun"
    (noun / "noun_rows").mkdir(parents=True)
    lab_run = tmp_path / "labels"
    (lab_run / "labels").mkdir(parents=True)
    (lab_run / "phase").mkdir()
    body_names = np.array(["world", "robot0_link0", "table", "akita_black_bowl_1_main", "akita_black_bowl_2_main"])
    geom_body = np.array([0, 1, 0, 2, 0, 3, 4])           # geom 1 robot, 3 table, 5 bowl1, 6 bowl2
    good = _grid_on(100, 140, 30, 70)
    bad = _grid_on(100, 140, 180, 220)
    index_rows, label_rows = [], []
    for task in (0, 1):
        stem = f"libero_spatial_t{task}_e1"
        frames = [10, 15, 20]
        np.savez(src / "episodes" / f"{stem}_geom.npz", model_geom_bodyid=geom_body, model_body_names=body_names,
                 obs_index=np.arange(25), **{"segmentation_geom/head": np.repeat(_seg()[None], 25, 0)})
        img = np.full((3, 16, 8, 768), 1e-4, np.float32)
        for k in range(3):
            # cell (10, 7) right on every request; cell (3, 0) right only in task 0; request 2 is not approach
            img[k, 10 - 2, 7, :256] = good.reshape(-1)
            img[k, 3 - 2, 0, :256] = (good if task == 0 else bad).reshape(-1)
            if k == 2:
                img[k, 10 - 2, 7, :256] = bad.reshape(-1)
        np.savez(noun / "noun_rows" / f"{stem}_noun.npz", noun_image=img, layers=np.arange(2, 18),
                 noun_ok=np.ones(3, bool), obs_index=np.array(frames))
        np.savez(lab_run / "phase" / f"phase_{stem}.npz", call_frames=np.array(frames),
                 **{"segP/head": np.repeat(_seg()[None], 3, 0)})
        index_rows.append({"suite": "libero_spatial", "task_id": task, "episode_idx": 1, "init_state_index": 1,
                           "source_run_dir": str(src), "geom_npz": f"episodes/{stem}_geom.npz",
                           "noun_npz": f"noun_rows/{stem}_noun.npz",
                           "target": {"target": "akita_black_bowl_1", "instances_geom_ids": INST},
                           "lookup_first": {"noun": "bowl"}})
        label_rows.append({"stem": stem, "frames": frames, "labels": ["approach", "approach", "grasp"]})
    (noun / "noun_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index_rows))
    (lab_run / "labels" / "request_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in label_rows))
    res = HS.select(noun, lab_run)
    assert res["counts"]["requests_used"] == 4 and res["counts"]["episodes"] == 2
    L = res["layers"].index(10)
    assert res["identity"][L][7] == 1.0 and res["identity"][res["layers"].index(3)][0] == 0.5
    assert res["decision"]["kept_L10H7"] and res["decision"]["spec"] == "noun_target:L10H7"
    assert res["rank_of_L10H7"] == 1
    rec = HS.select(noun, lab_run, seg_source="rollout")
    assert rec["identity"] == res["identity"]
    # init 0 / another suite is refused
    bad_rows = [dict(index_rows[0], init_state_index=0)]
    (noun / "noun_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bad_rows))
    with pytest.raises(ValueError, match="dev episode"):
        HS.select(noun, lab_run)
