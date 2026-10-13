"""SL4b: dev calibration of the Panda ATS attention cell -- pre-registration in code, replay options
(``--attention-cell``, ``--grounding-only``, ``--attention-source oracle``, several source runs), phase labels and
image-instant segmentation, and the evaluator's metric / selection / adoption / cause split (CPU; no GPU)."""
import importlib.util
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from AG3S.adapters.panda import attention as pa
from AG3S.adapters.panda import model as pm
from AG3S.adapters.panda import profile as pf
from benchmark.adapters.libero_panda import client_env
from benchmark.evaluators import ats_calibration as ac
from benchmark.evaluators import libero_phases as lp
from benchmark.evaluators import libero_scene_gt as G
from benchmark.runners import perception_replay as pr
from common.workspace import PROJECT_ROOT, shared_storage_root

LIBERO_PYTHON = PROJECT_ROOT / ".venv-libero/bin/python"
PHASE_EXPORT = PROJECT_ROOT / "benchmark/adapters/libero_panda/phase_export.py"
TASK_FILE = PROJECT_ROOT / "AG3S/ag3s/docs/handoff/SL4b.task.md"
SL3_GT = PROJECT_ROOT / "outputs/verify/SL3/raw/gt"  # verifier output (read only; tests skip without it)
SL3_SCORE = PROJECT_ROOT / "outputs/verify/SL3/scripts/score_attention.py"
DEV_RUN = "20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0"
AUDIT_RUN = "20261012T202459_309e4c6da477_sl3-attention-audit-dev"


def _storage_runs():
    base = shared_storage_root() / "results/benchmark"
    return (base / DEV_RUN).is_dir() and (base / AUDIT_RUN).is_dir()


needs_runs = pytest.mark.skipif(not _storage_runs(), reason="SL3 dev / audit runs not in this storage root")
needs_libero = pytest.mark.skipif(not LIBERO_PYTHON.exists(), reason="needs .venv-libero")


# --- pre-registration ------------------------------------------------------------------------------------
def _task_table_rows():
    text = TASK_FILE.read_text()
    section = text.split("## 사전 등록", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(tuple(cells))
    assert rows[0] == ("항목", "정의")
    return tuple(rows[1:])


def test_preregistration_is_the_task_table_verbatim_and_its_numbers():
    assert ac.PREREGISTRATION_ROWS == _task_table_rows()
    assert ac.ADOPTION_RULE_TEXT == dict(_task_table_rows())["채택 조건"]
    assert ac.TASK_IDS == tuple(range(10)) and ac.DEV_INIT_STATES == (0, 1, 2, 3, 4)
    assert ac.EVAL_PHASES == ("approach", "grasp") and ac.BASELINE_CELL == (8, 2) and ac.N_TOP_CELLS == 16
    assert ac.CENTROID_MAX_M == 0.030 and ac.NEAR_M == 0.010 and ac.VIS_TOL_M == 0.010
    assert (ac.ADOPT_MIN_GAIN_NUM, ac.ADOPT_MIN_GAIN_DEN, ac.ADOPT_MIN_TASKS) == (1, 10, 6)
    assert (ac.GRID_DENOISE_STEP, ac.GRID_AGGREGATION) == (0, "last")
    cell = ac.check_baseline_cell()  # the pre-registered L8H2 is V11AB's cell
    assert (cell["layer"], cell["head"]) == pr.v11ab_layer_head() == (8, 2)
    rec = ac.preregistration_record()
    assert [(r["항목"], r["정의"]) for r in rec["rows_verbatim"]] == list(ac.PREREGISTRATION_ROWS)
    assert rec["adoption_rule_text"] == ac.ADOPTION_RULE_TEXT and len(rec["rows_sha256"]) == 64
    assert rec["constants"]["adopt_min_gain"] == "1/10" and rec["constants"]["n_top_cells"] == 16
    assert set(rec["operational_definitions"]) >= {"evaluation_calls", "point_label", "majority", "centroid_error",
                                                   "selection_metric", "cause_split", "segmentation", "oracle"}
    json.dumps(rec, ensure_ascii=False)


# --- replay options ----------------------------------------------------------------------------------------
def test_parse_cell_and_attention_spec_rules():
    assert pr.parse_cell("8,2") == (8, 2) and pr.parse_cell(" 17 , 7 ") == (17, 7)
    for bad in ("8", "8,2,1", "-1,2", "a,b"):
        with pytest.raises(Exception):
            pr.parse_cell(bad)
    default, l8h2, l8h2_f16, other = (pr.AttentionSpec(), pr.AttentionSpec(cell=(8, 2)),
                                      pr.AttentionSpec(cell=(8, 2), from_pooled_f16=True), pr.AttentionSpec(cell=(3, 1)))
    assert default.resolved_cell() == l8h2.resolved_cell() == (8, 2)
    assert default.grid_key() == l8h2.grid_key() == "cell_grid_policy" and default.is_v11ab_cell() and l8h2.is_v11ab_cell()
    assert l8h2_f16.grid_key() == other.grid_key() == "pooled_f16" and not other.is_v11ab_cell()
    oracle = pr.AttentionSpec(source="oracle")
    assert oracle.grid_key() is None and oracle.resolved_cell() is None
    assert oracle.record()["privileged"] is True and default.record()["privileged"] is False
    for kwargs in ({"source": "oracle", "cell": (8, 2)}, {"source": "oracle", "mapping": "mirror"},
                   {"source": "segmentation"}, {"mapping": "flip"}):
        with pytest.raises(ValueError):
            pr.AttentionSpec(**kwargs)
    parser = pr.build_parser()
    base = ["--source-run", "a", "--attention-run", "b"]
    a0 = parser.parse_args(base)
    a1 = parser.parse_args(base + ["--attention-cell", "8,2"])
    a2 = parser.parse_args(base + ["--attention-source", "oracle", "--phases", "x", "--grounding-only"])
    assert pr.attention_spec(a0).grid_key() == pr.attention_spec(a1).grid_key() == "cell_grid_policy"
    assert not pr.is_sl4b(a0) and pr.is_sl4b(a1) and pr.is_sl4b(a2)
    assert pr.attention_spec(a2).source == "oracle" and a2.phases == ["x"] and a2.grounding_only
    assert pr.is_sl4b(parser.parse_args(base + ["--source-run", "c", "--attention-run", "d"]))


def _synthetic_attn(n=3, L=18, H=8, seed=0):
    rng = np.random.default_rng(seed)
    pooled = rng.random((n, L, H, 768)).astype(np.float16)
    return {"pooled_f16": pooled, "cell_grid_policy": rng.random((n, 2, 16, 16)).astype(np.float32),
            "obs_index": np.arange(n), "global_step": np.arange(n)}


def test_episode_attention_grids_cell_blocks_and_underflow():
    attn = _synthetic_attn()
    for spec in (pr.AttentionSpec(), pr.AttentionSpec(cell=(8, 2))):  # the SL4 grids, bit for bit
        grids, info = pr.episode_attention_grids(attn, spec)
        assert grids.dtype == np.float32 and np.array_equal(grids, attn["cell_grid_policy"])
        assert not info["invalid"].any()
    attn["pooled_f16"][1, 3, 1, 256:512] = 0  # float16 underflow of the wrist block (SL3 #6)
    grids, info = pr.episode_attention_grids(attn, pr.AttentionSpec(cell=(3, 1)))
    for cam, (b0, b1) in enumerate(((0, 256), (256, 512))):
        assert np.array_equal(grids[:, cam], attn["pooled_f16"][:, 3, 1, b0:b1].astype(np.float32).reshape(-1, 16, 16))
    assert info["invalid"].tolist() == [False, True, False] and info["block_sum"][1, 1] == 0.0
    assert not grids[1, 1].any()  # used as it is
    f16, _ = pr.episode_attention_grids(attn, pr.AttentionSpec(cell=(8, 2), from_pooled_f16=True))
    assert np.array_equal(f16[:, 0], attn["pooled_f16"][:, 8, 2, :256].astype(np.float32).reshape(-1, 16, 16))
    with pytest.raises(ValueError, match="outside"):
        pr.episode_attention_grids(attn, pr.AttentionSpec(cell=(18, 0)))
    with pytest.raises(ValueError, match="pooled_f16"):
        pr.episode_attention_grids({k: v for k, v in attn.items() if k != "pooled_f16"}, pr.AttentionSpec(cell=(3, 1)))


def test_several_source_runs_are_paired_in_order(monkeypatch, tmp_path):
    def fake(dev, audit, task_ids=None):
        eps = [{"stem": f"libero_spatial_t{t}_e{dev[-1]}", "task_id": t, "episode_idx": int(dev[-1]),
                "init_state_index": int(dev[-1]), "dev_run_dir": str(tmp_path / dev), "npz": f"{t}.npz"}
               for t in (task_ids or (0, 1))]
        return {"dev": {"run_dir": dev}, "attention": {"run_dir": audit}, "episodes": eps}

    monkeypatch.setattr(pr, "index_sources", fake)
    one = pr.index_source_pairs(["dev0"], ["aud0"])
    assert one["dev"] == {"run_dir": "dev0"} and len(one["pairs"]) == 1  # the SL4 layout
    two = pr.index_source_pairs(["dev0", "dev1"], ["aud0", "aud1"], task_ids=[0, 1])
    assert two["dev"] is None and [p["dev"]["run_dir"] for p in two["pairs"]] == ["dev0", "dev1"]
    assert [e["stem"] for e in two["episodes"]] == ["libero_spatial_t0_e0", "libero_spatial_t1_e0",
                                                    "libero_spatial_t0_e1", "libero_spatial_t1_e1"]
    assert pr.check_dev_split(two) == [0, 1]
    with pytest.raises(ValueError, match="more than one source run"):
        pr.index_source_pairs(["dev0", "dev0"], ["aud0", "aud1"])
    with pytest.raises(ValueError, match="pair up"):
        pr.index_source_pairs(["dev0", "dev1"], ["aud0"])
    with pytest.raises(ValueError, match="outside the dev split"):
        pr.check_dev_split(pr.index_source_pairs(["dev5"], ["aud5"]))

    # scene GT in several directories: each stem in exactly one
    for d in ("dev0", "dev1"):
        (tmp_path / d).mkdir()
        for t in (0, 1):
            np.savez(tmp_path / d / f"{t}.npz", chunk_t=np.array([10, 15]))
    for e in two["episodes"]:
        e.update(geom_npz_sha256="g" + e["stem"], npz_sha256="n" + e["stem"])
    dirs = [tmp_path / "gt_a", tmp_path / "gt_b"]
    for d in dirs:
        d.mkdir()
    for e in two["episodes"]:
        d = dirs[0] if e["stem"].endswith("e0") else dirs[1]
        prov = json.dumps({"geom_npz_sha256": e["geom_npz_sha256"], "npz_sha256": e["npz_sha256"]})
        np.savez(d / f"scene_{e['stem']}.npz", provenance_json=np.array(prov), frames=np.array([10, 15]))
    out = pr.index_scene_gt([str(d) for d in dirs], two)
    assert len(out["files"]) == 4 and out["dirs"] == [str(d.resolve()) for d in dirs] and "dir" not in out
    single = pr.index_scene_gt(str(dirs[0]), {"episodes": two["episodes"][:2]})
    assert single["dir"] == str(dirs[0].resolve())  # SL4 layout kept
    with pytest.raises(ValueError, match="no scene GT"):
        pr.index_scene_gt([str(dirs[0])], two)
    (dirs[1] / "scene_libero_spatial_t0_e0.npz").write_bytes((dirs[0] / "scene_libero_spatial_t0_e0.npz").read_bytes())
    with pytest.raises(ValueError, match="more than one directory"):
        pr.index_scene_gt([str(d) for d in dirs], two)


# --- phase labels -------------------------------------------------------------------------------------------
def _sl3_phases_of():
    if not SL3_SCORE.exists():
        return None
    spec = importlib.util.spec_from_file_location("sl3_score_attention", SL3_SCORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.phases_of


def test_phase_labels_rule():
    def track(z, contact):
        return {"target_xpos": np.stack([np.zeros(40), np.zeros(40), z], 1), "contact_grip_target": contact.copy(),
                "call_frames": np.array([10, 15, 20, 25, 30, 35])}

    frames = [10, 15, 20, 25, 30, 35]
    z, contact = np.zeros(40), np.zeros(40, bool)
    info, labels = lp.phase_labels(track(z, contact), frames)
    assert labels.tolist() == ["approach"] * 6 and info["first_contact"] is None  # never touched
    contact[18:31] = True
    z[24:] = 0.0101  # lift at 24 (> z0 + 10 mm), last contact 30
    info, labels = lp.phase_labels(track(z, contact), frames)
    assert (info["first_contact"], info["lift"], info["last_contact"]) == (18, 24, 30)
    assert labels.tolist() == ["approach", "approach", "grasp", "transport", "transport", "post_release"]
    z[24:] = 0.0100  # exactly z0 + 10 mm is not a lift: grasp until the end
    _, labels = lp.phase_labels(track(z, contact), [10, 20, 35])
    assert labels.tolist() == ["approach", "grasp", "grasp"]
    phases_of = _sl3_phases_of()
    if phases_of is not None:  # the SL3 verifier's own function gives the same labels
        for zz in (np.zeros(40), np.r_[np.zeros(24), np.full(16, 0.02)], np.r_[np.zeros(30), np.full(10, 0.02)]):
            tr = track(zz, contact)
            assert lp.phase_labels(tr, frames)[1].tolist() == phases_of(tr, frames)[1].tolist()


def test_phase_export_is_python38_and_uses_the_sl3_contact_rule():
    import ast

    ast.parse(PHASE_EXPORT.read_text(), feature_version=(3, 8))
    spec = importlib.util.spec_from_file_location("phase_export_mod", PHASE_EXPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gripper = np.array([False, True, False, False])
    robot = np.array([False, True, True, False])
    # target geom 0; contacts: target-gripper, gripper-target (reversed), target-robot arm, target-table, table-gripper
    pairs = [(0, 1), (1, 0), (0, 2), (0, 3), (3, 1)]
    assert module.contact_flags(pairs, {0}, gripper, robot) == (True, 2, True, 4)
    assert module.contact_flags([(0, 3)], {0}, gripper, robot) == (False, 0, False, 1)


@needs_runs
def test_phase_labels_of_init0_equal_the_sl3_series():
    if not SL3_GT.is_dir():
        pytest.skip("SL3 verifier GT export not present")
    series = json.loads((PROJECT_ROOT / "outputs/verify/SL3/scores/series.json").read_text())
    src = pr.index_source_pairs([DEV_RUN], [AUDIT_RUN], task_ids=[0, 4, 9])
    for e in src["episodes"]:
        with np.load(e["attn_npz"]) as a:
            obs = a["obs_index"].astype(np.int64)
        labels, info, track = ac.episode_phases([SL3_GT], e["stem"], obs, e["geom_npz_sha256"], segmentation=True)
        assert labels.tolist() == series[e["stem"]]["phase"]
        assert lp.segmentation_at(track, int(obs[0]), "head").shape == (256, 256)
        with pytest.raises(KeyError):
            lp.segmentation_at(track, int(obs[0]) + 1, "head")


# --- evaluator: per call, aggregate, adoption, selection, cause --------------------------------------------
NAMES = ("bowl_1", "bowl_2", "plate", "table")


def test_point_labels_majority_and_call_metric_categories():
    inf = np.inf
    D = np.array([[0.002, inf, inf, 0.009], [inf, 0.004, inf, inf], [inf, inf, inf, inf], [inf, inf, inf, inf],
                  [0.0101, inf, inf, inf]])
    robot = np.array([1.0, 1.0, 0.005, 0.5, 1.0])
    assert ac.label_points(D, NAMES, robot).tolist() == ["bowl_1", "bowl_2", "robot", "none", "none"]
    assert ac.label_points(D, NAMES).tolist() == ["bowl_1", "bowl_2", "none", "none", "none"]
    assert ac.majority_label(["bowl_1", "bowl_1", "plate"])[0] == "bowl_1"
    assert ac.majority_label(["bowl_1", "plate"])[0] is None  # a tie is not a majority
    assert ac.majority_label(["none", "none", "bowl_1"])[0] == "none" and ac.majority_label([])[0] is None

    gt_vis = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])  # visible GT target centroid (0.01, 0, 0)
    pts = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.5, 0.0, 0.0]])

    def call(labels, centroid, has=True, points=pts, vis=gt_vis):
        return ac.call_metrics(has_target=has, target_name="bowl_1", target_points=points, target_centroid=centroid,
                               point_labels=labels, dist_to_target=np.array([0.0, 0.005, 0.2])[:len(points)],
                               gt_target_visible=vis)

    ok = call(["bowl_1", "bowl_1", "plate"], [0.01, 0.0, 0.0293])
    assert ok["target_correct"] == 1 and ok["category"] == "correct" and ok["precision_hits"] == 2
    assert ok["recall"] == 1.0 and abs(ok["centroid_err_m"] - 0.0293) < 1e-12
    far = call(["bowl_1", "bowl_1", "plate"], [0.01, 0.0, 0.0301])
    assert far["target_correct"] == 0 and far["category"] == "centroid_far"
    wrong = call(["plate", "plate", "bowl_1"], [0.01, 0.0, 0.0])
    assert wrong["target_correct"] == 0 and wrong["category"] == "wrong_instance" and wrong["majority"] == "plate"
    tie = call(["bowl_1", "plate", "none"], [0.01, 0.0, 0.0])
    assert tie["category"] == "wrong_instance" and tie["majority"] is None
    invisible = call(["bowl_1"] * 3, [0.01, 0.0, 0.0], vis=np.zeros((0, 3)))
    assert invisible["category"] == "no_visible_gt" and invisible["target_correct"] == 0 and invisible["recall"] is None
    none = call([], [np.nan] * 3, has=False, points=np.zeros((0, 3)))
    assert none["category"] == "no_target" and none["target_correct"] == 0 and none["recall"] == 0.0

    rows = [dict(r, task=t, attention_invalid=False) for r, t in ((ok, 0), (far, 0), (wrong, 1), (none, 1))]
    agg = ac.aggregate(rows)
    assert agg["primary"] == {"correct": 1, "n": 4, "rate": 0.25}  # no-target calls stay in the denominator
    assert agg["per_task"] == {"0": {"correct": 1, "n": 2, "rate": 0.5}, "1": {"correct": 0, "n": 2, "rate": 0.0}}
    assert agg["precision_points"] == {"within_near": 6, "total": 9} and agg["has_target_calls"] == 3
    assert agg["recall_median_has_target_visible"] == 1.0 and agg["recall_median_no_target_as_0"] == 1.0
    assert agg["categories"] == {"correct": 1, "centroid_far": 1, "wrong_instance": 1, "no_target": 1}
    assert ac.majority_kind("plate") == "other_instance" and ac.majority_kind(None) == "tie"
    assert ac.majority_kind("none") == "none" and ac.majority_kind("robot") == "robot"


def _result(cell, per_task_correct, n_per_task=10):
    return {"cell": cell, "primary": {"correct": sum(per_task_correct), "n": n_per_task * len(per_task_correct)},
            "per_task": {str(t): {"correct": c, "n": n_per_task} for t, c in enumerate(per_task_correct)}}


def test_adoption_rule_boundaries_and_tie_break():
    base = [5] * 10  # 50 / 100
    res = {"L8H2": _result((8, 2), base),
           # +10 correct of 100 = exactly 10 %p, better or equal on 6 tasks -> satisfies
           "L3H1": _result((3, 1), [8, 8, 8, 8, 7, 5, 4, 4, 4, 4]),
           # +9 -> 9 %p: fails the gain
           "L2H0": _result((2, 0), [6, 6, 6, 6, 6, 6, 6, 6, 6, 5]),
           # +15 but only 5 tasks >= baseline: fails the task count
           "L1H1": _result((1, 1), [9, 9, 9, 9, 9, 4, 4, 4, 4, 4]),
           # same total and tasks as L3H1 but a higher layer: tie -> lower layer (L3H1) wins
           "L9H0": _result((9, 0), [8, 8, 8, 8, 7, 5, 4, 4, 4, 4])}
    d = ac.adoption_decision(res, "L8H2", ["L3H1", "L2H0", "L1H1", "L9H0"])
    rows = {r["label"]: r for r in d["candidates"]}
    assert rows["L3H1"]["gain_ok"] and rows["L3H1"]["tasks_ge_baseline"] == 6 and rows["L3H1"]["satisfies_rule"]
    assert not rows["L2H0"]["gain_ok"] and rows["L2H0"]["tasks_ok"]
    assert rows["L1H1"]["gain_ok"] and rows["L1H1"]["tasks_ge_baseline"] == 5 and not rows["L1H1"]["tasks_ok"]
    assert d["satisfying"] == ["L3H1", "L9H0"] and d["selected_by_rule"] == "L3H1" and d["selected_cell"] == [3, 1]
    best = dict(res, L7H7=_result((7, 7), [9, 8, 8, 8, 7, 5, 4, 4, 4, 4]))  # higher primary beats lower layer
    assert ac.adoption_decision(best, "L8H2", ["L3H1", "L7H7"])["selected_by_rule"] == "L7H7"
    none = ac.adoption_decision(res, "L8H2", ["L2H0", "L1H1"])
    assert none["selected_by_rule"] is None and "keep L8H2" in none["outcome"]
    assert d["rule_text"] == ac.ADOPTION_RULE_TEXT
    with pytest.raises(ValueError, match="different calls"):
        ac.adoption_decision({**res, "L4H4": _result((4, 4), [5] * 9)}, "L8H2", ["L4H4"])


def test_selection_mass_ranking_and_cause_split():
    N, L, H = 2, 4, 3
    pooled = np.zeros((N, L, H, 768), np.float16)
    covs = np.zeros((N, 2, 256))
    covs[:, 0, 5] = 1.0  # target covers head patch 5
    covs[:, 1, 7] = 0.5  # half of wrist patch 7
    pooled[:, 1, 2, 5] = 3.0  # cell (1, 2): all head mass on the target
    pooled[:, 1, 2, 256 + 7] = 1.0
    pooled[:, 1, 2, 256 + 8] = 1.0  # wrist: half the block on patch 7 (coverage 0.5) -> 0.25
    pooled[:, 0, 0, 0] = 1.0  # cell (0, 0): head on a non-target patch; wrist underflows
    pooled[:, 2, 1, 5] = 1.0
    pooled[:, 2, 1, 6] = 1.0
    pooled[:, 2, 1, 256 + 7] = 2.0  # cell (2, 1): head 0.5, wrist 0.5
    mass, invalid = ac.selection_mass(pooled, covs)
    assert mass.shape == (N, 2, L, H) and np.allclose(mass[:, 0, 1, 2], 1.0) and np.allclose(mass[:, 1, 1, 2], 0.25)
    assert invalid[:, 1, 0, 0].all() and not invalid[:, 0, 0, 0].any() and mass[:, 1, 0, 0].sum() == 0.0
    metric = mass.mean((0, 1))
    assert np.isclose(metric[1, 2], 0.625) and np.isclose(metric[2, 1], 0.5)
    ranked = ac.rank_cells(metric, exclude=(1, 2), k=3)
    assert ranked[0] == (2, 1) and (1, 2) not in ranked and ranked[1:] == [(0, 0), (0, 1)]  # ties: lower layer, head
    assert ac.rank_cells(metric, exclude=(-1, -1), k=2) == [(1, 2), (2, 1)]

    grid = np.zeros((2, 16, 16))
    cov = np.zeros((2, 16, 16))
    cov[0, 3, 4] = 0.2
    grid[0, 3, 4], grid[1, 9, 9] = 0.6, 0.8  # joint argmax on the wrist, off the target
    c = ac.attention_cause(grid, cov)
    assert c["argmax_camera"] == "left_wrist" and c["cause"] == "attention" and c["head_argmax_on_target"]
    grid[0, 3, 4] = 0.9
    c = ac.attention_cause(grid, cov)
    assert c["argmax_camera"] == "head" and c["argmax_patch"] == [3, 4] and c["cause"] == "grounding"
    rows = [{"category": "wrong_instance", "majority": m, "task": t, "cause": ac.attention_cause(g, cov)}
            for m, t, g in (("plate", 4, grid), (None, 5, np.flip(grid, 0)))]
    split = ac.cause_split(rows + [{"category": "correct"}])
    assert split["misselections"] == 2 and split["grounding_cause"] == 1 and split["attention_cause"] == 1
    assert split["by_majority_kind"] == {"other_instance": {"misselections": 1, "attention": 0, "grounding": 1},
                                         "tie": {"misselections": 1, "attention": 1, "grounding": 0}}


def test_pairwise_closest_points_and_near_distance_are_exact():
    rng = np.random.default_rng(5)
    tri = rng.normal(size=(40, 3, 3)) * 0.05
    p = rng.normal(size=(60, 3)) * 0.06
    allpairs = G.closest_points_on_triangles(p, tri[:, 0], tri[:, 1], tri[:, 2])  # (N, T, 3)
    i, j = np.repeat(np.arange(60), 40), np.tile(np.arange(40), 60)
    pair = G.closest_points_pairwise(p[i], tri[j, 0], tri[j, 1], tri[j, 2])
    assert np.array_equal(pair, allpairs.reshape(-1, 3))
    exact = G.point_triangles_distance(p, tri)
    for r in (0.005, 0.02, 0.1):
        near = G.point_triangles_distance_near(p, tri, r, pair_budget=97)
        assert np.array_equal(np.isfinite(near), exact <= r)
        assert np.allclose(near[exact <= r], exact[exact <= r], rtol=0, atol=1e-15)


# --- storage: the SL3 audit, the SL4 path, oracle, grounding-only on CPU ---------------------------------
@needs_runs
def test_v11ab_cell_options_read_the_sl4_grid_and_dry_run(capsys, monkeypatch):
    src = pr.index_sources(DEV_RUN, AUDIT_RUN, task_ids=[0])
    with np.load(src["episodes"][0]["attn_npz"]) as a:
        attn = {k: a[k] for k in a.files}
    sl4, _ = pr.episode_attention_grids(attn, pr.AttentionSpec())
    l8h2, _ = pr.episode_attention_grids(attn, pr.AttentionSpec(cell=(8, 2)))
    f16, info = pr.episode_attention_grids(attn, pr.AttentionSpec(cell=(8, 2), from_pooled_f16=True))
    assert np.array_equal(sl4, attn["cell_grid_policy"]) and np.array_equal(l8h2, sl4)
    assert np.abs(f16 - sl4).max() < 1e-6 and not info["invalid"].any()  # float16 rounding only

    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(PROJECT_ROOT))
    pr.main(["--source-run", DEV_RUN, "--attention-run", AUDIT_RUN, "--no-gt", "--task-ids", "0", "--limit-calls", "1",
             "--attention-cell", "8,2", "--grounding-only", "--dry-run"])
    text = capsys.readouterr().out
    out = json.loads(text[text.index("{"):])
    s = out["settings"]
    assert s["step"] == "SL4b" and s["study"] == "SL4b" and s["privileged_attention"] is False
    assert s["preregistration"]["rows_sha256"] == ac.preregistration_record()["rows_sha256"]
    assert s["options"]["attention"]["grid_array"] == "cell_grid_policy" and s["options"]["attention"]["is_v11ab_cell"]
    assert s["options"]["grounding_only"]["what"] == pr.GROUNDING_ONLY_NOTE
    assert out["checks"][0]["attention_equals_audit_opencv"] is True and out["checks"][0]["attention_invalid_calls"] == 0
    assert s["data_split"].startswith("dev")


@pytest.fixture(scope="module")
def phase_dir(tmp_path_factory):
    """phase_export.py (py3.8) of the init-0 episode of task 0 -- with the image-instant segmentation."""
    if not (_storage_runs() and LIBERO_PYTHON.exists()):
        pytest.skip("needs the SL3 dev run and .venv-libero")
    out = tmp_path_factory.mktemp("sl4b") / "phase"
    run = shared_storage_root() / "results/benchmark" / DEV_RUN
    proc = subprocess.run([str(LIBERO_PYTHON), str(PHASE_EXPORT), "--run", str(run), "--out", str(out), "--task-ids", "0"],
                          capture_output=True, text=True, timeout=900, env=client_env.client_environment())
    assert proc.returncode == 0, proc.stderr[-3000:]
    return out


def test_phase_export_reproduces_the_sl3_export(phase_dir):
    summary = json.loads((phase_dir / "phase_export.json").read_text())
    ep = summary["episodes"][0]
    assert ep["target"] == "akita_black_bowl_1" and ep["checks"]["call_frames"] == 21
    assert ep["checks"]["rgb_rerender_equal_call_frames"] == {"head": 21, "left_wrist": 21}
    track = lp.load_track(phase_dir / "phase_libero_spatial_t0_e0.npz", segmentation=True)
    assert track["segP/head"].shape == (21, 256, 256) and track["segP/head"].dtype == np.int16
    with pytest.raises(ValueError, match="different geometry npz"):
        lp.load_track(phase_dir / "phase_libero_spatial_t0_e0.npz", geom_npz_sha256="0" * 64)
    with pytest.raises(ValueError, match="chunk_t"):
        lp.load_track(phase_dir / "phase_libero_spatial_t0_e0.npz", chunk_t=[1, 2])
    with pytest.raises(subprocess.CalledProcessError):  # never overwrites
        subprocess.run([str(LIBERO_PYTHON), str(PHASE_EXPORT), "--run", str(shared_storage_root() / "results/benchmark" /
                        DEV_RUN), "--out", str(phase_dir), "--task-ids", "0"], capture_output=True, text=True,
                       timeout=900, check=True, env=client_env.client_environment())
    ref = SL3_GT / "gt_libero_spatial_t0_e0.npz"
    if ref.exists():  # bit-identical to the SL3 verifier export of the same episode
        with np.load(ref) as z:
            for key in ("target_xpos", "contact_grip_target", "call_frames", "chunk_t", "segP/head", "segP/left_wrist"):
                assert np.array_equal(track[key], z[key]), key


def test_oracle_dry_run_is_privileged(phase_dir, capsys, monkeypatch):
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(PROJECT_ROOT))
    base = ["--source-run", DEV_RUN, "--attention-run", AUDIT_RUN, "--no-gt", "--task-ids", "0", "--limit-calls", "1",
            "--attention-source", "oracle", "--dry-run"]
    with pytest.raises(ValueError, match="--phases"):
        pr.main(base)
    pr.main(base + ["--phases", str(phase_dir)])
    text = capsys.readouterr().out
    out = json.loads(text[text.index("{"):])
    s = out["settings"]
    assert s["step"] == "SL4b" and s["privileged_attention"] is True and s["privileged_inputs"].startswith("PRIVILEGED")
    assert s["options"]["attention"]["privileged"] is True and s["options"]["attention"]["cell"] is None
    assert "privileged" in s["condition"]
    check = out["checks"][0]
    assert check["oracle_target_pixels_call0"]["head"] > 0 and len(check["oracle_segmentation"]["sha256"]) == 64


def _cpu_ag3s(monkeypatch, *, grounding_only=False):
    import AG3S.ag3s.config as ag3s_config

    for switch in ("RECON_DEVICE", "SELF_FILTER_DEVICE", "SUPPORT_RANSAC_DEVICE"):  # CPU suite: never the GPU
        monkeypatch.setattr(ag3s_config, switch, "cpu")
    monkeypatch.setattr(pr, "_torch", lambda: None)
    ag3s, _ = pf.build_panda_ag3s(extra_overrides={"esdf": {"backend": "legacy"},
                                                   "constraint": {"target_field_policy": "relax"}})
    ag3s.strict_invariants = False
    if grounding_only:
        pr.install_grounding_only(ag3s)
    return ag3s


def _replay(ag3s, episode, out_dir, spec, *, grounding_only=False, calls=3, phase_dirs=None):
    out_dir.mkdir(parents=True)
    (out_dir / "perception").mkdir()
    frames = io.StringIO()
    opts = pr.FrameOptions(stage_probe=False, grounding_only=grounding_only)
    res = pr.run_episode(ag3s, episode, Path(episode["dev_run_dir"]), mapping=spec.mapping, opts=opts, gt_path=None,
                         samples_cfg={}, out_dir=out_dir, frames_out=frames, limit_calls=calls,
                         spheres_all=pm.build_panda_constraint_model(None), attention=spec, phase_dirs=phase_dirs)
    with np.load(out_dir / res["perception_npz"]) as z:
        return {k: z[k] for k in z.files}, [json.loads(l) for l in frames.getvalue().splitlines()]


TIMING = ("t_process_s", "t_record_s", "stage_ms", "probe_self_filter_ms_head", "probe_self_filter_ms_left_wrist",
          "gpu_max_allocated", "gpu_reserved")
GROUNDING = ("grounding_status", "has_target", "target_centroid", "target_confidence", "target_attention_score",
             "target_seed_camera", "target_supporting", "best_score", "n_clusters", "target_points",
             "target_points_offsets", "target_indices", "seed_indices", "seed_points", "seed_camera_mask",
             "seed_argmax_camera", "manipulated_centroid", "manipulated_radius", "manipulated_state", "cloud_points",
             "cloud_points_offsets", "cloud_camera_mask", "cloud_attention", "cloud_raw_attention", "cloud_support",
             "cloud_seed", "cloud_target", "robot_state9", "q7", "fingers", "sph_center", "sph_all_center",
             "attention_grid_policy", "attention_block_sum", "attention_invalid")


def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f":
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


@needs_runs
def test_cell_8_2_is_the_sl4_path_and_grounding_only_equals_the_full_grounding(tmp_path, monkeypatch):
    """Three CPU replays of the first 3 calls of t0: SL4 default, ``--attention-cell 8,2``, default + grounding-only."""
    episode = pr.index_sources(DEV_RUN, AUDIT_RUN, task_ids=[0])["episodes"][0]
    full, rows_full = _replay(_cpu_ag3s(monkeypatch), episode, tmp_path / "default", pr.AttentionSpec())
    cell, rows_cell = _replay(_cpu_ag3s(monkeypatch), episode, tmp_path / "cell82", pr.AttentionSpec(cell=(8, 2)))
    gonly, rows_g = _replay(_cpu_ag3s(monkeypatch, grounding_only=True), episode, tmp_path / "gonly", pr.AttentionSpec(),
                            grounding_only=True)
    assert len(full["call_k"]) == 3 and full["has_target"].any()  # a frame with a target is compared
    # --attention-cell 8,2 == the SL4 default: every recorded array except wall-clock timing, bit for bit
    assert set(full) == set(cell)
    differ = [k for k in full if k not in TIMING and not _equal(full[k], cell[k])]
    assert differ == [], differ
    assert str(full["attention_grid_array"]) == "cell_grid_policy" and tuple(cell["attention_cell"]) == (8, 2)
    # grounding-only: the same grounding / clouds / spheres; no field, no field queries
    differ = [k for k in GROUNDING if not _equal(full[k], gonly[k])]
    assert differ == [], differ
    assert "sph_esdf_main" in full and "sph_esdf_main" not in gonly and bool(gonly["grounding_only"])
    assert [r["grounding_status"] for r in rows_full] == [r["grounding_status"] for r in rows_g]
    assert all(pr.GROUNDING_ONLY_NOTE in r["notes"] for r in rows_g) and all(r["esdf"] == {} for r in rows_g)
    assert all(r["sphere_esdf_main_min_m"] is None for r in rows_g)
    assert all(r["sphere_esdf_main_min_m"] is not None for r in rows_full + rows_cell)


def test_oracle_attention_reaches_ag3s_at_full_resolution(phase_dir, tmp_path, monkeypatch):
    episode = pr.index_sources(DEV_RUN, AUDIT_RUN, task_ids=[0])["episodes"][0]
    track = pr.oracle_track([phase_dir], episode)
    with np.load(Path(episode["dev_run_dir"]) / episode["geom_npz"]) as g:
        geom = {k: g[k] for k in ("model_geom_bodyid", "model_body_names")}
    tg = pr.oracle_target_geoms(geom, episode["target"])
    maps = pr.oracle_attention(track, 10, tg)
    assert set(maps) == {"head", "left_wrist"} and maps["head"].shape == (256, 256) and maps["head"].max() == 1.0
    z, rows = _replay(_cpu_ag3s(monkeypatch, grounding_only=True), episode, tmp_path / "oracle",
                      pr.AttentionSpec(source="oracle"), grounding_only=True, calls=1, phase_dirs=[phase_dir])
    assert str(z["attention_source"]) == "oracle" and tuple(z["attention_cell"]) == (-1, -1)
    assert z["oracle_target_pixels"][0, 0] == int(maps["head"].sum()) and not z["attention_invalid"].any()
    assert "attention_grid_policy" not in z and json.loads(str(z["oracle_segmentation"]))["sha256"] == track["sha256"]
    assert rows[0]["fusion"]["cameras_with_attention"] == ["head", "left_wrist"]


def test_evaluate_roles_adoption_and_cause_split(monkeypatch, tmp_path):
    """Orchestration of ``evaluate`` on synthetic per-call rows: baseline / candidate / reference roles."""
    cells = {"base": ((8, 2), "cell_grid_policy", "vla"), "c1": ((3, 1), "pooled_f16", "vla"),
             "c2": ((5, 0), "pooled_f16", "vla"), "f16": ((8, 2), "pooled_f16", "vla"), "oracle": (None, None, "oracle")}
    infos = {}
    for name, (cell, grid, source) in cells.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_manifest.json").write_text("{}")
        label = ac.cell_label(cell) + ("-f16" if name == "f16" else "")
        infos[name] = ac.RunInfo(tmp_path / name, {}, label, cell, source, "unmirror", grid, True)
    # one evaluation call per task (10 tasks): c1 +2 of 10 (20 %p) and >= L8H2 on every task; c2 = L8H2
    correct = {"base": [1, 0, 0, 0, 1, 1, 1, 1, 1, 1], "c1": [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
               "c2": [1, 0, 0, 0, 1, 1, 1, 1, 1, 1], "f16": [1, 0, 0, 0, 1, 1, 1, 1, 1, 1], "oracle": [1] * 10}

    def rows_of(info, phase_dirs, cache, with_cause=False):
        name = info.run_dir.name
        out = []
        for j, ok in enumerate(correct[name]):
            r = {"target_correct": ok, "has_target": True, "n_target_points": 4, "precision_hits": 3 * ok,
                 "recall": 0.5, "centroid_err_m": 0.01, "majority": "bowl_1" if ok else "plate",
                 "category": "correct" if ok else "wrong_instance", "label_counts": {}, "run": info.label,
                 "stem": f"libero_spatial_t{j}_e0", "task": j, "init": 0, "call_k": 0, "obs_index": 10,
                 "phase": "approach", "attention_invalid": False}
            if with_cause and not ok:
                r["cause"] = {"cause": "attention" if j % 2 else "grounding", "argmax_camera": "head",
                              "head_argmax_on_target": False, "wrist_argmax_on_target": j == 3}
            out.append(r)
        return out

    monkeypatch.setattr(ac, "read_run", lambda v: infos[v])
    monkeypatch.setattr(ac, "evaluate_run", rows_of)
    monkeypatch.setattr(ac, "_episode_index", lambda info: [{"stem": f"libero_spatial_t{j}_e0"} for j in range(10)])
    selection = {"candidate_cells": [[8, 2], [3, 1], [5, 0]],
                 "episodes": [{"stem": f"libero_spatial_t{j}_e0"} for j in range(10)]}
    out, rows = ac.evaluate(["base", "c1", "c2", "f16", "oracle"], [], selection)
    roles = {k: v["role"] for k, v in out["runs"].items()}
    assert roles == {"L8H2": "baseline", "L3H1": "candidate", "L5H0": "candidate", "L8H2-f16": "reference",
                     "oracle": "reference (privileged)"}
    assert out["oracle_reference"]["oracle"]["primary"]["correct"] == 10
    a = out["adoption"]
    assert a["baseline_correct"] == 7 and a["satisfying"] == ["L3H1"] and a["selected_by_rule"] == "L3H1"
    assert {r["label"]: r["tasks_ge_baseline"] for r in a["candidates"]} == {"L3H1": 10, "L5H0": 10}
    cs = out["cause_split_L8H2"]
    assert cs["misselections"] == 3 and cs["attention_cause"] == 2 and cs["grounding_cause"] == 1
    assert cs["alt_either_camera_on_target"] == 1
    assert out["preregistration"]["rows_sha256"] == ac.preregistration_record()["rows_sha256"]
    assert not out["complete_dev_set_50"] and out["evaluation_calls"] == 10
    with pytest.raises(ValueError, match="no replay run for selected cells"):
        ac.evaluate(["base", "c1", "oracle"], [], selection)
    with pytest.raises(ValueError, match="exactly one L8H2"):
        ac.evaluate(["c1", "c2", "f16"], [], selection)
    with pytest.raises(ValueError, match="episodes differ"):
        ac.evaluate(["base", "c1", "c2"], [], dict(selection, episodes=selection["episodes"][:3]))
