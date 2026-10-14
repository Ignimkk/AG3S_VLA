"""T53 registry and file pairing (`t53_records`) on synthetic layouts — no MuJoCo, no GPU, no storage.

Pinned:
  - the 16 final-test episodes: layout slots 0..15 once each, four fruits, both arms, none in the VLA
    fine-tune range, scene id ``scene<N>``, role ``final_test``; key resolution (``1840`` == ``ep1840``,
    unknown / duplicate refused); the dataset-metadata check (and, when storage is present, the real file);
  - roots: unconfigured refused, ``--run-dir`` outside ``results/development`` refused, every output
    path under ``<run>/verify/T53/``;
  - locating: episode dir name variants, ambiguity refused (two dirs, two step runs, two trajectories),
    ``failed/`` never searched;
  - pairing on a complete synthetic rollout (step record + frames.jsonl + manifest + trajectory + server
    chunks + attention npz) finds nothing, and each single corruption is reported;
  - chunk segments split at ``t_step == 1`` and pairing a server segment by planning actions.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t51_records as rc51
from AG3S.ag3s.experiments.studies import t53_records as rc

KEY = "ep1840"
EP = 1840
PROMPT = rc.EXPECTED[EP][3]


@pytest.fixture(autouse=True)
def _isolated(tmp_path):
    rc.configure(t53_dir=tmp_path / "run" / "verify" / "T53")
    yield
    rc.configure()


# ------------------------------------------------------------------------------ registry


def test_registry_is_the_task_table():
    assert rc.EPISODES == (1840, 1809, 1842, 1843, 1860, 1845, 1862, 1815, 1848, 1849, 1866, 1851,
                           1804, 1853, 1822, 1839)
    assert rc.KEYS == tuple(f"ep{e}" for e in rc.EPISODES)
    eps = list(rc.REGISTRY.values())
    assert sorted(e.layout_slot for e in eps) == list(range(16))
    assert [e.layout_slot for e in eps] == list(range(16))  # task order = slot order
    assert {e.target for e in eps} == {"apple", "banana", "orange", "pear"}
    assert {e.used_arm for e in eps} == {"left", "right"}
    assert all(e.role == "final_test" for e in eps)
    assert rc.scene_of("ep1809") == "scene1809" and rc.target_of("ep1842") == "pear"
    prov = rc.provenance(rc.KEYS)
    assert prov["n_scenes"] == 16 and prov["vla_training_episodes"] == [] and prov["role"] == "final_test"
    for e in eps:  # every prompt names exactly its target
        assert rc.prompt_fruit(e.prompt) == e.target
    # no overlap with any record used for selection / diagnosis / calibration
    assert not set(rc.EPISODES) & set(rc51.EPISODE_OF.values())


def test_resolve_keys():
    assert rc.resolve_keys(None) == list(rc.KEYS)
    assert rc.resolve_keys(["1840", "ep1809"]) == ["ep1840", "ep1809"]
    with pytest.raises(SystemExit):
        rc.resolve_keys(["ep1807"])
    with pytest.raises(SystemExit):
        rc.resolve_keys(["ep1840", "1840"])
    with pytest.raises(KeyError):
        rc.episode_of("ep0")


def _dataset_rows():
    return {e: {"episode_index": e, "target_fruit": t, "used_arm": a, "layout_index": s, "prompt": p,
                "split": "test"} for e, (t, a, s, p) in rc.EXPECTED.items()}


def test_dataset_problems_synthetic():
    rows = _dataset_rows()
    assert rc.dataset_problems(rows) == []
    rows[1842] = {**rows[1842], "target_fruit": "apple"}
    rows.pop(1839)
    probs = rc.dataset_problems(rows)
    assert len(probs) == 2 and any("1842" in p for p in probs) and any("1839" in p for p in probs)
    rows = _dataset_rows()
    rows[1840] = {**rows[1840], "split": "train"}
    assert rc.dataset_problems(rows)


def test_dataset_problems_real_metadata():
    path = rc.storage_root() / rc.DATASET_REL / "meta" / "randomized_episodes.jsonl"
    if not path.is_file():
        pytest.skip("shared storage dataset metadata not present")
    assert rc.dataset_problems(rc.load_dataset_meta()) == []


# ------------------------------------------------------------------------------ roots


def test_unconfigured_and_storage_check(tmp_path):
    rc.configure()
    with pytest.raises(SystemExit):
        rc.t53_dir()
    with pytest.raises(SystemExit):  # not a run under results/development
        rc.configure(run_dir=tmp_path / "somewhere")
    rc.configure(run_dir=tmp_path / "r", check_storage=False)
    assert rc.t53_dir() == tmp_path / "r" / "verify" / "T53"
    with pytest.raises(ValueError):
        rc.configure(run_dir=tmp_path, t53_dir=tmp_path)


def test_run_dir_under_development_results(tmp_path, monkeypatch):
    import common.run_output as ro

    dev = tmp_path / "storage" / "results" / "development"
    (dev / "RUN").mkdir(parents=True)
    monkeypatch.setattr(ro, "development_results_root", lambda *a, **k: dev)
    rc.configure(run_dir=dev / "RUN")
    assert rc.t53_dir() == (dev / "RUN" / "verify" / "T53").resolve()
    with pytest.raises(SystemExit):  # nested deeper than one level is not a run
        rc.configure(run_dir=dev / "RUN" / "sub")


def test_output_paths_under_t53(tmp_path):
    t = rc.t53_dir()
    assert rc.npz_path(KEY) == t / "attention" / "attention_16d_ep1840.npz"
    assert rc.noun_path(KEY) == t / "noun" / "orig" / "ep1840.npz"
    assert rc.feat_path(KEY) == t / "feat" / "ep1840.npz"
    assert rc.gt_cache_path(KEY) == t / "gt" / "ep1840.npz"
    assert rc.grasp_path(KEY) == t / "gt" / "ep1840.grasp.npz"
    assert rc.out_dir("feat").is_dir()
    with pytest.raises(ValueError):
        rc.out_dir("figures")
    with pytest.raises(KeyError):
        rc.npz_path("ep1807")
    with pytest.raises(SystemExit):
        rc.load_gt(KEY)


# ------------------------------------------------------------------------------ synthetic rollout


def make_rollout(root: pathlib.Path, *, key=KEY, dirname=None, n=4, prompt=None, episode=None,
                 seed=7, frames=True, manifest_overrides=None, traj=True, chunks=True,
                 traj_starts=None, n_chunks=None, chunk_dir="server_constraints/run_0000",
                 n_steps_meta=True) -> pathlib.Path:
    e = rc.episode_of(key)
    prompt = e.prompt if prompt is None else prompt
    episode = e.episode if episode is None else episode
    d = root / (dirname or f"C1a_{key}_s{seed}")
    rec = d / "ag3s" / "run_0000"
    rec.mkdir(parents=True)
    argv = ["pi05_infer.py", "--episode-index", str(episode), "--policy-seed", str(seed)]
    meta = {"prompt": prompt, "argv": argv}
    if n_steps_meta:
        meta["n_steps"] = n
    (rec / "meta.json").write_text(json.dumps(meta))
    acts = [np.full((50, 16), i, np.float32) for i in range(n)]
    for i in range(n):
        np.savez(rec / f"step_{i:05d}.npz", t_step=np.asarray(i * 8), actions=acts[i])
    if frames:
        rows = []
        for i in range(n):
            rows.append({"kind": "observation", "t_step": i * 8})
            rows.append({"kind": "planning", "t_step": i * 8, "seq": i + 1, "actions": acts[i].tolist()})
            rows += [{"kind": "control", "t_step": i * 8 + j} for j in range(8)]
        (d / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        man = {"policy_seed": seed,
               "scene": {"episode_index": episode, "episode_target_fruit": e.target,
                         "episode_used_arm": e.used_arm},
               "policy": {"prompt": prompt}}
        for k, v in (manifest_overrides or {}).items():
            man[k] = {**man.get(k, {}), **v} if isinstance(v, dict) else v
        (d / "manifest.json").write_text(json.dumps(man))
    if traj:
        starts = [i * 8 for i in range(n)] if traj_starts is None else traj_starts
        np.savez_compressed(d / "traj.npz", executed_actions=np.zeros((8 * n, 16)),
                            chunk_start_steps=np.asarray(starts, np.int64), prompt=np.asarray(prompt),
                            policy_seed=np.asarray(seed, np.int64))
    if chunks:
        cd = d / chunk_dir
        cd.mkdir(parents=True)
        for i in range(n if n_chunks is None else n_chunks):
            np.savez_compressed(cd / f"chunk_{i:05d}.npz", t_step=np.asarray(i + 1),
                                refined_chunk=acts[i % n])
    return d


def write_attention(key: str, record: pathlib.Path, *, n: int, prompt: str, t_step=None):
    p = rc.npz_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p, attention=np.zeros((n, 3, 3, 18, 8, 3, 16, 16), np.float16),
        t_step=np.asarray(t_step if t_step is not None else [i * 8 for i in range(n)], np.int64),
        denoise_steps=np.asarray([0, 4, 9]), aggregations=np.asarray(["mean", "first", "last"]),
        cameras=np.asarray(list(hs.POLICY_CAMERAS)), prompt=np.asarray(prompt),
        noise_seeds=np.asarray([0]), records=np.asarray(str(record.resolve())),
        checkpoint=np.asarray("/s/checkpoints/rby1/" + rc51.CHECKPOINT_TAIL))
    return p


def test_complete_rollout_pairs():
    d = make_rollout(rc.t53_dir())
    files = rc.locate(KEY)
    assert files.episode_dir == d and files.record == d / "ag3s" / "run_0000"
    assert files.frames == d / "frames.jsonl" and files.manifest == d / "manifest.json"
    assert files.trajectory == d / "traj.npz"
    write_attention(KEY, files.record, n=4, prompt=PROMPT)
    info, probs = rc.pairing_report(KEY, require_chunks=True)
    assert probs == []
    assert info["n_frames"] == 4 and info["policy_seed"] == 7 and info["scene_id"] == "scene1840"
    assert info["frames"]["n_planning"] == 4 and info["frames"]["n_control"] == 32
    assert info["frames"]["actions_vs_record"] == {"compared": 4, "max_abs": 0.0, "n_over_1e-4": 0}
    assert info["chunks"]["n"] == 4 and info["attention"]["n_frames"] == 4
    assert rc.check_pairing(KEY)["key"] == KEY


@pytest.mark.parametrize("kw,needle", [
    ({"prompt": "Put the apple in the basket."}, "dataset prompt"),
    ({"episode": 1841}, "--episode-index"),
    ({"manifest_overrides": {"scene": {"episode_used_arm": "left"}}}, "episode_used_arm"),
    ({"manifest_overrides": {"policy": {"prompt": "x"}}}, "policy.prompt"),
    ({"traj_starts": [0, 8, 16]}, "chunk_start_steps"),
    ({"n_chunks": 3}, "server chunks"),
    ({"frames": False}, "no frames.jsonl"),
    ({"traj": False}, "no trajectory"),
])
def test_single_corruption_is_reported(kw, needle):
    make_rollout(rc.t53_dir(), **kw)
    info, probs = rc.pairing_report(KEY)
    assert probs and any(needle in p for p in probs), probs
    with pytest.raises(SystemExit):
        rc.check_pairing(KEY)


def test_frames_planning_count_mismatch():
    d = make_rollout(rc.t53_dir())
    rows = rc.read_frames(d / "frames.jsonl")
    rows = [r for r in rows if not (r["kind"] == "planning" and r["t_step"] == 24)]
    (d / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    _, probs = rc.pairing_report(KEY)
    assert any("planning t_step" in p for p in probs)


def test_meta_n_steps_mismatch_and_unclosed_writer():
    d = make_rollout(rc.t53_dir())
    (d / "ag3s" / "run_0000" / "step_00003.npz").unlink()
    _, probs = rc.pairing_report(KEY, require_frames=False, require_trajectory=False)
    assert any("n_steps" in p for p in probs)


def test_attention_mismatch_reported():
    d = make_rollout(rc.t53_dir())
    write_attention(KEY, d / "ag3s" / "run_0000", n=4, prompt=PROMPT, t_step=[0, 8, 16, 32])
    _, probs = rc.pairing_report(KEY)
    assert any("npz frames" in p for p in probs)
    write_attention(KEY, d, n=4, prompt=PROMPT)  # records field names another directory
    _, probs = rc.pairing_report(KEY)
    assert any("records=" in p for p in probs)


def test_episode_dir_variants_and_ambiguity():
    root = rc.t53_dir()
    make_rollout(root, dirname="ep1840")
    assert rc.episode_dir(KEY) == root / "ep1840"
    (root / "failed" / "C1a_ep1840_s1").mkdir(parents=True)  # never searched
    assert rc.episode_dir(KEY) == root / "ep1840"
    (root / "1840").mkdir()
    with pytest.raises(SystemExit, match="candidate episode directories"):
        rc.episode_dir(KEY)
    with pytest.raises(SystemExit, match="no episode directory"):
        rc.episode_dir("ep1809")


def test_two_step_runs_or_trajectories_refused():
    d = make_rollout(rc.t53_dir())
    (d / "ag3s" / "run_0001").mkdir()
    (d / "ag3s" / "run_0001" / "meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="step records"):
        rc.record_dir(KEY)
    d2 = make_rollout(rc.t53_dir(), key="ep1809")
    np.savez(d2 / "other.npz", executed_actions=np.zeros(1), chunk_start_steps=np.zeros(1))
    with pytest.raises(SystemExit, match="trajectory npz"):
        rc.trajectory_path("ep1809")
    np.savez(d2 / "unrelated.npz", x=np.zeros(1))  # not a trajectory: ignored
    (d2 / "other.npz").unlink()
    assert rc.trajectory_path("ep1809") == d2 / "traj.npz"


def test_record_directly_in_ag3s_dir(tmp_path):
    d = rc.t53_dir() / "ep1840" / "ag3s"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"prompt": PROMPT, "argv": ["--episode-index", "1840"]}))
    assert rc.record_dir(KEY) == d


# ------------------------------------------------------------------------------ chunks


def test_segments_split_at_restart(tmp_path):
    rec = tmp_path / "srv" / "server_constraints" / "run_0000"
    rec.mkdir(parents=True)
    for i, ts in enumerate([1, 2, 3, 1, 2, 1]):
        np.savez(rec / f"chunk_{i:05d}.npz", t_step=np.asarray(ts), refined_chunk=np.full((2, 2), i))
    segs = rc.segments([rec])
    assert [len(s) for s in segs] == [3, 2, 1]
    plan = [np.full((2, 2), 3.0), np.full((2, 2), 4.0)]
    assert rc.pair_segment(segs, plan) == 1
    assert rc.pair_segment(segs, plan, used=[1]) is None
    assert rc.pair_segment(segs, [np.full((2, 2), 3.0), np.full((2, 2), 4.1)]) is None


def test_chunks_from_server_roots_paired_by_actions(tmp_path):
    root = rc.t53_dir()
    make_rollout(root, chunks=False)
    srv = tmp_path / "servers" / "server_8290_x" / "server_constraints" / "run_0000"
    srv.mkdir(parents=True)
    k = 0
    for seg in ([9, 9], [0, 1, 2, 3]):  # an unrelated 2-chunk rollout first, then ours
        for j, v in enumerate(seg):
            np.savez(srv / f"chunk_{k:05d}.npz", t_step=np.asarray(j + 1),
                     refined_chunk=np.full((50, 16), v, np.float32))
            k += 1
    info, probs = rc.pairing_report(KEY, require_chunks=True, server_roots=[tmp_path / "servers"])
    assert probs == [] and info["chunks"]["n"] == 4 and "segment 1 of 2" in info["chunks"]["source"]
    _, probs = rc.pairing_report(KEY, require_chunks=True)
    assert any("no server constraint chunks" in p for p in probs)


def test_chunks_in_two_dirs_refused():
    d = make_rollout(rc.t53_dir())
    other = d / "server_constraints" / "run_0001"
    other.mkdir(parents=True)
    np.savez(other / "chunk_00000.npz", t_step=np.asarray(1))
    with pytest.raises(SystemExit, match="chunk files in 2 directories"):
        rc.locate_chunks(KEY)
