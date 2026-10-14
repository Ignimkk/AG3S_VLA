"""T55 re-grounded perception (`t55_perception`) — the serving AG3S re-run on re-sim observations.

Pinned (T55.task.md 추가 사전 고정 2, user decision 2026-10-13):
  - the server is built from the T53 rollout servers' own command line; servers whose SafePolicy arguments differ
    are refused (port / run dir / record path are the only ignored differences);
  - the policy stub keeps the request's result across `SafePolicy.reset` (a reset request resets before infer);
  - the perception-only refiner calls ``scene_fn`` exactly once per request, records a failure instead of raising,
    and returns the reference chunk;
  - the execution feedback is the client's (`ExecutionLog.close` -> `wire.make_exec_feedback`): none on the first
    request, float32 on the wire, the record's executed_chunk / hold fields when its executed flags match;
  - serving parity counts (state equality, attach / detach events, reach, manipulated id, centroid);
  - diagnostic attention sources (``server`` / ``t53``) are test E0 / orig only;
  - real (skipped without storage): a 3-request prefix of T53 ``ep1804`` re-grounded with the recorded serving
    attention reproduces the recorded latch state, reach and target centroid.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t55_perception as tp
from AG3S.ag3s.experiments.studies import t55_resim as rs

T53_RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/"
                       "20261013T100210+0900_4a540f1a18d7_cb037c425d2f")
LINE = ("/x/python -m AG3S.trajopt.serve_safe --config c --port {port} --esdf-backend curobo --voxel 0.020 "
        "--run-dir /r{port} --record-constraints /r{port}/sc {extra}\n")


def _proc(root: pathlib.Path, port: int, extra: str = "") -> None:
    d = root / "verify" / "T53" / "runs" / "C1a" / f"server_{port}_x"
    d.mkdir(parents=True, exist_ok=True)
    (d / "server_proc.txt").write_text("cwd=/x\nA=1\n" + LINE.format(port=port, extra=extra))


def test_serving_argv_ignores_ports_refuses_other_args(tmp_path):
    _proc(tmp_path, 8290)
    _proc(tmp_path, 8291)
    argv, prov = tp.serving_argv(tmp_path)
    assert argv[:2] == ["--config", "c"] and "--port" in argv and len(prov["server_procs"]) == 2
    assert tp._strip_argv(argv) == ["--config", "c", "--esdf-backend", "curobo", "--voxel", "0.020"]  # noqa: SLF001
    _proc(tmp_path, 8292, extra="--held-slip-detach")
    with pytest.raises(SystemExit, match="different SafePolicy arguments"):
        tp.serving_argv(tmp_path)
    with pytest.raises(SystemExit, match="no server"):
        tp.serving_argv(tmp_path / "nothing")


def test_stub_policy_survives_reset():
    s = tp.StubPolicy()
    with pytest.raises(RuntimeError):
        s.infer({})
    s.set_result({"actions": 1})
    s.reset()                                   # SafePolicy.reset() runs before this request's infer
    assert s.infer({}) == {"actions": 1}
    with pytest.raises(RuntimeError):
        s.infer({})                             # one result per request


def test_perception_only_refiner():
    calls = []
    r = tp.PerceptionOnlyRefiner(lambda ctx: calls.append(ctx))
    chunk = np.ones((4, 16), np.float32)
    out = r.refine(chunk, {"t_step": 3})
    assert calls == [{"t_step": 3}] and out.dtype == np.float64 and np.array_equal(out, chunk)
    assert r.last_result is None and r.last_failure is None and r.optimizer is None

    def boom(ctx):
        raise ValueError("no camera observations in the request")

    r2 = tp.PerceptionOnlyRefiner(boom)
    assert np.array_equal(r2.refine(chunk), chunk)
    assert r2.last_failure == "ValueError: no camera observations in the request" and r2.failures == 1
    r2.reset()
    assert r2.last_failure is None


def _steps(n=8, grip=0.3):
    return {"t": list(range(8, 8 + n)), "executed": [True] * (n - 2) + [False] * 2,
            "applied_gripper": [[1.0, grip]] * n, "applied_arm": [[0.1] * 14] * n}


def test_wire_feedback():
    fb0 = tp.wire_feedback(0, {}, (1.0, 1.0), horizon=8, record_fb=None, executed_rows=None)
    assert fb0["available"] is False and "no chunk has run yet" in fb0["reason"]
    rows = np.ones((8, 16))
    rows[:, 15] = 0.25
    fb = tp.wire_feedback(2, _steps(), (1.0, 1 / 3), horizon=8, record_fb=None, executed_rows=rows)
    assert fb["available"] and fb["seq"] == 2 and fb["t_step_start"] == 8 and fb["n_exec"] == 6 and fb["n_hold"] == 2
    assert fb["executed_chunk"] == "refined" and fb["hold_kind"] is None
    assert np.asarray(fb["measured_gripper"]).dtype == np.float32
    assert np.allclose(np.asarray(fb["planned_gripper"])[:, 1], 0.25)
    assert np.asarray(fb["applied_arm"]).shape == (8, 14)
    rec = {"available": True, "executed": [True] * 6 + [False] * 2, "executed_chunk": "reference",
           "hold_kind": "unsafe", "hold_reason": "r", "planned_gripper": [[1.0, 0.5]] * 8, "ipc": "ok"}
    fb2 = tp.wire_feedback(2, _steps(), (1.0, 0.4), horizon=8, record_fb=rec, executed_rows=rows)
    assert fb2["executed_chunk"] == "reference" and fb2["hold_kind"] == "unsafe"
    assert np.allclose(np.asarray(fb2["planned_gripper"])[:, 1], 0.5)
    other = dict(rec, executed=[True] * 8)          # the record describes another execution: not used
    fb3 = tp.wire_feedback(2, _steps(), (1.0, 0.4), horizon=8, record_fb=other, executed_rows=rows)
    assert fb3["executed_chunk"] == "refined"
    # the latch reads it like the server does
    from AG3S.trajopt.grasp_latch import grasp_signal_from_feedback

    sig = grasp_signal_from_feedback(fb, 1, 0.85)
    assert sig.closing_steps == 6 and sig.opening == pytest.approx(1 / 3, abs=1e-6)


def test_rows_and_serving_parity():
    rows = []
    states = ["searching", "latched", "closing", "held", "held", "placed"]
    for i, ph in enumerate(states):
        rows.append({"phase": ph, "phase_before": "", "attach": i == 3, "detach": i == 5, "rejected": False,
                     "revoked": False, "slipped": False, "reach_mm": float(i * 10) if i else float("nan"),
                     "manipulated_id": 1 if i else -1, "manipulated_state": "visible", "has_target": i > 0,
                     "target_centroid": np.full(3, float(i)) if i else np.full(3, np.nan), "manipulated_now": 1,
                     "scene_fn_failure": "", "grasp_json": "{}"})
    arr = tp.rows_to_arrays(rows)
    lat = tp.latch_arrays(arr)
    assert set(lat) == {f"latch_reground__{k}" for k in ("phase", "phase_before", "attach", "detach", "revoked",
                                                          "slipped")}
    assert arr["phase"].dtype.kind == "U" and arr["target_centroid"].shape == (6, 3)
    grasp = [{"state": s, "attach": i == 3, "detach": i == 4,
              "evidence": {"reach_mm": (i * 10 + 1.5) if i else None, "manipulated_id": 1 if i else None}}
             for i, s in enumerate(["searching", "latched", "closing", "held", "placed", "placed"])]
    cen = np.asarray([np.full(3, np.nan)] + [np.full(3, float(i)) + 0.001 for i in range(1, 6)])
    p = tp.serving_parity(arr, grasp, cen)
    assert p["state_equal"] == 5 and p["first_state_mismatch"] == 4
    assert p["attach"] == {"offline": [3], "recorded": [3]} and p["detach"] == {"offline": [5], "recorded": [4]}
    assert p["reach_mm_abs_diff"]["n"] == 5 and p["reach_mm_abs_diff"]["max"] == pytest.approx(1.5)
    assert p["manipulated_id_equal"] == 6 and p["state_confusion"]["placed->held"] == 1
    assert p["target_centroid_dist_m"]["max"] == pytest.approx(np.sqrt(3) * 0.001)


def test_plan_jobs_diagnostic_sources(tmp_path):
    rd = tmp_path / "resim"
    rd.mkdir()
    for tag in ("E0", "empty_L0O0_park"):
        np.savez(rd / f"k1__{tag}.npz", prompt=np.asarray("Put the apple in the basket."), target=np.asarray("apple"),
                 episode=np.asarray(7))
    meta = {7: {"non_target_fruits": ["banana", "pear"]}}
    jobs = tp.plan_jobs(rd, set_name="test", source="t55", attn_dir=tmp_path / "att", meta_ds=meta)
    assert [(j["tag"], [p["prompt_tag"] for p in j["prompts"]]) for j in jobs] == [
        ("E0", ["orig", "instr_banana", "instr_pear"]), ("empty_L0O0_park", ["orig"])]
    assert jobs[0]["prompts"][1]["attention"].endswith("k1__E0__instr_banana.npz")
    jd = tp.plan_jobs(rd, set_name="test", source="server", attn_dir=None, meta_ds=meta)
    assert [(j["tag"], [p["prompt_tag"] for p in j["prompts"]]) for j in jd] == [("E0", ["orig"])]
    with pytest.raises(SystemExit, match="diagnostic"):
        tp.plan_jobs(rd, set_name="dev", source="t53", attn_dir=None, meta_ds=meta)


@pytest.mark.skipif(not (T53_RUN / "verify" / "T53" / "runs" / "C1a" / "R").is_dir(), reason="T53 records not present")
def test_real_t53_prefix_regrounded_with_serving_attention(tmp_path):
    servers = sorted((T53_RUN / "verify/T53/runs/C1a").glob("server_829*_*"))
    src = rs.t53_source("ep1804", t53_run_dir=T53_RUN, episodes_root=T53_RUN / "verify/T53/runs/C1a/R",
                        server_roots=servers)
    n = 3
    H = 8
    short = rs.Source(**{**{f.name: getattr(src, f.name) for f in rs.dataclasses.fields(src)},
                         "step_files": src.step_files[:n], "t_steps": src.t_steps[:n], "actions": src.actions[:n * H],
                         "executed": src.executed[:n * H], "ref_measured": src.ref_measured[:n * H],
                         "ref_applied_gripper": src.ref_applied_gripper[:n * H], "ref_feedback": src.ref_feedback[:n],
                         "ref_grasp": src.ref_grasp[:n], "ref_chunk_files": src.ref_chunk_files[:n]})
    cl = rs.load_client(rs.default_client(T53_RUN))
    seen = []
    res = rs.simulate(cl, short, keep_images=True, observer=lambda k, ctx: seen.append((k, len(ctx["chunk_steps"]["t"]))))
    assert seen == [(0, 0), (1, 8), (2, 8)] and res["arrays"]["applied_arm"].shape == (n * H, 14)
    resim = tmp_path / "resim" / "test"
    resim.mkdir(parents=True)
    lat = rs.run_latches(short, res)
    rs.save_npz_atomic(resim / "ep1804__E0.npz", **rs.variant_payload(short, res, lat, tag="E0", window_start=None,
                                                                       meta={"baseline": "record"}))
    out = tmp_path / "perc"
    out.mkdir()
    serving, _ = tp.serving_argv(T53_RUN)
    job = tp.plan_jobs(resim, set_name="test", source="server", attn_dir=None)[0]
    orig_source = tp._source_for  # noqa: SLF001
    tp._source_for = lambda key, set_name, ns: short  # noqa: SLF001
    try:
        r = tp.run_job(job, source="server", backend="legacy", out_dir=out, client_path=rs.default_client(T53_RUN),
                       serving=serving, args_ns={"t53_run_dir": str(T53_RUN)}, force=True, run_meta={})
    finally:
        tp._source_for = orig_source  # noqa: SLF001
    assert "error" not in r
    meta = json.loads((out / "ep1804__E0__orig.json").read_text())
    p = meta["serving_parity"]
    assert meta["pairing"] == {"image_mismatch_frames": [], "qpos_mismatch_frames": []}
    assert p["state_equal"] == n and p["manipulated_id_equal"] == n and meta["scene_fn_failures"] == 0
    assert p["reach_mm_abs_diff"].get("max", 0.0) == 0.0
    with np.load(out / "ep1804__E0__orig.npz", allow_pickle=False) as z:
        assert z["latch_reground__phase"].tolist() == [str(g.get("state")) for g in short.ref_grasp]
