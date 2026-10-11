"""T43 Q on the records — off is the pre-T43 code bit for bit; on, the new rows see what the waypoints
missed in the two contact chunks and the re-solved plan clears them (fixed-input replay, CPU).

| chunk | recorded | waypoint rows on the recorded plan | path rows on the recorded plan |
|---|---|---|---|
| T40 G C1807 seq 28 (t=216, margin 0) | hand–bar contact t=218 | clear (+0.4 mm) | servo −10.6 · sweep −9.7 mm |
| T41 a C1807 seq 35 (t=272, margin 10 mm) | hand–bar contact t=276 | clear (+1.3 mm) | servo −16.5 · sweep −8.5 mm |

The executed outcome (MuJoCo, true mesh) is in `tests/sim/test_t43q_servo_sim.py`.
"""

from __future__ import annotations
from scripts.setup.reference_layout import expose_legacy_benchmark
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from tests.trajopt import _t43q_replay as RP

records = pytest.mark.skipif(not RP.available(), reason="T40 G / T41 a E3b 기록이 없다")
#: benchmark before T43 Q. The bit-identity test reverts **only** the files T43 Q changed to this
#: commit (and drops its two new modules), so edits other lanes make elsewhere do not enter the
#: comparison. If one of these files is later changed for another reason, re-pin.
PRE_T43 = "d309550"
CHANGED = ("config.py", "linearize.py", "refiner.py", "safe_policy.py", "serve_safe.py", "sqp.py")
NEW = ("servo.py", "sweep.py")
SERVO = {"limits": {"servo_model": True}}
SWEEP = {"reduction": {"sweep_check": True}}
BOTH = {**SERVO, **SWEEP}


def _has_commit(rev):
    return subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "cat-file", "-e", rev],
                          capture_output=True).returncode == 0


def _reverted_tree(root: pathlib.Path) -> pathlib.Path:
    """`root/benchmark` = the working tree with T43 Q's files at `PRE_T43` (symlinks elsewhere)."""
    src = pathlib.Path(_ag3s_workspace_path('legacy/benchmark'))
    pkg = root / "benchmark"
    pkg.mkdir(parents=True)
    for entry in src.iterdir():
        if entry.name != "trajopt":
            (pkg / entry.name).symlink_to(entry)
    traj = pkg / "trajopt"
    traj.mkdir()
    for entry in (src / "trajopt").iterdir():
        if entry.name in NEW or entry.name in CHANGED or entry.name == "__pycache__":
            continue
        (traj / entry.name).symlink_to(entry)
    for name in CHANGED:
        text = subprocess.run(["git", "-C", str(src), "show", f"{PRE_T43}:trajopt/{name}"],
                              capture_output=True, check=True, text=True).stdout
        (traj / name).write_text(text)
    expose_legacy_benchmark(root)
    return root


@records
@pytest.mark.skipif(not _has_commit(PRE_T43), reason=f"benchmark {PRE_T43} 가 없다")
def test_off_replay_of_recorded_chunks_is_bit_identical_to_the_pre_t43_code(tmp_path):
    root = _reverted_tree(tmp_path / "head")
    out = tmp_path / "head.npz"
    env = {**os.environ, "AG3S_ASSET_ROOT": _ag3s_workspace_path("legacy/pi05_TO_hybrid"), "AG3S_CLIENT_SCRIPT": _ag3s_workspace_path("integration/rby1/bringup/pi05_infer.py"), "PYTHONPATH": f"{root}:{_ag3s_workspace_path('')}"}
    proc = subprocess.run([sys.executable, str(pathlib.Path(RP.__file__)), str(out)], env=env,
                          cwd=_ag3s_workspace_path(''), capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-3000:]
    head = np.load(out)
    meta = json.loads(str(head["meta"]))
    assert meta.pop("benchmark_file").startswith(str(root))       # the reverted code really ran
    for name, seq in RP.IDENTITY:
        r = RP.replay(name, seq)
        assert r["chunk"].tobytes() == head[f"{name}_{seq}_chunk"].tobytes(), (name, seq)
        m = meta[f"{name}_{seq}"]
        assert (m["status"], m["clearance_m"], m["returned"]) == (
            r["status"], r["clearance_m"], r["metrics"].get("returned")), (name, seq)
        assert m["metric_keys"] == sorted(r["metrics"]), (name, seq)   # no new keys when off


#: (run, seq, obstacle margin of the replayed scene, tolerance on the recorded waypoint clearance)
CASES = [("T40C1807", 28, 0.0, 1e-3), ("T41aC1807", 35, None, 1e-3)]


@records
@pytest.mark.parametrize("name, seq, margin, waypoint_m", CASES[:1] + [
    ("T41aC1807", 35, 0.010, 1e-3)])
def test_the_new_rows_see_the_contact_the_waypoint_rows_missed(name, seq, margin, waypoint_m):
    recorded = RP.npz(name, seq)["refined_chunk"]
    for overrides in (SERVO, SWEEP, BOTH):
        c = RP.check_plan(name, seq, recorded, overrides=overrides, obstacle_margin=margin)
        assert c["waypoint_clearance_m"] > 0.0                                    # waypoints: clear
        rec = (RP.summary(name, seq).get("to") or {}).get("metrics") or {}
        if margin == RP.RUNS[name][3]:                    # the replayed scene is the recorded one
            assert c["waypoint_clearance_m"] == pytest.approx(rec["clearance_m"], abs=waypoint_m)
        assert c["path_clearance_m"] < -0.005, overrides                         # path: > 5 mm in
        worst = c["path"]["worst"]
        assert worst["link"] in ("ee_finger_l1", "ee_finger_l2", "ee_left")      # the hand
        assert 0 <= worst["step"] < 8
    servo = RP.check_plan(name, seq, recorded, overrides=SERVO, obstacle_margin=margin)
    assert servo["servo_state"].source.startswith("propagated")               # velocity from feedback


@records
@pytest.mark.parametrize("name, seq, margin, _w", CASES + [("T40C1807", 28, 0.010, None)])
def test_the_resolved_plan_clears_the_path_rows(name, seq, margin, _w):
    for overrides in (SERVO, BOTH):
        r = RP.replay(name, seq, overrides=overrides, obstacle_margin=margin)
        pc = r["metrics"]["path_check"]
        assert r["safe"], (overrides, r["status"], r["result"].notes)
        assert pc["clearance_m"] >= -1e-4 and r["clearance_m"] >= -1e-4
        json.dumps({"path_check": pc, "initial_source": r["metrics"].get("initial_source")})
        if overrides is BOTH:
            assert pc["max_sweep_mm"] <= 12.0 + 2.5          # one 2 ms sub-step of rounding
            assert pc["cap_bound_steps"] == []


@records
def test_the_t41a_off_replay_still_crosses_and_on_does_not():
    off = RP.replay("T41aC1807", 35)
    assert off["safe"]                                                         # waypoint-only: ok
    c = RP.check_plan("T41aC1807", 35, off["chunk"], overrides=SERVO)
    assert c["path_clearance_m"] < -0.005                                      # but the path is not
    on = RP.replay("T41aC1807", 35, overrides=SERVO)
    assert on["safe"] and on["metrics"]["path_check"]["clearance_m"] >= -1e-4
