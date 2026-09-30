"""T34 J2 — 거짓 attach 회수 (`attach_revoked`), T33 기록 재생.

T33 의 두 서버 기록(E3a 8204 · E3b 8206)을 **요청 순서 그대로** `GraspLatch(evidence=True)` 에 다시
넣는다. latch 가 live 에서 받은 입력을 기록에서 그대로 읽는다:

* 실행 피드백 → `grasp_signal_from_feedback(summary.exec_feedback, 0, 0.85)` (측정 개도 · 실행 명령)
* reach · 조작 대상 id/상태 → 그 요청의 `summary.grasp.evidence` (live latch 가 본 값)
* J2 관측 (`held_observation`) → **직전 요청**의 `summary.manipulated` (그 프레임에 관측됐으면) 와 그
  프레임의 planning 행 q, attach 청크의 `grasp.attached.held` (T33 은 J1 전이라 기록된 구 = 맞춤 구 =
  self-filter 구의 중심), attach 직전 프레임의 manipulated centroid — `AG3S.held_observation` 과 같은
  식 (`held_object.held_observation_check`).

잠금은 첫 요청 전에 같은 이름 3 번으로 건다 (기록의 LATCHED 진입보다 이르지만, attach 판단은
closing 이후의 증거만 보므로 attach 시각은 기록과 같아야 한다 — 테스트가 그것을 먼저 확인한다).
기록이 이 머신에 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

V = pathlib.Path("/mnt/dev/work/outputs/verify/T33")
PORT = {"E3a": 8204, "E3b": 8206}
MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")

pytestmark = pytest.mark.skipif(
    not ((V / "p/E3b_analysis.json").exists() and MODEL.exists()
         and (V / "E3b_rec/server_8206/server_constraints/run_0000/chunk_00316.npz").exists()),
    reason="T33 records / model not present")


_ROBOT: dict = {}


def _robot(cond: str, run: str):
    if cond in _ROBOT:
        return _ROBOT[cond]
    from benchmark.trajopt.experiments import fixed_input_ablation as T

    rec = V / f"{cond}_rec"
    port = PORT[cond]
    args = T.build_parser().parse_args([
        "--constraints", str(rec / f"server_{port}/server_constraints/run_0000"),
        "--frames", str(rec / run / "frames.jsonl"),
        "--server-log", str(rec / f"serve_safe_{port}.log"), "--out", "/dev/null"])
    settings = T.resolve_settings(args, T.parse_server_log(args.server_log),
                                  json.loads((rec / run / "manifest.json").read_text()))
    robot, qadr, _ = T.build_robot(settings)
    _ROBOT[cond] = (robot, qadr, T)
    return _ROBOT[cond]


def _chunks(cond: str, run: str):
    A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
    rows = {int(x["seq"]): x for x in (json.loads(line) for line in
                                        (V / f"{cond}_rec" / run / "frames.jsonl").read_text()
                                        .splitlines() if line.strip())
            if x.get("kind") == "planning"}
    out = []
    for c in A["chunks"]:
        if not c.get("server_chunk"):
            continue
        z = np.load(V / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)
        out.append({"t": int(c["t"]), "summary": json.loads(str(z["summary_json"])),
                    "row": rows[int(c["seq"])]})
    return out


def replay(cond: str, run: str, config=None):
    """→ list of per-request dicts `{t, state, attach, detach, revoked, reasons, recorded}`."""
    from benchmark.ag3s.robot_models.held_object import held_observation_check
    from benchmark.trajopt.grasp_latch import GraspLatch, LatchConfig, grasp_signal_from_feedback

    robot, qadr, T = _robot(cond, run)
    chunks = _chunks(cond, run)
    latch = GraspLatch(config or LatchConfig(), evidence=True)
    for _ in range(3):
        latch.update(label="obj0", score=1.0, runner_up=0.0)
    out = []
    ref = None                      # set at the replayed attach
    for i, c in enumerate(chunks):
        s = c["summary"]
        g = s.get("grasp") or {}
        ev = g.get("evidence") or {}
        sig = grasp_signal_from_feedback(s.get("exec_feedback"), 0, 0.85)
        held_obs = None
        if latch.holding and ref is not None and i > 0:
            prev = chunks[i - 1]
            m = prev["summary"].get("manipulated") or {}
            if (m.get("last_seen_frame") is not None and m.get("last_seen_frame") == m.get("frame")
                    and int(m["frame"]) >= ref["frame"]):
                q = T.q_now_from_qpos(prev["row"]["qpos"], qadr)
                Tq = robot.link_pose(q, ref["link"])
                held_obs = held_observation_check(
                    observed_centroid=m["centroid"],
                    held_centre=Tq[:3, :3] @ ref["local"] + Tq[:3, 3],
                    held_centre_at_attach=ref["at_attach"],
                    object_centroid_at_attach=ref["object_centroid"])
                held_obs["held_radius_m"] = ref["radius"]
        reach = None if ev.get("reach_mm") is None else float(ev["reach_mm"]) / 1000.0
        event = latch.observe_grasp(sig, reach_m=reach, manipulated_id=ev.get("manipulated_id"),
                                    manipulated_state=ev.get("manipulated_state"),
                                    held_observation=held_obs)
        if event.attach:
            held = (g.get("attached") or {}).get("held") if g.get("attach") else None
            if held is None:
                # replay attached where the record did not: use the next recorded held record
                held = next(((x["summary"].get("grasp") or {}).get("attached") or {}).get("held")
                            for x in chunks[i:] if (x["summary"].get("grasp") or {}).get("attach"))
            link = held["frame_link"]
            local = np.asarray(held["spheres_parent_frame"][0]["centre_m"], np.float64)
            q_now = T.q_now_from_qpos(c["row"]["qpos"], qadr)
            Tn = robot.link_pose(q_now, link)
            mprev = chunks[i - 1]["summary"].get("manipulated") or {}
            ref = {"frame": int(g.get("alignment", {}).get("pipeline_frame", s["t_step"] - 1)),
                   "link": link, "local": local,
                   "radius": float(held["spheres_parent_frame"][0]["radius_mm"]) / 1000.0,
                   "at_attach": Tn[:3, :3] @ local + Tn[:3, 3],
                   "object_centroid": np.asarray(mprev["centroid"], np.float64)}
        out.append({"t": c["t"], "state": event.phase.value, "attach": bool(event.attach),
                    "detach": bool(event.detach), "revoked": bool(event.revoked),
                    "reasons": list(((event.evidence or {}).get("revoke") or {}).get("reasons") or ()),
                    "checks": (event.evidence or {}).get("hold_checks"),
                    "window": (event.evidence or {}).get("hold_window"),
                    "held_observation": held_obs,
                    "recorded": {"state": g.get("state"), "attach": bool(g.get("attach")),
                                 "detach": bool(g.get("detach"))}})
    return out


def _events(res):
    return ([r["t"] for r in res if r["attach"]], [r["t"] for r in res if r["detach"]],
            [(r["t"], tuple(r["reasons"])) for r in res if r["revoked"]])


@pytest.fixture(scope="module")
def e3b_1808_r1():
    return replay("E3b", "E3b_ep1808_r1")


def test_replay_attaches_where_the_record_did(e3b_1808_r1):
    """The replay is faithful up to the attach: same attach request as T33 (t=128)."""
    rec = [r["t"] for r in e3b_1808_r1 if r["recorded"]["attach"]]
    att, _, _ = _events(e3b_1808_r1)
    assert rec == [128] and att == [128]


def test_false_attach_ep1808_r1_is_revoked_on_the_first_executed_chunk(e3b_1808_r1):
    """E3b ep1808 r1: attached at t=128, the fingers then followed the command down (0.684 →
    0.555 at the next request, command 0.552) — revoked at t=136 (T33: detached at t=232)."""
    att, det, rev = _events(e3b_1808_r1)
    assert rev and rev[0][0] == 136
    assert set(rev[0][1]) == {"opening_dropped", "not_blocked"}
    r = next(x for x in e3b_1808_r1 if x["t"] == 136)
    assert r["state"] == "latched" and r["detach"]
    assert r["window"]["exec_chunks"] == 1
    assert r["checks"]["opening_dropped"] >= 0.12 and r["checks"]["gap_to_command"] < 0.01
    # no second attach for the rest of the run (the hand never closes on the apple again)
    assert att == [128]
    # the recorded detach at t=232 was the 2-frame open rule; here nothing is held any more
    assert [x["t"] for x in e3b_1808_r1 if x["recorded"]["detach"]] == [232]


def test_false_attach_is_also_caught_by_the_observation_alone():
    """Opening checks off (ablation): the apple is seen on the table at t=152 (frame after the
    lift), 206–238 mm from the held sphere, and did not rise with the 224 mm lift — revoked at the
    next request, t=160, the 4th executed chunk after attach (window 4)."""
    from benchmark.trajopt.grasp_latch import LatchConfig

    res = replay("E3b", "E3b_ep1808_r1", LatchConfig(revoke_use_opening=False))
    _, _, rev = _events(res)
    assert rev and rev[0][0] == 160
    assert set(rev[0][1]) == {"observed_elsewhere", "not_following"}
    r = next(x for x in res if x["t"] == 160)
    assert r["window"]["exec_chunks"] == 4
    assert r["held_observation"]["centre_distance_m"] > 0.15
    assert r["held_observation"]["hand_rise_m"] > 0.15
    assert abs(r["held_observation"]["object_rise_m"]) < 0.01


@pytest.mark.parametrize("cond,run,attach_t,detach_t", [
    ("E3a", "E3a_ep1807_r1", 104, 208),
    ("E3a", "E3a_ep1807_r2", 120, 224),
    ("E3b", "E3b_ep1807_r1", 104, 216),
    ("E3b", "E3b_ep1807_r2", 136, 240),
    ("E3b", "E3b_ep1808_r2", 400, 528),
])
def test_successful_runs_are_never_revoked(cond, run, attach_t, detach_t):
    """The true attaches of every successful T33 run: same attach, no revoke, the recorded
    release (2 frames open) at the same request."""
    res = replay(cond, run)
    att, det, rev = _events(res)
    assert att == [attach_t]
    assert rev == []
    assert det == [detach_t]
    # the window closed on each of them with the opening still held
    worst = [x["checks"] for x in res if x["checks"] and (x["window"] or {}).get("active")]
    assert worst and max(c.get("opening_dropped", 0.0) for c in worst) < 0.02
    assert min(c["gap_to_command"] for c in worst if "gap_to_command" in c) > 0.10


def test_the_held_deadlock_run_is_not_revoked():
    """E3b ep1800 r2: a real grasp that never lifted (56 HOLD chunks). No executed chunk → the
    window never runs out, the opening stays at 0.716–0.720 → never revoked."""
    res = replay("E3b", "E3b_ep1800_r2")
    att, det, rev = _events(res)
    assert att == [152] and rev == [] and det == []


def test_drop_right_after_the_lift_is_revoked_and_a_late_drop_is_not():
    """E3a ep1800 r1 (failed run): lifted 10 mm, the apple fell back at the next chunk (opening
    0.686 → 0.551 = command 0.550) — revoked at t=160, the 2nd executed chunk. E3a ep1800 r2: the
    apple fell at the 6th executed chunk (t=200), outside the window — the recorded open rule
    releases it (t=264)."""
    _, _, rev = _events(replay("E3a", "E3a_ep1800_r1"))
    assert rev and rev[0][0] == 160
    att, det, rev2 = _events(replay("E3a", "E3a_ep1800_r2"))
    assert att == [152] and rev2 == [] and det == [264]
