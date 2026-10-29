"""T23 측정 #3·#4 — MuJoCo 단위 실험: HOLD 동안 legacy (매 step rby1_state 재명령) vs fixed (HoldController q_hold).

실제 서버 없음. `pi05_infer.py` (서버 사본) 에서 import 해 직접 부르는 것:
  apply_action, reset_randomized_scene, load_randomized_episode, hold_keyframe_pose, 상수 (CTRL_HZ,
  RBY1_GRIPPER_OPEN, *_ARM_JOINTS/ACTS, GRIPPER_*), MODEL_XML_TRANSPORT.
`AG3S.trajopt.client.HoldController` 를 그대로 쓴다 (mode="legacy" / "fixed").
**재현한 것 (main() 안의 closure 라 import 불가):** rby1_state(), applied_ctrl(), idx/act 사상, 씬 초기화 순서
  (main 의 948–1062 줄과 같은 순서로 옮겼다), 제어 루프의 HOLD 분기 (1742–1757: hold_action(last_command=applied_ctrl(),
  measured=rby1_state()) → apply_action → 33 × mj_step).
시작 상태: T16 을 처음부터 다시 돌린다 — t=0..71 에 T16 planning 행의 실행 청크(actions[t%8])를 그대로 적용.
  T16 control 기록의 qpos (apply_action 직후·mj_step 전) 와 step 마다 비교해 재현 오차를 남긴다.
  (T16 은 local PC mujoco 3.10.0, 여기는 3.11.0.)
t=72 부터 8 chunk × 8 step = 64 control step (각 33 sim step) 을 두 방식으로.
시나리오 G (합성, #4): t=72 에서 한 step 을 팔=직전 명령, gripper=0.0(닫힘 명령) 으로 실행한 뒤 HOLD 63 step.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('integration/rby1/bringup'))
import mujoco  # noqa: E402
import pi05_infer as P  # noqa: E402
from AG3S.trajopt.client import HoldController  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as fia  # noqa: E402

RUN = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t16'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T23/m3'))
T_HOLD = 72
N_HOLD = 64
EPISODE = 1807

rows = [json.loads(l) for l in (RUN / "execute_ep1807/frames.jsonl").read_text().splitlines() if l.strip()]
plan = {int(r["t_step"]): r for r in rows if r["kind"] == "planning"}
ctrl_rows = {int(r["t_step"]): r for r in rows if r["kind"] == "control"}


def setup():
    m = mujoco.MjModel.from_xml_path(P.MODEL_XML_TRANSPORT)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(m, d, key)
    P.hold_keyframe_pose(m, d)
    arm6 = {}
    for side, jn in (("left", P.LEFT_ARM_JOINTS[6]), ("right", P.RIGHT_ARM_JOINTS[6])):
        arm6[side] = float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]])
    for i in range(m.nu):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    for side, an in (("left", P.LEFT_ARM_ACTS[6]), ("right", P.RIGHT_ARM_ACTS[6])):
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an)] = arm6[side]
    P.reset_randomized_scene(m, d, P.load_randomized_episode(EPISODE))
    mujoco.mj_forward(m, d)
    J = lambda j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
    A = lambda a: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    idx = {"right_q": [J(j) for j in P.RIGHT_ARM_JOINTS], "left_q": [J(j) for j in P.LEFT_ARM_JOINTS],
           "right_grip_q": J(P.GRIPPER_R_JOINT), "left_grip_q": J(P.GRIPPER_L_JOINT)}
    act = {"right_a": [A(a) for a in P.RIGHT_ARM_ACTS], "left_a": [A(a) for a in P.LEFT_ARM_ACTS],
           "right_grip_a": A(P.GRIPPER_R_ACT), "left_grip_a": A(P.GRIPPER_L_ACT),
           "left_arm6_hold": arm6["left"], "right_arm6_hold": arm6["right"]}
    return m, d, idx, act


def closures(m, d, idx, act):
    n = 7

    def rby1_state():
        left = np.asarray([d.qpos[i] for i in idx["left_q"][:n]])
        right = np.asarray([d.qpos[i] for i in idx["right_q"][:n]])
        lg = float(abs(d.qpos[idx["left_grip_q"]]) / abs(P.RBY1_GRIPPER_OPEN))
        rg = float(abs(d.qpos[idx["right_grip_q"]]) / abs(P.RBY1_GRIPPER_OPEN))
        return np.concatenate([left, [lg], right, [rg]])

    def applied_ctrl():
        arm = np.asarray([d.ctrl[a] for a in act["left_a"][:n]] + [d.ctrl[a] for a in act["right_a"][:n]])
        grip = np.asarray([d.ctrl[act["left_grip_a"]], d.ctrl[act["right_grip_a"]]]) / P.RBY1_GRIPPER_OPEN
        return {"arm": arm, "gripper": grip}
    return rby1_state, applied_ctrl


def step_physics(m, d, steps):
    for _ in range(steps):
        mujoco.mj_step(m, d)


# 손가락 구: T16 제약 모델 (T24 도구가 로그에서 다시 짓는 것과 같은 모델)
args = fia.build_parser().parse_args(["--constraints", str(RUN / "server_constraints/run_0000"),
                                       "--frames", str(RUN / "execute_ep1807/frames.jsonl"),
                                       "--server-log", str(RUN / "serve_safe.log"), "--out", "/dev/null"])
settings = fia.resolve_settings(args, fia.parse_server_log(args.server_log),
                                json.loads((RUN / "execute_ep1807/manifest.json").read_text()))
robot, qadr, _ = fia.build_robot(settings)
names = [str(x) for x in robot.sphere_link_names]
finger_idx = np.asarray([i for i, x in enumerate(names) if x in ("ee_finger_l1", "ee_finger_l2")])


def finger_centre(qpos):
    q = fia.q_now_from_qpos(qpos, qadr)
    c, _ = robot.sphere_centers_numeric(q)
    return np.asarray(c)[finger_idx].mean(axis=0)


WARMUP_STEPS = 1000  # --start-delay 2.0 s (기본) 의 명목 sim step 수. 실제는 벽시계 페이싱이라 기록에 없다.


def run(mode, scenario="hold", init="record"):
    m, d, idx, act = setup()
    step_physics(m, d, WARMUP_STEPS)
    rby1_state, applied_ctrl = closures(m, d, idx, act)
    spa = max(1, int(round(1.0 / (P.CTRL_HZ * m.opt.timestep))))
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    apple_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    bl1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l1")
    bl2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l2")
    prefix_err = []
    for t in range(T_HOLD):
        chunk = plan[8 * (t // 8)]
        assert ctrl_rows[t]["executed"], t
        P.apply_action("rby1_16d", np.asarray(chunk["actions"][t % 8], np.float64), d, idx, act)
        rec = np.asarray(ctrl_rows[t]["qpos"])
        prefix_err.append(float(np.abs(np.asarray(d.qpos) - rec).max()))
        step_physics(m, d, spa)
    if init == "record":
        # lead 지시: T16 t=72 자세(qpos)로 시작. qvel 은 기록에 없어 0. ctrl 은 재생이 t=71 에 넣은 값(= T16 의 t=71 명령).
        d.qpos[:] = np.asarray(ctrl_rows[T_HOLD]["qpos"], np.float64)
        d.qvel[:] = 0.0
        d.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(m, d)
    hold = HoldController(mode=mode)
    hold.begin_chunk(executed=False, reason="unit test", kinds=["collision"])
    log = []
    for i in range(N_HOLD):
        t = T_HOLD + i
        if i > 0 and i % 8 == 0:
            hold.begin_chunk(executed=False, reason="unit test", kinds=["collision"])
        if scenario == "close_then_hold" and i == 0:
            hold.release()
            last = applied_ctrl()
            action = np.concatenate([last["arm"][:7], [0.0], last["arm"][7:], [last["gripper"][1]]])
            executed = True
        else:
            action = hold.hold_action(last_command=applied_ctrl(), measured=rby1_state())
            executed = False
        P.apply_action("rby1_16d", action, d, idx, act)
        ap = applied_ctrl()
        hrec = hold.record(measured=rby1_state()) if hold.holding else None
        qpos_cmd = np.asarray(d.qpos).copy()  # apply_action 직후·mj_step 전 (T16 control 기록과 같은 순간)
        mujoco.mj_kinematics(m, d)
        R, p = d.xmat[base_id].reshape(3, 3), d.xpos[base_id].copy()
        log.append({
            "t": t, "executed": executed,
            "qpos": qpos_cmd.tolist(),
            "applied_arm": ap["arm"].tolist(), "applied_gripper": ap["gripper"].tolist(),
            "ctrl_left_grip": float(d.ctrl[act["left_grip_a"]]), "ctrl_right_grip": float(d.ctrl[act["right_grip_a"]]),
            "measured_state": rby1_state().tolist(),
            "finger_centre_base": finger_centre(qpos_cmd).tolist(),
            "finger_bodies_world": [d.xpos[bl1].tolist(), d.xpos[bl2].tolist()],
            "apple_world": d.xpos[apple_id].tolist(), "base_world": p.tolist(), "base_R": R.tolist(),
            "hold": hrec,
            "recorded_t16_qpos": (ctrl_rows[t]["qpos"] if t in ctrl_rows else None),
        })
        step_physics(m, d, spa)
    return {"mode": mode, "scenario": scenario, "init": init, "warmup_steps": WARMUP_STEPS, "sim_steps_per_action": spa, "timestep": m.opt.timestep,
            "prefix_qpos_max_abs_err_per_t": prefix_err, "hold_stats": dict(hold.stats), "log": log}


if __name__ == "__main__":
    out = {"mujoco": mujoco.__version__, "model_xml": P.MODEL_XML_TRANSPORT, "episode": EPISODE,
           "t_hold": T_HOLD, "n_hold_steps": N_HOLD, "runs": {}}
    for init in ("record", "replay"):
      for mode in ("legacy", "fixed"):
        for sc in ("hold", "close_then_hold"):
            r = run(mode, sc, init)
            out["runs"][f"{init}|{mode}|{sc}"] = r
            fc = np.asarray([e["finger_centre_base"] for e in r["log"]])
            dr = np.linalg.norm(fc - fc[0], axis=1) * 1000
            print(init, mode, sc, "prefix err max", max(r["prefix_qpos_max_abs_err_per_t"]),
                  "drift mm @8,24,63:", dr[8].round(2), dr[24].round(2), dr[-1].round(2),
                  "gripL ctrl first/last", r["log"][0]["ctrl_left_grip"], r["log"][-1]["ctrl_left_grip"], flush=True)
    (OUT / "hold_drift.json").write_text(json.dumps(out))
    print("wrote", OUT / "hold_drift.json")
