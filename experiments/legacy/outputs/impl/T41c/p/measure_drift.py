"""T41 c — HOLD drift in MuJoCo from replayed closed-loop states (implementer unit-level numbers).

For each (run, t0): replay the recorded run bit for bit to row t0, then HOLD for 64 control steps
with HoldController in `fixed` and `measured`. Writes outputs/impl/T41c/drift.json.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from tests.sim import _hold_replay as hr

OUT = pathlib.Path(_ag3s_workspace_path('outputs/impl/T41c'))
CASES = [("B/E3b_ep1807_s18071", [184, 200, 216]),
         ("B/E3b_ep1968_s19681", None),
         ("B/E3b_ep1828_s18281", None),
         ("C/E3b_ep1982_s19822", [176])]
N_STEPS = 64


def carry_rows(control):
    """chunk-boundary rows with the apple > 0.15 m above its start and the left gripper commanded < 0.9."""
    z0 = control[0]["qpos"][40]
    out = [r["t_step"] for r in control if r["t_step"] % 8 == 0 and r["t_step"] > 0
           and r["qpos"][40] > z0 + 0.15 and r["applied_ctrl"]["gripper"][0] < 0.9]
    return out


def summary(res, steps=(1, 8, 64)):
    rows = res["rows"]
    s = {"mode": res["mode"], "collided": res["collided"],
         "first_contact_step_substep": res["first_contact_step_substep"],
         "contact_pairs": res["contact_pairs"], "min_robot_clearance_mm":
         None if res["min_robot_clearance_m"] is None else res["min_robot_clearance_m"] * 1e3}
    for k in steps:
        r = rows[k - 1]
        s[f"step{k}"] = {"arm_dev_from_entry_deg": float(np.degrees(r["arm_dev_from_entry_rad"])),
                         "ee_disp_mm": r["ee_disp_mm"], "fruit_in_hand_shift_mm": r["fruit_in_hand_shift_mm"],
                         "fruit_z_m": r["fruit_z_m"], "grip_meas_norm": r["grip_meas_norm"],
                         "grip_ctrl_norm": r["grip_ctrl_norm"]}
    s["max_over_64"] = {"arm_dev_from_entry_deg": float(np.degrees(max(r["arm_dev_from_entry_rad"] for r in rows))),
                        "ee_disp_mm": max(r["ee_disp_mm"] for r in rows),
                        "fruit_in_hand_shift_mm": max(r["fruit_in_hand_shift_mm"] for r in rows)}
    s["grip_ctrl_constant"] = bool(np.all(res["ctrl_rows"][:, 7] == res["ctrl_rows"][0, 7]))
    s["q_hold_source"] = res["records"][0]["source"]
    s["entry_command_dev_deg"] = float(np.degrees(np.max(np.abs(res["last_command_entry"]["arm"] - res["arm_entry"]))))
    s["per_step"] = [{k: v for k, v in r.items()} for r in rows]
    return s


def main():
    client = hr.load_client()
    out = {"_comment": "implementer T41 c unit-level numbers: replay (bit-identical to every recorded row up "
                       "to t0) then HOLD 64 control steps (33 sim steps each, 2.2 s... 4.2 s sim) per mode. "
                       "arm_dev = max |q_arm - q_arm(t0)| over the 14 arm joints; ee = EE_BODY_L world "
                       "displacement; fruit_in_hand = apple position change in the EE_BODY_L frame.",
           "n_steps": N_STEPS, "cases": []}
    for name, t0s in CASES:
        run = hr.load_run(hr.RUNS / name)
        if t0s is None:
            rows = carry_rows(run["control"])
            t0s = [rows[len(rows) // 2]] if rows else []
        for t0 in t0s:
            local = hr.reset(client, run)
            hr.replay_to(client, local, run["control"], t0)
            case = {"run": name, "t0": t0, "modes": {}}
            for mode in ("fixed", "measured"):
                res = hr.hold(client, local, mode, N_STEPS)
                case["modes"][mode] = summary(res)
                s = case["modes"][mode]
                print(name, t0, mode, "cmd-meas %.2f deg" % s["entry_command_dev_deg"],
                      "| step8 arm %.3f deg ee %.1f mm | step64 arm %.3f deg ee %.1f mm | fruit shift max %.1f mm z64 %.3f"
                      % (s["step8"]["arm_dev_from_entry_deg"], s["step8"]["ee_disp_mm"],
                         s["step64"]["arm_dev_from_entry_deg"], s["step64"]["ee_disp_mm"],
                         s["max_over_64"]["fruit_in_hand_shift_mm"], s["step64"]["fruit_z_m"]),
                      "| collided", s["collided"], s["first_contact_step_substep"], "minclr", s["min_robot_clearance_mm"],
                      flush=True)
            out["cases"].append(case)
    (OUT / "drift.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
