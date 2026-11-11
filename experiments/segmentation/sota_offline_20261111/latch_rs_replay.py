#!/usr/bin/env python3
"""(2026-11-12 dev-57, offline CPU) Replay the recorded opening-rule latch inputs (calls.jsonl `grasp` block) through
`GraspLatch` with and without SL-SOTA11 `opening_rule_reach_sticky`. An ESTIMATE up to the first changed decision: the
recorded closed loop is the old latch's. Reports, per episode, the first call the decisions differ (a new attach) and what
the fingers did afterwards (the opening staying >= 0.03 with a closed command = something is in the hand).
usage (worktree code): PYTHONPATH=<checkout with SL-SOTA11> python latch_rs_replay.py --out FILE.json <run glob> [<run glob> ...]"""
import ast, glob, json, os, re, sys
import dataclasses

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return x or {}


def main():
    from AG3S.adapters.panda.server import panda_latch_config
    from AG3S.trajopt.grasp_latch import GraspLatch, GraspPhase, GraspSignal
    args = sys.argv[1:]
    out_path = None
    if args and args[0] == "--out":
        out_path = args[1]
        args = args[2:]
    base = dataclasses.replace(panda_latch_config(opening_rule=True, opening_rule_moved=0.05), reach_sticky=True, drop_from_open=True)
    cfgs = {"off": base, "on": dataclasses.replace(base, opening_rule_reach_sticky=True)}
    res = []
    n_ep = n_same = 0
    for pat in args:
        for d in sorted(glob.glob(B + pat)):
            if not os.path.isfile(d + "/episodes.jsonl"):
                continue
            name = re.sub(r"-(vla_v11ab|pi05).*", "", os.path.basename(d).split("_", 2)[2])
            for l in open(d + "/episodes.jsonl"):
                e = json.loads(l)
                fs = glob.glob(f"{d}/episodes/*_t{e.get('task_id')}_e{e['episode_idx']}_calls.jsonl")
                if not fs:
                    continue
                calls = [json.loads(x) for x in open(fs[0])]
                rows = {}
                for key, cfg in cfgs.items():
                    latch = GraspLatch(cfg, evidence=True, external_destination=True)
                    seq = []
                    for c in calls:
                        g = c.get("grasp") or {}
                        ev = lit(g.get("evidence"))
                        if not ev:
                            seq.append((c["seq"], g.get("state"), False, False))
                            continue
                        if latch.phase is GraspPhase.SEARCHING and g.get("state_before") not in (None, "searching"):
                            latch.phase = GraspPhase.LATCHED
                        rule = lit(ev.get("opening_rule"))
                        sig = GraspSignal(available=bool(ev.get("feedback_available")), feedback_seq=ev.get("feedback_seq"),
                                          n_steps=int(ev.get("n_steps") or 0), n_exec=int(ev.get("n_exec") or 0),
                                          closing_steps=int(ev.get("closing_steps") or 0),
                                          last_command=rule.get("last_command") if ev.get("n_exec") else None,
                                          opening=ev.get("opening_after"))
                        reach = ev.get("reach_mm")
                        al = lit(g.get("alignment"))
                        event = latch.observe_grasp(sig, reach_m=None if reach is None else float(reach) / 1000.0,
                                                    manipulated_id=ev.get("manipulated_id"),
                                                    manipulated_state=ev.get("manipulated_state"), placed=False,
                                                    held_observation=None,
                                                    reach_context={"geometry_frame": al.get("geometry_frame"),
                                                                   "reach_state": al.get("reach_state", "now")})
                        if event.attach and key == "off" and g.get("rejected"):
                            event = latch.revert_attach("recorded AG3S.attach rejection")
                        seq.append((c["seq"], latch.phase.value, bool(event.attach), bool(event.detach)))
                    rows[key] = seq
                rec_states = [(c.get("grasp") or {}).get("state") for c in calls]
                agree = sum(a == b[1] for a, b in zip(rec_states, rows["off"]))
                n_ep += 1
                first = next((i for i, (a, b) in enumerate(zip(rows["off"], rows["on"])) if a[1:] != b[1:]), None)
                sl = e.get("safelibero") or {}
                ok, col = bool(e.get("official_success")), bool(sl.get("collided"))
                item = {"run": name, "task": e.get("task_id"), "init": e.get("init_state_index"),
                        "outcome": "SS" if ok and not col else "S+C" if ok else "F+C" if col else "F",
                        "calls": len(calls), "off_agrees_with_record": agree, "first_change_seq": None}
                if agree == len(calls):
                    n_same += 1
                if first is not None:
                    c = calls[first]
                    ev = lit((c.get("grasp") or {}).get("evidence"))
                    st = lit(ev.get("reach_sticky")) if False else None
                    after = [lit((x.get("grasp") or {}).get("evidence")).get("opening_after") for x in calls[first:first + 12]]
                    held_like = sum(1 for o in after if o is not None and 0.03 <= o < 0.5)
                    item.update(first_change_seq=c["seq"], on=rows["on"][first][1:], off=rows["off"][first][1:],
                                reach_mm=ev.get("reach_mm"), opening=ev.get("opening_after"),
                                openings_next12=[None if o is None else round(o, 3) for o in after], held_like_next12=held_like,
                                recorded_state_to_end=sorted(set(rec_states[first:])))
                    print(f"{name} t{item['task']} e{item['init']} {item['outcome']}: first change at seq {c['seq']} on={rows['on'][first][1:]} "
                          f"off={rows['off'][first][1:]} reach {ev.get('reach_mm')} opening {ev.get('opening_after')} held-like next12 {held_like} "
                          f"recorded to end {item['recorded_state_to_end']}", flush=True)
                res.append(item)
    print(f"episodes {n_ep}, off replay identical to the record in {n_same}; changed by the flag {sum(1 for r in res if r['first_change_seq'] is not None)}")
    if out_path:
        json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
