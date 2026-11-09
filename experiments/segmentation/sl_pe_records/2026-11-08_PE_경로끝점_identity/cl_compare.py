"""Closed loop per episode, SL-PE runs vs gt24att on the same task / seed / init (2026-11-08, dev-30).
success, calls whose executed action the TO changed (pre-held / held), HOLD calls, first `close_seed_switch` call, identity at
the first closing call (admissibility decision), whether an attach happened. Writes cl_compare.json + prints a table.
usage: python cl_compare.py <run tag glob, e.g. pe123b>"""
import json
import pathlib
import sys

import numpy as np

R = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
HERE = pathlib.Path(__file__).resolve().parent


def chunks_by_episode(run, order):
    ch, ep_i, last = {}, -1, None
    for c in sorted((run / "server_constraints" / "run_0000").glob("chunk_*.npz")):
        ts = int(np.load(c)["t_step"])
        if last is None or ts <= last:
            ep_i += 1
        last = ts
        if ep_i >= len(order):   # episode still running (no episodes.jsonl line yet)
            break
        ch.setdefault(order[ep_i], []).append((ts, c))
    return ch


def episode_stats(run, stem, ch):
    calls = [json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")]
    st = lambda r: (r.get("grasp") or {}).get("state")
    pre = [r for r in calls if st(r) in ("searching", "latched", "closing")]
    held = [r for r in calls if st(r) == "held"]
    chg = lambda rs: sum(float(r.get("max_abs_action_change_executed") or 0) > 1e-6 for r in rs)
    out = {"calls": len(calls), "changed_pre": chg(pre), "changed_held": chg(held),
           "hold": sum(bool(r.get("hold")) for r in calls), "attached": any(st(r) == "held" for r in calls),
           "first_seed_switch": None, "first_closing": next((r["seq"] for r in calls if st(r) == "closing"), None)}
    for ts, cp in ch.get(stem, []):
        s = json.loads(str(np.load(cp)["summary_json"]))
        dec = (s.get("admissibility") or {}).get("decision") or {}
        if dec.get("reason") == "close_seed_switch" and out["first_seed_switch"] is None:
            out["first_seed_switch"] = ts
    return out


def load(run):
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    ch = chunks_by_episode(run, order)
    return {o: {"success": bool(e["official_success"]), "seed": e["seed"], **episode_stats(run, o, ch)}
            for o, e in zip(order, eps)}


def base_run(seed, safelibero):
    pat = (f"*gt24att-safelibero11-t3-*" if safelibero else f"*gt24att-libero9s{seed}-*")
    runs = [p for p in sorted(R.glob(pat)) if (p / "episodes.jsonl").is_file() and not p.name.startswith("INVALID")]
    return runs[-1] if runs else None


def main():
    tag = sys.argv[1]
    rows, out = [], {}
    for run in sorted(R.glob(f"2*gtto-{tag}-*")):
        if not (run / "episodes.jsonl").is_file():
            continue
        sl = "safelibero" in run.name
        new = load(run)
        seed = next(iter(new.values()))["seed"]
        b = base_run(seed, sl)
        old = load(b) if b else {}
        for stem, n in new.items():
            o = old.get(stem, {})
            out[f"{run.name[:40]}/{stem}"] = {"new": n, "gt24att": o}
            rows.append(f"| {'SafeLIBERO' if sl else 'LIBERO'} s{seed} | {stem.split('spatial_')[-1]} | "
                        f"{'ok' if o.get('success') else 'FAIL'} → {'ok' if n['success'] else 'FAIL'} | "
                        f"{o.get('changed_pre', '–')} / {o.get('changed_held', '–')} / {o.get('hold', '–')} → "
                        f"{n['changed_pre']} / {n['changed_held']} / {n['hold']} | "
                        f"{n['first_seed_switch']} / {n['first_closing']} | {o.get('attached')} → {n['attached']} |")
    print("| bench | episode | success gt24att → new | changed pre / held / HOLD (gt24att → new) | seed switch / first closing call | attached |")
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))
    (HERE / f"cl_compare_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
