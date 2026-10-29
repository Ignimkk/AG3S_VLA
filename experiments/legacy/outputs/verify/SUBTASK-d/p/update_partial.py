"""SUBTASK-d — rebuild the run list of AG3S/ag3s/docs/handoff/SUBTASK-d.verify.partial.json from the run directories
(copied from T39 p/update_partial.py; stages v0 / runs, conds E3b / E3bG). Keeps every other top-level key. flock-safe."""
from common.workspace import workspace_path as _ag3s_workspace_path
import fcntl, json, pathlib, re, datetime
ROOT = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-d'))
PARTIAL = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/SUBTASK-d.verify.partial.json'))
LOCK = ROOT / "p" / ".partial.lock"
PAT = re.compile(r"^(E3b|E3bG)_ep(\d+)_s(\d+)\.start$")


def ts(p):
    try:
        return datetime.datetime.fromisoformat(p.read_text().splitlines()[0].strip())
    except Exception:
        return None


def scan():
    runs = {}
    for stage in ("v0", "runs"):
        for g in sorted((ROOT / stage).glob("*")) if (ROOT / stage).exists() else []:
            if not g.is_dir():
                continue
            for f in sorted(g.glob("*.start")):
                m = PAT.match(f.name)
                if not m:
                    continue
                name = f.name[:-6]; d = g / name
                st, en = ts(f), ts(g / f"{name}.end")
                port = next((l.split("=", 1)[1] for l in f.read_text().splitlines() if l.startswith("port=")), None)
                status = "done" if (g / f"{name}.done").exists() else ("failed" if en else "running")
                runs[f"{stage}/{g.name}/{name}"] = {
                    "stage": stage, "group": g.name, "cond": m.group(1), "episode": int(m.group(2)), "seed": int(m.group(3)),
                    "status": status, "run_dir": str(d), "port": None if port is None else int(port),
                    "start": st.isoformat() if st else None, "end": en.isoformat() if en else None,
                    "wall_s": (en - st).total_seconds() if (st and en and en >= st) else None,
                    "n_failed_attempts": len(list((g / "failed").glob(f"{name}.client.log.*"))) if (g / "failed").exists() else 0}
    return runs


def main():
    LOCK.touch(exist_ok=True)
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {"step": "SUBTASK-d"}
        runs = scan()
        cur["runs"] = runs
        cnt = {}
        for r in runs.values():
            k = f'{r["stage"]}:{r["cond"]}:{r["status"]}'
            cnt[k] = cnt.get(k, 0) + 1
        cur["run_counts"] = dict(sorted(cnt.items()))
        cur["v1_expected_runs"] = 48
        cur["v1_done"] = sum(1 for r in runs.values() if r["stage"] == "runs" and r["cond"] == "E3bG" and r["status"] == "done")
        cur["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        tmp = PARTIAL.with_suffix(".json.tmp"); tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False)); tmp.replace(PARTIAL)
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(cur["updated"], cur["run_counts"], "v1_done", cur["v1_done"])


if __name__ == "__main__":
    main()
