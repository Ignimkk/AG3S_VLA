"""T41a (copy of T40/p/update_partial.py) — rebuild the run list of AG3S/ag3s/docs/handoff/T43W.verify.partial.json from the run directories
(copy of T39/p/update_partial.py). Keeps every other top-level key. Safe to call from several chains at once (flock).
Scans outputs/verify/T40/G/runs/<COND>/<STAGE>/<COND>_ep<EP>_s<SEED>.start ; STAGE = H (T43 P hurdle) | B (no obstacle); status = done (NAME.done, check_run passed) | failed (.end without .done) | running (.start only).
wall_s = .end - .start. check_run mode (full | obstacle_stop) from NAME.check.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import fcntl, json, pathlib, re, datetime
G = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/GD/runs'))
PARTIAL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/GD/partial.json'))
LOCK = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/GD/p/.partial.lock'))
PAT = re.compile(r"^(V7)_ep(\d+)_s(\d+)\.start$")


def ts(p):
    try:
        return datetime.datetime.fromisoformat(p.read_text().splitlines()[0].strip())
    except Exception:
        return None


def scan():
    runs = {}
    for cdir in sorted(G.glob("*")) if G.exists() else []:
        for g in sorted(p for p in cdir.glob("*") if p.is_dir() and not p.name.startswith(("server_", "failed"))):
            for f in sorted(g.glob("*.start")):
                m = PAT.match(f.name)
                if not m:
                    continue
                name = f.name[:-6]; d = g / name
                st, en = ts(f), ts(g / f"{name}.end")
                lines = f.read_text().splitlines()
                port = next((l.split("=", 1)[1] for l in lines if l.startswith("port=")), None)
                obst = next((l.split("=", 1)[1] for l in lines if l.startswith("obstacle=")), None)
                status = "done" if (g / f"{name}.done").exists() else ("failed" if en else "running")
                mode = None
                try:
                    mode = json.loads((g / f"{name}.check.json").read_text())["checks"].get("mode")
                except Exception:
                    pass
                runs[f"{g.name}/{name}"] = {
                    "stage": g.name, "cond": m.group(1), "episode": int(m.group(2)), "seed": int(m.group(3)),
                    "status": status, "check_mode": mode, "run_dir": str(d), "port": None if port is None else int(port),
                    "obstacle": obst, "start": st.isoformat() if st else None, "end": en.isoformat() if en else None,
                    "wall_s": (en - st).total_seconds() if (st and en and en >= st) else None,
                    "n_failed_attempts": len(list((g / "failed").glob(f"{name}.client.log.*"))) if (g / "failed").exists() else 0}
    return runs


def main():
    LOCK.touch(exist_ok=True)
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {
            "_comment": "ag3s-verifier partial record for T43GD (numbers and paths only). runs is rebuilt by outputs/verify/T43/GD/p/update_partial.py after every run.",
            "step": "T43GD"}
        runs = scan()
        cur["runs"] = runs
        cnt = {}
        for r in runs.values():
            k = f'{r["stage"]}:{r["cond"]}:{r["status"]}'
            cnt[k] = cnt.get(k, 0) + 1
        cur["run_counts"] = dict(sorted(cnt.items()))
        cur["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        tmp = PARTIAL.with_suffix(".json.tmp"); tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False)); tmp.replace(PARTIAL)
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(cur["updated"], cur["run_counts"])


if __name__ == "__main__":
    main()
