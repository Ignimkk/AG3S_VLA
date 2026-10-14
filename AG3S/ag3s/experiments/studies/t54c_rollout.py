"""T54c — rollout launcher template: the T53 ``p/`` scripts re-pointed at a new run, ports 8292 · 8293, new episodes.

`T54c.task.md` "기록 조건 / 구현": the T54c rollouts are recorded exactly as T53's (AG3S ON, current serving =
C1a flag set, ``clustering.placed_no_seed_certify`` off, policy seed ``<N>1``, 600 step, the same client /
server record flags, GPU lock) — so the launcher is **generated from the T53 run's own ``p/``** (passed as an
argument; never a constant), with only these changes, each one counted (a count mismatch = the source
changed = refusal):

==========================  =========================================================================
file                        change
==========================  =========================================================================
every copied file           T53 run path → the new run (``…/verify/T53`` → ``…/verify/T54c``)
``env.sh``                  ``V=$RUN/verify/T54c``; ``side_of`` = the 16 T54c episodes (dataset
                            ``used_arm``); ``port_ok`` = 8292 · 8293
``*.sh``                    lock variable ``T53_GPU_LOCK_HELD`` → ``T54C_GPU_LOCK_HELD``; port messages
``run_chain.sh``            baseline md5 list ``p/code_md5_t53.txt`` → ``p/code_md5_t54c.txt``
``run_client.sh``           message "not in the T54c list"
``update_partial.py``       ``"step": "T54c"``
``launch_t54c.sh``          from ``launch_t53.sh``: the two per-port lists (even slot → 8292, odd →
                            8293, seed ``<N>1``), ports, ``829[23]`` server count
new ``make_snapshot.sh``    ``git archive`` of the checkout HEAD into ``$RUN/snapshot`` (refused unless HEAD
                            is ``cf38c9db`` or a descendant whose changes since it are docs-only), then
                            ``snapshot_head.txt``, ``p/code_md5_t54c.txt`` and ``md5_start.txt``
new ``md5_files.sh``        ``start`` / ``end`` md5 of the six serving files (main tree and snapshot)
``../seeds.json``           ``{slot, episode, seed, port}`` (written before any rollout)
==========================  =========================================================================

Not copied: ``launch_t53.sh`` (→ ``launch_t54c.sh``), ``launch_all.sh`` / ``launch_cl.sh`` (CL3 leftovers),
``build_verify_t53.py`` / ``fig_t53.py`` (T53 verifier products), ``code_md5_t53.txt`` (made again from the
new snapshot), lock files, ``__pycache__``. The analysis scripts (``analyze_cl.py``, ``agg_cl.py``,
``replay_contacts.py`` …) are copied with only the run-path change; they resolve their inputs relative to
their own directory and read ``seeds.json``. ``p/GENERATED.json`` lists source md5s and every count.

usage (verifier)::

    R=$(.venv-openpi-live/bin/python -m common.run_output)
    .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t54c_rollout --run-dir $R \\
        --src-p <T53 run>/verify/T53/p
    $R/verify/T54c/p/make_snapshot.sh      # once, CPU
    nohup setsid $R/verify/T54c/p/launch_t54c.sh > $R/verify/T54c/launch_t54c.out 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
from typing import Mapping, Optional, Sequence

from AG3S.ag3s.experiments.studies import t54c_records as reg

STEP = "T54c"
#: HEAD the rollouts must run (or a docs-only descendant) — `T54c.task.md` "기록 조건".
BASE_HEAD = "cf38c9dbca7b2c6d78935a67a1edc6fc2ed6d3ac"
SKIP = {"launch_t53.sh", "launch_all.sh", "launch_cl.sh", "build_verify_t53.py", "fig_t53.py",
        "code_md5_t53.txt", ".partial.lock", ".server_start.lock"}
LOCK_OLD, LOCK_NEW = "T53_GPU_LOCK_HELD", "T54C_GPU_LOCK_HELD"
#: the six files whose md5 T53 recorded at start / end (`AG3S/ag3s/runtime/pipeline.py`: the path
#: T53's md5_end.txt corrected).
MD5_FILES = ("AG3S/ag3s/runtime/pipeline.py", "AG3S/ag3s/config.py", "AG3S/ag3s/fields/curobo_builder.py",
             "AG3S/trajopt/serve_safe.py", "AG3S/trajopt/wire.py", "integration/rby1/bringup/pi05_infer.py")


class TemplateError(SystemExit):
    pass


def _sub(text: str, old: str, new: str, *, count: Optional[int] = None, at_least: int = 1,
         what: str = "") -> tuple[str, int]:
    n = text.count(old)
    if (count is not None and n != count) or n < at_least:
        raise TemplateError(f"{what}: {old!r} found {n}x (expected "
                            f"{count if count is not None else f'>= {at_least}'}) — the source p/ changed")
    return text.replace(old, new), n


def side_of_line(expected: Mapping[int, tuple]) -> str:
    left = [str(e) for e, v in expected.items() if v[1] == "left"]
    right = [str(e) for e, v in expected.items() if v[1] == "right"]
    if len(left) + len(right) != len(expected):
        raise TemplateError("used_arm must be left / right for every episode")
    cases = []
    if left:
        cases.append(f"{'|'.join(left)}) echo left;;")
    if right:
        cases.append(f"{'|'.join(right)}) echo right;;")
    return ("side_of() { case $1 in " + " ".join(cases)
            + ' *) echo "BADEP" >&2; echo BADEP;; esac; }')


def port_lists(seeds: Sequence[Mapping]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {p: [] for p in reg.PORTS}
    for x in seeds:
        out[int(x["port"])].append(str(x["episode"]))
    return out


def md5(path: pathlib.Path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def header(src: pathlib.Path) -> str:
    return (f"# {STEP}: generated by AG3S.ag3s.experiments.studies.t54c_rollout from {src} "
            f"({_dt.date.today().isoformat()}); changes listed in GENERATED.json\n")


def _with_header(text: str, src: pathlib.Path) -> str:
    if text.startswith("#!"):
        first, _, rest = text.partition("\n")
        return first + "\n" + header(src) + rest
    return header(src) + text


MAKE_SNAPSHOT = r"""#!/bin/bash
{header}# Freeze the code the servers / clients run: git archive of the checkout HEAD into $O1 (once, CPU, before launch_t54c.sh).
# Refused unless HEAD is BASE_HEAD or a descendant whose changes since BASE_HEAD are docs-only (*.md, AG3S/ag3s/docs/, doc/, paper/, papers/).
source "$(dirname "$(readlink -f "$0")")/env.sh"
BASE_HEAD={base}
[ -e "$O1" ] && {{ echo "REFUSED: $O1 exists (never overwritten)"; exit 2; }}
cd "$PROJ" || exit 2
H=$(git rev-parse HEAD)
git merge-base --is-ancestor $BASE_HEAD $H || {{ echo "REFUSED: HEAD $H is not $BASE_HEAD or a descendant"; exit 3; }}
NONDOC=$(git diff --name-only $BASE_HEAD $H | grep -vE '\.md$|^AG3S/ag3s/docs/|^doc/|^paper/|^papers/')
[ -n "$NONDOC" ] && {{ echo "REFUSED: non-docs changes since $BASE_HEAD:"; echo "$NONDOC"; exit 3; }}
mkdir -p "$O1" && git archive $H | tar -x -C "$O1" || {{ echo "REFUSED: git archive failed"; exit 4; }}
{{ echo $H; git status --short; }} > $V/snapshot_head.txt
$P/code_md5.sh > $P/code_md5_t54c.txt
$P/md5_files.sh start > $V/md5_start.txt
echo "snapshot $H -> $O1 ($(wc -l < $P/code_md5_t54c.txt) md5 lines) $(date -Iseconds)"
"""

MD5_FILES_SH = r"""#!/bin/bash
{header}# usage: md5_files.sh start|end  -> stdout. md5 of the six serving files in the main tree and in the snapshot.
source "$(dirname "$(readlink -f "$0")")/env.sh"
F="{files}"
echo "# $1 $(date -Iseconds) main-tree HEAD=$(git -C $PROJ rev-parse HEAD) snapshot=git archive $(head -1 $V/snapshot_head.txt 2>/dev/null)"
echo "## main tree"; (cd $PROJ && md5sum $F)
echo "## snapshot"; (cd $O1 && md5sum $F)
"""


def write_rollout_p(src_p: pathlib.Path, run_dir: pathlib.Path, *,
                    expected: Optional[Mapping[int, tuple]] = None, check_storage: bool = True,
                    force: bool = False) -> dict:
    """Write ``<run_dir>/verify/T54c/p/`` (+ ``../seeds.json``) from the T53 ``p/`` (module doc).
    Returns the GENERATED record. Refuses an existing ``p/`` unless `force`."""
    src_p = pathlib.Path(src_p).resolve()
    expected = dict(reg.EXPECTED if expected is None else expected)
    if len(expected) != reg.N_SLOTS:
        raise TemplateError(f"{len(expected)} episodes in the registry (need {reg.N_SLOTS})")
    env_src = (src_p / "env.sh").read_text()
    old_run = next((ln.split("=", 1)[1].strip() for ln in env_src.splitlines() if ln.startswith("RUN=")),
                   None)
    if not old_run:
        raise TemplateError(f"{src_p}/env.sh has no RUN= line")
    if check_storage:
        from common.run_output import resolve_output_path

        try:
            v = resolve_output_path(reg.VERIFY_SUBDIR, run_dir=run_dir, default_name=reg.VERIFY_SUBDIR)
        except ValueError as exc:
            raise TemplateError(f"--run-dir {run_dir}: {exc}") from exc
        run = pathlib.Path(run_dir).resolve()
    else:
        run = pathlib.Path(run_dir)
        v = run / reg.VERIFY_SUBDIR
    if str(run) == old_run:
        raise TemplateError("--run-dir is the T53 run")
    dst = v / "p"
    if dst.exists() and any(dst.iterdir()) and not force:
        raise TemplateError(f"{dst} exists and is not empty (never overwritten; --force to replace)")
    dst.mkdir(parents=True, exist_ok=True)
    seeds = reg.seeds_table(expected)
    lists = port_lists(seeds)
    p0, p1 = reg.PORTS
    record: dict = {"step": STEP, "generated_kst": _dt.datetime.now(
        _dt.timezone(_dt.timedelta(hours=9))).isoformat(), "src_p": str(src_p), "old_run": old_run,
        "run": str(run), "ports": list(reg.PORTS), "base_head": BASE_HEAD, "seeds": seeds,
        "files": {}, "skipped": []}
    for f in sorted(src_p.iterdir()):
        if f.is_dir() or f.name in SKIP:
            record["skipped"].append(f.name)
            continue
        text = f.read_text()
        subs = {}
        text, subs["run_verify"] = _sub(text, old_run + "/verify/T53", str(run) + "/" + reg.VERIFY_SUBDIR,
                                        at_least=0, what=f.name)
        text, subs["run"] = _sub(text, old_run, str(run), at_least=0, what=f.name)
        if f.suffix == ".sh":
            text, subs["lock"] = _sub(text, LOCK_OLD, LOCK_NEW, at_least=0, what=f.name)
        if f.name == "env.sh":
            text, subs["V"] = _sub(text, "V=$RUN/verify/T53", "V=$RUN/" + reg.VERIFY_SUBDIR, count=1,
                                   what=f.name)
            side = [ln for ln in text.splitlines() if ln.startswith("side_of()")]
            if len(side) != 1:
                raise TemplateError("env.sh: expected one side_of() line")
            text, subs["side_of"] = _sub(text, side[0], side_of_line(expected), count=1, what=f.name)
            text, subs["port_ok"] = _sub(text, 'port_ok() { [ "$1" = 8290 ] || [ "$1" = 8291 ]; }',
                                         f'port_ok() {{ [ "$1" = {p0} ] || [ "$1" = {p1} ]; }}',
                                         count=1, what=f.name)
            text, subs["port_comment"] = _sub(text, "# T53 ports: 8290 and 8291 ONLY",
                                              f"# T54c ports: {p0} and {p1} ONLY", count=1, what=f.name)
        if f.name in ("run_chain.sh", "start_server.sh"):
            text, subs["port_msg"] = _sub(text, "(8290 / 8291 only)", f"({p0} / {p1} only)", count=1,
                                          what=f.name)
        if f.name == "run_chain.sh":
            text, subs["md5_list"] = _sub(text, "code_md5_t53.txt", "code_md5_t54c.txt", count=3,
                                          what=f.name)
            text, subs["lock_count"] = _sub(text, LOCK_NEW, LOCK_NEW, count=3, what=f.name)  # header comment + 2 checks
        if f.name == "run_client.sh":
            text, subs["list_msg"] = _sub(text, "not in the T53 list", "not in the T54c list", count=1,
                                          what=f.name)
        if f.name == "update_partial.py":
            text, subs["step"] = _sub(text, '"step": "T53"', '"step": "T54c"', count=1, what=f.name)
        if f.suffix == ".sh":
            text = _with_header(text, f)
        out = dst / f.name
        out.write_text(text)
        out.chmod(f.stat().st_mode & 0o777)
        record["files"][f.name] = {"src_md5": md5(f), "dst_md5": md5(out),
                                   "substitutions": {k: n for k, n in subs.items() if n}}
    # launch_t54c.sh from launch_t53.sh
    lsrc = src_p / "launch_t53.sh"
    text = lsrc.read_text()
    subs = {}
    for old_line, port in (("for e in 1840 1842 1860 1862 1848 1866 1804 1822; do", p0),
                           ("for e in 1809 1843 1845 1815 1849 1851 1853 1839; do", p1)):
        text, subs[f"list_{port}"] = _sub(text, old_line, f"for e in {' '.join(lists[port])}; do", count=1,
                                          what="launch_t53.sh")
    text, subs["lock"] = _sub(text, LOCK_OLD, LOCK_NEW, count=1, what="launch_t53.sh")
    text, subs["srv_count"] = _sub(text, "--port 829[01]", f"--port 829[{p0 % 10}{p1 % 10}]", count=1,
                                   what="launch_t53.sh")
    text, subs["port0"] = _sub(text, "8290", str(p0), what="launch_t53.sh")
    text, subs["port1"] = _sub(text, "8291", str(p1), what="launch_t53.sh")
    text, subs["name"] = _sub(text, "launch_t53", "launch_t54c", what="launch_t53.sh")
    text = _with_header(text, lsrc).replace("# T53 rollouts (verifier", "# T54c rollouts (copy of T53 p/launch_t53.sh; verifier", 1)
    out = dst / "launch_t54c.sh"
    out.write_text(text)
    out.chmod(0o755)
    record["files"]["launch_t54c.sh"] = {"src": "launch_t53.sh", "src_md5": md5(lsrc), "dst_md5": md5(out),
                                         "substitutions": subs}
    for name, body in (("make_snapshot.sh", MAKE_SNAPSHOT.format(header=header(src_p), base=BASE_HEAD)),
                       ("md5_files.sh", MD5_FILES_SH.format(header=header(src_p), files=" ".join(MD5_FILES)))):
        out = dst / name
        out.write_text(body)
        out.chmod(0o755)
        record["files"][name] = {"new": True, "dst_md5": md5(out)}
    seeds_p = v / "seeds.json"
    if seeds_p.exists() and not force:
        raise TemplateError(f"{seeds_p} exists (never overwritten)")
    seeds_p.write_text(json.dumps({
        "_comment": (f"{STEP} pre-registered policy seeds (t54c_rollout, written before any rollout). One seed "
                     f"per episode, seed = episode_index * 10 + 1. Port split: layout slot even -> {p0}, odd -> "
                     f"{p1}. Episodes = t54c_records.EXPECTED (selection rule in T54c.task.md). Arm C1a = "
                     "current serving, clustering.placed_no_seed_certify off (default)."),
        "episodes": seeds}, indent=1))
    record["seeds_json"] = str(seeds_p)
    (dst / "GENERATED.json").write_text(json.dumps(record, indent=1))
    return record


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, help="the new T54c run (results/development/<run>)")
    ap.add_argument("--src-p", required=True, help="the T53 run's verify/T53/p directory (read only)")
    ap.add_argument("--force", action="store_true", help="replace an existing p/ (never by default)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    rec = write_rollout_p(pathlib.Path(args.src_p), pathlib.Path(args.run_dir), force=args.force)
    print(json.dumps({k: rec[k] for k in ("run", "ports", "seeds_json")}, indent=1))
    print(json.dumps({f: r.get("substitutions") for f, r in rec["files"].items()}, indent=1))


__all__ = ["BASE_HEAD", "MD5_FILES", "TemplateError", "port_lists", "side_of_line", "write_rollout_p"]


if __name__ == "__main__":
    main()
