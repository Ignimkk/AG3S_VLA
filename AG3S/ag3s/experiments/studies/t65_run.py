"""T65 — orchestrate the fixed-input replay: one `serve_safe` per (attention source × record group), the recorded
requests re-sent in order (`t65_replay_client`), constraint chunks written under this run (`handoff/T65.task.md`).

Server argv = the recorded one, with only ``--attention-source`` / ``--port`` / ``--run-dir`` / ``--record-constraints``
replaced:
* regression groups ``REG_k0`` / ``REG_k1``: ``server_proc.txt`` of the group's first server (the k-groups' two
  servers differ only in ``--port``);
* occlusion groups ``OCC/k0`` / ``OCC/k1``: ``BASE`` + ``SAFE`` of ``occ/launch_occ_CORE.sh`` (no server_proc.txt
  there), evaluated with the same ``SN`` / ``RUN`` / ``PORT``.
Code and environment = the core-evaluation snapshot (``$CORE/snapshot_core``), as the recorded servers ran.

usage: python t65_run.py --out B_RUN --sources noun_target:L10H7 action:last:L8H2 action:mean:L10H7
                         [--groups REG_k0 REG_k1 OCC/k0 OCC/k1] [--port 8297] [--limit N] [--clients NAME ...]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import socket
import subprocess
import sys
import time

CORE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261028T224856+0900_b3b69311f948_8784a1265958")
V = CORE / "verify/AG3S-CORE-EVAL"
SN = CORE / "snapshot_core"
CO = pathlib.Path("/mnt/dev/AG3S_VLA-development")
PY = CO / ".venv-openpi-live/bin/python"
GROUPS = ("REG_k0", "REG_k1", "OCC/k0", "OCC/k1")
REPLACED = ("--attention-source", "--port", "--run-dir", "--record-constraints")


def tag(source: str) -> str:
    return source.replace(":", "_")


def env_for(out: pathlib.Path) -> dict:
    e = dict(os.environ)
    e.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
              "XLA_FLAGS": "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0",
              "PYTHONUNBUFFERED": "1", "AG3S_STORAGE_ROOT": "/mnt/dev/AG3S_VLA-storage",
              "OPENPI_DATA_HOME": str(CO / ".cache/openpi"), "JAX_COMPILATION_CACHE_DIR": str(CO / ".cache/jax"),
              "AG3S_RUN_DIR": str(out), "MUJOCO_GL": "osmesa",
              "PYTHONPATH": ":".join([str(SN), str(SN / "integration/rby1/manipulation/src"),
                                      str(SN / "third_party/openpi/src"),
                                      str(SN / "third_party/openpi/packages/openpi-client/src")])})
    return e


def recorded_argv(group: str) -> list[str]:
    """The serve_safe arguments after ``-m AG3S.trajopt.serve_safe`` as the recorded server ran them."""
    if group.startswith("REG"):
        sp = sorted((V / group / "runs").glob("server_*/server_proc.txt"))[0]
        cmd = shlex.split(sp.read_text().strip().splitlines()[-1])
        return cmd[cmd.index("AG3S.trajopt.serve_safe") + 1:]
    sh = (V / "occ/launch_occ_CORE.sh").read_text().splitlines()
    block = "\n".join(l for l in sh if l.startswith(("CKPT=", "BASE=", "SAFE=", "  --")))
    script = (f"AG3S_STORAGE_ROOT=/mnt/dev/AG3S_VLA-storage; SN={SN}; RUN={CORE}; PORT=0\n{block}\n"
              'echo "$BASE $SAFE"')
    return shlex.split(subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout)


def server_argv(group: str, source: str, port: int, out: pathlib.Path, rec: pathlib.Path) -> list[str]:
    a, out_a, skip = recorded_argv(group), [], False
    for x in a:
        if skip:
            skip = False
            continue
        if x in REPLACED:
            skip = True
            continue
        out_a.append(x)
    return out_a + ["--attention-source", source, "--port", str(port), "--run-dir", str(out),
                    "--record-constraints", str(rec)]


def clients(group: str) -> list[pathlib.Path]:
    if group.startswith("REG"):
        return sorted(p for p in (V / group / "runs/B_ROI_FPR").glob("B_ROI_FPR_ep*") if p.is_dir())
    order = sorted((V / group).glob("server_*/order.txt"))[0].read_text().split()
    return [V / group / "runs" / n for n in order]


def port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_one(out: pathlib.Path, source: str, group: str, port: int, limit, only) -> dict:
    d = out / "verify/T65" / tag(source) / group.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    if port_open(port):
        raise SystemExit(f"port {port} busy")
    argv = server_argv(group, source, port, out, d / "server_constraints")
    (d / "server_argv.json").write_text(json.dumps(argv, indent=1))
    env = env_for(out)
    log = open(d / "serve_safe.log", "w")
    srv = subprocess.Popen([str(PY), "-m", "AG3S.trajopt.serve_safe", *argv], cwd=SN, env=env, stdout=log,
                           stderr=subprocess.STDOUT)
    t0 = time.time()
    while not port_open(port):
        if srv.poll() is not None:
            raise SystemExit(f"server died (see {d / 'serve_safe.log'})")
        if time.time() - t0 > 1800:
            srv.terminate()
            raise SystemExit("server not up in 30 min")
        time.sleep(5)
    print(f"[t65] {tag(source)} {group} server up {time.time() - t0:.0f}s", flush=True)
    res = []
    try:
        for c in clients(group):
            if only and c.name not in only:
                continue
            (d / "order.txt").open("a").write(c.name + "\n")
            cmd = [str(PY), str(CO / "AG3S/ag3s/experiments/studies/t65_replay_client.py"), "--client-dir", str(c),
                   "--port", str(port)] + (["--limit", str(limit)] if limit else [])
            r = subprocess.run(cmd, cwd=SN, env=env, capture_output=True, text=True)
            (d / f"{c.name}.client.log").write_text(r.stdout + r.stderr)
            if r.returncode != 0:
                raise SystemExit(f"replay client failed for {c.name} (see {d / (c.name + '.client.log')})")
            res.append(json.loads(r.stdout.strip().splitlines()[-1]))
            print(f"[t65] {c.name}: {res[-1]['sent']} requests", flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            srv.kill()
    (d / "replay.json").write_text(json.dumps(res, indent=1))
    return {"dir": str(d), "clients": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--port", type=int, default=8297)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clients", nargs="*", default=None)
    a = ap.parse_args()
    for s in a.sources:
        for g in a.groups:
            print(json.dumps(run_one(a.out, s, g, a.port, a.limit, a.clients))[:300], flush=True)


if __name__ == "__main__":
    main()
