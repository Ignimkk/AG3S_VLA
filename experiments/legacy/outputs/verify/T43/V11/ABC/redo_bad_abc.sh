#!/bin/bash
# V11ABC: audit finished runs for a dropped server connection ([safe] error/timeout/stale > 0 or no line) and for missing .done; move bad ones to <stage>/failed/ and rerun (port ${PORT:-8286}, stages ${STAGES:-H B}, <= 3 passes).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/p/env.sh
PORT=${PORT:-8286}; export STAGES=${STAGES:-"H B"}; ARM=V11ABC; R=$V/runs/$ARM; CH=$V/ABC/run_chain_abc.sh
for pass in 1 2 3; do
  ITEMS=$(python3 - <<'P'
import re, pathlib, shutil, datetime
R = pathlib.Path("${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/runs/V11ABC"); P = pathlib.Path("${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/p/lists")
spec = {}
for f in ("hurdle.txt", "clear.txt"):
    for l in (P / f).read_text().split():
        ep, sd, st = l.split(":")[:3]; spec[(st, ep, sd)] = l
ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S"); out = []
import os
for (st, ep, sd), item in spec.items():
    if st not in os.environ["STAGES"].split(): continue
    nm = f"V11ABC_ep{ep}_s{sd}"; d = R / st
    bad = False
    if not (d / f"{nm}.done").exists():
        bad = True   # missing or never completed
        if not (d / f"{nm}.start").exists(): out.append(item); continue
    else:
        log = (d / f"{nm}.client.log").read_text(errors="replace")
        m = re.search(r"\[safe\] (\d+) chunks: safe (\d+), unsafe (\d+), timeout (\d+), stale (\d+), error (\d+)", log)
        bad = m is None or int(m.group(4)) or int(m.group(5)) or int(m.group(6))
    if bad:
        (d / "failed").mkdir(exist_ok=True)
        for suf in ("", ".client.log", ".start", ".end", ".check.json", ".done", ".safe_audit.txt"):
            f = d / f"{nm}{suf}"
            if f.exists(): shutil.move(str(f), str(d / "failed" / f"{f.name}.{ts}"))
        out.append(item)
print(" ".join(out))
P
)
  [ -z "$ITEMS" ] && { echo "pass $pass: no bad/missing runs"; break; }
  echo "pass $pass $(date -Iseconds): rerun [$(echo $ITEMS | tr ' ' '\n' | cut -d: -f1-3 | tr '\n' ' ')]"
  EST_GB=28 $CH $ARM $PORT $R "$ITEMS"; echo "redo pass $pass rc=$? $(date -Iseconds)"
done
echo "redo end $(date -Iseconds)"
