#!/bin/bash
# V11AB: audit finished runs for a dropped server connection (client log "[safe] N chunks: ... error N>0 / timeout / stale", which check_run.py does not fail), move them to <stage>/failed/ and rerun them
# with the same run_chain.sh (one server, port 8286; at most 3 passes). usage: redo_bad.sh
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/p/env.sh
ARM=V11AB; R=$V/runs/$ARM
for pass in 1 2 3; do
  ITEMS=$(python3 - <<'P'
import re, pathlib, shutil, datetime
R = pathlib.Path("${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/runs/V11AB"); P = pathlib.Path("${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/p/lists")
spec = {}
for l in (P / "hurdle.txt").read_text().split():
    ep, sd, st = l.split(":")[:3]; spec[(st, ep, sd)] = l
for l in (P / "clear.txt").read_text().split():
    ep, sd, st = l.split(":")[:3]; spec[(st, ep, sd)] = l
ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S"); out = []
for st in ("H", "B"):
    for done in sorted((R / st).glob("V11AB_ep*_s*.done")):
        nm = done.stem; log = (R / st / f"{nm}.client.log").read_text(errors="replace")
        m = re.search(r"\[safe\] (\d+) chunks: safe (\d+), unsafe (\d+), timeout (\d+), stale (\d+), error (\d+)", log)
        if m is None or int(m.group(4)) or int(m.group(5)) or int(m.group(6)):
            fd = R / st / "failed"; fd.mkdir(exist_ok=True)
            for suf in ("", ".client.log", ".start", ".end", ".check.json", ".done"):
                f = R / st / f"{nm}{suf}"
                if f.exists():
                    shutil.move(str(f), str(fd / f"{f.name}.{ts}"))
            ep, sd = re.match(r"V11AB_ep(\d+)_s(\d+)", nm).groups(); out.append(spec[(st, ep, sd)])
print(" ".join(out))
P
)
  [ -z "$ITEMS" ] && { echo "pass $pass: no bad runs"; break; }
  echo "pass $pass $(date -Iseconds): rerun [$(echo $ITEMS | tr ' ' '\n' | cut -d: -f1-3 | tr '\n' ' ')]"
  # hurdle items first, clear items after, each with its own chain call (run_chain.sh takes one OUTDIR per call; stage is in the item)
  EST_GB=28 $P/run_chain.sh $ARM 8286 $R "$ITEMS"; echo "redo pass $pass rc=$? $(date -Iseconds)"
done
echo "redo end $(date -Iseconds)"
