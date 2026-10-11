"""T39 S2-1: write the s2_1 section of T39.verify.partial.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import fcntl, json, pathlib, datetime, glob
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39')); S = T / "s21"
PARTIAL = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T39.verify.partial.json'))
def iso(p): return datetime.datetime.fromisoformat(pathlib.Path(p).read_text().splitlines()[0].strip())
srv = {}
for g in ("E0_a", "E0_b", "E3b_a", "E3b_b"):
    rec = pathlib.Path(sorted(glob.glob(str(S / g / "server_*")))[0])
    before = int((rec / "gpu_used_before_mib").read_text())
    samples = [int(l.split()[1]) for l in (rec / "gpu_samples.txt").read_text().splitlines() if len(l.split()) == 2]
    c = g.split("_")[0]; name = f"{c}_ep1807_s18071"
    log = (S / g / f"chain_{rec.name.split('_')[1]}.log").read_text()
    srv[g] = {"server_dir": str(rec), "port": int(rec.name.split("_")[1]), "gpu_used_before_mib": before, "gpu_peak_total_mib": max(samples),
              "gpu_peak_delta_mib": max(samples) - before, "n_samples_2s": len(samples),
              "startup_s": int(log.split("startup_s=")[1].split()[0]),
              "run_wall_s": (iso(S / g / f"{name}.end") - iso(S / g / f"{name}.start")).total_seconds(),
              "run_dir": str(S / g / name)}
cmp_ = {c: json.loads((S / f"compare_{c}.json").read_text()) for c in ("E0", "E3b")}
s21 = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "episode": 1807, "seed": 18071, "servers_one_at_a_time": True, "xla_flags": "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0",
       "servers": srv,
       "bit_identical": {c: {"all": d["all_bit_identical"], "traj_npz": {k: v["equal"] for k, v in d["traj_npz"].items() if isinstance(v, dict)},
                             "frames": {k: v["equal"] for k, v in d["frames"].items()},
                             "server_record": {k: v["equal"] for k, v in (d.get("server_record") or {}).items() if isinstance(v, dict)},
                             "file": str(S / f"compare_{c}.json")} for c, d in cmp_.items()},
       "gpu_note_fact": "other-session usage 105544 MiB during E0_a/E0_b, 0 MiB from E3b_a on (released between 09:24:46 and 09:25 UTC)",
       "est_gb_set": {"E0": 10, "E3b": 22, "rule": "peak delta GiB x 1.15, rounded up"}}
with open(T / "p/.partial.lock", "a") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    cur = json.loads(PARTIAL.read_text()); cur["s2_1"] = s21
    tmp = PARTIAL.with_suffix(".json.tmp"); tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False)); tmp.replace(PARTIAL)
print(json.dumps({g: {k: v[k] for k in ("gpu_peak_delta_mib", "run_wall_s", "startup_s")} for g, v in srv.items()}), {c: v["all"] for c, v in s21["bit_identical"].items()})
