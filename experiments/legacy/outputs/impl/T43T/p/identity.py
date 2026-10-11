"""T43 T — off bit-identity probe: the same harness against whichever `benchmark` is on PYTHONPATH.
Writes an npz: refined chunks of held deadlock chunks (TO, flags off) and the grounding replay's per-chunk
manipulated record + decision (fix 3 flags off) for T41 c C1982 seq 1-47 and D1982 seq 1-47."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import recov_replay as RR
import ground_replay as G

TO_CHUNKS = (("T42_R1908", 16), ("T42_R1908", 18), ("T42_L1992", 16), ("T42_L1992", 24))
GR_RUNS = (_ag3s_workspace_path('outputs/verify/T41/c/runs/E3b/C/E3b_ep1982_s19822'),
           _ag3s_workspace_path('outputs/verify/T41/c/runs/E3b/D/E3b_ep1982_s19822'))


def main(out):
    import benchmark
    arrays, meta = {}, {"benchmark": benchmark.__file__}
    for key, seq in TO_CHUNKS:
        r = RR.Run(key).solve(seq)
        arrays[f"to_{key}_{seq}"] = r["chunk"]
        meta[f"to_{key}_{seq}"] = [r["status"], r["clearance_mm"], r["reasons"]]
    for rd in GR_RUNS:
        rows = []
        for seq, s, m, last, ref in G.replay(rd, overrides={"clustering": {"subtask_gate": False}}, seqs=range(1, 48)):
            rows.append([seq, None if m is None else m.record(), None if last is None else
                         [last.mode, last.rank, last.reason, last.streak, last.frozen, last.hand_occluded], ref])
        meta["gr_" + rd.split("/")[-2]] = json.loads(json.dumps(rows, default=str))
    np.savez(out, meta=json.dumps(meta, sort_keys=True, default=str), **arrays)


if __name__ == "__main__":
    main(sys.argv[1])
