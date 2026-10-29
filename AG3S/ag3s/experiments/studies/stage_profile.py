"""AG3S-GPU-STAGES M1 — where does target grounding spend its time? (offline replay on recorded inputs, not closed-loop)

The server records, per chunk, the attention point cloud grounding consumed (`attention_cloud`, `attention_values`,
`attention_raw`) and the support mask it excluded (`support_mask`). Those are fed, chunk by chunk in server order,
into `ground_target` with the clustering settings of the recorded server argv (`server_proc.txt`, built through
`serve_safe.ag3s_from_args`) and that AG3S object's own `TargetConfirm`. The `note_*` inputs the pipeline gives the
confirm (hand, latch, support planes, subtask) are **not** replayed, so the confirm's decisions can differ from the
server's; the parts below the confirm (seeds, growth, components, split, scoring) see the server's exact input.

Timed parts (exclusive wall ms; a part's own time excludes the timed parts nested in it):
  seeds        `extract_seeds`
  grow         `grow_region` (KD-tree build + wave queries)
  components   `dbscan` / `connected_components_3d`
  split        `_split_groups` (attention split)
  score        `_score_cluster`, all clusters
  confirm      `TargetConfirm.select` + `TargetConfirm.observe` + `follow_kept`
  other        the rest of `ground_target`
Per chunk also: the replay's seed / target indices against the recorded ones (fidelity), and SHA-1 of the
grown set and labels (CPU vs GPU equivalence when a device path exists).

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa .venv-openpi-live/bin/python \
      -m AG3S.ag3s.experiments.studies.stage_profile --index <e4_index.json> --out <dir> [--keys …]
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
import time

import numpy as np

PARTS = ("seeds", "grow", "components", "split", "score", "confirm", "other")


class Clock:
    """Exclusive wall time per part (a stack: nested parts are subtracted from their parent)."""

    def __init__(self):
        self.t: dict[str, float] = {}
        self.stack: list[list] = []

    def wrap(self, part, fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            self.stack.append([part, 0.0])
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = (time.perf_counter() - t0) * 1000.0
                _, nested = self.stack.pop()
                self.t[part] = self.t.get(part, 0.0) + dt - nested
                if self.stack:
                    self.stack[-1][1] += dt
        return inner

    def pop(self):
        out, self.t = self.t, {}
        return out


def _sha(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def install(clock: Clock, sink: dict):
    from AG3S.ag3s.stages import target_grounding as tg

    tg.extract_seeds = clock.wrap("seeds", tg.extract_seeds)

    def grow(fn):
        def inner(*a, **k):
            out = fn(*a, **k)
            sink["grown_sha"], sink["n_grown"] = _sha(out), int(np.size(out))
            return out
        return inner

    def comp(fn):
        def inner(*a, **k):
            out = fn(*a, **k)
            sink["labels_sha"], sink["n_labels"] = _sha(out), int(len(np.unique(out))) if np.size(out) else 0
            return out
        return inner

    tg.grow_region = clock.wrap("grow", grow(tg.grow_region))
    tg.dbscan = clock.wrap("components", comp(tg.dbscan))
    tg.connected_components_3d = clock.wrap("components", comp(tg.connected_components_3d))
    tg._split_groups = clock.wrap("split", tg._split_groups)
    tg._score_cluster = clock.wrap("score", tg._score_cluster)
    for name in ("select", "observe", "follow_kept"):
        setattr(tg.TargetConfirm, name, clock.wrap("confirm", getattr(tg.TargetConfirm, name)))
    return tg


def server_argv(first_chunk: pathlib.Path) -> list[str]:
    from AG3S.trajopt.experiments import ta_offline as TA

    _, cmdline = TA.server_args(first_chunk.parent)
    import shlex
    return shlex.split(cmdline)


def chunks_of(entry: dict) -> list[pathlib.Path]:
    seg = entry["server_segment"]
    if isinstance(seg, str):
        import ast
        seg = ast.literal_eval(seg)
    first = pathlib.Path(seg["first"])
    i0 = int(first.stem.split("_")[1])
    return [first.parent / f"chunk_{i0 + n:05d}.npz" for n in range(int(seg["n"]))]


def run_episode(key: str, entry: dict, clock: Clock, sink: dict, tg, repeat: int = 1) -> dict:
    from AG3S.ag3s.types import AttentionPointCloud, PointCloud
    from AG3S.trajopt import serve_safe

    files = chunks_of(entry)
    argv = server_argv(files[0])
    args = serve_safe.build_parser().parse_args(argv)
    ag3s = serve_safe.ag3s_from_args(args, plan_horizon_steps=None, rows_per_step=None)
    cfg = ag3s.config
    out = {"key": key, "server_argv": argv, "chunks": []}
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            if "attention_cloud" not in z.files:
                out["chunks"].append({"file": f.name, "skipped": "no attention_cloud in the record"})
                continue
            pts = np.asarray(z["attention_cloud"], np.float64)
            att = np.asarray(z["attention_values"], np.float32)
            raw = np.asarray(z["attention_raw"], np.float32) if "attention_raw" in z.files else None
            sup = np.asarray(z["support_mask"], bool)
            rec_seeds = np.asarray(z["seed_indices"]) if "seed_indices" in z.files else None
            rec_target = (np.asarray(z["target_point_indices"]) if "target_point_indices" in z.files
                          else np.zeros(0, np.int64))
            summ = json.loads(str(z["summary_json"]))
        apc = AttentionPointCloud(PointCloud(points=pts), att, raw)
        rows = []
        for r in range(repeat):
            sink.clear()
            clock.pop()
            t0 = time.perf_counter()
            res = tg.ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                                   seed_threshold=cfg.attention.seed_threshold, exclude_mask=sup,
                                   min_radius=cfg.geometry.min_radius, timestamp=0.0,
                                   confirm=ag3s._target_confirm if r == 0 else None)
            wall = (time.perf_counter() - t0) * 1000.0
            parts = clock.pop()
            parts["other"] = wall - sum(parts.values())
            rows.append({"wall_ms": wall, "parts": parts})
        tgt = None if res.target is None else np.asarray(res.target.point_indices)
        seeds = np.asarray(res.seed_indices) if res.seed_indices is not None else np.zeros(0)
        out["chunks"].append({
            "file": f.name, "n_points": int(len(pts)), "n_seeds": int(seeds.size),
            "server_ms": float((summ.get("profile_ms") or {}).get("target_grounding", np.nan)),
            "runs": rows, "status": res.status.value, **dict(sink),
            "seeds_sha": _sha(seeds),
            "seeds_match_record": None if rec_seeds is None else bool(np.array_equal(seeds, rec_seeds)),
            "target_sha": None if tgt is None else _sha(tgt),
            "target_match_record": bool(tgt is not None and np.array_equal(tgt, rec_target))
                                   or bool(tgt is None and rec_target.size == 0),
            "n_clusters": len(res.clusters or ()),
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="json {runs: {key: {run_dir, server_segment}}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--repeat", type=int, default=1,
                    help="extra stateless re-runs per chunk (confirm=None) for timing spread")
    cli = ap.parse_args(argv)
    runs = json.load(open(cli.index))["runs"]
    keys = cli.keys or list(runs)
    outdir = pathlib.Path(cli.out)
    outdir.mkdir(parents=True, exist_ok=True)
    clock, sink = Clock(), {}
    tg = install(clock, sink)
    for key in keys:
        t0 = time.perf_counter()
        res = run_episode(key, runs[key], clock, sink, tg, repeat=max(1, cli.repeat))
        path = outdir / (key.replace("/", "__") + ".json")
        path.write_text(json.dumps(res, indent=1, default=float))
        print(f"{key}: {len(res['chunks'])} chunks, {time.perf_counter() - t0:.1f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
