"""T57-A — flag-on parity and shadow cost of the serving T57 additions (A1 writes it; the verifier / lead runs it).

`AG3S/ag3s/docs/handoff/T57-A.task.md` §2 · §3 · §4: GPU checks are prepared as commands, not run by the
implementer. Every GPU stage drives the **serving** objects — `attention_policy.AttentionPolicy` with the
`attention_source.JaxPrefixBackend` (`--attention-source` / `--task-gate` path), on the attention model
`serve_safe` loads (`pi05_attention.load_policy`, ``return_attn_probs=True``) — with a stub in place of the
action policy (only its `_input_transform` is used; the served action chunk is not part of these checks).

Stages
------
``noun`` (GPU)     T54c recorded frames (`t54c_records`, the T54c run): the server's ``noun_target:L10H7`` maps
                   vs the T54c noun extract (`t54c_extract` → T49 format, query ``target``, cell L10H7 —
                   the `t54_arms.SOURCES["noun"]` map). The extract stores float16: reported are the exact
                   match rate of ``float16(server)`` with the stored value, max |Δ| (server float32 vs stored),
                   per-camera argmax agreement, and the token index / prefix row of the target noun. Writes
                   ``noun/<key>.npz`` (server maps per selected frame) for ``replay``.
``replay`` (CPU)   AG3S on those frames with the server's maps, arm **N**, condition ``current`` (`t54_study.
                   _job`, the T54c replay job unchanged; c1a profile from the record's server_proc.txt, legacy
                   ESDF on CPU) vs the stored T54c N rows (`<T54c run>/verify/T54c/replay/test/rows/
                   <key>__current.jsonl`). Two inputs: ``f16`` (the server map rounded to the stored float16 —
                   rows must equal the stored rows when the maps match) and ``f32`` (the map as served).
                   Timing fields are excluded from the row comparison.
``probe`` (GPU)    T55 **test** E0 re-sim frames (`verify/T55/resim/test/<key>__E0.npz`, the prompt of the
                   re-sim): the served v5 row and p (`result["task_probe"]`) vs the offline T55 features
                   (`verify/T55/feat/test/<key>__E0__orig.npz` ``noun_v[:, 0, layer 5]``) and `t55_study`'s p
                   (`load_probe` + `noun_matrix("v5")`). The V row is gathered both from the action-source
                   prefix cache (the gate arm's configuration) and from the noun-source pass; the noun pass's
                   kv_L4 subtask feature / label is compared with the sampler's (stage 1 must not move it).
``cost`` (GPU)     per-request wall time of `AttentionPolicy.infer` (stub action policy) on the same frames for
                   the configurations default (no flag) · probe only (= shadow / gate on the default source) ·
                   noun source · noun source + probe: median / p95 ms and the paired per-frame extra over the
                   default (warm-up calls excluded). The SafePolicy side of the gate (`_task_gate`) is timed on
                   CPU in the same stage.

Usage (one GPU job at a time, the shared lock, nothing else on the GPU)::

    cd <checkout>; source scripts/setup/env.sh; cd <checkout>   # env.sh cd's to the main checkout
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"
    M=AG3S.ag3s.experiments.studies.t57a_parity; L=/tmp/ag3s-vla-gpu-validation.lock
    T54C=$AG3S_STORAGE_ROOT/results/development/20261014T133115+0900_cf38c9dbca7b_5697bbdb5190
    flock $L .venv-openpi-live/bin/python -m $M --stage noun  --t54c-run $T54C
    flock $L .venv-openpi-live/bin/python -m $M --stage probe
    flock $L .venv-openpi-live/bin/python -m $M --stage cost  --t54c-run $T54C
    MUJOCO_GL=osmesa OMP_NUM_THREADS=2 .venv-ag3s/bin/python -m $M --stage replay --t54c-run $T54C

Outputs: ``$AG3S_RUN_DIR/verify/T57-A/{noun,replay,probe,cost}/`` (+ one ``<stage>.json`` each). Numbers only —
the verdict is the verifier's.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

STEP = "T57-A"
GPU_STAGES = ("noun", "probe", "cost")
STAGES = ("noun", "replay", "probe", "cost")
GPU_LOCK = "/tmp/ag3s-vla-gpu-validation.lock"
CONFIG = "pi05_rby1_randomized_pick_place_16d_lora"
#: T54c records used by default (5 records x 5 frames).
DEFAULT_RECORDS = ("ep1806", "ep1811", "ep1813", "ep1814", "ep1816")
DEFAULT_FRAMES = (0, 12, 24, 40, 60)
T55_RUN = "20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c"
DEFAULT_T55_KEYS = ("ep1804", "ep1809")
NOUN_SPEC = "noun_target:L10H7"
TIMING_KEYS = ("ag3s_ms", "ms_lift_ground", "ms_primary_lift_posthoc", "ms_profile", "ms_secondary_lift")
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def kst_now() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat()


def out_dir(args, sub: str = "") -> pathlib.Path:
    run = args.run_dir or os.environ.get("AG3S_RUN_DIR")
    if not run:
        raise SystemExit("set AG3S_RUN_DIR (python -m common.run_output) or pass --run-dir")
    d = pathlib.Path(run) / "verify" / STEP / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: pathlib.Path, payload: Any) -> None:
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm

    path.write_text(json.dumps(bm.jsonable(payload), indent=1, allow_nan=True))
    print(f"wrote {path}")


def run_info(stage: str, args) -> dict:
    from AG3S.ag3s.experiments.studies.t47_head_selection import git_state

    return {"step": STEP, "stage": stage, "argv": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
            "started_kst": kst_now(), "git": git_state(), "python": sys.executable}


# ------------------------------------------------------------------------------ GPU session


def gpu_session(args) -> dict:
    """Guard (lock held by the caller's flock, no other compute process), load the attention model once."""
    from AG3S.ag3s.experiments.studies import t53_extract as ex
    from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne

    if not args.no_lock_check and not ex.lock_is_held(GPU_LOCK):
        raise SystemExit(f"run under: flock {GPU_LOCK} <this command> (--no-lock-check only on the lead's word)")
    pids = ne.gpu_guard(args)  # nvidia-smi empty, XLA env, jax on the GPU
    from AG3S.ag3s.experiments.sources.pi05_attention import load_policy
    from AG3S.ag3s.experiments.studies import t51_extract as ex51
    from AG3S.trajopt.attention_source import NounTokens

    ckpt = ex51.checkpoint_path(args)
    t0 = time.perf_counter()
    attn_policy = load_policy(args.config, str(ckpt))
    load_s = time.perf_counter() - t0
    return {"pids": pids, "ckpt": str(ckpt), "attn_policy": attn_policy, "load_s": load_s,
            "tokens": NounTokens.from_config(args.config)}


class _StubPolicy:
    """The served action policy, reduced to its input transform (actions are not part of these checks)."""

    def __init__(self, input_transform):
        self._input_transform = input_transform

    @staticmethod
    def infer(obs, **_):
        return {"actions": np.zeros((50, 16), np.float32)}


def serving_policy(s: dict, *, source: Optional[str], probe, subtask_probe: Any = "default", sampler=None):
    """`AttentionPolicy` exactly as `serve_safe.main` builds it for these flags (one shared sampler)."""
    from AG3S.trajopt.attention_policy import AttentionPolicy

    kw: dict = {}
    if source is not None or probe is not None:
        kw = {"attention_source": source, "task_probe": probe, "noun_tokens": s["tokens"]}
    return AttentionPolicy(_StubPolicy(s["attn_policy"]._input_transform), s["attn_policy"]._model,  # noqa: SLF001
                           subtask_probe=subtask_probe, sampler=sampler, **kw)


def step_obs(step, prompt: str) -> dict:
    """A request's policy observation from a recorded step — `pi05_attention.observation_from_step`'s
    dict (images HWC -> CHW, the recorded state as is)."""
    return {"state": step.state, "images": {c: np.transpose(step.images[c], (2, 0, 1)) for c in CAMS},
            "prompt": prompt}


# ------------------------------------------------------------------------------ T54c registry


def t54c_inputs(args, key: str) -> dict:
    """`t54_study.record_inputs` for one T54c test record (paths, target, profile, server_proc)."""
    from AG3S.ag3s.experiments.studies import t54_study as st

    sargv = ["--run-dir", str(out_dir(args, "replay")), "--registry", "t54c", "--set", "test",
             "--stage", "inputs", "--records-run", str(args.t54c_run), "--arms", "N"]
    sargs = st.build_parser().parse_args(sargv)
    sargs._t53_cfg = st.configure_t53(sargs.t53_run, sargs.registry)  # noqa: SLF001
    info = st.record_inputs("test", key, sargs)
    info["_t53_cfg"] = sargs._t53_cfg                                   # noqa: SLF001
    return info


def stored_noun_map(ex: dict, f: int, layer: int = 10, head: int = 7) -> np.ndarray:
    """The T54c noun extract's target map at the cell, ``(3, 16, 16)`` float16 (as stored)."""
    qi = [str(q) for q in ex["query_names"]].index("target")
    return np.asarray(ex["maps"][f, qi, layer, head])


# ------------------------------------------------------------------------------ stage noun


def stage_noun(args) -> dict:
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import load_extract

    s = gpu_session(args)
    ap = serving_policy(s, source=NOUN_SPEC, probe=None)
    d = out_dir(args, "noun")
    per_record = {}
    for key in args.records:
        info = t54c_inputs(args, key)
        run = load_run(info["record"])
        ex = load_extract(pathlib.Path(info["noun"]))
        frames = [f for f in args.frames if f < len(run)]
        served, rows = [], []
        for f in frames:
            step = run.steps[f]
            t0 = time.perf_counter()
            res = ap.infer(step_obs(step, run.prompt))
            ms = (time.perf_counter() - t0) * 1e3
            src = res.get("attention_source", {})
            m = np.stack([np.asarray(res["attention"][c], np.float32) for c in CAMS]) \
                if res.get("attention") else np.zeros((3, 16, 16), np.float32)
            ref16 = stored_noun_map(ex, f)
            ref = ref16.astype(np.float32)
            served.append(m)
            rows.append({
                "frame": f, "t_step": int(step.t_step), "ms": ms, "noun": src.get("noun"),
                "registered_target": info["target"], "noun_ok": src.get("noun_ok"),
                "token_index": src.get("token_index"), "prefix_row": src.get("prefix_row"),
                "stored_token_index": int(ex["token_index"][f, 0]), "stored_prefix_row": int(ex["prefix_row"][f, 0]),
                "f16_exact_share": float((m.astype(np.float16) == ref16).mean()),
                "max_abs_vs_stored": float(np.abs(m - ref).max()),
                "max_abs_f16_rounded": float(np.abs(m.astype(np.float16).astype(np.float32) - ref).max()),
                "max_rel_vs_stored": float(np.abs(m - ref).max() / max(float(np.abs(ref).max()), 1e-30)),
                "argmax_same": [bool(int(np.argmax(m[c])) == int(np.argmax(ref[c]))) for c in range(3)],
                "mass_served": [float(m[c].sum()) for c in range(3)],
                "mass_stored": [float(ref[c].sum()) for c in range(3)],
            })
            print(f"[noun] {key} f{f}: f16 exact {rows[-1]['f16_exact_share']:.4f} max|Δ| "
                  f"{rows[-1]['max_abs_vs_stored']:.3g} token {rows[-1]['token_index']}/"
                  f"{rows[-1]['stored_token_index']} {ms:.0f} ms", flush=True)
        np.savez_compressed(d / f"{key}.npz", frames=np.asarray(frames, np.int64), maps=np.stack(served),
                            prompt=np.asarray(run.prompt), spec=np.asarray(NOUN_SPEC))
        per_record[key] = {"noun_extract": info["noun"], "record": info["record"], "frames": rows}
    flat = [r for v in per_record.values() for r in v["frames"]]
    payload = {**run_info("noun", args), "checkpoint": s["ckpt"], "load_s": s["load_s"],
               "gpu_compute_pids_before": s["pids"], "spec": NOUN_SPEC, "records": per_record,
               "summary": {"n_frames": len(flat),
                           "f16_exact_share_min": min((r["f16_exact_share"] for r in flat), default=None),
                           "max_abs_vs_stored_max": max((r["max_abs_vs_stored"] for r in flat), default=None),
                           "token_index_all_equal": all(r["token_index"] == r["stored_token_index"] for r in flat),
                           "noun_is_registered_target": all(r["noun"] == r["registered_target"] for r in flat),
                           "argmax_same_share": float(np.mean([a for r in flat for a in r["argmax_same"]]))
                           if flat else None},
               "note": "the stored maps are float16; the served map is float32 (raw)"}
    write_json(out_dir(args) / "noun.json", payload)
    return payload


# ------------------------------------------------------------------------------ stage replay (CPU)


def stored_noun_maps(ex: dict, layer: int = 10, head: int = 7) -> np.ndarray:
    """``(F, 3, 16, 16)`` float32 = `t54_arms.noun_map(block_fc(ex, "target"), SOURCES["noun"])` (σ0, k1: the
    cell as float64 then float32) without materialising every cell in float64."""
    qi = [str(q) for q in ex["query_names"]].index("target")
    return np.asarray(ex["maps"][:, qi, layer, head], np.float64).astype(np.float32)


def _row_diff(a: dict, b: dict) -> list[str]:
    keys = sorted((set(a) | set(b)) - set(TIMING_KEYS))
    bad = []
    for k in keys:
        x, y = a.get(k), b.get(k)
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        if json.dumps(x, sort_keys=True, default=str) != json.dumps(y, sort_keys=True, default=str):
            bad.append(k)
    return bad


def stage_replay(args) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies import t54_arms as arms_mod
    from AG3S.ag3s.experiments.studies import t54_replay as rp
    from AG3S.ag3s.experiments.studies import t54_study as st
    from AG3S.ag3s.experiments.studies.t49_noun_query import block_fc
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import load_extract

    nd = out_dir(args, "noun")
    d = out_dir(args, "replay")
    stored_root = pathlib.Path(args.t54c_run) / "verify" / "T54c" / "replay" / "test" / "rows"
    arm = arms_mod.ARMS["N"]
    result = {}
    for key in args.records:
        z_path = nd / f"{key}.npz"
        if not z_path.is_file():
            raise SystemExit(f"no served maps {z_path} — run --stage noun first")
        with np.load(z_path, allow_pickle=False) as z:
            frames = [int(f) for f in z["frames"]]
            served = np.asarray(z["maps"], np.float32)
        info = t54c_inputs(args, key)
        if info["problems"]:
            raise SystemExit(f"[{key}] input problems: {info['problems']}")
        ex = load_extract(pathlib.Path(info["noun"]))
        base = stored_noun_maps(ex)
        # the lean extraction is `t54_arms.noun_map` (checked on the selected frames)
        mini = {"maps": ex["maps"][frames], "query_names": ex["query_names"]}
        if not np.array_equal(base[frames], arms_mod.noun_map(block_fc(mini, "target"), arms_mod.SOURCES["noun"])):
            raise SystemExit("stored_noun_maps differs from t54_arms.noun_map")
        del ex, mini
        stored = {}
        with open(stored_root / f"{key}__current.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                if r["arm"] == "N" and int(r["frame"]) in frames:
                    stored[int(r["frame"])] = r
        build, prov = rp.make_builder(info["profile"], pathlib.Path(info["record"]),
                                      server_proc=None if info["server_proc"] is None
                                      else pathlib.Path(info["server_proc"]))
        per_variant = {}
        for variant in ("f16", "f32"):
            maps = base.copy()
            for f, m in zip(frames, served):
                maps[f] = m.astype(np.float16).astype(np.float32) if variant == "f16" else m
            rows = rp.run_record(pathlib.Path(info["record"]), target=info["target"], maps={"noun": maps},
                                 arms=[arm], condition="current", build=build, freeze=None, frames=frames,
                                 log=lambda msg: None)
            # serialised exactly as `t54_study._job` wrote the stored rows
            got = {int(r["frame"]): json.loads(json.dumps(st.jsonable({"record": key, **r}), allow_nan=True))
                   for r in rows[arm.name]}
            with open(d / f"{key}__{variant}__current.jsonl", "w") as fh:
                for f in frames:
                    fh.write(json.dumps(got[f], allow_nan=True) + "\n")
            cmp = {}
            for f in frames:
                if f not in stored:
                    cmp[f] = {"stored_row": False}
                    continue
                g = got[f]
                diff = _row_diff(g, stored[f])
                cmp[f] = {"equal": not diff, "diff_keys": diff,
                          "chosen_rank": [g.get("chosen_rank"), stored[f].get("chosen_rank")],
                          "on_target": [g.get("on_target"), stored[f].get("on_target")],
                          "identity_correct": [g.get("identity_correct"), stored[f].get("identity_correct")]}
            per_variant[variant] = cmp
            print(f"[replay] {key} {variant}: equal {sum(1 for c in cmp.values() if c.get('equal'))}/{len(cmp)}",
                  flush=True)
        result[key] = {"profile": prov.get("profile"), **per_variant}
    payload = {**run_info("replay", args), "stored_rows": str(stored_root), "records": result,
               "timing_keys_excluded": list(TIMING_KEYS)}
    write_json(out_dir(args) / "replay.json", payload)
    return payload


# ------------------------------------------------------------------------------ stage probe


def t55_dir(args) -> pathlib.Path:
    return pathlib.Path(args.t55_run_dir or (pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
                                             / "results" / "development" / T55_RUN)) / "verify" / "T55"


def stage_probe(args) -> dict:
    from AG3S.ag3s.experiments.studies import t55_study as st55
    from AG3S.trajopt.attention_source import (
        DEFAULT_SOURCE, load_noun_probe, noun_lookup, parse_attention_source)

    root = t55_dir(args)
    probe = load_noun_probe(root / "noun_probe.npz")
    model, feat = st55.load_probe(root / "noun_probe.npz")
    s = gpu_session(args)
    ap = serving_policy(s, source=None, probe=probe)                       # gate arm: default source + probe
    ap_n = serving_policy(s, source=NOUN_SPEC, probe=probe, sampler=ap._sampler)  # noqa: SLF001
    noun_src = parse_attention_source(NOUN_SPEC)
    rows = []
    for key in args.t55_keys:
        src = root / "resim" / "test" / f"{key}__E0.npz"
        fz = root / "feat" / "test" / f"{key}__E0__orig.npz"
        with np.load(src, allow_pickle=False) as z:
            images = {c: np.asarray(z[f"image_{c}"]) for c in CAMS}
            states = np.asarray(z["state"], np.float64)
            prompt = str(z["prompt"])
        with np.load(fz, allow_pickle=False) as z:
            layers = [int(x) for x in z["noun_layers"]]
            off_v = np.asarray(z["noun_v"][:, 0, layers.index(probe.layer)], np.float32)
            ok = np.asarray(z["noun_ok"], bool)
        off_p = model.proba(off_v.astype(np.float64))[:, 1]
        frames = [i for i in np.flatnonzero(ok)[::max(1, int(args.probe_stride))]][: args.probe_frames]
        for i in frames:
            # `t48_hsr_features.FeatureExtractor.inputs`: state float32, images CHW, the re-sim prompt
            obs = {"state": np.asarray(states[i], np.float32),
                   "images": {c: np.transpose(images[c][i], (2, 0, 1)) for c in CAMS}, "prompt": prompt}
            res = ap.infer(dict(obs))
            tp = res.get("task_probe", {})
            inputs = ap._policy._input_transform(dict(obs))                # noqa: SLF001
            look = noun_lookup(inputs, prompt, s["tokens"])
            o_a = ap._backend.run(inputs, source=DEFAULT_SOURCE, rows=look.rows, want_feature=True,  # noqa: SLF001
                                  v_layer=probe.layer)
            o_n = ap_n._backend.run(inputs, source=noun_src, rows=look.rows, want_feature=True,       # noqa: SLF001
                                    v_layer=probe.layer)
            v_a, v_n = o_a["v"][0], o_n["v"][0]
            # stage 1 side effect: the kv_L4 subtask feature of the noun pass (another executable) vs the
            # sampler's (`_jit_prefix`) — the subtask label must not move
            sub = ap.subtask_probe
            lab_a = sub.predict(o_a["feature"]) if sub is not None else None
            lab_n = sub.predict(o_n["feature"]) if sub is not None else None
            r = {"key": key, "frame": int(i), "p_served": tp.get("p"), "p_offline": float(off_p[i]),
                 "p_abs_diff": None if tp.get("p") is None else abs(float(tp["p"]) - float(off_p[i])),
                 "pass_same": None if tp.get("p") is None else
                 bool((tp["p"] >= 0.3) == (off_p[i] >= 0.3)),
                 "noun_ok": tp.get("noun_ok"), "noun": tp.get("noun"),
                 "v_action_vs_offline_max_abs": float(np.abs(v_a - off_v[i]).max()),
                 "v_noun_vs_offline_max_abs": float(np.abs(v_n - off_v[i]).max()),
                 "v_action_vs_noun_max_abs": float(np.abs(v_a - v_n).max()),
                 "p_from_v_noun": probe.p(v_n),
                 "kv_l4_noun_vs_action_max_abs": float(np.abs(np.asarray(o_a["feature"]) - np.asarray(o_n["feature"])).max()),
                 "subtask_argmax_same": None if lab_a is None else lab_a["argmax"] == lab_n["argmax"],
                 "subtask_p_action": None if lab_a is None else lab_a["p"],
                 "subtask_p_noun": None if lab_n is None else lab_n["p"]}
            rows.append(r)
            print(f"[probe] {key} f{i}: p {r['p_served']} vs {r['p_offline']:.6g} |Δv| {r['v_action_vs_offline_max_abs']:.3g}",
                  flush=True)
    payload = {**run_info("probe", args), "checkpoint": s["ckpt"], "probe": probe.record(), "t55": str(root),
               "rows": rows,
               "summary": {"n": len(rows),
                           "p_abs_diff_max": max((r["p_abs_diff"] for r in rows if r["p_abs_diff"] is not None),
                                                 default=None),
                           "pass_same_share": float(np.mean([r["pass_same"] for r in rows if r["pass_same"] is not None]))
                           if rows else None,
                           "v_action_vs_offline_max_abs": max((r["v_action_vs_offline_max_abs"] for r in rows),
                                                              default=None),
                           "v_noun_vs_offline_max_abs": max((r["v_noun_vs_offline_max_abs"] for r in rows),
                                                            default=None),
                           "kv_l4_noun_vs_action_max_abs": max((r["kv_l4_noun_vs_action_max_abs"] for r in rows),
                                                               default=None),
                           "subtask_argmax_same_share": float(np.mean([r["subtask_argmax_same"] for r in rows
                                                                       if r["subtask_argmax_same"] is not None]))
                           if rows else None}}
    write_json(out_dir(args) / "probe.json", payload)
    return payload


# ------------------------------------------------------------------------------ stage cost


def _summ(ms: Sequence[float]) -> dict:
    a = np.asarray(list(ms), np.float64)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "max_ms": float(a.max())}


def gate_cpu_cost(n: int = 2000) -> dict:
    """`SafePolicy._task_gate` alone (probe dict -> record / reason), CPU."""
    import types as _t

    from AG3S.trajopt.grasp_latch import GraspPhase
    from AG3S.trajopt.safe_policy import SafePolicy

    sp = SafePolicy.__new__(SafePolicy)
    sp.task_gate, sp.task_gate_tau = "on", 0.3
    sp.follow_to_limit, sp._last_follow_to, sp._follow_to_streak = None, {}, 0  # noqa: SLF001
    sp._latch = _t.SimpleNamespace(phase=GraspPhase.HELD)                   # noqa: SLF001
    sp._pending = {"scene": {"seq": 1}, "task_probe": {"p": 0.2, "ok": True, "noun_ok": True,  # noqa: SLF001
                                                       "noun": "apple", "feature_sha1": "x"}}
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        sp._task_gate([])                                                  # noqa: SLF001
        t.append((time.perf_counter() - t0) * 1e3)
    return _summ(t)


def stage_cost(args) -> dict:
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    from AG3S.trajopt.attention_source import load_noun_probe

    probe = load_noun_probe(t55_dir(args) / "noun_probe.npz")
    s = gpu_session(args)
    base = serving_policy(s, source=None, probe=None)
    sampler = base._sampler                                                 # noqa: SLF001
    configs = {"default": base,
               "probe_default_source": serving_policy(s, source=None, probe=probe, sampler=sampler),
               "noun_source": serving_policy(s, source=NOUN_SPEC, probe=None, sampler=sampler),
               "noun_source_probe": serving_policy(s, source=NOUN_SPEC, probe=probe, sampler=sampler)}
    obs = []
    for key in args.records:
        info = t54c_inputs(args, key)
        run = load_run(info["record"])
        obs += [step_obs(run.steps[f], run.prompt) for f in args.frames if f < len(run)]
    warm = int(args.warmup)
    for name, ap in configs.items():  # compile every executable first
        for o in obs[:warm]:
            ap.infer(dict(o))
    times: dict[str, list[float]] = {k: [] for k in configs}
    for _ in range(int(args.repeats)):
        for o in obs:
            for name, ap in configs.items():  # interleaved: drift hits every configuration alike
                t0 = time.perf_counter()
                ap.infer(dict(o))
                times[name].append((time.perf_counter() - t0) * 1e3)
    extra = {name: _summ(np.asarray(times[name]) - np.asarray(times["default"])) for name in configs
             if name != "default"}
    payload = {**run_info("cost", args), "checkpoint": s["ckpt"], "n_requests": len(obs) * int(args.repeats),
               "per_config": {k: _summ(v) for k, v in times.items()}, "paired_extra_over_default": extra,
               "gate_cpu_ms": gate_cpu_cost(),
               "note": "AttentionPolicy.infer with a stub action policy (the action inference is the same in "
                       "every configuration and is not timed); shadow cost = probe_default_source − default"}
    write_json(out_dir(args) / "cost.json", payload)
    return payload


# ------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--run-dir", default=None, help="default $AG3S_RUN_DIR")
    ap.add_argument("--t54c-run", default=None, help="the T54c run (records, noun extract, stored rows)")
    ap.add_argument("--t55-run-dir", default=None, help=f"default $AG3S_STORAGE_ROOT/results/development/{T55_RUN}")
    ap.add_argument("--records", nargs="*", default=list(DEFAULT_RECORDS))
    ap.add_argument("--frames", type=int, nargs="*", default=list(DEFAULT_FRAMES))
    ap.add_argument("--t55-keys", nargs="*", default=list(DEFAULT_T55_KEYS))
    ap.add_argument("--probe-frames", type=int, default=8, help="E0 frames per T55 key (noun_ok frames)")
    ap.add_argument("--probe-stride", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--checkpoint", default=None, help="default: the T47 / T49 / T51 checkpoint step")
    ap.add_argument("--no-lock-check", action="store_true")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    return ap


def validate_args(ap: argparse.ArgumentParser, args) -> None:
    if args.stage in ("noun", "replay", "cost") and not args.t54c_run:
        ap.error(f"--stage {args.stage} needs --t54c-run <the T54c run>")
    if args.t54c_run and not (pathlib.Path(args.t54c_run) / "verify" / "T54c").is_dir():
        ap.error(f"{args.t54c_run} has no verify/T54c")
    if not args.frames or min(args.frames) < 0:
        ap.error("--frames: non-negative frame indices")
    if args.stage == "probe" and not (t55_dir(args) / "noun_probe.npz").is_file():
        ap.error(f"no T55 probe under {t55_dir(args)}")


def main(argv: Optional[Sequence[str]] = None) -> dict:
    ap = build_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    return {"noun": stage_noun, "replay": stage_replay, "probe": stage_probe, "cost": stage_cost}[args.stage](args)


if __name__ == "__main__":
    main()
