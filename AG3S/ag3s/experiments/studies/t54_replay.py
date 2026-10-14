"""T54 — the MuJoCo replay half: one row per (frame, arm, condition) on the serving grounding path.

Frame loop = T47b's (`t47b_replay._frames` · `_ground` · `t47_row`; render osmesa, CPU): the live
serving path `wire.pack_request -> strip_request -> unpack_camera_observations(attention=...) ->
AG3S.process_multi_debug`, with `t54_arms.arm_hooks` around the call.

Conditions (`T54.task.md` 사전 고정 2)
    ``current`` (primary)  every frame on its own: **one** AG3S per job, `AG3S.reset()` before every
                           (frame, arm) call — no `TargetConfirm` history, no destination registry,
                           no latch / subtask state, no previous-frame target guard. Only that
                           frame's attention and cloud.
    ``hold``               the episode in time order, `TargetConfirm` on, **one AG3S per arm** (each
                           arm accumulates its own history; nothing of one arm reaches another). When
                           the serving chunk records are available (T53), before each call every
                           arm receives the **same recorded** latch / freeze inputs `SafePolicy` gave
                           the serving AG3S (`freeze_inputs`): `set_grasp_active`, `set_grasp_latched`,
                           `set_subtask`, `set_placed`. `attach` / `detach` are **not** replayed (the
                           attached geometry is the serving arm's own target, not robot feedback).
                           Without chunk records (the dev records) nothing is injected — that is the
                           T47b / T51 / T52 replay loop exactly, the one the dev parity compares with.

Profiles (the AG3S each job builds; the same for every arm)
    ``t52``  `serve_safe.build_ag3s` at the server defaults, legacy ESDF, subtask gate off — the
             factory of every T47–T52 replay (`t47b_replay._build`). Dev records / parity.
    ``c1a``  the T53 rollout server's own flags (`server_proc.txt` → `serve_safe.build_parser` →
             the kwargs `serve_safe.main` passes to `build_ag3s`, `t54_replay.c1a_build_kwargs`),
             except the ESDF backend: legacy (CPU) instead of cuRobo. Grounding (fusion → support →
             lifting → grounding) runs before the field; the backend only moves what the exclusion
             gate feeds back to `TargetConfirm` in the hold condition (a stated limitation).

CPU only. Run by the verifier; the implementer only smoke-runs a few frames.
"""

from __future__ import annotations

import json
import pathlib
import shlex
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t54_arms as arms_mod

CONDITIONS = ("current", "hold")
PROFILES = ("t52", "c1a")


# ------------------------------------------------------------------------------- profiles


def server_argv(proc_txt: pathlib.Path) -> list[str]:
    """The serve_safe argument list from a T53 `server_proc.txt` (the line with the module)."""
    lines = [ln for ln in pathlib.Path(proc_txt).read_text().splitlines()
             if "AG3S.trajopt.serve_safe" in ln]
    if len(lines) != 1:
        raise SystemExit(f"{proc_txt}: expected one serve_safe command line, found {len(lines)}")
    argv = shlex.split(lines[0])
    return argv[argv.index("AG3S.trajopt.serve_safe") + 1:]


def c1a_build_kwargs(argv: Sequence[str], *, backend: str = "legacy") -> tuple[str, dict, dict]:
    """``(model_xml, build_ag3s kwargs, record)`` for a serve_safe argv — the keyword set
    `serve_safe.main` passes (`serve_safe.py` `SafePolicy(... ag3s=build_ag3s(...))`), with the ESDF
    `backend` replaced (fine / TSDF voxel dropped with it) and the two log-only arguments
    (`plan_horizon_steps`, `rows_per_step`) left at None."""
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    args = ap.parse_args(list(argv))
    ss.reject_bad_flag_combinations(ap, args)
    ss.resolve_subtask_gate(args)
    identity = {**ss.t43t_identity_section(args), **ss.split_identity_section(args),
                **ss.placed_no_seed_section(args)}
    target_exclusion = {**ss.target_exclusion_section(args), **ss.target_free_coarse_section(args),
                        **ss.target_free_coarse_sign_section(args),
                        **ss.target_free_coarse_support_section(args)}
    legacy = backend == "legacy"
    # Field-only settings the legacy builder refuses (`pipeline._build_esdf`: a target-free layer is
    # cuRobo-only) — both are read only after grounding (`pipeline.py` field / constraint stages), so
    # the replay drops them on the CPU backend and records that it did.
    field_policy = "relax" if legacy else args.target_field_policy
    kw = dict(voxel=args.voxel, range_max=args.range_max, links=args.links, backend=backend,
              fine_voxel=None if legacy else args.fine_voxel,
              tsdf_voxel=None if legacy else args.tsdf_voxel,
              attached_sign_threshold=args.attached_sign_threshold,
              max_field_age_sec=args.max_field_age_sec, exclude_links=args.exclude_links,
              constraint_sphere_options=ss.sphere_options(args),
              gripper_cover=ss.gripper_cover_options(args),
              finger_cover=ss.finger_cover_options(args), palm_cover=ss.palm_cover_options(args),
              self_collision=not args.no_self_collision,
              target_field_policy=field_policy,
              self_filter=ss.self_filter_options(args), subtask_gate=args.subtask_gate,
              subtask_probe_path=args.subtask_probe, held_body_cover=args.held_body_cover,
              identity=identity or None,
              target_exclusion=None if legacy else (target_exclusion or None),
              contact=ss.nonauthorized_target_margin_section(args) or None,
              held_free_observed_pad=(None if args.held_free_observed_mm is None
                                      else float(args.held_free_observed_mm) / 1000.0))
    record = {"server_esdf_backend": args.esdf_backend, "replay_esdf_backend": backend,
              "identity": identity, "subtask_gate": bool(args.subtask_gate),
              "placed_no_seed_certify": bool(identity.get("placed_no_seed_certify", False)),
              "field_only_overrides": ({} if not legacy else {
                  "target_field_policy": [args.target_field_policy, field_policy],
                  "target_exclusion": [target_exclusion, None],
                  "fine_voxel": [args.fine_voxel, None], "tsdf_voxel": [args.tsdf_voxel, None]})}
    return str(args.model_xml), kw, record


def t52_build_kwargs() -> dict:
    """`t47b_replay._build` / T52 `run_replay.sh`: server defaults, legacy, subtask gate off."""
    return dict(voxel=0.020, range_max=2.0, links="gripper", backend="legacy", subtask_gate=False)


def make_builder(profile: str, record_path: pathlib.Path, *, server_proc: Optional[pathlib.Path] = None
                 ) -> tuple[Callable[[], Any], dict]:
    """``(build() -> AG3S, provenance)``. The model XML is the record's (`resolve_asset`), as every
    T47–T52 replay; the c1a profile checks it names the same file as the server's."""
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    from AG3S.ag3s.runtime.asset_path import resolve_asset
    from AG3S.trajopt.serve_safe import build_ag3s

    record = load_run(record_path)
    model_xml = str(resolve_asset(record.model_xml, what="transport model"))
    if profile == "t52":
        kw, prov = t52_build_kwargs(), {"profile": "t52"}
    elif profile == "c1a":
        if server_proc is None:
            raise SystemExit("profile c1a needs the T53 server_proc.txt")
        argv = server_argv(server_proc)
        server_xml, kw, rec = c1a_build_kwargs(argv)
        if pathlib.Path(server_xml).name != pathlib.Path(model_xml).name:
            raise SystemExit(f"server model {server_xml} and record model {model_xml} differ")
        if rec["placed_no_seed_certify"]:
            raise SystemExit("T54 fixes clustering.placed_no_seed_certify = off; the server had it on")
        prov = {"profile": "c1a", "server_proc": str(server_proc), "server_argv": argv, **rec}
    else:
        raise ValueError(f"profile {profile!r} not in {PROFILES}")
    prov["model_xml"] = model_xml
    prov["build_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool, type(None), dict, list))
                                else repr(v)) for k, v in kw.items()}
    return (lambda: build_ag3s(model_xml, **kw)), prov


# ------------------------------------------------------------------------ freeze injection

#: `SafePolicy._scene_fn` order (`safe_policy.py`): set_grasp_active, set_grasp_latched, set_subtask,
#: set_placed, then `process_multi_debug`.
FREEZE_KEYS = ("grasp_active", "grasp_latched", "subtask_p", "placed")


def freeze_inputs(summary: Mapping[str, Any], previous: Optional[Mapping[str, Any]] = None) -> dict:
    """The latch / freeze inputs the serving AG3S got for one chunk, read from its recorded summary.

    * ``grasp_active`` = ``exclusion.grasp_active`` (the AG3S flag during that call; `SafePolicy`
      sets it from the latch phase ∈ {closing, held}).
    * ``grasp_latched`` = ``admissibility.decision.latched`` (what `TargetConfirm` held during that
      call; phase ∈ {latched, closing}). When the call made no decision (no cluster), it is inferred
      from the grasp record (``source_latched`` = ``fallback``): closing if ``grasp_active`` and nothing
      attached, else ``grasp.state_before == "latched"``.
    * ``subtask_p`` = ``subtask.p`` (the probe's probabilities for that request, or None).
    * ``placed`` = ``subtask.placed_seen`` (sticky after the call; `note_placed` is sticky, so feeding
      the after-call value reproduces the before-call stickiness at every call).
    Also kept for checks: ``frozen_recorded`` (``decision.frozen``), ``attached_label``, grasp states.
    """
    ex = summary.get("exclusion") or {}
    adm = summary.get("admissibility") or {}
    dec = adm.get("decision") or {}
    grasp = summary.get("grasp") or {}
    sub = summary.get("subtask") or {}
    ga = ex.get("grasp_active")
    ga_source = "exclusion"
    if ga is None:
        ga, ga_source = False, "missing"
    attached = ex.get("attached_label")
    if "latched" in dec and dec.get("latched") is not None:
        latched, l_source = bool(dec["latched"]), "decision"
    else:
        if ga:
            latched = attached is None
        else:
            latched = grasp.get("state_before") == "latched"
        l_source = "fallback"
    p = sub.get("p")
    return {
        "grasp_active": bool(ga), "grasp_latched": bool(latched),
        "subtask_p": None if p is None else {str(k): float(v) for k, v in p.items()},
        "placed": bool(sub.get("placed_seen", False)),
        "source_grasp_active": ga_source, "source_latched": l_source,
        "frozen_recorded": dec.get("frozen"), "attached_label": attached,
        "grasp_state_before": grasp.get("state_before"), "grasp_state": grasp.get("state"),
        "grasp_attach": bool(grasp.get("attach", False)), "grasp_detach": bool(grasp.get("detach", False)),
        "chunk_t_step": summary.get("t_step"), "chunk_index": summary.get("chunk_index"),
    }


def read_summary(chunk: pathlib.Path) -> dict:
    with np.load(chunk, allow_pickle=False) as z:
        return json.loads(str(z["summary_json"]))


def freeze_from_chunks(chunks: Sequence[pathlib.Path]) -> list[dict]:
    out, prev = [], None
    for c in chunks:
        f = freeze_inputs(read_summary(c), prev)
        f["chunk"] = str(c)
        out.append(f)
        prev = f
    return out


def apply_freeze(ag, f: Optional[Mapping[str, Any]]) -> None:
    """Give one recorded chunk's latch inputs to an AG3S, in `SafePolicy._scene_fn` order."""
    if f is None:
        return
    ag.set_grasp_active(bool(f["grasp_active"]))
    ag.set_grasp_latched(bool(f["grasp_latched"]))
    ag.set_subtask(f["subtask_p"])
    ag.set_placed(bool(f["placed"]))


def freeze_summary(freeze: Sequence[Mapping[str, Any]]) -> dict:
    n = len(freeze)
    return {
        "n": n,
        "grasp_active": int(sum(f["grasp_active"] for f in freeze)),
        "grasp_latched": int(sum(f["grasp_latched"] for f in freeze)),
        "placed": int(sum(f["placed"] for f in freeze)),
        "subtask_label_frames": int(sum(f["subtask_p"] is not None for f in freeze)),
        "latched_fallback": int(sum(f["source_latched"] == "fallback" for f in freeze)),
        "grasp_active_missing": int(sum(f["source_grasp_active"] == "missing" for f in freeze)),
        "attached_frames": int(sum(f["attached_label"] is not None for f in freeze)),
        "attach_events": int(sum(f["grasp_attach"] for f in freeze)),
        "detach_events": int(sum(f["grasp_detach"] for f in freeze)),
        # frozen in serving = grasp_active or attached; the replay never attaches, so these differ
        # exactly on attached frames whose grasp_active was False.
        "frozen_recorded_ne_grasp_active": int(sum(
            f["frozen_recorded"] is not None and bool(f["frozen_recorded"]) != bool(f["grasp_active"])
            for f in freeze)),
    }


# ----------------------------------------------------------------------------- frame loop


def camera_maps(maps: np.ndarray, i: int, mj_cams: Sequence[str]) -> dict:
    """``{mujoco camera: float32 16x16}`` of frame `i` (the `attention=` of `unpack_camera_observations`)."""
    return {mj: np.asarray(maps[i, ci], np.float32) for ci, mj in enumerate(mj_cams)}


def camera_id_maps(att: Mapping[str, np.ndarray]) -> dict:
    """mujoco camera name → `CameraID` keyed maps, by `wire.unpack_camera_observations`' own table."""
    from AG3S.ag3s.experiments.sources.mujoco_source import CAMERA_MOUNTS
    from AG3S.ag3s.types import CameraID

    return {CameraID.parse(CAMERA_MOUNTS.get(cam, (None, cam))[1]): m for cam, m in att.items()}


def ground_arm(ag, fr: dict, arm: arms_mod.ArmSpec, maps: Mapping[str, np.ndarray], *,
               label_of, gt_centre, on_target_radius: float = 0.10,
               membership_radius: float = 0.005, ground_fn: Optional[Callable] = None) -> dict:
    """One hooked serving call for one arm on one frame -> row fields (no frame / condition keys).
    `ground_fn(ag, fr, att) -> (cs, debug, ms)` defaults to `t47b_replay._ground` (the serving call)."""
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm
    from AG3S.ag3s.experiments.studies import t47b_replay as br

    ground_fn = ground_fn or br._ground

    i, mj = fr["i"], fr["mj_cams"]
    att = camera_maps(maps[arm.primary], i, mj)
    secondary = {s: camera_id_maps(camera_maps(maps[s], i, mj)) for s in arm.secondary_sources()}
    rec = arms_mod.CallRecord()
    confirm = ag._target_confirm
    frame_before = int(getattr(confirm, "_frame", -1))
    with arms_mod.arm_hooks(arm, secondary, rec):
        cs, debug, ms = ground_fn(ag, fr, att)
    selected = int(getattr(confirm, "_frame", -1)) != frame_before
    row, arrays = br.t47_row(cs, debug, label_of=label_of, gt_centre=gt_centre,
                             on_target_radius=on_target_radius)
    has = bool(cs.has_target)
    chosen = np.zeros(len(arrays["pts"]), bool)
    membership = "none"
    if has:
        chosen, membership = bm.membership_mask(arrays["pts"], np.asarray(cs.target.points),
                                                np.asarray(cs.target.point_indices),
                                                radius=membership_radius)
    b2 = bm.frame_b2(arrays["labels"], support_mask=arrays["support"], seed_indices=arrays["seeds"],
                     chosen_mask=chosen, has_target=has, ball=None)
    gr = debug.get("grounding")
    ac = debug.get("attention_cloud")
    primary_att = None if ac is None else np.asarray(ac.attention)
    choice_att = (rec.secondary[arm.choice_source].attention
                  if arm.choice_source != arm.primary else None)
    adm = confirm.last_admissibility if (selected and gr is not None and gr.clusters) else None
    clusters = arms_mod.cluster_rows(gr, arrays["labels"], primary_att, choice_att, adm)
    decision = confirm.last if selected else None
    prof = dict(getattr(cs, "profile", {}) or {})
    ms_primary = arms_mod.posthoc_lift_ms(rec, camera_id_maps(att))
    lift_ground = (ms_primary + rec.ms_secondary_lift + float(prof.get("attention_lifting", 0.0))
                   + float(prof.get("target_grounding", float("nan"))))
    return {
        **row, "ag3s_ms": ms, **b2, "chosen_membership": membership,
        "coverage": ((1.0 - b2["target_residual_rate"]) if b2["n_target_obs"] else float("nan")),
        **arms_mod.candidate_fields(clusters),
        "chosen_rank": arms_mod.chosen_rank(gr, cs),
        "clusters": clusters,
        "decision_mode": None if decision is None else decision.mode,
        "decision_reason": None if decision is None else decision.reason,
        "decision_frozen": None if decision is None else bool(decision.frozen),
        "select_called": bool(selected),
        "ms_profile": {k: float(v) for k, v in prof.items()},
        "ms_primary_lift_posthoc": ms_primary,
        "ms_lift_ground": lift_ground,
        **arms_mod.hook_fields(rec),
    }


def run_frames(frames: Iterable[dict], *, n_f: int, maps: Mapping[str, np.ndarray],
               arms: Sequence[arms_mod.ArmSpec], condition: str, build: Callable[[], Any],
               freeze: Optional[Sequence[Mapping[str, Any]]] = None,
               ground_fn: Optional[Callable] = None,
               log: Callable[[str], None] = print) -> dict[str, list[dict]]:
    """``{arm name: [row per frame]}`` for one record under one condition (module doc). `frames`
    yields `t47b_replay._frames` dicts (``i``, ``n_f``, ``step``, ``phase``, ``mj_cams``, ``label_of``,
    ``gt_centre``, plus what `ground_fn` reads)."""
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm

    if condition not in CONDITIONS:
        raise ValueError(f"condition {condition!r} not in {CONDITIONS}")
    if freeze is not None and condition == "hold" and len(freeze) != n_f:
        raise ValueError(f"{len(freeze)} freeze records for {n_f} frames")
    if condition == "current":
        shared = build()
        ags = {a.name: shared for a in arms}
    else:
        ags = {a.name: build() for a in arms}
    rows: dict[str, list[dict]] = {a.name: [] for a in arms}
    t0 = time.time()
    for fr in frames:
        i, step = fr["i"], fr["step"]
        f = freeze[i] if (freeze is not None and condition == "hold") else None
        marks = []
        for a in arms:
            ag = ags[a.name]
            if condition == "current":
                ag.reset()
            else:
                apply_freeze(ag, f)
            r = ground_arm(ag, fr, a, maps, label_of=fr["label_of"], gt_centre=fr["gt_centre"],
                           ground_fn=ground_fn)
            rows[a.name].append({"frame": i, "t_step": int(step.t_step), "phase": fr["phase"],
                                 "phase_group": bm.phase_group(fr["phase"]), "arm": a.name,
                                 "condition": condition, "freeze_injected": f is not None,
                                 **({} if f is None else {"freeze": {k: f[k] for k in FREEZE_KEYS}}),
                                 **r})
            marks.append("I" if r["identity_correct"] else ("s" if r["has_target"] else "-"))
        log(f"  t54 {condition} frame {i + 1}/{fr['n_f']} t={step.t_step} "
            f"{''.join(marks)}  ({time.time() - t0:.0f}s)")
    return rows


def run_record(record_path: pathlib.Path, *, target: str, maps: Mapping[str, np.ndarray],
               arms: Sequence[arms_mod.ArmSpec], condition: str, build: Callable[[], Any],
               freeze: Optional[Sequence[Mapping[str, Any]]] = None,
               frames: Optional[Sequence[int]] = None, label_radius: float = 0.003,
               log: Callable[[str], None] = print) -> dict[str, list[dict]]:
    """`run_frames` over the MuJoCo replay of one step record (`t47b_replay._frames`, osmesa)."""
    from AG3S.ag3s.experiments.sources.policy_record import POLICY_CAMERA_NAMES, load_run
    from AG3S.ag3s.experiments.studies import t47b_replay as br

    n_f = len(load_run(record_path))
    need = arms_mod.sources_needed(arms)
    br._check_sources({s: maps[s] for s in need}, n_f, len(POLICY_CAMERA_NAMES))
    return run_frames(br._frames(record_path, target=target, frames=frames,
                                 label_radius=label_radius, log=log),
                      n_f=n_f, maps=maps, arms=arms, condition=condition, build=build,
                      freeze=freeze, log=log)


__all__ = ["CONDITIONS", "FREEZE_KEYS", "PROFILES", "apply_freeze", "c1a_build_kwargs",
           "camera_id_maps", "camera_maps", "freeze_from_chunks", "freeze_inputs",
           "freeze_summary", "ground_arm", "make_builder", "read_summary", "run_frames", "run_record",
           "server_argv", "t52_build_kwargs"]
