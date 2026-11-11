"""The V11AB AG3S, built for the Panda (SL4): V11AB's settings, Panda geometry, nothing else.

V11AB serves AG3S through ``AG3S/trajopt/serve_safe.py``: ``scripts/benchmark/v11ab/start_server.sh``
gives the flags, ``serve_safe.main`` turns them into ``build_ag3s(...)`` keywords, and ``build_ag3s``
turns those into an `AG3SConfig` plus the two RB-Y1 robot models and the attached-slot links. This
module repeats that chain for the Panda **without changing a line of AG3S core**:

1. :func:`v11ab_server_flags` reads the V11AB flags out of ``start_server.sh`` / ``env.sh`` (text only;
   the scripts are never executed -- ``env.sh`` allocates a run directory when sourced);
2. :func:`parse_v11ab_args` parses them with **serve_safe's own** ``build_parser`` and runs its own
   ``reject_bad_flag_combinations`` / ``resolve_subtask_gate``;
3. :func:`v11ab_build_kwargs` evaluates the keyword expressions of the one ``build_ag3s(...)`` call in
   ``serve_safe.main`` -- read from its AST, so a changed call is followed, not re-typed;
4. :func:`v11ab_ag3s_config` assembles the config dict exactly as ``build_ag3s`` does. This is the one
   unavoidable copy (``build_ag3s`` builds it in the middle of building the RB-Y1 models);
   ``tests/benchmark/test_panda_sl4.py`` runs the real ``build_ag3s`` with the same keywords and
   compares the two configs field by field;
5. :func:`panda_ag3s_config` applies the Panda overrides (:data:`PANDA_OVERRIDES`) and
   :func:`build_panda_ag3s` constructs `AG3S` with the Panda models and attached-slot links.

Panda differences (all injected; :func:`diff_against_v11ab` must return only the config rows,
:func:`constructor_differences` lists the rest):

==============================  ================================================  ==========================================
what                            V11AB (RB-Y1)                                      Panda (LIBERO)
==============================  ================================================  ==========================================
``robot_model`` (self-filter)   RB-Y1 URDF chain + MJCF covers (194 spheres)        ``build_panda_self_filter_model()`` --
                                                                                  franka.yml 61 + ``confined_hand_mount``
                                                                                  cover (4271 spheres, SL3 0.1)
``constraint_robot_model``      ``--links gripper`` = ``GRIPPER_LINKS`` (palms +   ``PANDA_GRIPPER_LINKS`` (``panda_hand``,
                                fingers), ``--capsule-radius-scale 0.05 ...``,      fingers; 22 franka.yml spheres); the
                                ``--finger-cover both``                            RB-Y1 sphere / finger-cover build options
                                                                                  are not applied (franka spheres are
                                                                                  zero-length role-``collision`` capsules:
                                                                                  a 0.05 radius scale would shrink them to
                                                                                  5 %; the finger cover is RB-Y1 geometry)
``attached_parent_links``       ``grasp_parent_links()`` = (``ee_left``,           ``grasp_parent_links(PANDA_GRASP_LINKS,
                                ``ee_right``) -- the palms                          PANDA_HOLD_LINKS)`` = (``panda_hand``,)
``contact.contact_links``       ``DEFAULT_CONTACT_LINKS`` (``ee_*`` names, left    ``{"left": PANDA_CONTACT_LINKS["left"]}``
                                and right)                                         -- one arm, the left manipulator (SL2)
``esdf.bounds_lower`` z         ``default_bounds()`` z 0.0 (base frame; RB-Y1      -0.10 m. LIBERO table top is at base z
                                table well above its base)                         -0.012 m (SL1 verify): a 0.0 floor cuts
                                                                                  the table out of the field. Every other
                                                                                  bound stays ``default_bounds()``
==============================  ================================================  ==========================================

``esdf.bounds_upper`` is written explicitly because AG3S requires the two bounds together; its value
is ``default_bounds()[1]`` (the V11AB effective value) and :func:`diff_against_v11ab` compares
**effective** bounds (``None`` resolved to ``default_bounds()``, as ``curobo_builder`` resolves it).

V11AB flags that are not AG3S settings (TO: ``--esdf-margin``, ``--obstacle-margin``, ``--plan-horizon``,
``--lookahead*``, ``--qp-eps``, ``--w-deflection-rate``; recording: ``--record-constraints``) are parsed
and kept in :func:`manifest_entry` but play no part in building AG3S here. The closed-loop server
(``AG3S/adapters/panda/server.py``) applies them: the TO flags through ``serve_safe.trajopt_config_from_args``
(SL-5), ``--record-constraints`` / ``--record-constraints-esdf`` through ``server.constraint_recorder`` (SL5j).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import pathlib
import re
import shlex
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from AG3S.adapters.panda import model as pm
from common.workspace import shared_storage_root

PROJECT_ROOT = pm.PROJECT_ROOT
V11AB_SCRIPT_DIR = PROJECT_ROOT / "scripts/benchmark/v11ab"
START_SERVER = V11AB_SCRIPT_DIR / "start_server.sh"
ENV_SH = V11AB_SCRIPT_DIR / "env.sh"
RUN_CLIENT = V11AB_SCRIPT_DIR / "run_client.sh"
SERVE_SAFE = PROJECT_ROOT / "AG3S/trajopt/serve_safe.py"
#: The V11 arm this profile reproduces (user decision: V11AB).
V11AB_ARM = "V11AB"
#: ``build_ag3s`` keywords that only feed a start-up log line (``announce_row_budget``) and need the TO
#: config of SL-5; not evaluated here.
LOG_ONLY_BUILD_KEYWORDS = ("plan_horizon_steps", "rows_per_step")
#: ``build_ag3s`` keywords added by the AG3S final-configuration merge (main 3ff415d5, 2026-10-29) that
#: :func:`v11ab_config_dict` does not assemble. Off (None / empty) they change nothing; on, the copy would drop them, so
#: it refuses instead until their Panda port is decided (palm/wrist/shell covers, target exclusion incl. ROI carve,
#: non-authorised target margin, confidence gate).
#: SL6-A A10 (2026-10-29): ``palm_shell`` (A5) and ``target_exclusion`` (A2: target exclusion, ROI carve, non-target
#: evidence) are ported -- :func:`v11ab_config_dict` / :func:`build_panda_ag3s` apply them as ``build_ag3s`` does.
UNPORTED_BUILD_KEYWORDS = ("palm_cover", "wrist_cover", "confidence_gate")   # SL15: ``contact`` (T44e A9) is ported

#: Panda grasp links per hand (``safe_policy.DEFAULT_GRASP_LINKS`` shape): reach is measured from the
#: first finger, the pair is the two fingers.
PANDA_GRASP_LINKS: dict[str, tuple] = {
    pm.MANIPULATOR_SIDE: ("panda_leftfinger", ("panda_leftfinger", "panda_rightfinger"))}
#: Link a held object is attached to (``safe_policy.DEFAULT_HOLD_LINKS``: the palm, T32 H2).
PANDA_HOLD_LINKS: dict[str, str] = {pm.MANIPULATOR_SIDE: "panda_hand"}
#: ESDF workspace box floor for the Panda (base frame, m). Only z of ``bounds_lower`` changes.
PANDA_ESDF_BOUNDS_LOWER_Z = -0.10
ESDF_BOUNDS_CHOICES = ("panda", "v11ab")

#: Why each Panda config override exists (manifest + docstring table above).
OVERRIDE_REASONS: dict[str, str] = {
    "contact.contact_links": (
        "RB-Y1 link names (ee_left/ee_finger_l1/l2, ee_right/...) do not exist on the Panda; the single Panda arm "
        "is the left manipulator (SL2) and its contact-authorized links are the palm + both fingers "
        "(PANDA_CONTACT_LINKS, the Panda analogue of DEFAULT_CONTACT_LINKS)"),
    "esdf.bounds_lower": (
        "z only: V11AB default_bounds() floor is base z 0.0 (RB-Y1 table well above its base); the LIBERO table "
        "top is at base z -0.012 m (SL1 verify), so the default box cuts the table out of the field. "
        f"Floor lowered to {PANDA_ESDF_BOUNDS_LOWER_Z} m; x/y unchanged"),
}


# --- V11AB flags (text of the launcher scripts) -----------------------------------------------------------
def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _shell_text(path: pathlib.Path) -> str:
    """Script text with backslash-newline continuations joined and comments dropped (outside quotes)."""
    text = pathlib.Path(path).read_text().replace("\\\n", "")
    lines = []
    for line in text.splitlines():
        out, quote = [], None
        for ch in line:
            if quote is None and ch == "#":
                break
            if ch in "\"'":
                quote = None if quote == ch else (ch if quote is None else quote)
            out.append(ch)
        lines.append("".join(out))
    return "\n".join(lines)


def _assignments(text: str) -> dict[str, str]:
    """First ``NAME="value"`` of every name (double-quoted, no nesting)."""
    out: dict[str, str] = {}
    for name, value in re.findall(r'(?:^|[\s;])([A-Z][A-Z0-9_]*)="([^"]*)"', text, flags=re.M):
        out.setdefault(name, value)
    return out


def _expand(value: str, variables: Mapping[str, str]) -> str:
    def sub(match):
        name = match.group(1) or match.group(2)
        if name not in variables:
            raise ValueError(f"V11AB launcher: unresolved shell variable ${name} in {value!r}")
        return variables[name]

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", sub, value)


def ta_flags(arm: str = V11AB_ARM, env_sh: pathlib.Path = ENV_SH) -> str:
    """``env.sh``'s ``ta_flags ARM`` (the per-arm TO flags), read from its text."""
    text = _shell_text(env_sh)
    variables = _assignments(text)
    line = next((l for l in text.splitlines() if re.match(r"\s*ta_flags\(\)", l)), None)
    if line is None:
        raise ValueError(f"{env_sh}: no ta_flags() definition")
    cases = dict(re.findall(r'([A-Za-z0-9_]+)\)\s*echo\s+"([^"]*)";;', line))
    if arm not in cases:
        raise ValueError(f"{env_sh}: ta_flags has no arm {arm!r} (have {sorted(cases)})")
    return _expand(cases[arm], variables)


def v11ab_server_flags(arm: str = V11AB_ARM, start_server: pathlib.Path = START_SERVER,
                       env_sh: pathlib.Path = ENV_SH) -> list[str]:
    """The serve_safe flags ``start_server.sh ARM`` passes as ``$FLAGS`` (``$REC`` recording excluded)."""
    text = _shell_text(start_server)
    variables = _assignments(text)
    flags_line = None
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9_|]+)\)\s*FLAGS=\"([^\"]*)\"", line)
        if m and arm in m.group(1).split("|"):
            flags_line = m.group(2)
            break
    if flags_line is None:
        raise ValueError(f"{start_server}: no FLAGS for arm {arm!r}")
    flags_line = flags_line.replace("$(ta_flags $C)", ta_flags(arm, env_sh))
    if "$(" in flags_line:
        raise ValueError(f"{start_server}: unsupported command substitution in FLAGS: {flags_line!r}")
    return shlex.split(_expand(flags_line, variables))


def v11ab_required_args(start_server: pathlib.Path = START_SERVER, env_sh: pathlib.Path = ENV_SH) -> dict[str, str]:
    """``--config`` / ``--model-xml`` of the V11AB server (``--checkpoint`` is irrelevant here)."""
    text = _shell_text(start_server)
    config = re.search(r"serve_safe\s+.*?--config\s+(\S+)", text)
    env = _assignments(_shell_text(env_sh))
    model_xml = env.get("MODEL_XML") or re.search(r"MODEL_XML=(\S+)", _shell_text(env_sh)).group(1)
    model_xml = model_xml.replace("$O1", str(PROJECT_ROOT)).replace("${AG3S_PROJECT_ROOT}", str(PROJECT_ROOT))
    if config is None:
        raise ValueError(f"{start_server}: no serve_safe --config")
    return {"config": config.group(1), "model_xml": model_xml}


def v11ab_client_phase(run_client: pathlib.Path = RUN_CLIENT) -> str:
    """``--safe-phase`` the V11AB client sends (``run_client.sh`` ``SB``)."""
    m = re.search(r"--safe-phase\s+(\S+)", _shell_text(run_client))
    if m is None:
        raise ValueError(f"{run_client}: no --safe-phase")
    return m.group(1)


#: AG3S versions the benchmark serves: serve_safe flags added to the V11AB argv. ``V11AB`` is the paired-v2
#: baseline (unchanged algorithm); every ``V11AB-pb<N>`` is an improvement made on feature/panda-benchmark,
#: recorded in AG3S_BENCHMARK_LOG.md (cause, change, commit, before/after). Flags only add behaviour that is off
#: in V11AB, so the baseline stays bit-identical.
AG3S_VERSIONS: dict[str, tuple[str, ...]] = {
    "V11AB": (),
    "V11AB-pb1": ("--reach-at-capture",),
    # pb2 = pb1 + minimal intervention gate at 10 mm (the response model's prediction error on unchanged chunks is
    # ~3 mm median, 7.5 mm max (SL5a), so a reference within 10 mm of a constraint is still optimized).
    "V11AB-pb2": ("--reach-at-capture", "--minimal-intervention-mm", "10"),
    # pb3 = pb2 + the arm links in the constraint model (panda_link1..7 + gripper, 59 franka.yml spheres instead of the
    # gripper's 22): SL5d pb2-C, 5 of 11 SafeLIBERO Level I collisions were robot0_link5..7 touching the obstacle,
    # which the gripper-only model cannot see.
    "V11AB-pb3": ("--reach-at-capture", "--minimal-intervention-mm", "10", "--links", "arms"),
    # pb4 = pb3 + held exemption (user 2026-10-17 (1)): while HELD the hand's rows on the object it holds are dropped.
    # SL5e: in LIBERO 10, 34 of 102 reference-violating calls were the hand at the bowl it was carrying.
    "V11AB-pb4": ("--reach-at-capture", "--minimal-intervention-mm", "10", "--links", "arms", "--held-hand-exempt"),
    # pb5 = pb4 + grasp zone (user 2026-10-17 (2)): LATCHED/CLOSING take V11AB's phase rule from the latch
    # (pre_grasp / grasp, client fixed at approach) and the hand's contact links may come 10 mm closer to the target and
    # the support right under it within 100 mm (horizontal) of the object. SL5e: 68 of 102 calls were the gripper at
    # the bowl (23) or the support under it (45) while latched/closing; 10 mm = 2 fine voxels (the support tolerance).
    "V11AB-pb5": ("--reach-at-capture", "--minimal-intervention-mm", "10", "--links", "arms", "--held-hand-exempt",
                  "--latch-phase", "--grasp-zone-mm", "10"),
    # pb6 "파지 인식 수정판" = pb5 + the latch recognises the real grasp (SL5i, user 2026-10-22). SL5g pb5-C: the bowl was held
    # while the latch stayed CLOSING on 189 LI / 37 LIBERO 10 calls, failing reach (179 / 33) and dropped (42 / 36).
    # (a) reach sticky: the reach that passed on entering CLOSING (LI t2e4 91.1 mm, LIBERO t5e0 81.7 mm) was re-measured
    # on a later capture (123 / 133 mm) and stayed failed; within one attempt it now stays passed (AG3S.attach too).
    # (b) drop from open: t5e0 latched with the fingers already at 0.072, so 'dropped' measured from 0.0719 instead of
    # the last open opening (~1.0).
    "V11AB-pb6": ("--reach-at-capture", "--minimal-intervention-mm", "10", "--links", "arms", "--held-hand-exempt",
                  "--latch-phase", "--grasp-zone-mm", "10", "--grasp-reach-sticky", "--grasp-drop-from-open"),
    # pb7 "파지 인식 + closing 면제판" = pb6 + the held exemption also while CLOSING once the fingers stopped on something
    # (closed, not_empty, blocked): SL5h D4, TO pushed the hand up on the grasped bowl (up 1431 rows, down 243) before
    # the latch reached HELD.
    "V11AB-pb7": ("--reach-at-capture", "--minimal-intervention-mm", "10", "--links", "arms", "--held-hand-exempt",
                  "--latch-phase", "--grasp-zone-mm", "10", "--grasp-reach-sticky", "--grasp-drop-from-open",
                  "--closing-hand-exempt"),
}


# --- SL6-A: the AG3S final configuration on the Panda ("Panda 최종 구성", task SL6 §1) --------------------------------
#: The RB-Y1 final server argv (``AG3S-CORE-EVAL-A6.impl.md`` §3) minus the V11AB server flags it starts with (= the
#: final-configuration flags added since V11AB), minus G1 / A4 (:data:`G1A4_FLAGS`), plus the Panda decisions of task
#: SL6 §6: ``--hsr6-objective-scope nbc`` (A9), the Panda attention cell ``noun_target:L8H6`` (SL6-B head re-selection,
#: replaces RB-Y1's ``L10H7``) and A12 ``--continuity-after-grasp --deflection-rate-after-grasp``. ``--hsr6-probe`` is
#: added by :func:`version_server_flags` (default :data:`PANDA_HSR6_PROBE`, SL6-B B4); the Panda server refuses the HSR6
#: flags without a probe (A10).
#: ``--plan-horizon 16`` stays as in V11AB; the Panda server replaces the horizon (H 10, K 5, plan 10, ``PANDA_HORIZON``).
#: No pb1-pb7 flag and no :data:`V11AB_PRE_MERGE_DEFAULTS` (ROI carve on, BLAS 1): user ruling 2026-10-29.
FINAL_CONFIG_FLAGS: tuple[str, ...] = (
    "--to-fast-eval", "--esdf-host-mirror", "--target-exclusion", "ball-owned", "--owned-max-dist-voxels", "3",
    "--attention-split-radius-mm", "12", "--attention-split-ratio-source", "raw", "--target-permission-strict",
    "--attention-source", "noun_target:L8H6", "--attention-split-partition", "--split-supersedes-anchor",
    "--wrist-track-scale", "0.1", "--neighbour-contact-mm", "20", "--nontarget-evidence", "surface",
    "--placed-no-seed-certify", "--kept-geometry-follow", "--roi-carve", "--roi-carve-footprint",
    "--roi-carve-footprint-pad-px", "0", "--roi-carve-footprint-robot", "ignore", "--roi-carve-device", "cuda",
    "--blas-threads", "1", "--to-multi-seed", "16", "--to-gpu-seeds", "curobo", "--to-gpu-seeds-check", "first-pass",
    "--palm-shell", "on", "--palm-shell-margin", "5", "--hsr6-attach-assist", "--hsr6-held-gate",
    "--hsr6-target-exempt", "--hsr6-objective", "--neighbour-palm-contact-mm", "10",
    "--hsr6-objective-scope", "nbc",
    # A12 (user ruling 2026-10-29 ~20:40 KST, task SL6 §0 / §1): continuity and deflection-rate terms off before the
    # grasp (latch searching / latched / closing), on from held -- not in the RB-Y1 measured final configuration
    "--continuity-after-grasp", "--deflection-rate-after-grasp")
#: The geometric grasp correction (G1) and the grasp-goal term (A4), as in the RB-Y1 final argv (task SL6 §6 "작동판").
G1A4_FLAGS: tuple[str, ...] = (
    "--grasp-anchor", "2000", "--grasp-anchor-path-weight", "0", "--geo-grasp", "--geo-goal-mode", "correct",
    "--geo-track-scale", "0.25", "--geo-continuity-off")
#: SL6-C (user ruling 2026-10-30, task SL6 §0 / §8): the SL6 smoke failures repeated the causes pb1 / pb6 / pb7 fixed
#: (the latch did not recognise the held bowl: reach measured at the occluded bowl's last-seen centroid, the latch
#: stayed CLOSING, TrajOpt pushed the hand up before HELD). pb2-pb5 are not added.
GRASP_RECOGNITION_PB_FLAGS: tuple[str, ...] = (
    "--reach-at-capture",                                    # pb1
    "--grasp-reach-sticky", "--grasp-drop-from-open",        # pb6
    "--closing-hand-exempt")                                 # pb7
#: "Panda 최종 구성 2판" = the G1 · A4-removed final version without its ``--attention-source`` (back to the default
#: ``action:last:L8H2``, the V11AB-pb7 path: the noun cell L8H6 could not tell the two black bowls apart, SL6 §6.1)
#: + :data:`GRASP_RECOGNITION_PB_FLAGS`.
_i = FINAL_CONFIG_FLAGS.index("--attention-source")
FINAL_CONFIG_V2_FLAGS: tuple[str, ...] = FINAL_CONFIG_FLAGS[:_i] + FINAL_CONFIG_FLAGS[_i + 2:] + GRASP_RECOGNITION_PB_FLAGS
del _i
#: SL6-E (user ruling 2026-10-30 ~07:00 KST, task SL6 §0 / §9): the SL6-C task-1 collisions followed TrajOpt corrections
#: that conversion C saturated, 4 of 2판's 6 collisions were arm links outside the gripper-only model, and 2판 had only
#: pb1 / pb6 / pb7's own flags of the cumulative V11AB-pb7 (pb2-pb5 missing). The continuity and deflection-rate terms
#: are switched off in every phase (t1e1's correction came on the first HELD call, when both terms switched on).
CONTINUITY_DEFLECTION_OFF_FLAGS: tuple[str, ...] = ("--w-continuity", "0", "--w-deflection-rate", "0")
_after = ("--continuity-after-grasp", "--deflection-rate-after-grasp")
_v2_kept = tuple(t for t in FINAL_CONFIG_V2_FLAGS if t not in _after + GRASP_RECOGNITION_PB_FLAGS)
#: "Panda 최종 구성 3판" = 2판 − the two after-grasp flags + the nine V11AB-pb7 flags + both terms off.
FINAL_CONFIG_V3_FLAGS: tuple[str, ...] = _v2_kept + AG3S_VERSIONS["V11AB-pb7"] + CONTINUITY_DEFLECTION_OFF_FLAGS
del _after, _v2_kept
#: SL6-F (user ruling 2026-10-30 ~10:10 KST, task SL6 §0 / §10): 3판 with the TrajOpt settings of AG3S-PRE-CORE-TO E0
#: (the TO baseline before E1-E3): no GPU multi-seed (A2), no minimal intervention gate (pb2), continuity and
#: deflection-rate terms from HELD only at the V11AB weights (0.5 / 20). Robot model, grasp rules, field unchanged.
_E0_DROPPED = {"--to-multi-seed": 1, "--to-gpu-seeds": 1, "--to-gpu-seeds-check": 1, "--minimal-intervention-mm": 1,
               "--w-continuity": 1, "--w-deflection-rate": 1}            # flag -> number of value tokens


def _drop_flags(flags, dropped):
    out, i = [], 0
    while i < len(flags):
        if flags[i] in dropped:
            i += 1 + dropped[flags[i]]
        else:
            out.append(flags[i]); i += 1
    return tuple(out)


FINAL_CONFIG_V3_E0TO_FLAGS: tuple[str, ...] = _drop_flags(FINAL_CONFIG_V3_FLAGS, _E0_DROPPED) + (
    "--continuity-after-grasp", "--deflection-rate-after-grasp")
#: SL6-H (user ruling 2026-10-30 ~12:00 KST, task SL6 §0 / §11): E0판 + HSR6 ⑤ placement contact -- on a call with
#: HSR6 = placement ∧ (latch HELD ∨ (CLOSING ∧ closed ∧ blocked)) the attached held object's query rows are not blocked
#: by the surfaces beneath them (support, the object under the held object's footprint, e.g. the plate); hand / finger
#: / arm rows and surfaces beside or above stay constrained. Diagnosis: 5 of E0판's 10 failures were TrajOpt turning
#: the VLA's descent over the plate upwards.
FINAL_CONFIG_V3_E0TO_PC_FLAGS: tuple[str, ...] = FINAL_CONFIG_V3_E0TO_FLAGS + ("--hsr6-placement-contact",)
#: Versions that are the final configuration (no `V11AB_PRE_MERGE_DEFAULTS`, the Panda HSR6 probe required).
FINAL_CONFIG_VERSIONS: tuple[str, ...] = ("AG3S-final-panda", "AG3S-final-panda-g1a4", "AG3S-final-panda-v2",
                                          "AG3S-final-panda-v3", "AG3S-final-panda-v3-e0to",
                                          "AG3S-final-panda-v3-e0to-pc")
AG3S_VERSIONS["AG3S-final-panda"] = FINAL_CONFIG_FLAGS                       # G1 · A4 removed ("제거판")
AG3S_VERSIONS["AG3S-final-panda-g1a4"] = FINAL_CONFIG_FLAGS + G1A4_FLAGS     # G1 · A4 kept ("작동판")
AG3S_VERSIONS["AG3S-final-panda-v2"] = FINAL_CONFIG_V2_FLAGS                 # "2판" (SL6-C)
AG3S_VERSIONS["AG3S-final-panda-v3"] = FINAL_CONFIG_V3_FLAGS                 # "3판" (SL6-E)
AG3S_VERSIONS["AG3S-final-panda-v3-e0to"] = FINAL_CONFIG_V3_E0TO_FLAGS       # "3판 + TO E0" (SL6-F)
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"] = FINAL_CONFIG_V3_E0TO_PC_FLAGS  # "E0판 + placement 접촉 허용" (SL6-H)
# --- SL7: the development final configuration "P0m" on the Panda (task SL7, user ruling 2026-11-03) --------------------
#: Development's final configuration P0m (``P0_FINAL_CONFIG_FOR_BENCHMARK_20261103_KO.md``) = P0 target extent from the
#: ATS surface extraction + the grasp latch as the 3-stage gate label (T92). On the Panda it is built from the
#: final-configuration flags (:data:`FINAL_CONFIG_FLAGS`) by *removing* what P0m does not use and *adding* the P0 flags:
#:   removed  HSR6 rules ``--hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective
#:            --hsr6-objective-scope nbc`` (no ``--hsr6-probe`` either), ``--nontarget-evidence surface`` (the surface
#:            extraction replaces it), TrajOpt multi-seed ``--to-multi-seed 16 --to-gpu-seeds curobo
#:            --to-gpu-seeds-check first-pass`` (E0), and (D1, user 2026-11-03) ``--neighbour-contact-mm 20`` /
#:            ``--neighbour-palm-contact-mm 10`` (finger / palm neighbour allowance);
#:   added    ``--target-extent attention-surface --surface-target-check overlap --surface-keep-last
#:            --surface-grasp-hold --surface-no-ball --attention-surface-device cuda`` (+ ``--surface-hull-fill
#:            --surface-contact-allowance-mm 10`` in :data:`P0M_FLAGS`), ``--subtask-from-latch``.
#: Everything else is the Panda final version's value (finger cover, palm shell 5 mm, ROI carve, obstacle margin, wrist
#: tracking 0.1, after-grasp continuity / deflection-rate, hold-follow ...): task SL7 0.6, a change of those is a user
#: ruling. The attention head stays the Panda head ``noun_target:L8H6`` (SL7 0.2).
P0M_REMOVED_FLAGS: dict[str, int] = {      # flag -> number of value tokens
    "--hsr6-attach-assist": 0, "--hsr6-held-gate": 0, "--hsr6-target-exempt": 0, "--hsr6-objective": 0,
    "--hsr6-objective-scope": 1, "--nontarget-evidence": 1, "--to-multi-seed": 1, "--to-gpu-seeds": 1,
    "--to-gpu-seeds-check": 1, "--neighbour-contact-mm": 1, "--neighbour-palm-contact-mm": 1}
#: P0 target extent: ATS surface extraction on the GPU, no ball (T78 / T78b / T78f of development).
P0M_SURFACE_FLAGS: tuple[str, ...] = (
    "--target-extent", "attention-surface", "--surface-target-check", "overlap", "--surface-keep-last",
    "--surface-grasp-hold", "--surface-no-ball", "--attention-surface-device", "cuda")
#: pre-grasp hull fill + the 10 mm target contact allowance of the hand layer (development P0m; the "nohull" version
#: leaves them out -- development T92b).
P0M_HULL_FLAGS: tuple[str, ...] = ("--surface-hull-fill", "--surface-contact-allowance-mm", "10")
#: the 3-stage gate label from the grasp latch (T92); the kv_L4_v1 probe does not exist on the Panda.
P0M_SUBTASK_FLAGS: tuple[str, ...] = ("--subtask-from-latch",)
P0M_FLAGS: tuple[str, ...] = (_drop_flags(FINAL_CONFIG_FLAGS, P0M_REMOVED_FLAGS) + P0M_SURFACE_FLAGS + P0M_HULL_FLAGS
                              + P0M_SUBTASK_FLAGS)
P0M_NOHULL_FLAGS: tuple[str, ...] = (_drop_flags(FINAL_CONFIG_FLAGS, P0M_REMOVED_FLAGS) + P0M_SURFACE_FLAGS
                                     + P0M_SUBTASK_FLAGS)
#: the D1 control: P0m + the neighbour allowances that D1 switched off (development MuJoCo demo profile before D1;
#: measures what the D1 switch-off costs on LIBERO).
P0M_NBC_FLAGS: tuple[str, ...] = P0M_FLAGS + ("--neighbour-contact-mm", "20", "--neighbour-palm-contact-mm", "10")
#: SL7b (user ruling 2026-11-04 via development d2, task SL7b): the LIBERO pilot's P0m 0/9 came with saturated executed
#: actions; the TrajOpt side is split in two versions, target extent and phase labels stay P0m's.
#: (c) "termsoff": the continuity / deflection-rate terms off in every phase (as 3판 SL6-E), not only before the grasp.
P0M_TERMSOFF_FLAGS: tuple[str, ...] = (tuple(t for t in P0M_FLAGS if t not in ("--continuity-after-grasp",
                                                                               "--deflection-rate-after-grasp"))
                                       + CONTINUITY_DEFLECTION_OFF_FLAGS)
#: (b) "pandafix": the Panda corrections pc판 carries beyond P0m (V11AB-pb7 without the pb2 minimal-intervention gate that
#: E0 dropped): pb1 reach at capture, arm links in the collision model, held / closing hand exemptions, latch phase,
#: grasp zone, pb6 sticky reach and drop from open. pc판's HSR6 placement contact is not taken (HSR6 is off in P0m).
P0M_PANDA_FIX_FLAGS: tuple[str, ...] = (
    "--reach-at-capture", "--links", "arms", "--held-hand-exempt", "--latch-phase", "--grasp-zone-mm", "10",
    "--grasp-reach-sticky", "--grasp-drop-from-open", "--closing-hand-exempt")
P0M_PANDAFIX_FLAGS: tuple[str, ...] = P0M_FLAGS + P0M_PANDA_FIX_FLAGS
#: P0m versions: final-configuration base argv (no ``V11AB_PRE_MERGE_DEFAULTS``), **no** HSR6 probe.
#: (d) the chosen TrajOpt side + pass-through when the VLA reference is clear (T88, SL7b): both bases registered, the
#: SL7b rule picks the one that runs ((c) on a tie).
P0M_PASS_FLAGS: tuple[str, ...] = ("--passthrough-when-clear",)
#: SL7c ① (user approval 2026-11-04 via development d2, task SL7c): diagnostic only, **not an adoption candidate** -- P0m
#: with the target extent back to 1판's ball (``--target-exclusion ball-owned`` + ``--nontarget-evidence surface``): the
#: surface-extraction flags and hull fill / contact allowance (both surface-only) are removed, everything else is P0m's.
P0M_BALL_FLAGS: tuple[str, ...] = (_drop_flags(FINAL_CONFIG_FLAGS, {k: n for k, n in P0M_REMOVED_FLAGS.items()
                                                                     if k != "--nontarget-evidence"})
                                   + P0M_SUBTASK_FLAGS)
P0M_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-p0m", "AG3S-final-panda-p0m-nohull", "AG3S-final-panda-p0m-nbc",
                                 "AG3S-final-panda-p0m-termsoff", "AG3S-final-panda-p0m-pandafix",
                                 "AG3S-final-panda-p0m-termsoff-pass", "AG3S-final-panda-p0m-pandafix-pass",
                                 "AG3S-final-panda-p0m-ball", "AG3S-final-panda-p0m-l8h2",
                                 "AG3S-final-panda-p0m-gate")
AG3S_VERSIONS["AG3S-final-panda-p0m"] = P0M_FLAGS                       # P0m (hull fill + contact allowance 10 mm)
AG3S_VERSIONS["AG3S-final-panda-p0m-nohull"] = P0M_NOHULL_FLAGS         # P0m without hull fill / allowance (T92b control)
AG3S_VERSIONS["AG3S-final-panda-p0m-nbc"] = P0M_NBC_FLAGS               # P0m + neighbour allowances (D1 control)
AG3S_VERSIONS["AG3S-final-panda-p0m-termsoff"] = P0M_TERMSOFF_FLAGS     # (c) P0m + continuity / deflection terms off
AG3S_VERSIONS["AG3S-final-panda-p0m-pandafix"] = P0M_PANDAFIX_FLAGS     # (b) P0m + pc판 Panda corrections (no HSR6)
AG3S_VERSIONS["AG3S-final-panda-p0m-termsoff-pass"] = P0M_TERMSOFF_FLAGS + P0M_PASS_FLAGS   # (d) on (c)
AG3S_VERSIONS["AG3S-final-panda-p0m-pandafix-pass"] = P0M_PANDAFIX_FLAGS + P0M_PASS_FLAGS   # (d) on (b)
AG3S_VERSIONS["AG3S-final-panda-p0m-ball"] = P0M_BALL_FLAGS                 # SL7c ① ball extent (diagnostic)
#: SL7d (user ruling 2026-11-04, task SL7d): (가) P0m with the attention cell back to the Panda ATS signal of pc판
#: (``action:last:L8H2``, SL6-C user ruling 2026-10-30) -- P0m's ATS and surface seed map share it; everything else P0m's.
_i = P0M_FLAGS.index("--attention-source")
P0M_L8H2_FLAGS: tuple[str, ...] = P0M_FLAGS[:_i + 1] + ("action:last:L8H2",) + P0M_FLAGS[_i + 2:]
del _i
AG3S_VERSIONS["AG3S-final-panda-p0m-l8h2"] = P0M_L8H2_FLAGS               # SL7d (가)
AG3S_VERSIONS["AG3S-final-panda-p0m-gate"] = P0M_FLAGS + ("--surface-seed-ats-gate",)   # SL7f offline (L8H6 rows)
#: (나) pc판 + the latch label as the 3-stage gate label (T92); HSR6 rules and probe unchanged.
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-latch"] = FINAL_CONFIG_V3_E0TO_PC_FLAGS + P0M_SUBTASK_FLAGS   # SL7d (나)
#: final-configuration base argv + the HSR6 probe like :data:`FINAL_CONFIG_VERSIONS` (kept apart so that list stays the SL6 one)
PC_LATCH_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc-latch",)
#: SL7e (lead self-ruling 2026-11-04 in the user's autonomous window, task SL7e):
#: (C) "pc-surface" = pc판 + the P0 surface-extraction target extent (with hull fill / contact allowance, as P0m) on the
#: Panda ATS cell made explicit (``--attention-source action:last:L8H2``, the seed map); ``--nontarget-evidence`` is
#: removed (the config refuses it with the surface extent). HSR6 rules and probe stay. Candidate, not adopted.
_pc = list(FINAL_CONFIG_V3_E0TO_PC_FLAGS)
_j = _pc.index("--nontarget-evidence")
PC_SURFACE_FLAGS: tuple[str, ...] = (tuple(_pc[:_j] + _pc[_j + 2:]) + ("--attention-source", "action:last:L8H2")
                                     + P0M_SURFACE_FLAGS + P0M_HULL_FLAGS)
#: (A) "pc-nohsr6" = pc판 minus every HSR6 rule (attach assist, held gate, target exempt, objective + scope,
#: placement contact) -- diagnostic only; no probe.
PC_HSR6_FLAGS: dict[str, int] = {"--hsr6-attach-assist": 0, "--hsr6-held-gate": 0, "--hsr6-target-exempt": 0,
                                 "--hsr6-objective": 0, "--hsr6-objective-scope": 1, "--hsr6-placement-contact": 0}
PC_NOHSR6_FLAGS: tuple[str, ...] = _drop_flags(FINAL_CONFIG_V3_E0TO_PC_FLAGS, PC_HSR6_FLAGS)
del _pc, _j
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-surface"] = PC_SURFACE_FLAGS     # SL7e (C) candidate
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-nohsr6"] = PC_NOHSR6_FLAGS       # SL7e (A) diagnostic
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + ("AG3S-final-panda-v3-e0to-pc-surface",)   # probe-taking pc variants
PC_NOPROBE_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc-nohsr6",)
#: SL7f (lead self-ruling D4, 2026-11-04): the surface seed map kept on the ATS target's patches (`--surface-seed-ats-gate`)
#: -- on pc-surface (candidate) and on P0m (offline SafeLIBERO replay with the stored L8H6 rows).
SEED_GATE_FLAGS: tuple[str, ...] = ("--surface-seed-ats-gate",)
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-surface-gate"] = PC_SURFACE_FLAGS + SEED_GATE_FLAGS   # SL7f candidate
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + ("AG3S-final-panda-v3-e0to-pc-surface-gate",)
#: SL7g (lead self-ruling D5, 2026-11-04): surface where accepted, ball otherwise -- pc판 + the surface extent with the
#: ATS overlap check but without `--surface-keep-last` / `--surface-grasp-hold` / `--surface-no-ball` (and hull fill /
#: contact allowance, which need them): a request without an accepted region falls back to the `estimated` (ball) extent.
PC_SURFACE_FALLBACK_FLAGS: tuple[str, ...] = (
    tuple(PC_SURFACE_FLAGS[:len(PC_SURFACE_FLAGS) - len(P0M_SURFACE_FLAGS) - len(P0M_HULL_FLAGS)])
    + ("--target-extent", "attention-surface", "--surface-target-check", "overlap", "--attention-surface-device", "cuda"))
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-surface-fb"] = PC_SURFACE_FALLBACK_FLAGS                       # SL7g
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-surface-fb-gate"] = PC_SURFACE_FALLBACK_FLAGS + SEED_GATE_FLAGS  # SL7g
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + ("AG3S-final-panda-v3-e0to-pc-surface-fb", "AG3S-final-panda-v3-e0to-pc-surface-fb-gate")
#: SL7j (lead self-ruling D12, 2026-11-04): the two non-surface differences between pc판 and `-pc-surface-fb`, one at a
#: time on pc판 -- (a) the explicit `--attention-source action:last:L8H2` (the extended path instead of the default one),
#: (b) without `--nontarget-evidence surface`. Diagnostic only (task 7 of LIBERO spatial diverged at call 4).
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-l8h2explicit"] = FINAL_CONFIG_V3_E0TO_PC_FLAGS + (
    "--attention-source", "action:last:L8H2")
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-nonte"] = _drop_flags(
    FINAL_CONFIG_V3_E0TO_PC_FLAGS, {"--nontarget-evidence": 1})
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + ("AG3S-final-panda-v3-e0to-pc-l8h2explicit", "AG3S-final-panda-v3-e0to-pc-nonte")
#: SL8b (user ruling 2026-11-04 15:1x KST, task SL8b): loosen the lock on the first-chosen object, pc판 + (idwin) the
#: window-vote switch W=5, M=3, (idattn) the seed-attention release K=4, F=0.7, R=15 mm, (idboth) both. Candidates for the
#: offline replay, then the closed loop; all three flags default off, so pc판 itself is unchanged.
ID_WINDOW_FLAGS: tuple[str, ...] = ("--identity-switch-window", "5", "--identity-switch-votes", "3")
ID_ATTENTION_FLAGS: tuple[str, ...] = ("--identity-attention-release", "4", "--identity-attention-off-frac", "0.7",
                                       "--identity-attention-radius-mm", "15")
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-idwin"] = FINAL_CONFIG_V3_E0TO_PC_FLAGS + ID_WINDOW_FLAGS
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-idattn"] = FINAL_CONFIG_V3_E0TO_PC_FLAGS + ID_ATTENTION_FLAGS
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-idboth"] = FINAL_CONFIG_V3_E0TO_PC_FLAGS + ID_WINDOW_FLAGS + ID_ATTENTION_FLAGS
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + ("AG3S-final-panda-v3-e0to-pc-idwin", "AG3S-final-panda-v3-e0to-pc-idattn",
                                         "AG3S-final-panda-v3-e0to-pc-idboth")
#: SL9 (task SL9 §1-2, lead 2026-11-04; benchmark user ruling 17:3x-17:5x KST): the development proposals A (target as a
#: closed solid, T96 / T96c / T96d) and B (removed-seed sign fix, T96e) as ablations on the three bases. A replaces hull
#: fill + the 10 mm contact allowance (the config refuses it together with them, so it cannot sit on the fallback base);
#: B is one flag on any attention-surface base. C (real-size palm cover) is not applicable on the Panda (task SL9 §4-1).
SOLID_REMOVED_FLAGS: dict[str, int] = {"--surface-hull-fill": 0, "--surface-contact-allowance-mm": 1}
SOLID_FLAGS: tuple[str, ...] = ("--surface-solid-owned", "--surface-solid-protect-mm", "5", "--surface-solid-dilate-mm", "0",
                                "--surface-solid-dn-ever-target", "--surface-solid-shadow-mm", "80")
SIGN_FREE_FLAGS: tuple[str, ...] = ("--surface-removed-sign-free",)
_SL9 = {
    # name: (base flags, A, B)
    "AG3S-final-panda-p0m-B": (P0M_FLAGS, False, True),
    "AG3S-final-panda-p0m-A": (P0M_FLAGS, True, False),
    "AG3S-final-panda-p0m-AB": (P0M_FLAGS, True, True),
    "AG3S-final-panda-v3-e0to-pc-surface-fb-B": (PC_SURFACE_FALLBACK_FLAGS, False, True),
    "AG3S-final-panda-v3-e0to-pc-surface-B": (PC_SURFACE_FLAGS, False, True),
    "AG3S-final-panda-v3-e0to-pc-surface-A": (PC_SURFACE_FLAGS, True, False),
    "AG3S-final-panda-v3-e0to-pc-surface-AB": (PC_SURFACE_FLAGS, True, True),
}
for _n, (_base, _a, _b) in _SL9.items():
    AG3S_VERSIONS[_n] = ((_drop_flags(_base, SOLID_REMOVED_FLAGS) + SOLID_FLAGS) if _a else tuple(_base)) \
        + (SIGN_FREE_FLAGS if _b else ())
del _n, _base, _a, _b
#: kept apart from :data:`P0M_VERSIONS` / the pc lists' dest scan so that an older version's recorded flag values keep
#: exactly the keys they had (:func:`version_flag_values` adds the SL9 dests only when one is set); same handling otherwise:
#: the P0m ones take no HSR6 probe, the pc-surface ones take it like their base.
P0M_SL9_VERSIONS: tuple[str, ...] = tuple(n for n in _SL9 if "-p0m-" in n)
PC_SL9_VERSIONS: tuple[str, ...] = tuple(n for n in _SL9 if "-pc-" in n)
SL9_VERSIONS: tuple[str, ...] = P0M_SL9_VERSIONS + PC_SL9_VERSIONS
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + PC_SL9_VERSIONS
#: SL10 §2.2 (task SL10, lead 2026-11-04): the Panda controller's action limit as TrajOpt rows (``--to-action-box``,
#: margin 0.05; ``AG3S/trajopt/action_box.py``) on pc판 and on pc판 + surface extraction -- the base flags plus this one flag.
ACTION_BOX_FLAGS: tuple[str, ...] = ("--to-action-box",)
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-abox"] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]) + ACTION_BOX_FLAGS
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-surface-abox"] = (
    tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-surface"]) + ACTION_BOX_FLAGS)
#: probe-taking pc variants like the SL9 ones; kept apart so that an earlier version's recorded flag values keep their keys
#: (:func:`version_flag_values` adds ``to_action_box`` only when it is set).
SL10_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc-abox", "AG3S-final-panda-v3-e0to-pc-surface-abox")
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + SL10_VERSIONS
#: SL10b (task SL10b, lead 2026-11-05): "쥔 뒤 변화율 항 끔" -- pc판 / pc판 + surface with the two after-grasp flags removed and the
#: continuity / deflection-rate terms off in every phase (the P0M_TERMSOFF_FLAGS construction, applied to the pc bases). Before the
#: grasp both bases already have the terms off, so the difference lives in the grasping / held calls only. Every dest is one an
#: earlier version already sets, so the recorded flag values keep their keys.
_AFTER_GRASP_FLAGS = ("--continuity-after-grasp", "--deflection-rate-after-grasp")
SL10B_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc-termsoff", "AG3S-final-panda-v3-e0to-pc-surface-termsoff")
for _v in SL10B_VERSIONS:
    _base = _v[:-len("-termsoff")]
    assert all(t in AG3S_VERSIONS[_base] for t in _AFTER_GRASP_FLAGS), _base
    AG3S_VERSIONS[_v] = tuple(t for t in AG3S_VERSIONS[_base] if t not in _AFTER_GRASP_FLAGS) + CONTINUITY_DEFLECTION_OFF_FLAGS
del _v, _base
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + SL10B_VERSIONS
#: SL10d (task SL10d, lead 2026-11-05): "쥔 물체 비우기 = 충돌 구" -- pc판 / pc판 + surface plus ``--held-free-query-spheres`` (the held
#: volume the TSDF frees also covers the spheres the solver is queried with). The base flags plus this one flag; the dest is a new one,
#: so :func:`version_flag_values` adds it only when it is set.
HELD_FREE_QUERY_FLAGS: tuple[str, ...] = ("--held-free-query-spheres",)
SL10D_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc-hfq", "AG3S-final-panda-v3-e0to-pc-surface-hfq")
for _v in SL10D_VERSIONS:
    AG3S_VERSIONS[_v] = tuple(AG3S_VERSIONS[_v[:-len("-hfq")]]) + HELD_FREE_QUERY_FLAGS
del _v
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + SL10D_VERSIONS
#: SL10e (task SL10e, lead 2026-11-05): "들어 올림 때도 아래 받침면 허용" -- pc판 / pc판 + surface plus ``--hsr6-placement-contact-lift``
#: (HSR6 ⑤'s phase condition placement -> placement or lift). The base flags plus this one flag; the dest is a new one, so
#: :func:`version_flag_values` adds it only when it is set.
PLACEMENT_LIFT_FLAGS: tuple[str, ...] = ("--hsr6-placement-contact-lift",)
SL10E_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc-plift", "AG3S-final-panda-v3-e0to-pc-surface-plift")
for _v in SL10E_VERSIONS:
    AG3S_VERSIONS[_v] = tuple(AG3S_VERSIONS[_v[:-len("-plift")]]) + PLACEMENT_LIFT_FLAGS
del _v
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + SL10E_VERSIONS
#: SL10f (task SL10f, lead 2026-11-05): "미인증 HOLD 풀기 진단" -- pc판 plus the existing serve_safe flag ``--allow-uncertified``
#: (``safety.require_certified_geometry=False``: an uncertified call is ``uncertified_waived`` -> execute instead of HOLD). Diagnostic
#: only, not a candidate. No new code path; the dest is new to the version records, so :func:`version_flag_values` adds it only when set.
UNCERTIFIED_WAIVE_FLAGS: tuple[str, ...] = ("--allow-uncertified",)
SL10F_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc-uncwaive",)
for _v in SL10F_VERSIONS:
    AG3S_VERSIONS[_v] = tuple(AG3S_VERSIONS[_v[:-len("-uncwaive")]]) + UNCERTIFIED_WAIVE_FLAGS
del _v
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + SL10F_VERSIONS
#: SL11 (task SL11, lead 2026-11-05): "장애물 없는 실패의 기본 원인 고치기 -- 통과 판". Three pc-based versions:
#:   ``-pc-mi10``       pc판 + the existing pb2 gate ``--minimal-intervention-mm 10`` (the VLA plan clears every certified row by
#:                      10 mm or more -> it is the plan, so the conversion passes the policy action bit for bit);
#:   ``-pc-mi10-sdev``  mi10 + ``--smooth-on-deviation`` (smoothness on D2(Q - Q_ref), zero for a smooth VLA plan);
#:   ``-pc-observe``    pc판 + ``--observe-only`` (diagnostic: the client always gets the policy action, never a HOLD; the verdict
#:                      and the converted actions the server computed are recorded).
#: ``--minimal-intervention-mm`` is a V11AB-family dest (every version records it, None here); the two new dests join a record only
#: when set (see :func:`version_flag_values`).
MINIMAL_INTERVENTION_FLAGS: tuple[str, ...] = ("--minimal-intervention-mm", "10")
SMOOTH_ON_DEVIATION_FLAGS: tuple[str, ...] = ("--smooth-on-deviation",)
OBSERVE_ONLY_FLAGS: tuple[str, ...] = ("--observe-only",)
SL11_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc-mi10", "AG3S-final-panda-v3-e0to-pc-mi10-sdev",
                                  "AG3S-final-panda-v3-e0to-pc-observe")
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-mi10"] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]) + MINIMAL_INTERVENTION_FLAGS
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-mi10-sdev"] = (
    tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-mi10"]) + SMOOTH_ON_DEVIATION_FLAGS)
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-observe"] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]) + OBSERVE_ONLY_FLAGS
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + SL11_VERSIONS
#: SL11 (task SL11, user ruling 2026-11-05): "fork 응답 모델 맞추기" -- any Panda version ``X`` can be run as ``X-lsfork`` = ``X`` plus
#: ``--response-model libero_safety_fork`` (conversion C uses the LIBERO-Safety fork controller's lag model; LIBERO-Safety client with
#: ``libero_to_safety`` only -- the server refuses other clients). Generic: :func:`register_lsfork` adds the entry for one base version
#: and gives it the base's final-version / probe membership (:func:`_lsfork_base`); to run another version on LIBERO-Safety add its
#: name to :data:`LSFORK_BASES` (or call ``register_lsfork``) after that version is registered.
LSFORK_SUFFIX = "-lsfork"
LSFORK_FLAGS: tuple[str, ...] = ("--response-model", "libero_safety_fork")
LSFORK_VERSIONS: tuple[str, ...] = ()


def _lsfork_base(version: str) -> str:
    """The version an ``-lsfork`` entry is built on (any other name: itself)."""
    version = str(version)
    return version[:-len(LSFORK_SUFFIX)] if version in LSFORK_VERSIONS else version


def register_lsfork(base: str) -> str:
    """Register ``base + "-lsfork"`` (idempotent); returns its name. ``base`` must be a registered Panda version."""
    global LSFORK_VERSIONS, PC_LATCH_VERSIONS
    if base not in AG3S_VERSIONS or base.endswith(LSFORK_SUFFIX):
        raise ValueError(f"cannot add {LSFORK_SUFFIX} to {base!r}")
    name = base + LSFORK_SUFFIX
    if name not in AG3S_VERSIONS:
        AG3S_VERSIONS[name] = tuple(AG3S_VERSIONS[base]) + LSFORK_FLAGS
        LSFORK_VERSIONS = LSFORK_VERSIONS + (name,)
    return name


LSFORK_BASES: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc", "AG3S-final-panda-v3-e0to-pc-mi10",
                                 "AG3S-final-panda-v3-e0to-pc-mi10-sdev")
for _b in LSFORK_BASES:
    register_lsfork(_b)
del _b
#: SL11 §5 (task SL11 §5, user ruling 2026-11-05): the grasp latch measures the opening as the gap of the two fingers.
#:   ``-pc-gap``       pc판 + ``--gripper-gap-opening``;   ``-pc-mi10-gap``   ``-pc-mi10`` + the flag.
#: Registered after the ``-lsfork`` loop, so they have no ``-lsfork`` twin (add one with ``register_lsfork``/``LSFORK_BASES`` if wanted).
#: They are not in :data:`SL11_VERSIONS` (that tuple is the three pass versions); their record dest joins by its own group
#: (:func:`version_flag_values`), so the record of every earlier version, sdev / observe / lsfork included, keeps its keys.
GAP_OPENING_FLAGS: tuple[str, ...] = ("--gripper-gap-opening",)
SL11_GAP_VERSIONS: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc-gap", "AG3S-final-panda-v3-e0to-pc-mi10-gap")
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-gap"] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]) + GAP_OPENING_FLAGS
AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-mi10-gap"] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc-mi10"]) + GAP_OPENING_FLAGS
PC_LATCH_VERSIONS = PC_LATCH_VERSIONS + SL11_GAP_VERSIONS
#: SL12 (task SL12, lead 2026-11-05): "정답 target 진단" -- the RB-Y1 T46e visible-target oracle on the Panda (diagnostic, not a
#: candidate). ``-pc-oraclegt`` = pc판, ``-pc-mi10-oraclegt`` = ``-pc-mi10``, each with ``--target-extent oracle-visible
#: --oracle-extent-identity gt``: the fine target-free layer removes the GT target's visible surface (+ truncation band) from the
#: client's MuJoCo code image (``benchmark/adapters/libero_panda/oracle_codes.py``; the LIBERO / LIBERO-Safety clients send it
#: with ``--oracle-segmentation``, which the launchers pass for these versions). One more change, forced by the config:
#: ``--nontarget-evidence surface`` is removed (the oracle replaces the ball-owned split it acts on; the config refuses both).
#: The ATS cell (``--attention-source`` default ``action:last:L8H2``), HSR6 rules and probe, the ball for the coarse layer and the
#: permission, and everything else are the base version's -- no RB-Y1-specific attention prerequisite applies (oracle-visible reads
#: no attention). ``--oracle-extent-identity gt``: the removed object is always the GT target (the ATS pick only decides the
#: manipulated geometry / latch, recorded as ``identity_matches_gt``). Its dests join a record only when set.
ORACLE_GT_FLAGS: tuple[str, ...] = ("--target-extent", "oracle-visible", "--oracle-extent-identity", "gt")
SL12_ORACLE_SUFFIX = "-oraclegt"
SL12_ORACLE_BASES: tuple[str, ...] = ("AG3S-final-panda-v3-e0to-pc", "AG3S-final-panda-v3-e0to-pc-mi10")
SL12_VERSIONS: tuple[str, ...] = tuple(b + SL12_ORACLE_SUFFIX for b in SL12_ORACLE_BASES)
for _b in SL12_ORACLE_BASES:
    AG3S_VERSIONS[_b + SL12_ORACLE_SUFFIX] = _drop_flags(tuple(AG3S_VERSIONS[_b]), {"--nontarget-evidence": 1}) + ORACLE_GT_FLAGS
del _b


#: SL13 (task SL13, lead 2026-11-05): "정답 target 부피 지우기" -- ``-pc-oraclevol`` = ``-pc-oraclegt`` + ``--oracle-target-volume``,
#: ``-pc-mi10-oraclevol`` = ``-pc-mi10-oraclegt`` + the flag: the oracle-visible removal also takes every seed of the fine target-free
#: layer inside the GT target volume (convex hull of the code-1 samples' xy + one voxel, ``z_floor`` .. top + one voxel), whoever owns it.
#: Diagnostic, not a candidate. Their record dest (``oracle_target_volume``) joins a record only when set (:func:`version_flag_values`).
ORACLE_VOLUME_FLAGS: tuple[str, ...] = ("--oracle-target-volume",)
SL13_ORACLE_VOLUME_SUFFIX = "-oraclevol"
SL13_VERSIONS: tuple[str, ...] = tuple(b + SL13_ORACLE_VOLUME_SUFFIX for b in SL12_ORACLE_BASES)
for _b in SL12_ORACLE_BASES:
    AG3S_VERSIONS[_b + SL13_ORACLE_VOLUME_SUFFIX] = tuple(AG3S_VERSIONS[_b + SL12_ORACLE_SUFFIX]) + ORACLE_VOLUME_FLAGS
del _b

#: SL14 (task SL14, lead 2026-11-05): "층별 원인 진단" -- two diagnostic / repair versions of the pc판 grasp latch, each = pc판 + one flag.
#:   ``-pc-oraclelatch``  + ``--oracle-latch``: the latch's attach / detach follow the simulator's ground truth (the LIBERO / LIBERO-Safety clients
#:                        send ``ag3s/oracle_grasp`` with ``--oracle-grasp``, which the launchers pass for this version). Diagnostic, not a candidate.
#:   ``-pc-rearm``        + ``--placed-rearm``: PLACED is not absorbing (an open hand after a release may grasp again). Deployable (no oracle).
#: Like SL12 / SL13 they are not in :data:`PC_LATCH_VERSIONS` (so no earlier list slice moves), have no ``-lsfork`` twin, and their record dests
#: join a record only when set (:func:`version_flag_values`).
ORACLE_LATCH_FLAGS: tuple[str, ...] = ("--oracle-latch",)
PLACED_REARM_FLAGS: tuple[str, ...] = ("--placed-rearm",)
OPENING_RULE_FLAGS: tuple[str, ...] = ("--grasp-opening-rule",)
SL14_ORACLE_LATCH_VERSION = "AG3S-final-panda-v3-e0to-pc-oraclelatch"
SL14_REARM_VERSION = "AG3S-final-panda-v3-e0to-pc-rearm"
SL14_OPENRULE_VERSION = "AG3S-final-panda-v3-e0to-pc-openrule"
SL14_VERSIONS: tuple[str, ...] = (SL14_ORACLE_LATCH_VERSION, SL14_REARM_VERSION, SL14_OPENRULE_VERSION)
AG3S_VERSIONS[SL14_ORACLE_LATCH_VERSION] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]) + ORACLE_LATCH_FLAGS
AG3S_VERSIONS[SL14_REARM_VERSION] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]) + PLACED_REARM_FLAGS
AG3S_VERSIONS[SL14_OPENRULE_VERSION] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]) + OPENING_RULE_FLAGS


#: SL15 (task SL15, lead 2026-11-05): "비권한 link 의 target 여유" -- ``--nonauthorized-target-margin obstacle`` (T44e A9) on the two pc
#: versions that carry a grasp-latch repair: the links that may not touch the target (``panda_link1..7`` with ``--links arms``) keep the
#: obstacle rows' clearance from the target instead of ``geometry.safety_margin`` (50 mm). Contact permission itself is unchanged.
#:   ``-pc-ntm``           = ``-pc``           + the flag.
#:   ``-pc-openrule-ntm``  = ``-pc-openrule``  + the flag.
#: Like SL14 they are not in :data:`PC_LATCH_VERSIONS`, have no ``-lsfork`` twin, and the record dest (``nonauthorized_target_margin``)
#: joins a record only when set (:func:`version_flag_values`). The flag was refused by the Panda server until SL15 (it is now in
#: ``server.PORTED_FINAL_CONFIG_DESTS``; the config reaches AG3S through ``v11ab_config_dict`` as ``contact.nonauthorized_target_margin``).
NONAUTHORIZED_TARGET_MARGIN_FLAGS: tuple[str, ...] = ("--nonauthorized-target-margin", "obstacle")
SL15_NTM_VERSION = "AG3S-final-panda-v3-e0to-pc-ntm"
SL15_OPENRULE_NTM_VERSION = "AG3S-final-panda-v3-e0to-pc-openrule-ntm"
SL15_NTM_VERSIONS: tuple[str, ...] = (SL15_NTM_VERSION, SL15_OPENRULE_NTM_VERSION)
AG3S_VERSIONS[SL15_NTM_VERSION] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]) + NONAUTHORIZED_TARGET_MARGIN_FLAGS
AG3S_VERSIONS[SL15_OPENRULE_NTM_VERSION] = tuple(AG3S_VERSIONS[SL14_OPENRULE_VERSION]) + NONAUTHORIZED_TARGET_MARGIN_FLAGS


#: SL15 sfhand (task SL15, lead 2026-11-05): "자기 손이 self-filter 를 새어 나간다" -- the constraint-recorded run
#: (``sl15-DREC``) shows that, during the CLOSING latch phase, most robot rows near occupied voxels are ``panda_hand`` rows whose
#: nearest MuJoCo body is the robot's own hand (``gripper0_right_gripper``): hand surface the inflation-0 self-filter let into the
#: field, so TrajOpt avoids its own hand. ``-pc-sfhand`` = ``-pc`` + a per-link self-filter margin on ``panda_hand`` only
#: (``--self-filter-inflation-link panda_hand=<m>``, ``pointcloud.self_filter_inflation_by_link``; the other links stay at the
#: default 0). The value is the p95 leak distance of the hand points, rounded up to 1 mm and capped at 15 mm
#: (``AG3S/trajopt/experiments/panda_sfhand_leak.py``; ``AG3S/ag3s/docs/handoff/SL15.sfhand.impl.md``).
#: Like the other SL14 / SL15 versions it is not in :data:`PC_LATCH_VERSIONS`, has no ``-lsfork`` twin, and the record dest
#: (``self_filter_inflation_link``) joins a record only when set (:func:`version_flag_values`).
SL15_SFHAND_PANDA_HAND_M: float = 0.004
SELF_FILTER_HAND_FLAGS: tuple[str, ...] = ("--self-filter-inflation-link", f"panda_hand={SL15_SFHAND_PANDA_HAND_M:g}")
SL15_SFHAND_VERSION = "AG3S-final-panda-v3-e0to-pc-sfhand"
SL15_SFHAND_VERSIONS: tuple[str, ...] = (SL15_SFHAND_VERSION,)
AG3S_VERSIONS[SL15_SFHAND_VERSION] = tuple(AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]) + SELF_FILTER_HAND_FLAGS


#: SL16 HSC (task SL16 §2, lead 2026-11-06; user: "쥔 물체와 받침면의 접촉은 당연히 허용해야 합니다"): while the call is actually holding
#: (latch HELD, the HELD that ``--grasp-opening-rule`` sets included; CLOSING only with ``closed`` ∧ ``blocked``) the held rows and the grasping
#: hand's finger / palm rows may touch the surface beneath them in **every** HSR6 phase (``--held-support-contact``; the finger / palm overlap
#: is ``--held-support-contact-finger-mm``, default 10). ``-pc-openrule-hsc`` = ``-pc-openrule`` + the flag. Like the other SL14 / SL15 versions it
#: is not in :data:`PC_LATCH_VERSIONS`, has no ``-lsfork`` twin, and the record dest (``held_support_contact``) joins a record only when set
#: (:func:`version_flag_values`). The run conversion D is chosen by the launcher, not by the version (task SL16 §2).
HELD_SUPPORT_CONTACT_FLAGS: tuple[str, ...] = ("--held-support-contact",)
SL16_HSC_VERSION = "AG3S-final-panda-v3-e0to-pc-openrule-hsc"
SL16_HSC_VERSIONS: tuple[str, ...] = (SL16_HSC_VERSION,)
AG3S_VERSIONS[SL16_HSC_VERSION] = tuple(AG3S_VERSIONS[SL14_OPENRULE_VERSION]) + HELD_SUPPORT_CONTACT_FLAGS
#: SL16 HVC (task SL16, lead 2026-11-06): the occupied FINE voxels the MuJoCo-GT hand owns sit 13.6 mm (median) / 29 mm (p95) from the
#: self-filter's hand spheres, while the depth leak is 4 mm (p95; SL15 sfhand) -- the field holds hand-owned voxels the filter does not
#: reach, and TrajOpt avoids them. ``AG3S/trajopt/experiments/panda_hvc_origin.py`` split them by the fixed rule (current / trail /
#: smear; ``AG3S/ag3s/docs/handoff/SL16.hvc.data.json``): trail share 0.21 < 0.5 -> mode ``current``, r = p90 of the hand-voxel distance
#: (23.6 mm) capped at 20 mm. ``-pc-openrule-hvc`` = ``-pc-openrule`` + ``--hand-voxel-carve current --hand-voxel-carve-mm 20``. Like
#: the other SL14 / SL15 versions: not in :data:`PC_LATCH_VERSIONS`, no ``-lsfork`` twin, and the record dests join a record only when
#: set (:func:`version_flag_values`). Panda only (the RB-Y1 server refuses the flags).
SL16_HVC_MODE = "current"
SL16_HVC_MM = 20.0
HAND_VOXEL_CARVE_FLAGS: tuple[str, ...] = ("--hand-voxel-carve", SL16_HVC_MODE, "--hand-voxel-carve-mm", f"{SL16_HVC_MM:g}")
SL16_HVC_VERSION = "AG3S-final-panda-v3-e0to-pc-openrule-hvc"
SL16_HVC_VERSIONS: tuple[str, ...] = (SL16_HVC_VERSION,)
AG3S_VERSIONS[SL16_HVC_VERSION] = tuple(AG3S_VERSIONS[SL14_OPENRULE_VERSION]) + HAND_VOXEL_CARVE_FLAGS


def needs_oracle_grasp(version: str) -> bool:
    """SL14: whether the version's server runs ``--oracle-latch`` (the client must send ``ag3s/oracle_grasp``, ``--oracle-grasp``)."""
    return "--oracle-latch" in AG3S_VERSIONS.get(str(version), ())


def is_gtto_version(version: str) -> bool:
    """SL-GTTO (2026-11-06 merge): one of the common-TO GT-perception versions (``GT_VERSIONS`` ... ``GT9_VERSIONS``). They
    run ``--target-extent oracle-visible`` with their own client path (``--oracle-seg``, ``gtto_oracle_codes.py``), so the SL12
    client path (:func:`needs_oracle_segmentation`) leaves them out and :func:`version_requires_oracle_segmentation` takes them."""
    return str(version) in (GT_VERSIONS + GT2_VERSIONS + GT3_VERSIONS + GT4_VERSIONS + GT5_VERSIONS + GT6_VERSIONS
                            + GT7_VERSIONS + GT8_VERSIONS + GT9_VERSIONS + GT10_VERSIONS + GT11_VERSIONS
                            + GT12_VERSIONS + GT13_VERSIONS + GT14_VERSIONS + GT15_VERSIONS)


def needs_oracle_segmentation(version: str) -> bool:
    """SL12: whether the version's server runs ``--target-extent oracle-visible`` (the client must send the GT code image).
    The SL-GTTO versions are not SL12 ones (:func:`is_gtto_version`; their client flag is ``--oracle-seg``)."""
    if is_gtto_version(version):
        return False
    flags = AG3S_VERSIONS.get(str(version), ())
    return "--target-extent" in flags and flags[list(flags).index("--target-extent") + 1] == "oracle-visible"


# --- SL-GTTO — begin: the common TO with GT perception (task SL-GTTO, lead 2026-11-04; flag table: SL-GTTO.impl.md) ----
#: The common TO candidate (task §1) = the Panda final flags (:data:`FINAL_CONFIG_FLAGS`) minus every AG3S-perception
#: flag (attention source / attention split / split-supersedes-anchor / non-target evidence), minus the HSR6 rules (no
#: probe), minus TrajOpt multi-seed (E0), minus the D1 neighbour allowances; plus the grasp handling of pc판 / P0m
#: (:data:`P0M_PANDA_FIX_FLAGS`, :data:`P0M_SUBTASK_FLAGS`); plus the GT perception (task §2): the visible-target
#: oracle extent with the GT identity and the GT grounding input. Every other flag keeps the final version's value
#: (ambiguous -> base value, task §1).
GT_REMOVED_FLAGS: dict[str, int] = {      # flag -> number of value tokens
    # AG3S perception (identity / extent from attention): replaced by the GT perception below
    "--attention-source": 1, "--attention-split-radius-mm": 1, "--attention-split-ratio-source": 1,
    "--attention-split-partition": 0, "--split-supersedes-anchor": 0, "--nontarget-evidence": 1,
    # HSR6 rules (no probe)
    "--hsr6-attach-assist": 0, "--hsr6-held-gate": 0, "--hsr6-target-exempt": 0, "--hsr6-objective": 0,
    "--hsr6-objective-scope": 1,
    # TrajOpt multi-seed (E0: one VLA chunk as the seed)
    "--to-multi-seed": 1, "--to-gpu-seeds": 1, "--to-gpu-seeds-check": 1,
    # scene relaxations (D1)
    "--neighbour-contact-mm": 1, "--neighbour-palm-contact-mm": 1}
#: GT perception (task §2): the fine target-free layer from the GT code images (T46e, identity = the GT target) and the
#: grounding input = this request's code-1 points (``clustering.grounding_source = oracle``).
GT_PERCEPTION_FLAGS: tuple[str, ...] = ("--target-extent", "oracle-visible", "--oracle-extent-identity", "gt",
                                        "--grounding-source", "oracle")
COMMONTO_GT_FLAGS: tuple[str, ...] = (_drop_flags(FINAL_CONFIG_FLAGS, GT_REMOVED_FLAGS) + P0M_PANDA_FIX_FLAGS
                                      + P0M_SUBTASK_FLAGS + GT_PERCEPTION_FLAGS)
#: Final-configuration base argv, no HSR6 probe, no ``--attention-source`` per run (no ATS in the grounding).
GT_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt", "AG3S-commonto-gt-pass")
AG3S_VERSIONS["AG3S-commonto-gt"] = COMMONTO_GT_FLAGS                              # main version (task §5)
AG3S_VERSIONS["AG3S-commonto-gt-pass"] = COMMONTO_GT_FLAGS + P0M_PASS_FLAGS        # + pass-through when clear (T88)


def version_requires_oracle_segmentation(version: str) -> bool:
    """SL-GTTO — the version's server needs the client's GT code images (``--target-extent oracle-visible``: the
    server metadata says ``oracle_segmentation_required``); the launchers then pass ``--oracle-seg
    --capture-segmentation`` to the client."""
    if not is_gtto_version(version):   # 2026-11-06 merge: the SL12 / SL13 oracle versions use --oracle-segmentation
        return False
    flags = list(AG3S_VERSIONS.get(str(version), ()))
    return any(flags[i] == "--target-extent" and flags[i + 1] == "oracle-visible" for i in range(len(flags) - 1))
# --- SL-GTTO — end
# --- SL-GTTO2 — begin: the GT identity / extent reaches the TO (task SL-GTTO2 §1, lead 2026-11-05) ----------------------
#: F1 ``--oracle-admissible``: the GT cluster is not refused by TargetConfirm admissibility / the exclusion gate
#: (recorded only); F2 ``--oracle-held-target``: while closing / held, the GT target's pixels are masked out of the TSDF.
GT2_FLAGS: tuple[str, ...] = ("--oracle-admissible", "--oracle-held-target")
#: Same base argv and per-run rules as :data:`GT_VERSIONS` (kept apart so their recorded flag keys stay as they were).
GT2_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2",)
AG3S_VERSIONS["AG3S-commonto-gt2"] = COMMONTO_GT_FLAGS + GT2_FLAGS                   # SL-GTTO2 main version
# --- SL-GTTO2 — end
# --- SL-GTTO3 — begin: TrajOpt-side variants of the GT2 version (lead self-ruling G26, 2026-11-05 03:5x KST) ----------
#: gt2 smoke: once the TrajOpt bends the VLA path (reference deviation >= 0.05), conversion C saturates (median max
#: |correction| 3.9 at 0.05-0.2, 12.3 at >= 0.2) -- in held / placed and, with F1, latched calls. SL7b ruled the TO out
#: under P0m only because P0m's perception blocked first; the same two TO changes are tried here on the GT2 version.
#: (termsoff) the continuity / deflection-rate terms off in every phase (as 3판 SL6-E / SL7b (c)); (pass) VLA chunk
#: through untouched when it has no violation (T88, SL7b (d)); (termsoff-pass) both.
_AFTER_GRASP_TERMS = ("--continuity-after-grasp", "--deflection-rate-after-grasp")
GT3_TERMSOFF_FLAGS: tuple[str, ...] = (tuple(t for t in COMMONTO_GT_FLAGS + GT2_FLAGS if t not in _AFTER_GRASP_TERMS)
                                       + CONTINUITY_DEFLECTION_OFF_FLAGS)
GT3_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff", "AG3S-commonto-gt2-pass",
                                 "AG3S-commonto-gt2-termsoff-pass")
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff"] = GT3_TERMSOFF_FLAGS
AG3S_VERSIONS["AG3S-commonto-gt2-pass"] = COMMONTO_GT_FLAGS + GT2_FLAGS + P0M_PASS_FLAGS
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass"] = GT3_TERMSOFF_FLAGS + P0M_PASS_FLAGS
del _AFTER_GRASP_TERMS
# --- SL-GTTO3 — end
# --- SL-GTTO4 — begin: the held object may rest on what is beneath it (lead self-ruling G29, 2026-11-05 07:3x KST) ------
#: gt2 30 scenes: in held calls the VLA reference brings the attached held model's bottom within 15 mm of the support
#: plane in most calls, and the TO deviation there is 3-10x the other held calls (median 0.18-0.29 vs 0.02-0.06); the
#: model's bottom sits 15-100 mm below the true object's (sim). F3 ``--held-beneath-contact`` = SL6-H's placement-contact
#: rule on the grasp latch alone (HELD + attached), no HSR6 phase.
GT4_FLAGS: tuple[str, ...] = ("--held-beneath-contact",)
GT4_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-hbc", "AG3S-commonto-gt2-termsoff-hbc",
                                 "AG3S-commonto-gt2-termsoff-pass-hbc")
AG3S_VERSIONS["AG3S-commonto-gt2-hbc"] = COMMONTO_GT_FLAGS + GT2_FLAGS + GT4_FLAGS
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-hbc"] = GT3_TERMSOFF_FLAGS + GT4_FLAGS
#: G31 (07:4x): + the SL-GTTO3 winner (termsoff-pass, LIBERO 4/9) with F3, added before any F3 result.
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-hbc"] = GT3_TERMSOFF_FLAGS + P0M_PASS_FLAGS + GT4_FLAGS
# --- SL-GTTO4 — end
# --- SL-GTTO5 — begin: minimal-intervention common TO + final-approach release (user ruling 2026-11-05 ~09:3x) -------
#: Base = gt2-termsoff-pass (no continuity / deflection-rate terms, VLA chunk through untouched when clear: the
#: literature's min-norm safety filters, SL-GTTO3 best 4/9). F4 ``--grasp-zone-release``: LATCHED / CLOSING hand rows
#: within the grasp zone radius (100 mm, horizontal) of the manipulated object are released (cuRobo plan_grasp).
GT5_FLAGS: tuple[str, ...] = ("--grasp-zone-release",)
GT5_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr"] = GT3_TERMSOFF_FLAGS + P0M_PASS_FLAGS + GT5_FLAGS
# --- SL-GTTO5 — end
# --- SL-GTTO6 — begin: occluded_target frames certified for the TO (lead self-ruling G40, 2026-11-05 11:0x KST) -------
#: gt5zr LIBERO 9: in HELD calls the pass-through was refused 67x for "geometry not certified" (6 collision, 6 limit):
#: the held bowl is hidden by the hand (no_cluster), which the verdict executes as occluded_target. F5
#: ``--occluded-certify`` gives the TO the verdict's reading.
GT6_FLAGS: tuple[str, ...] = ("--occluded-certify",)
GT6_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-oc",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-oc"] = GT3_TERMSOFF_FLAGS + P0M_PASS_FLAGS + GT5_FLAGS + GT6_FLAGS
#: G42 (11:5x): zr-oc held pass-through refusals are now collision 32 / limit 14 (the held model's rows, HOLD rows
#: `attached_point`) -> + F3 ``--held-beneath-contact`` (SL-GTTO4) so that a held row on the surface beneath can pass.
GT6_VERSIONS = GT6_VERSIONS + ("AG3S-commonto-gt2-termsoff-pass-zr-oc-hbc",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-oc-hbc"] = (GT3_TERMSOFF_FLAGS + P0M_PASS_FLAGS + GT5_FLAGS + GT6_FLAGS
                                                              + GT4_FLAGS)
# --- SL-GTTO6 — end
# --- SL-GTTO7 — begin: the grasping hand's rows join the held-beneath rule (lead self-ruling G44, 2026-11-05 12:3x) --
#: zr-oc-hbc LIBERO 9: held HOLD rows are the held object (attached_point) and the hand / fingers on a surface that is
#: not the held object (nearest_is_not_manipulated) -- placing brings the fingers down onto the plate / support.
GT7_FLAGS: tuple[str, ...] = ("--held-beneath-hand",)
GT7_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-oc-hbch",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-oc-hbch"] = (GT3_TERMSOFF_FLAGS + P0M_PASS_FLAGS + GT5_FLAGS
                                                               + GT6_FLAGS + GT4_FLAGS + GT7_FLAGS)
# --- SL-GTTO7 — end
# --- SL-GTTO8 — begin: let the SQP converge when it does intervene (lead self-ruling G45, 2026-11-05 14:0x) --
#: gt8hand / zr: every call whose pass-through is refused for a collision stops on the 3-iteration cap (100 %); the
#: refined path then moves 2-3x the reference violation (median) and conversion C saturates 60-95 % of rows. F6 runs the
#: SQP to 8 iterations (min = max, so the 50 ms wall-clock budget cannot stop it earlier) on the zr version.
GT8_FLAGS: tuple[str, ...] = ("--sqp-min-iterations", "8", "--sqp-max-iterations", "8")
GT8_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-it8",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-it8"] = GT3_TERMSOFF_FLAGS + P0M_PASS_FLAGS + GT5_FLAGS + GT8_FLAGS
# --- SL-GTTO8 — end
# --- SL-GTTO9 — begin: the TO in the policy's action space (lead self-ruling G47, 2026-11-05 17:4x) --
#: F3 / F3b / F5 / F6 changed which rows bind or how long the SQP runs and none moved success; what every intervening call
#: shares is a correction 2-3x the violation that conversion C cannot realise (beta 0.071: 3.6 mm of path per unit action
#: in one step). F7 ``--ee-action-qp``: min ||Da||^2 over the OSC action rows with the response model in the map, on zr.
GT9_FLAGS: tuple[str, ...] = ("--ee-action-qp",)
GT9_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee"] = GT3_TERMSOFF_FLAGS + P0M_PASS_FLAGS + GT5_FLAGS + GT9_FLAGS
# --- SL-GTTO9 — end
# --- SL-GTTO10 — begin: held object, destination contact and grasp phase on F7 (lead self-ruling G53, 2026-11-07;
#: user cautions 1-3 of 11-07 00:1x). F7 LIBERO-Safety human_safety record run (G52/G53, `--record-constraints full`):
#: t0e0's 76 carry HOLD rows = the held bowl's query sphere against the human hand the goal puts it on (no destination was
#: named); t0e2's 58 = the hand's rows against the robot hand's own voxels, with the held bowl's voxels around the hand.
#:   F8a (caution 2): ``--oracle-destination`` (GT destination injected each frame, T26 path) + ``--destination-margin-mm 0``
#:        (held rows may touch it, robot links keep their margin) + F3 ``--held-beneath-contact`` (held rows over a surface
#:        beneath them cannot bind: the sphere model of the held object is larger than the object).
#:   F8b (caution 1): SL10d ``--held-free-query-spheres`` (the TSDF frees the held query spheres too) + SL16 HVC
#:        ``--hand-voxel-carve current --hand-voxel-carve-mm 20`` (the hand's own voxels).
#:   F8c (caution 3): SL14 ``--grasp-opening-rule`` (attach / detach on the gripper command and opening; re-grasp allowed).
GT10_FLAGS: tuple[str, ...] = ("--oracle-destination", "--destination-margin-mm", "0", "--held-beneath-contact",
                               "--held-free-query-spheres", "--hand-voxel-carve", "current", "--hand-voxel-carve-mm", "20",
                               "--grasp-opening-rule")
GT10_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"] = (GT3_TERMSOFF_FLAGS + P0M_PASS_FLAGS + GT5_FLAGS + GT9_FLAGS
                                                            + GT10_FLAGS)
# --- SL-GTTO10 — end
# --- SL-GTTO11 — begin: F9a = F8 + the scaled action QP (approved plan item (a), lead self-ruling G54, 2026-11-07)
#: (c') the opening rule's `moved` check (F8 LS t0e3: attach on the first close-command call, fingers still open).
GT11_FLAGS: tuple[str, ...] = ("--ee-action-qp-scaled", "--grasp-opening-rule-moved", "0.05")
GT11_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-qs",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-qs"] = tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"]) + GT11_FLAGS
# --- SL-GTTO11 — end
# --- SL-GTTO12 — begin: F8 observe-only (user ruling 2026-11-07 19:1x: a LIBERO episode the VLA alone succeeds must succeed
#: with the module too -> catalogue every call where F8 would change / stop the VLA chunk on the VLA's own path).
#: SL11 ``--observe-only``: the client executes the policy chunk bit for bit, the server records what it would have sent.
GT12_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-obs",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-obs"] = (tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"])
                                                                + OBSERVE_ONLY_FLAGS)
# --- SL-GTTO12 — end
# --- SL-GTTO13 — begin: F10 = F8 + F5 ``--occluded-certify`` (lead G68, 2026-11-07): on the VLA's own LIBERO path (GT12
#: observe) F8's would-be changes are 'geometry not certified' calls, all of them occluded_target frames (the held object
#: hidden by the hand) -- the TO then refuses the pass-through and runs the QP on a clear chunk. F5 gives the TO the
#: verdict's reading of such a frame; the collision check is unchanged.
GT13_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-oc",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-oc"] = (tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"])
                                                               + GT6_FLAGS)
# --- SL-GTTO13 — end
# --- SL-GTTO14 — begin: F11 = F8 loosened for LIBERO parity (user ruling 2026-11-07 19:1x; lead G70). Each part answers one
#: kind of false positive on the VLA's own collision-free LIBERO path (GT12 observe, calibration seeds 10 · 11):
#:   F5 ``--occluded-certify``        the held object hidden by the hand -> 'geometry not certified' (118 + 19 calls)
#:   (c') ``--grasp-opening-rule-moved 0.05``  attach with open fingers (s11 t1 / t2, held model -83 / -84 mm)
#:   ``--oracle-robot-mask``          arm / hand rows against the robot's own voxels (s11 t1 / t2)
#:   ``--violation-tolerance-mm 25``  model violation of a collision-free path (sphere model + 5 mm field), max 23.9 mm
#:   ``--passthrough-limit-tolerance 0.01``  a 0.0005 acceleration overshoot refusing the pass-through (s11 t4)
GT14_FLAGS: tuple[str, ...] = ("--occluded-certify", "--grasp-opening-rule-moved", "0.05", "--oracle-robot-mask",
                               "--violation-tolerance-mm", "25", "--passthrough-limit-tolerance", "0.01")
GT14_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose", "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose-obs")
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose"] = (tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"])
                                                                  + GT14_FLAGS)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose-obs"] = (tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose"])
                                                                      + OBSERVE_ONLY_FLAGS)   # calibration check
# --- SL-GTTO14 — end
# --- SL-GTTO15 — begin: F12 = F11 with the pass-through's joint-limit tolerance 0.2 (lead G72, 2026-11-07 20:3x). F11 observe on
#: the calibration seeds 10 · 11: the 7 calls F11 still changes on a VLA-success episode are all joint-limit refusals of the
#: VLA's own chunk (max overshoot velocity 0.055 · acceleration 0.128 · position 0.02); 0.2 = the smallest of 0.01 · 0.02 ·
#: 0.05 · 0.1 · 0.2 above 0.128. Everything else is F11's.
GT15_FLAGS: tuple[str, ...] = tuple("0.2" if (i > 0 and GT14_FLAGS[i - 1] == "--passthrough-limit-tolerance") else t
                                    for i, t in enumerate(GT14_FLAGS))
GT15_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2"] = (tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"])
                                                                   + GT15_FLAGS)
# --- SL-GTTO15 — end
# --- SL-GTTO16 — begin (A, user ruling 2026-11-08 00:2x): F13 = F12 whose tau does not cover obstacle-object rows. F12 passed a
#: held-object <-> moka-pot row at -20.4 mm (SafeLIBERO t2e0 obs 75, collision 2 steps later) because tau 25 mm is blind to what
#: the row touches; LIBERO's tau-passed rows were fingers <-> target / destination / table and link7 <-> target. Registered in
#: GT15_VERSIONS (the LIBERO-parity family) so every GT15 rule (final base argv, no probe, test exclusions) applies unchanged.
GT16_FLAGS: tuple[str, ...] = GT15_FLAGS + ("--passthrough-obstacle-tolerance-mm", "0")
GT16_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst",
                                  "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-obs")
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst"] = (tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"])
                                                                        + GT16_FLAGS)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-obs"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst"]) + OBSERVE_ONLY_FLAGS)   # calibration
#: A' (2026-11-08 01:3x, F13 observe seed 10: 8 refused calls on VLA-success paths, owners by MuJoCo geom = plate (destination,
#: label missed) 3 · cabinet top under the bowl 3 · cookies box 1 (7.1 mm) · none 1): the beneath rule + tau_obs 10 mm (= the
#: obstacle margin: an obstacle row may use its 10 mm buffer, never reach the model's contact level).
GT16_FLAGS_B: tuple[str, ...] = GT15_FLAGS + ("--passthrough-obstacle-tolerance-mm", "10", "--passthrough-obstacle-beneath")
GT16_VERSIONS = GT16_VERSIONS + ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2",
                                 "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2-obs")
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"]) + GT16_FLAGS_B)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2-obs"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2"]) + OBSERVE_ONLY_FLAGS)
GT15_VERSIONS = GT15_VERSIONS + GT16_VERSIONS
# --- SL-GTTO16 — end
# --- SL-GTTO17 — begin (B, user ruling 2026-11-08): F14 = F13 whose held object is a shell of 20 mm voxel spheres on its observed
#: points (radius 17.3 mm) instead of the two ~62 mm body-cover spheres (bowl 108 x 105 x 55 mm: bottoms ~35 mm below it).
GT17_FLAGS: tuple[str, ...] = GT16_FLAGS + ("--held-point-model-mm", "20")
GT17_FLAGS_B: tuple[str, ...] = GT16_FLAGS_B + ("--held-point-model-mm", "20")   # B on A' (2026-11-08 01:3x)
GT17_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-pts",
                                  "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-pts-obs")
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-pts"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"]) + GT17_FLAGS)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-pts-obs"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-pts"]) + OBSERVE_ONLY_FLAGS)
GT17_VERSIONS = GT17_VERSIONS + ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2-pts",
                                 "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2-pts-obs")
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2-pts"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"]) + GT17_FLAGS_B)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2-pts-obs"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2-pts"]) + OBSERVE_ONLY_FLAGS)
GT15_VERSIONS = GT15_VERSIONS + GT17_VERSIONS
# --- SL-GTTO17 — end
# --- SL-GTTO18 — begin (user ruling 2026-11-08: "결과 좋으면 GT 사용하지 않고 AG3S 사용해서 각 벤치마크"): F14 without GT. The GT
#: perception (oracle-visible extent, GT identity, oracle grounding) and the GT-only flags (oracle admissible / held target /
#: destination / robot mask) are removed; the final configuration's attention grounding comes back (split, partition,
#: supersedes-anchor, non-target evidence; ``--attention-source`` left at the server default ``action:last:L8H2`` -- the noun
#: cell L8H6 could not tell the two black bowls apart, SL6 §6.1). Everything else (TO, A, B, latch) is F14's.
_NONGT_DROP = {"--target-extent": 1, "--oracle-extent-identity": 1, "--grounding-source": 1, "--oracle-admissible": 0,
               "--oracle-held-target": 0, "--oracle-destination": 0, "--oracle-robot-mask": 0}
ATT_PERCEPTION_FLAGS: tuple[str, ...] = ("--attention-split-radius-mm", "12", "--attention-split-ratio-source", "raw",
                                         "--attention-split-partition", "--split-supersedes-anchor",
                                         "--nontarget-evidence", "surface")
GT18_VERSIONS: tuple[str, ...] = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst-pts",
                                  "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst2-pts")
for _v in GT18_VERSIONS:
    AG3S_VERSIONS[_v] = (_drop_flags(tuple(AG3S_VERSIONS[_v.replace("-att-", "-gt2-")]), _NONGT_DROP)
                         + ATT_PERCEPTION_FLAGS)
del _v
GT15_VERSIONS = GT15_VERSIONS + GT18_VERSIONS
# --- SL-GTTO19 (A'', lead G84, 2026-11-08 02:3x): B' LIBERO s7 failed t0 (held point-model row vs the plate = destination,
#: 17.3 mm) and t4 (finger vs the cabinet top under the bowl, 101 mm from its centroid, 16.0 mm) -- both are the TO class
#: rule calling a destination / the target's support an obstacle. A'' takes the user's GT definition: obstacle objects =
#: GT objects (code >= 3) other than the target, the destination and the object the target rests on.
GT19_FLAGS: tuple[str, ...] = GT15_FLAGS + ("--passthrough-obstacle-tolerance-mm", "10", "--oracle-obstacle-objects")
GT19_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3",
                                  "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-obs",
                                  "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts")
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"]) + GT19_FLAGS)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-obs"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3"]) + OBSERVE_ONLY_FLAGS)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts"] = (
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3"]) + ("--held-point-model-mm", "20"))
GT15_VERSIONS = GT15_VERSIONS + GT19_VERSIONS
#: SL-GTTO20 (2026-11-08 03:3x): B'' without GT -- the GT perception / GT-only flags dropped (as SL-GTTO18), the attention
#: grounding back, and the obstacle-object surface from AG3S's own perception (``--scene-obstacle-objects``).
GT20_VERSIONS: tuple[str, ...] = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts",)
AG3S_VERSIONS["AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts"] = (
    _drop_flags(tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts"]),
                {"--target-extent": 1, "--oracle-extent-identity": 1, "--grounding-source": 1, "--oracle-admissible": 0,
                 "--oracle-held-target": 0, "--oracle-destination": 0, "--oracle-robot-mask": 0,
                 "--oracle-obstacle-objects": 0})
    + ATT_PERCEPTION_FLAGS + ("--scene-obstacle-objects",))
GT15_VERSIONS = GT15_VERSIONS + GT20_VERSIONS
#: SL-GTTO21 (lead G88, 2026-11-08 03:5x; dev-49 review §14): SL16 HVC (``--hand-voxel-carve current 20``) frees real obstacle
#: voxels next to the hand (SafeLIBERO s10 t2/t3 F8 records: 38 % of freed voxels were obstacles; t2e1 hand -1..-13 mm inside
#: the box while the field read +8..+38 mm), and ``--held-free-query-spheres`` frees whatever lies inside the held query
#: spheres. In the GT versions the robot's own pixels (--oracle-robot-mask) and the held object's (--oracle-held-target) are
#: already kept out of the TSDF, so both carves are dropped.
_CARVE_DROP = {"--hand-voxel-carve": 1, "--hand-voxel-carve-mm": 1, "--held-free-query-spheres": 0}
#: + (G88b) the target's support only from under its base (`--oracle-obstacle-support under`): "near" took the moka pot standing
#: next to the bowl as the bowl's support (B'' SafeLIBERO t2e0: no obstacle row before the collision at 77).
GT21_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve"] = _drop_flags(
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts"]), _CARVE_DROP) + (
    "--oracle-obstacle-support", "under")
GT15_VERSIONS = GT15_VERSIONS + GT21_VERSIONS
#: SL-GTTO22 (G89): B4 = B3 + the obstacle-object rows over the whole chunk (one call earlier).
GT22_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh"] = tuple(
    AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve"]) + ("--passthrough-obstacle-full-horizon",)
GT15_VERSIONS = GT15_VERSIONS + GT22_VERSIONS
#: SL-GTTO23 (G91, 04:3x): B5 = B4 with the general tau 25 -> 100 mm. With the GT obstacle-object rows strict (tau_obs 10 mm; the
#: LIBERO-Safety human is an object, `left_hand_1` code 6), the general tau only guards contact surfaces (target / support /
#: table / destination); every refusal of that kind in B2's evaluation was model error on a path the VLA completes (27-54 mm:
#: link7 <-> target while grasping, hand <-> table on the coarse tier while holding). The later flag wins (argparse).
GT23_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100",)
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100"] = tuple(
    AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh"]) + ("--violation-tolerance-mm", "100")
GT15_VERSIONS = GT15_VERSIONS + GT23_VERSIONS
#: SL-GTTO24 (G92): B5 without GT -- GT perception / GT-only flags dropped, attention grounding back, the obstacle-object surface
#: from AG3S's own perception (--scene-obstacle-objects, base-only support rule).
GT24_VERSIONS: tuple[str, ...] = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100",)
AG3S_VERSIONS["AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100"] = (
    _drop_flags(tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100"]),
                {"--target-extent": 1, "--oracle-extent-identity": 1, "--grounding-source": 1, "--oracle-admissible": 0,
                 "--oracle-held-target": 0, "--oracle-destination": 0, "--oracle-robot-mask": 0,
                 "--oracle-obstacle-objects": 0, "--oracle-obstacle-support": 1})
    + ATT_PERCEPTION_FLAGS + ("--scene-obstacle-objects",))
GT15_VERSIONS = GT15_VERSIONS + GT24_VERSIONS
#: SL-PE (dev-30, 2026-11-08 20:xx, decision log G150-): B5 without GT + the VLA's grasp point as the target identity.
#: pe1 = + --close-seed-grounding (SL-PE1); pe123 = + --destination-after-held (PE2) + --scene-obstacle-held-exclude (PE3).
PE_VERSIONS: tuple[str, ...] = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe1",
                                "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123")
AG3S_VERSIONS[PE_VERSIONS[0]] = tuple(AG3S_VERSIONS[GT24_VERSIONS[0]]) + ("--close-seed-grounding",)
AG3S_VERSIONS[PE_VERSIONS[1]] = tuple(AG3S_VERSIONS[PE_VERSIONS[0]]) + (
    "--destination-after-held", "--scene-obstacle-held-exclude")
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS
#: SL-PE v3 (dev-30, 2026-11-08 22:xx, G157): pe12345 = pe123 + --passthrough-obstacle-class-tests (PE4: the Panda held-object
#: / support / destination row classes are not overridden by the scene obstacle set) + --destination-held-leader (PE5: the
#: plate registers as the destination while held). PE1 itself also changed in v3 (graded seed attention, kept support plane).
PE_VERSIONS = PE_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345",)
AG3S_VERSIONS[PE_VERSIONS[2]] = tuple(AG3S_VERSIONS[PE_VERSIONS[1]]) + (
    "--passthrough-obstacle-class-tests", "--destination-held-leader")
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS[2:]
#: SL-PE (G158, 2026-11-08 22:4x): pe123456 = pe12345 + --no-target-certify (PE6: no HOLD on the first no-target calls; every
#: surface an obstacle). pe125cls = the user-requested check of the earlier Panda method alone (G75 A'): the obstacle-object rows
#: from the TO's row classes + the beneath rule, no scene obstacle set (so no PE3 / PE4); PE1 · PE2 · PE5 kept.
PE_VERSIONS = PE_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123456",
                             "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst2-pts-nocarve-fh-t100-pe125cls")
AG3S_VERSIONS[PE_VERSIONS[3]] = tuple(AG3S_VERSIONS[PE_VERSIONS[2]]) + ("--no-target-certify",)
AG3S_VERSIONS[PE_VERSIONS[4]] = tuple(x for x in AG3S_VERSIONS[PE_VERSIONS[2]] if x not in (
    "--scene-obstacle-objects", "--scene-obstacle-held-exclude", "--passthrough-obstacle-class-tests")) + (
    "--passthrough-obstacle-beneath",)
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS[3:]
#: SL-PE7 (G159, 2026-11-09 00:3x): pe1234567 = pe123456 + --close-seed-attention-ratio 5 (the seed cut must carry attention).
PE_VERSIONS = PE_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe1234567",)
AG3S_VERSIONS[PE_VERSIONS[5]] = tuple(AG3S_VERSIONS[PE_VERSIONS[3]]) + ("--close-seed-attention-ratio", "5")
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS[5:]
#: SL-PE8 (dev-4a, 2026-11-09 03:xx, G162): pe123458 = pe12345 + --scene-obstacle-held-motion (the held object = the points
#: that moved with the hand; the lifted bowl stayed an obstacle object while carrying / placing, LIBERO t9).
PE_VERSIONS = PE_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458",)
AG3S_VERSIONS[PE_VERSIONS[6]] = tuple(AG3S_VERSIONS[PE_VERSIONS[2]]) + ("--scene-obstacle-held-motion",)
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS[6:]
#: SL-PE (G162, 2026-11-09 03:4x): pe12345678 = pe1234567 (PE6 + PE7) + PE8, to decide PE6 · PE7 on top of PE8.
PE_VERSIONS = PE_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345678",)
AG3S_VERSIONS[PE_VERSIONS[7]] = tuple(AG3S_VERSIONS[PE_VERSIONS[5]]) + ("--scene-obstacle-held-motion",)
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS[7:]
#: SL-PE9 (dev-4a, 2026-11-09 04:xx, G163): pe1234589 = pe123458 + --attention-peak-seed 5 (before the chunk carries a close
#: command, the object under the attention peak is cut out of the cabinet / drawer it merged with).
PE_VERSIONS = PE_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe1234589",)
AG3S_VERSIONS[PE_VERSIONS[8]] = tuple(AG3S_VERSIONS[PE_VERSIONS[6]]) + ("--attention-peak-seed", "5")
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS[8:]
#: SL-PE (G163, 2026-11-09 04:1x): pe12345789 = pe1234589 + --close-seed-attention-ratio 5 (PE7, vets identity-changing grasp
#: seed cuts; removed the wine-bottle collision of pe12345 SafeLIBERO t3e3). PE6 (--no-target-certify) left out.
PE_VERSIONS = PE_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345789",)
AG3S_VERSIONS[PE_VERSIONS[9]] = tuple(AG3S_VERSIONS[PE_VERSIONS[8]]) + ("--close-seed-attention-ratio", "5")
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS[9:]
#: SL-PE11 (dev-4a, 2026-11-09 08:xx, G167, patch only): pe1234578911 = pe12345789 + --scene-obstacle-target-cut (the
#: manipulated object leaves the obstacle set by points, not with the obstacle touching it).
PE_VERSIONS = PE_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe1234578911",)
AG3S_VERSIONS[PE_VERSIONS[10]] = tuple(AG3S_VERSIONS[PE_VERSIONS[9]]) + ("--scene-obstacle-target-cut",)
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS[10:]
#: SL-PE11 alone (dev-4a, 2026-11-09 12:1x, G170): pe12345811 = final pe123458 + --scene-obstacle-target-cut (no PE7 / PE9).
#: pe123458 E1a / SafeLIBERO dev: the obstacle left the obstacle-object set with the target's component in 6 / 17 and 2 / 4
#: collided episodes; the target cut keeps it on every such call (offline, e1a_decomp/e1a_pe11_offline.py).
PE_VERSIONS = PE_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811",)
AG3S_VERSIONS[PE_VERSIONS[11]] = tuple(AG3S_VERSIONS[PE_VERSIONS[6]]) + ("--scene-obstacle-target-cut",)
GT15_VERSIONS = GT15_VERSIONS + PE_VERSIONS[11:]
#: SL-QE2 (dev-e5, 2026-11-09 01:xx, G111): B5 (GT) / B5 without GT with the action QP bending the position only
#: (--ee-action-qp-translation-only; E1a task 3: the F7 QP turned rx / ry / rz by ~1.0 and the grasp pose was lost, G110).
QE2_VERSIONS: tuple[str, ...] = ("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100-tro",
                                 "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-tro")
for _v in QE2_VERSIONS:
    AG3S_VERSIONS[_v] = tuple(AG3S_VERSIONS[_v[:-len("-tro")]]) + ("--ee-action-qp-translation-only",)
del _v
GT15_VERSIONS = GT15_VERSIONS + QE2_VERSIONS
#: SL-PE x SL-QE2 (dev-4a, 2026-11-09 12:xx, user-approved): the GT-free final pe123458 with the translation-only action QP
#: (B5-tro: E1a SSR 13 -> 20, G169). Same base + flag pattern, so test_sl_qe2 covers it.
QE2_VERSIONS = QE2_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458-tro",)
AG3S_VERSIONS[QE2_VERSIONS[-1]] = tuple(AG3S_VERSIONS[QE2_VERSIONS[-1][:-len("-tro")]]) + ("--ee-action-qp-translation-only",)
GT15_VERSIONS = GT15_VERSIONS + QE2_VERSIONS[-1:]
QE2_VERSIONS = QE2_VERSIONS + ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-tro",)
AG3S_VERSIONS[QE2_VERSIONS[-1]] = tuple(AG3S_VERSIONS[QE2_VERSIONS[-1][:-len("-tro")]]) + ("--ee-action-qp-translation-only",)
GT15_VERSIONS = GT15_VERSIONS + QE2_VERSIONS[-1:]
#: SOTA clearance candidates (dev-57, 2026-11-11, AG3S_PLAN_SCHEDULE_20261110_KO.md §9): the obstacle-object row of the
#: pass-through. pe123458 E1a 40: 5 collided episodes passed through with that row 6-9.7 mm inside its 10 mm margin (A) and
#: 3 touched right after an action-QP solve (B) -- with tau_obs 10 mm the passed chunk may have 0 mm modelled clearance.
#: ``-to5`` / ``-to0`` = --passthrough-obstacle-tolerance-mm 5 / 0 (argparse: the later value wins). Name =
#: <base>[-to<k>][-tro]; all default off. (A larger --obstacle-margin is NOT registered here: it is a V11AB base option,
#: and a GT15 version that sets it makes version_flag_values add the final-version keys to every V11AB record.)
SOTA_VERSIONS: tuple[str, ...] = ()
_SOTA_SUFFIXES: dict[str, tuple[str, ...]] = {
    "-to5": ("--passthrough-obstacle-tolerance-mm", "5"),
    "-to0": ("--passthrough-obstacle-tolerance-mm", "0"),
}
for _b in (PE_VERSIONS[6], PE_VERSIONS[11]):   # pe123458, pe12345811
    for _suffix, _flags in _SOTA_SUFFIXES.items():
        AG3S_VERSIONS[_b + _suffix] = tuple(AG3S_VERSIONS[_b]) + _flags
        AG3S_VERSIONS[_b + _suffix + "-tro"] = AG3S_VERSIONS[_b + _suffix] + ("--ee-action-qp-translation-only",)
        SOTA_VERSIONS = SOTA_VERSIONS + (_b + _suffix, _b + _suffix + "-tro")
del _b, _suffix, _flags
#: SL-SOTA1 (dev-57, 2026-11-11): the obstacle-object rows' own margin on top of ``-to0`` -- ``-om<mm>`` =
#: --obstacle-row-margin-mm (pass-through refused below floor + mm, the action QP holds those rows there), ``p<w>`` =
#: --obstacle-row-priority (their slack weight), ``s<mm>`` = --obstacle-surface-tol-mm (row <-> scene obstacle set). E1a
#: pe123458 (outputs/2026-11-11_SOTA_개선/offline): the world model misses 12-27 mm of an obstacle under the hand, the
#: executed motion is 10-15 mm off the QP plan, the obstacle set is 21-40 mm from the surface the hand touches.
#: Name = <base>-to0-om<mm>[p<w>][s<mm>][-tro]; all default off.
SOTA_OM_VERSIONS: tuple[str, ...] = ()
_SOTA_OM_SUFFIXES: dict[str, tuple[str, ...]] = {
    "-om15": ("--obstacle-row-margin-mm", "15"),
    "-om15p10": ("--obstacle-row-margin-mm", "15", "--obstacle-row-priority", "10"),
    "-om15p10s40": ("--obstacle-row-margin-mm", "15", "--obstacle-row-priority", "10", "--obstacle-surface-tol-mm", "40"),
    "-om25p10s40": ("--obstacle-row-margin-mm", "25", "--obstacle-row-priority", "10", "--obstacle-surface-tol-mm", "40"),
}
for _b in (PE_VERSIONS[6] + "-to0", PE_VERSIONS[11] + "-to0"):
    for _suffix, _flags in _SOTA_OM_SUFFIXES.items():
        AG3S_VERSIONS[_b + _suffix] = tuple(AG3S_VERSIONS[_b]) + _flags
        AG3S_VERSIONS[_b + _suffix + "-tro"] = AG3S_VERSIONS[_b + _suffix] + ("--ee-action-qp-translation-only",)
        SOTA_OM_VERSIONS = SOTA_OM_VERSIONS + (_b + _suffix, _b + _suffix + "-tro")
del _b, _suffix, _flags
SOTA_VERSIONS = SOTA_VERSIONS + SOTA_OM_VERSIONS
#: ``-ntc`` = --no-target-certify (SL-PE6, G158: a no-target frame is certified with every surface an obstacle instead of a
#: HOLD). pe123458 / pe123458-tro E1a t0 e7: `no_target` (grounding no_admissible) from call 0 -> 60 HOLDs, the arm never
#: moved (pi0.5 alone succeeds). PE6 was only evaluated together with PE7. Name = <SOTA base>-ntc[-tro].
SOTA_NTC_VERSIONS: tuple[str, ...] = ()
for _b in ((PE_VERSIONS[6], PE_VERSIONS[11], PE_VERSIONS[6] + "-to0", PE_VERSIONS[11] + "-to0")
           + tuple(v for v in SOTA_OM_VERSIONS if not v.endswith("-tro"))):
    AG3S_VERSIONS[_b + "-ntc"] = tuple(AG3S_VERSIONS[_b]) + ("--no-target-certify",)
    AG3S_VERSIONS[_b + "-ntc-tro"] = AG3S_VERSIONS[_b + "-ntc"] + ("--ee-action-qp-translation-only",)
    SOTA_NTC_VERSIONS = SOTA_NTC_VERSIONS + (_b + "-ntc", _b + "-ntc-tro")
del _b
SOTA_VERSIONS = SOTA_VERSIONS + SOTA_NTC_VERSIONS
#: SL-SOTA2 / SL-SOTA3 (dev-57, 2026-11-11) on the obstacle-row margin versions: ``-pt<n>`` = --obstacle-row-patience n (the
#: margin is dropped after n consecutive calls inside it, until a chunk clears it again), ``-pg<g*10>`` =
#: --ee-action-qp-progress g (the blocked correction prefers up while held / toward the target before the grasp). E1a task 0
#: of ...-om15p10s40-ntc-tro: CAR 3 -> 7, SSR 1 -> 4 of 10, and 5 episodes out of steps in front of what blocked them
#: (the mug under the descent, the forearm at the moka pot / wine bottle, the target / plate read as an obstacle).
#: Name = <-om* base>[-ntc][-pt<n>][-pg<k>][-tro].
SOTA_STALL_VERSIONS: tuple[str, ...] = ()
_SOTA_STALL_SUFFIXES: dict[str, tuple[str, ...]] = {
    "-pt6": ("--obstacle-row-patience", "6"),
    "-pg4": ("--ee-action-qp-progress", "0.4"),
    "-pt6-pg4": ("--obstacle-row-patience", "6", "--ee-action-qp-progress", "0.4"),
    "-pt3": ("--obstacle-row-patience", "3"),
    "-pt3-pg4": ("--obstacle-row-patience", "3", "--ee-action-qp-progress", "0.4"),
}
for _b in tuple(v for v in SOTA_OM_VERSIONS + SOTA_NTC_VERSIONS if "-om" in v and not v.endswith("-tro")):
    for _suffix, _flags in _SOTA_STALL_SUFFIXES.items():
        AG3S_VERSIONS[_b + _suffix] = tuple(AG3S_VERSIONS[_b]) + _flags
        AG3S_VERSIONS[_b + _suffix + "-tro"] = AG3S_VERSIONS[_b + _suffix] + ("--ee-action-qp-translation-only",)
        SOTA_STALL_VERSIONS = SOTA_STALL_VERSIONS + (_b + _suffix, _b + _suffix + "-tro")
del _b, _suffix, _flags
SOTA_VERSIONS = SOTA_VERSIONS + SOTA_STALL_VERSIONS
#: SL-SOTA4 (dev-57, 2026-11-11): ``-gr`` = --ee-action-qp-gate-rows (the action QP holds the ESDF rows that are not
#: obstacle-object rows at -tau, what the pass-through test accepts of them, instead of 0). E1a ...-om15p10s40-ntc-tro:
#: all 21 failures ran out of steps; in the held phase the chunk was refused for a held row inside the plate's margin and
#: the QP then pulled back every model violation the test had been accepting (link7 against the held bowl 68 mm -> |Da| 1.5).
#: Registered on pe123458 / pe12345811 (± -ntc; tau_obs 10 mm as adopted) and on every ``-to0`` SOTA base (with / without
#: the margin, -ntc, -pt / -pg). Name = <base>[...]-gr[-tro].
SOTA_GATE_VERSIONS: tuple[str, ...] = ()
for _b in ((PE_VERSIONS[6], PE_VERSIONS[11], PE_VERSIONS[6] + "-ntc", PE_VERSIONS[11] + "-ntc")
           + tuple(v for v in SOTA_VERSIONS if "-to0" in v and not v.endswith("-tro"))):
    AG3S_VERSIONS[_b + "-gr"] = tuple(AG3S_VERSIONS[_b]) + ("--ee-action-qp-gate-rows",)
    AG3S_VERSIONS[_b + "-gr-tro"] = AG3S_VERSIONS[_b + "-gr"] + ("--ee-action-qp-translation-only",)
    SOTA_GATE_VERSIONS = SOTA_GATE_VERSIONS + (_b + "-gr", _b + "-gr-tro")
del _b
SOTA_VERSIONS = SOTA_VERSIONS + SOTA_GATE_VERSIONS
GT15_VERSIONS = GT15_VERSIONS + SOTA_VERSIONS
#: (G88c, 03:5x) the A-alone ablation queued as ``…-obst3`` (gtto23, not started yet) gets the same two fixes, so A alone and
#: A + B differ only by the held point model: ``…-obst3`` := A'' + no carve + support "under" (``…-obst3-pts`` = B'' keeps
#: the flags it was evaluated with -- it was built above from the earlier definition).
AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3"] = _drop_flags(
    tuple(AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3"]), _CARVE_DROP) + (
    "--oracle-obstacle-support", "under")   # the same final base argv / no-probe rules (no GT despite the family)
del _NONGT_DROP
# --- SL-GTTO18 — end
#: SL6-B B4 — the Panda HSR6 probe (5 classes, no retreat; labels approved by the user 2026-10-29), the default
#: ``--hsr6-probe`` of the final versions, and its SHA-256 (recorded by the server, checked by the test when present).
#: SL7 (0.7): found under the shared storage root (``AG3S_STORAGE_ROOT``, default the checkout's sibling
#: ``AG3S_VLA-storage`` -- ``common.workspace.shared_storage_root``), not at a fixed path; with the default root the
#: string is the one the earlier versions recorded.
PANDA_HSR6_PROBE_REL = ("results/benchmark/20261029T192223_5ea56af44722_sl6b-hsr6-panda-labels/"
                        "probe/hsr6_panda_kv_L4.npz")
PANDA_HSR6_PROBE = str(shared_storage_root() / PANDA_HSR6_PROBE_REL)
PANDA_HSR6_PROBE_SHA256 = "e2bdd1290f5770ce16ad0235ff261a1d343757e37a321047708cd85ded498029"
#: Options a final version takes per run (not part of the version's identity in :func:`match_ag3s_version`).
FINAL_RUN_PARAMETERS: tuple[str, ...] = ("hsr6_probe", "attention_source")


def is_final_version(version: str) -> bool:
    """A version on the final-configuration base argv: the SL6 final versions and (SL7) the P0m versions."""
    version = _lsfork_base(version)   # SL11: X-lsfork is as final as X
    return (str(version) in FINAL_CONFIG_VERSIONS or str(version) in P0M_VERSIONS or str(version) in P0M_SL9_VERSIONS
            or str(version) in PC_LATCH_VERSIONS or str(version) in PC_NOPROBE_VERSIONS   # SL7d / SL7e / SL9
            or str(version) in SL12_VERSIONS or str(version) in SL13_VERSIONS   # SL12 / SL13
            or str(version) in SL14_VERSIONS or str(version) in SL15_NTM_VERSIONS
            or str(version) in SL15_SFHAND_VERSIONS
            or str(version) in SL16_HSC_VERSIONS or str(version) in SL16_HVC_VERSIONS   # SL14 / SL15 / SL16
            or str(version) in GT_VERSIONS or str(version) in GT2_VERSIONS              # SL-GTTO / SL-GTTO2
            or str(version) in GT3_VERSIONS or str(version) in GT4_VERSIONS              # SL-GTTO3 / SL-GTTO4
            or str(version) in GT5_VERSIONS or str(version) in GT6_VERSIONS              # SL-GTTO5 / 6
            or str(version) in GT7_VERSIONS or str(version) in GT8_VERSIONS              # SL-GTTO7 / 8
            or str(version) in GT9_VERSIONS or str(version) in GT10_VERSIONS            # SL-GTTO9 / 10
            or str(version) in GT11_VERSIONS or str(version) in GT12_VERSIONS            # SL-GTTO11 / 12
            or str(version) in GT13_VERSIONS or str(version) in GT14_VERSIONS            # SL-GTTO13 / 14
            or str(version) in GT15_VERSIONS)                                            # SL-GTTO15


def version_server_flags(version: str, *, hsr6_probe: Optional[str] = None,
                         attention_source: Optional[str] = None) -> list[str]:
    """The version's serve_safe flags. A final version (SL6-A) takes ``hsr6_probe`` (default :data:`PANDA_HSR6_PROBE`)
    and may replace its ``--attention-source`` cell; the V11AB family takes neither."""
    if version not in AG3S_VERSIONS:
        raise ValueError(f"unknown AG3S version {version!r}; known: {sorted(AG3S_VERSIONS)}")
    flags = list(AG3S_VERSIONS[version])
    if not is_final_version(version):
        if hsr6_probe is not None or attention_source is not None:
            raise ValueError(f"{version}: --hsr6-probe / --attention-source apply to {list(FINAL_CONFIG_VERSIONS)} only")
        return flags
    base = _lsfork_base(version)   # SL11: the probe rules of X-lsfork are X's
    if (version in GT_VERSIONS or version in GT2_VERSIONS or version in GT3_VERSIONS
            or version in GT4_VERSIONS or version in GT5_VERSIONS or version in GT6_VERSIONS
            or version in GT7_VERSIONS or version in GT8_VERSIONS or version in GT9_VERSIONS
            or version in GT10_VERSIONS or version in GT11_VERSIONS or version in GT12_VERSIONS
            or version in GT13_VERSIONS or version in GT14_VERSIONS or version in GT15_VERSIONS):   # SL-GTTO(2-11): no HSR6 rule -> no probe; GT grounding -> no cell
        if hsr6_probe is not None or attention_source is not None:
            raise ValueError(f"{version}: the GT-perception versions use no HSR6 rule and no ATS grounding, so "
                             "--hsr6-probe / --attention-source have nothing to feed")
        return flags
    if base in P0M_VERSIONS or base in P0M_SL9_VERSIONS or base in PC_NOPROBE_VERSIONS:   # SL7 / SL7e / SL9: no HSR6 rule -> no probe
        if hsr6_probe is not None:
            raise ValueError(f"{version}: the P0m versions use no HSR6 rule, so --hsr6-probe has nothing to feed")
        if attention_source is not None:
            flags[flags.index("--attention-source") + 1] = str(attention_source)
        return flags
    hsr6_probe = PANDA_HSR6_PROBE if hsr6_probe is None else hsr6_probe
    if not hsr6_probe:
        raise ValueError(f"{version} needs the Panda HSR6 probe (--hsr6-probe FILE; the RB-Y1 probe is refused on the "
                         "Panda)")
    if attention_source is not None:
        if "--attention-source" in flags:
            flags[flags.index("--attention-source") + 1] = str(attention_source)
        else:                                                    # a version on the default cell (SL6-C v2)
            flags += ["--attention-source", str(attention_source)]
    return flags + ["--hsr6-probe", str(hsr6_probe)]


def version_server_argv(version: str, arm: str = V11AB_ARM) -> list[str]:
    """The base argv a version is served with: the V11AB argv, with :data:`V11AB_PRE_MERGE_DEFAULTS` for the V11AB
    family and without them for a final version (SL6-A A1)."""
    return v11ab_server_argv(arm, pre_merge=not is_final_version(version))


def _version_dests(names) -> list[str]:
    from AG3S.trajopt import serve_safe as ss

    actions = ss.build_parser()._option_string_actions
    return sorted({actions[tok].dest for n in names for tok in AG3S_VERSIONS[n] if tok.startswith("--")})


def version_flag_values(args: argparse.Namespace) -> dict[str, Any]:
    """``{dest: value}`` of ``args`` for every serve_safe option any V11AB-family :data:`AG3S_VERSIONS` entry sets,
    plus (SL6-A) every option of the final versions when ``args`` sets one of those away from its parser default --
    so a V11AB / pb server records exactly the keys it recorded before the final versions existed."""
    from AG3S.trajopt import serve_safe as ss

    family = [n for n in AG3S_VERSIONS if not is_final_version(n)]
    dests = _version_dests(family)
    final = [d for d in _version_dests(FINAL_CONFIG_VERSIONS) if d not in set(dests)]
    p0m = [d for d in _version_dests(P0M_VERSIONS) if d not in set(dests) | set(final)]   # SL7: P0m-only options
    sl9 = [d for d in _version_dests(SL9_VERSIONS) if d not in set(dests) | set(final) | set(p0m)]   # SL9: A / B options
    sl10 = [d for d in _version_dests(SL10_VERSIONS) if d not in set(dests) | set(final) | set(p0m) | set(sl9)]   # SL10
    sl10d = [d for d in _version_dests(SL10D_VERSIONS) if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10)]   # SL10d
    sl10e = [d for d in _version_dests(SL10E_VERSIONS)
             if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d)]   # SL10e
    sl10f = [d for d in _version_dests(SL10F_VERSIONS)
             if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e)]   # SL10f
    sl11 = [d for d in _version_dests(tuple(SL11_VERSIONS) + tuple(LSFORK_VERSIONS))
            if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f)]   # SL11
    sl11gap = [d for d in _version_dests(SL11_GAP_VERSIONS)
               if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11)]   # SL11 §5
    sl12 = [d for d in _version_dests(SL12_VERSIONS)
            if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11) | set(sl11gap)]   # SL12
    sl13 = [d for d in _version_dests(SL13_VERSIONS)
            if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11) | set(sl11gap) | set(sl12)]   # SL13
    sl14 = [d for d in _version_dests((SL14_ORACLE_LATCH_VERSION, SL14_REARM_VERSION))
            if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11) | set(sl11gap) | set(sl12) | set(sl13)]   # SL14
    sl14ks = [d for d in _version_dests((SL14_OPENRULE_VERSION,))
              if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11) | set(sl11gap) | set(sl12) | set(sl13) | set(sl14)]   # SL14 KS
    sl15 = [d for d in _version_dests(SL15_NTM_VERSIONS)
            if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11) | set(sl11gap) | set(sl12) | set(sl13) | set(sl14) | set(sl14ks)]   # SL15
    sl15sf = [d for d in _version_dests(SL15_SFHAND_VERSIONS)
              if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11) | set(sl11gap) | set(sl12) | set(sl13) | set(sl14) | set(sl14ks) | set(sl15)]   # SL15 sfhand
    sl16hsc = [d for d in _version_dests(SL16_HSC_VERSIONS)
               if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11) | set(sl11gap) | set(sl12) | set(sl13) | set(sl14) | set(sl14ks) | set(sl15) | set(sl15sf)]   # SL16 HSC
    sl16hvc = [d for d in _version_dests(SL16_HVC_VERSIONS)
               if d not in set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11) | set(sl11gap) | set(sl12) | set(sl13) | set(sl14) | set(sl14ks) | set(sl15) | set(sl15sf) | set(sl16hsc)]   # SL16 HVC
    parser = ss.build_parser()
    actions = parser._option_string_actions
    pinned = {actions[t].dest for t in V11AB_PRE_MERGE_DEFAULTS if t.startswith("--")}   # set on every V11AB argv
    # SL6-E: a final version may re-set a V11AB base option (3판 --w-deflection-rate 0 over the base 20); the base
    # value alone is not a final option
    pinned |= {actions[t].dest for t in v11ab_server_argv(pre_merge=False) if t in actions}
    if any(getattr(args, d, None) != parser.get_default(d) for d in final if d not in pinned):
        dests = sorted(set(dests) | set(final))
    # SL7: the P0m-only options (surface extent, latch gate label) join only when one is set -- every earlier version's
    # record keeps exactly the keys it had
    if any(getattr(args, d, None) != parser.get_default(d) for d in p0m):
        dests = sorted(set(dests) | set(final) | set(p0m))
    # SL9: the A / B options join only when one is set -- every earlier version's record keeps exactly the keys it had
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl9):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(sl9))
    # SL10: the action-box option joins only when it is set -- every earlier version's record keeps exactly the keys it had
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl10):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10))
    # SL10d: the held-free option joins only when it is set (it is the only new dest of those versions)
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl10d):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d))
    # SL10e: the placement-contact-lift option joins only when it is set (it is the only new dest of those versions)
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl10e):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e))
    # SL10f: the uncertified-waive option joins only when it is set (it is the only new dest of that version)
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl10f):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f))
    # SL11: the smooth-on-deviation / observe-only / response-model options join only when one is set
    # (minimal_intervention_mm is a V11AB dest)
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl11):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(sl9) | set(sl10) | set(sl10d) | set(sl10e) | set(sl10f) | set(sl11))
    # SL12: the oracle identity / reach dests join only when one is set (target_extent itself is a P0m dest)
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl12):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(sl12))
    # SL13: the target-volume dest joins only when it is set -- no other version's record gains a key (the oracle dests of its
    # base version join as SL12's do)
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl13):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(sl12) | set(sl13))
    # SL11 §5: the gap-opening dest joins only when it is set -- no other version's record gains a key
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl11gap):
        dests = sorted(set(dests) | set(sl11gap))
    # SL14: the oracle-latch / placed-rearm dests join only when one is set -- no other version's record gains a key
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl14):
        dests = sorted(set(dests) | set(sl14))
    # SL14 KS: the opening-rule dest joins only when it is set -- the oracle-latch / rearm records keep their keys
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl14ks):
        dests = sorted(set(dests) | set(sl14ks))
    # SL15: the nonauthorized-target-margin dest joins only when it is set -- the -pc / -openrule records keep their keys
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl15):
        dests = sorted(set(dests) | set(sl15))
    # SL15 sfhand: the self-filter inflation dest joins only when it is set -- the other records keep their keys
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl15sf):
        dests = sorted(set(dests) | set(sl15sf))
    # SL16 HSC: the held-support-contact dest joins only when it is set -- the other records keep their keys
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl16hsc):
        dests = sorted(set(dests) | set(sl16hsc))
    # SL16 HVC: the hand-voxel-carve dests join only when one is set -- the other records keep their keys
    if any(getattr(args, d, None) != parser.get_default(d) for d in sl16hvc):
        dests = sorted(set(dests) | set(sl16hvc))
    # SL-GTTO: the GT-perception-only options join only when one is set (every earlier version's record unchanged)
    gt = [d for d in _version_dests(GT_VERSIONS) if d not in set(dests) | set(final) | set(p0m)]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(gt))
    # SL-GTTO2: the F1 / F2 options join only when one is set (AG3S-commonto-gt keeps its keys)
    gt2 = [d for d in _version_dests(GT2_VERSIONS) if d not in set(dests) | set(final) | set(p0m) | set(gt)]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt2):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(gt) | set(gt2))
    # SL-GTTO4: the F3 option joins only when set (every earlier version's record unchanged)
    gt4 = [d for d in _version_dests(GT4_VERSIONS) if d not in set(dests) | set(final) | set(p0m) | set(gt) | set(gt2)]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt4):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(gt) | set(gt2) | set(gt4))
    # SL-GTTO5: the F4 option joins only when set
    gt5 = [d for d in _version_dests(GT5_VERSIONS)
           if d not in set(dests) | set(final) | set(p0m) | set(gt) | set(gt2) | set(gt4)]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt5):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(gt) | set(gt2) | set(gt4) | set(gt5))
    # SL-GTTO6: the F5 option joins only when set
    gt6 = [d for d in _version_dests(GT6_VERSIONS)
           if d not in set(dests) | set(final) | set(p0m) | set(gt) | set(gt2) | set(gt4) | set(gt5)]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt6):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(gt) | set(gt2) | set(gt4) | set(gt5) | set(gt6))
    # SL-GTTO7: the F3b option joins only when set
    gt7 = [d for d in _version_dests(GT7_VERSIONS)
           if d not in set(dests) | set(final) | set(p0m) | set(gt) | set(gt2) | set(gt4) | set(gt5) | set(gt6)]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt7):
        dests = sorted(set(dests) | set(final) | set(p0m) | set(gt) | set(gt2) | set(gt4) | set(gt5) | set(gt6)
                       | set(gt7))
    # SL-GTTO8: the SQP iteration options join only when set (every earlier version's record unchanged)
    seen8 = set(dests) | set(final) | set(p0m) | set(gt) | set(gt2) | set(gt4) | set(gt5) | set(gt6) | set(gt7)
    gt8 = [d for d in _version_dests(GT8_VERSIONS) if d not in seen8]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt8):
        dests = sorted(seen8 | set(gt8))
    # SL-GTTO9: the F7 option joins only when set
    seen9 = set(dests) | seen8 | set(gt8)
    gt9 = [d for d in _version_dests(GT9_VERSIONS) if d not in seen9]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt9):
        dests = sorted(seen9 | set(gt9))
    # SL-GTTO10: the F8a options join only when set (the remote flags it also uses have their own blocks above)
    seen10 = set(dests) | seen9 | set(gt9)
    gt10 = [d for d in _version_dests(GT10_VERSIONS) if d not in seen10]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt10):
        dests = sorted(seen10 | set(gt10))
    # SL-GTTO11: the F9a option joins only when set
    seen11 = set(dests) | seen10 | set(gt10)
    gt11 = [d for d in _version_dests(GT11_VERSIONS) if d not in seen11]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt11):
        dests = sorted(seen11 | set(gt11))
    # SL-GTTO14: the F11 options join only when set
    seen14 = set(dests) | seen11 | set(gt11)
    gt14 = [d for d in _version_dests(GT14_VERSIONS) if d not in seen14]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt14):
        dests = sorted(seen14 | set(gt14))
    # SL-GTTO16+: the options of the later LIBERO-parity versions (registered in GT15_VERSIONS) join only when set
    seen16 = set(dests) | seen14 | set(gt14)
    gt16 = [d for d in _version_dests(GT15_VERSIONS) if d not in seen16]
    if any(getattr(args, d, None) != parser.get_default(d) for d in gt16):
        dests = sorted(seen16 | set(gt16))
    return {d: _jsonable(getattr(args, d, None)) for d in dests}


def match_ag3s_version(args: argparse.Namespace) -> Optional[str]:
    """The :data:`AG3S_VERSIONS` name whose flags give ``args``'s :func:`version_flag_values`, else ``None``.

    The server only sees the parsed flags (the launcher's ``--ag3s-version`` name is not in its argv); this reads
    the name back for the record by parsing each version's argv (:func:`version_server_argv` + its flags) the way the
    server is started. A final version's per-run options (:data:`FINAL_RUN_PARAMETERS`) are not compared; a version
    whose argv this host cannot parse (e.g. ``--to-gpu-seeds`` without a GPU) is skipped.
    """
    current = {k: v for k, v in version_flag_values(args).items() if k not in FINAL_RUN_PARAMETERS}
    for name, flags in AG3S_VERSIONS.items():
        try:
            parsed = parse_v11ab_args(argv=version_server_argv(name) + list(flags))
        except SystemExit:
            continue
        values = {k: v for k, v in version_flag_values(parsed).items() if k not in FINAL_RUN_PARAMETERS}
        if values == current:
            return name
    return None


#: serve_safe flags that keep V11AB's own defaults on the code after the AG3S final-configuration merge (main
#: 3ff415d5, 2026-10-29). That code turns the ROI carve on by default for a cuRobo field
#: (`serve_safe.roi_carve_serving_default`) and limits OpenBLAS to 1 thread (`--blas-threads`, default 1); V11AB
#: (``start_server.sh``) ran with neither. Pinned here so "V11AB" and every V11AB-pb<N> stay the pre-merge server.
V11AB_PRE_MERGE_DEFAULTS: tuple[str, ...] = ("--no-roi-carve", "--blas-threads", "0")


def v11ab_server_argv(arm: str = V11AB_ARM, *, pre_merge: bool = True) -> list[str]:
    """The V11AB server argv (+ :data:`V11AB_PRE_MERGE_DEFAULTS`; ``pre_merge=False``: without them -- the base of
    the final-configuration versions, SL6-A)."""
    req = v11ab_required_args()
    return ["--config", req["config"], "--checkpoint", "<not loaded: perception only>",
            "--model-xml", req["model_xml"], "--port", "0", *v11ab_server_flags(arm),
            *(V11AB_PRE_MERGE_DEFAULTS if pre_merge else ())]


def parse_v11ab_args(arm: str = V11AB_ARM, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """serve_safe's parser + its own flag checks + its subtask-gate resolution, as ``main`` runs them."""
    from AG3S.trajopt import serve_safe as ss

    from AG3S.adapters.panda.attention import PANDA_SEED_MAP_LABELS

    ap = ss.build_parser()
    args = ap.parse_args(list(argv) if argv is not None else v11ab_server_argv(arm))
    # SL7: this is the Panda profile -- the attention-surface extent takes the Panda head (L8H6) as its seed map
    ss.reject_bad_flag_combinations(ap, args, seed_map_labels=PANDA_SEED_MAP_LABELS)
    ss.resolve_subtask_gate(args)
    return args


def _calls(fn: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]


def _main_build_call() -> ast.Call:
    """The served AG3S's ``build_ag3s(...)`` call. Since the AG3S final-configuration merge (main 3ff415d5,
    2026-10-29) ``serve_safe.main`` builds it through ``ag3s_from_args`` (one call), which holds the one
    ``build_ag3s`` call; before it the call was in ``main`` itself. Both layouts are checked, never guessed."""
    tree = ast.parse(SERVE_SAFE.read_text())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    in_main = _calls(fns["main"], "build_ag3s")
    if in_main:
        if len(in_main) != 1:
            raise ValueError(f"serve_safe.main has {len(in_main)} build_ag3s calls; expected 1")
        return in_main[0]
    via = _calls(fns["main"], "ag3s_from_args")
    calls = _calls(fns["ag3s_from_args"], "build_ag3s") if "ag3s_from_args" in fns else []
    if len(via) != 1 or len(calls) != 1:
        raise ValueError(f"serve_safe.main has {len(via)} ag3s_from_args calls and ag3s_from_args "
                         f"{len(calls)} build_ag3s calls; expected 1 and 1")
    return calls[0]


def v11ab_build_call_source() -> dict[str, str]:
    """``{keyword: source expression}`` of the served ``build_ag3s(...)`` call (``**`` entries under ``"**<i>"``)."""
    return {(k.arg if k.arg is not None else f"**{i}"): ast.unparse(k.value)
            for i, k in enumerate(_main_build_call().keywords)}


def v11ab_build_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The keywords the server passes to ``build_ag3s`` for ``args`` (log-only ones omitted).

    Each expression is evaluated in serve_safe's module namespace with ``args`` bound -- the call is read,
    not restated.
    """
    from AG3S.trajopt import serve_safe as ss

    namespace = dict(vars(ss))
    namespace["args"] = args
    out = {}
    for keyword in _main_build_call().keywords:
        if keyword.arg in LOG_ONLY_BUILD_KEYWORDS:
            continue
        value = eval(compile(ast.Expression(keyword.value), str(SERVE_SAFE), "eval"), namespace)  # noqa: S307
        if keyword.arg is None:   # ``**{...}`` (a key only when its flag is on)
            out.update(value)
        else:
            out[keyword.arg] = value
    return out


# --- config ---------------------------------------------------------------------------------------------
def v11ab_config_dict(kw: Mapping[str, Any]) -> tuple[dict, Optional[dict]]:
    """``(AG3SConfig.from_dict input, identity overrides)`` -- the ``build_ag3s`` assembly, verbatim.

    Copy of ``serve_safe.build_ag3s`` (config part). The test compares the resulting config with the one
    the real ``build_ag3s`` hands to ``AG3S`` for the same keywords.
    """
    from AG3S.trajopt import serve_safe as ss

    unported = {k: kw[k] for k in UNPORTED_BUILD_KEYWORDS if kw.get(k) not in (None, {}, False)}
    if unported:
        raise ValueError(f"build_ag3s keywords not ported to the Panda profile yet: {unported} (AG3S final-configuration "
                         "merge 2026-10-29; this copy would drop them silently)")
    policy = ss.resolve_target_field_policy(kw.get("target_field_policy", "relax"))
    constraint_section: dict = {}
    if not kw.get("self_collision", True):
        constraint_section["self_collision"] = False
    if policy != "relax":
        constraint_section["target_field_policy"] = policy
    backend = kw.get("backend", "legacy")
    esdf = {"voxel_size": kw["voxel"], "max_distance": 0.4,
            "exclude_support_surfaces": False,
            "backend": backend,
            "attached_sign_threshold_voxels": kw.get("attached_sign_threshold", 1.5)}
    if backend == "curobo":
        esdf["fine_voxel_size"] = kw.get("fine_voxel") or None
        esdf["tsdf_voxel_size"] = kw.get("tsdf_voxel") or None
    if kw.get("target_exclusion"):
        # SL6-A A2 — `build_ag3s`: `esdf.update(target_exclusion)` (only the keys a flag gave; none = as before)
        esdf.update(dict(kw["target_exclusion"]))
    max_age = kw.get("max_field_age_sec")
    timing = {} if max_age is None else {"max_field_age_sec": max_age}
    data = {
        "collision_backend": "esdf",
        "pointcloud": {"range_max": kw["range_max"], **dict(kw.get("self_filter") or {})},
        "esdf": esdf,
        **({"timing": timing} if timing else {}),
        **({"constraint": constraint_section} if constraint_section else {}),
        # SL15 -- `build_ag3s`'s ``contact`` section (T44e A9, only the key a flag gave; none = as before). `panda_overrides`
        # later replaces only ``contact.contact_links``, so the two do not collide.
        **({"contact": dict(kw["contact"])} if kw.get("contact") else {}),
        **ss.subtask_config_section(kw.get("subtask_gate"), kw.get("subtask_probe_path")),
    }
    return data, (dict(kw["identity"]) if kw.get("identity") else None)


def v11ab_ag3s_config(kw: Optional[Mapping[str, Any]] = None):
    """The `AG3SConfig` V11AB's ``build_ag3s`` builds (``kw`` default: :func:`v11ab_build_kwargs` of V11AB)."""
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.trajopt import serve_safe as ss

    kw = v11ab_build_kwargs(parse_v11ab_args()) if kw is None else kw
    data, identity = v11ab_config_dict(kw)
    config = AG3SConfig.from_dict(data)
    if identity:
        config = config.with_overrides({"clustering": dict(identity)})
    ss.assert_ag3s_config(config, collision_backend="esdf", esdf_backend=kw.get("backend", "legacy"),
                          clustering=dict(identity or {}))
    return config


def default_esdf_bounds() -> tuple[tuple[float, ...], tuple[float, ...]]:
    from AG3S.ag3s.fields.esdf import default_bounds

    lo, hi = default_bounds()
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def panda_esdf_bounds() -> tuple[tuple[float, ...], tuple[float, ...]]:
    lo, hi = default_esdf_bounds()
    return (lo[0], lo[1], float(PANDA_ESDF_BOUNDS_LOWER_Z)), hi


def panda_overrides(esdf_bounds: str = "panda") -> dict[str, Any]:
    """The config override dict (``AG3SConfig.with_overrides`` shape)."""
    if esdf_bounds not in ESDF_BOUNDS_CHOICES:
        raise ValueError(f"esdf_bounds must be one of {ESDF_BOUNDS_CHOICES}, got {esdf_bounds!r}")
    out: dict[str, Any] = {"contact": {"contact_links": {pm.MANIPULATOR_SIDE: tuple(pm.PANDA_CONTACT_LINKS[pm.MANIPULATOR_SIDE])}}}
    if esdf_bounds == "panda":
        lo, hi = panda_esdf_bounds()
        out["esdf"] = {"bounds_lower": lo, "bounds_upper": hi}
    return out


def panda_ag3s_config(v11ab=None, *, esdf_bounds: str = "panda", extra_overrides: Optional[Mapping[str, Any]] = None):
    """V11AB config + :func:`panda_overrides` (+ ``extra_overrides``: tests / diagnostics only)."""
    from AG3S.trajopt import serve_safe as ss

    v11ab = v11ab_ag3s_config() if v11ab is None else v11ab
    config = v11ab.with_overrides(panda_overrides(esdf_bounds))
    if extra_overrides:
        config = config.with_overrides(dict(extra_overrides))
    else:
        ss.assert_ag3s_config(config, collision_backend=v11ab.collision_backend, esdf_backend=v11ab.esdf.backend)
    return config


def _flatten(prefix: str, value: Any, out: dict) -> None:
    if isinstance(value, Mapping) and value and not prefix.endswith(("contact_links", "phase_rules")):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def effective_config_dict(config) -> dict[str, Any]:
    """``config.to_dict()`` flattened to dotted keys, ESDF bounds resolved as ``curobo_builder`` does."""
    data = _jsonable(config.to_dict())
    if data["esdf"].get("bounds_lower") is None:
        lo, hi = default_esdf_bounds()
        data["esdf"]["bounds_lower"], data["esdf"]["bounds_upper"] = list(lo), list(hi)
    flat: dict[str, Any] = {}
    _flatten("", data, flat)
    return flat


def diff_against_v11ab(panda=None, v11ab=None) -> list[dict[str, Any]]:
    """Config rows where the Panda config differs from V11AB's (effective values): key, both values, reason.

    With the default profile this is exactly :data:`OVERRIDE_REASONS`' keys.
    """
    v11ab = v11ab_ag3s_config() if v11ab is None else v11ab
    panda = panda_ag3s_config(v11ab) if panda is None else panda
    a, b = effective_config_dict(v11ab), effective_config_dict(panda)
    rows = []
    for key in sorted(set(a) | set(b)):
        if a.get(key, "<absent>") != b.get(key, "<absent>"):
            row = {"key": key, "v11ab": a.get(key, "<absent>"), "panda": b.get(key, "<absent>"),
                   "reason": OVERRIDE_REASONS.get(key, "UNDECLARED")}
            if key == "esdf.bounds_lower":
                row["axes_changed"] = [ax for ax, x, y in zip("xyz", row["v11ab"], row["panda"]) if x != y]
            rows.append(row)
    return rows


def panda_attached_parent_links() -> tuple[str, ...]:
    from AG3S.trajopt.safe_policy import grasp_parent_links

    return grasp_parent_links(PANDA_GRASP_LINKS, PANDA_HOLD_LINKS)


def constructor_differences(kw: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    """What differs outside the config: models, attached slots, RB-Y1 build options not applied."""
    from AG3S.ag3s.config import DEFAULT_CONTACT_LINKS
    from AG3S.trajopt.safe_policy import grasp_parent_links

    kw = v11ab_build_kwargs(parse_v11ab_args()) if kw is None else kw
    return [
        {"what": "robot_model (self-filter)", "v11ab": "grounding_report.build_robot_model(TransportScene(model_xml))",
         "panda": f"model.build_panda_self_filter_model() (cover {pm.DEFAULT_SELF_FILTER_COVER})",
         "reason": "robot geometry (SL2/SL3)"},
        {"what": "constraint_robot_model", "v11ab": f"build_constraint_robot_model(links={kw.get('links')!r})",
         "panda": f"model.build_panda_constraint_model(panda_constraint_link_filter({kw.get('links')!r}))",
         "reason": "robot geometry; same link group name (SL2 link-set table)"},
        {"what": "attached_parent_links", "v11ab": list(grasp_parent_links()), "panda": list(panda_attached_parent_links()),
         "reason": "the palm a held object attaches to (T32 H2), Panda link name"},
        {"what": "constraint_sphere_options (not applied)", "v11ab": _jsonable(kw.get("constraint_sphere_options")),
         "panda": None,
         "reason": "RB-Y1 capsule-thinning knobs; franka.yml spheres are zero-length role-collision capsules, a 0.05 "
                   "radius scale would shrink them to 5 % (SL2 impl)"},
        {"what": "finger_cover (not applied)", "v11ab": _jsonable(kw.get("finger_cover")), "panda": None,
         "reason": "RB-Y1 finger-body cover (gripper_cover.finger_body_capsules); the Panda fingers keep their "
                   "franka.yml spheres (SL2)"},
        {"what": "gripper_cover", "v11ab": _jsonable(kw.get("gripper_cover")), "panda": None, "reason": "not set in V11AB either"},
        {"what": "contact links read by the start-up log", "v11ab": _jsonable(DEFAULT_CONTACT_LINKS),
         "panda": _jsonable(pm.PANDA_CONTACT_LINKS), "reason": "log only in build_ag3s; the config row is the decision"},
    ]


# --- AG3S --------------------------------------------------------------------------------------------------
def build_panda_ag3s(*, esdf_bounds: str = "panda", extra_overrides: Optional[Mapping[str, Any]] = None,
                     kw: Optional[Mapping[str, Any]] = None, models: Optional[pm.PandaModels] = None):
    """`AG3S` as V11AB's ``build_ag3s`` makes it, with the Panda injections. Returns ``(ag3s, info)``.

    Post-construction switches are set the way ``build_ag3s`` sets them (``held_body_cover``,
    ``held_free_observed_pad``). ``extra_overrides`` is for tests / diagnostics only (recorded in ``info``).
    """
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt import serve_safe as ss

    kw = v11ab_build_kwargs(parse_v11ab_args()) if kw is None else dict(kw)
    v11ab = v11ab_ag3s_config(kw)
    config = panda_ag3s_config(v11ab, esdf_bounds=esdf_bounds, extra_overrides=extra_overrides)
    # SL6-A A5 — `--palm-shell` (`kw["palm_shell"]`, key only when on): the Panda palm shell appended to the
    # constraint model (`model.build_panda_constraint_model`). Off: the V11AB models, unchanged.
    shell = {"palm_shell": dict(kw["palm_shell"])} if kw.get("palm_shell") else {}
    models = models or pm.build_panda_models(kw.get("links", "gripper"), **shell)
    parents = panda_attached_parent_links()
    ag3s = AG3S(config, robot_model=models.self_filter, constraint_robot_model=models.constraint,
                attached_parent_links=parents)
    if not extra_overrides:
        _, identity = v11ab_config_dict(kw)
        ss.assert_ag3s_config(ag3s.config, collision_backend="esdf", esdf_backend=kw.get("backend", "legacy"),
                              clustering=dict(identity or {}))
    target_exclusion = dict(kw.get("target_exclusion") or {})
    if target_exclusion:
        # SL6-A A2 — `build_ag3s`'s survival check: the keys a flag gave must be in the AG3S the server runs.
        lost = {k: getattr(ag3s.config.esdf, k) for k, v in target_exclusion.items()
                if getattr(ag3s.config.esdf, k) != v}
        if lost:
            raise RuntimeError(f"AG3S esdf config is not what the server asked for: {lost}, asked {target_exclusion}")
    contact = dict(kw.get("contact") or {})
    if contact:
        # SL15 -- `build_ag3s`'s survival check for the `contact` section (T44e A9).
        lost = {k: getattr(ag3s.config.contact, k) for k, v in contact.items() if getattr(ag3s.config.contact, k) != v}
        if lost:
            raise RuntimeError(f"AG3S contact config is not what the server asked for: {lost}, asked {contact}")
    self_filter = dict(kw.get("self_filter") or {})
    if self_filter:
        # SL15 sfhand -- survival check for the `pointcloud` self-filter keys a flag gave (--self-filter-inflation-link,
        # --self-filter-target-guard): the server must run the filter it was asked for, with the per-link table that
        # reaches `robot_sphere_mask` (``AG3S.self_filter_settings``; a link name the Panda self-filter model lacks already raised).
        lost = {k: getattr(ag3s.config.pointcloud, k) for k, v in self_filter.items() if getattr(ag3s.config.pointcloud, k) != v}
        if lost:
            raise RuntimeError(f"AG3S pointcloud config is not what the server asked for: {lost}, asked {self_filter}")
    if kw.get("held_body_cover"):
        ag3s.held_body_cover = True
    if kw.get("held_free_observed_pad") is not None:
        ag3s.held_free_observed_pad = float(kw["held_free_observed_pad"])
    if kw.get("reach_at_capture"):
        ag3s.reach_at_capture = True
    if kw.get("held_free_query_spheres"):   # SL10d -- key in `info` only when on
        ag3s.held_free_query_spheres = True
    if kw.get("hand_voxel_carve"):   # SL16 HVC -- key in `info` only when on
        ag3s.hand_voxel_carve = dict(kw["hand_voxel_carve"])
    if kw.get("held_point_model") is not None:   # SL-GTTO17 (B) -- key in `info` only when on
        ag3s.held_point_model_m = float(kw["held_point_model"])
    if getattr(ag3s.config.esdf, "target_extent", "estimated") == "attention_surface":
        # SL7 -- the surface selection needs head / left_wrist / right_wrist views; the Panda has no right wrist
        # (pi05_libero's third image is a masked zero image), so that view is a dummy (attention_surface.dummy_view).
        ag3s.surface_pad_missing_cameras = True
    info = {"esdf_bounds": esdf_bounds, "extra_overrides": _jsonable(dict(extra_overrides or {})),
            "n_self_filter_spheres": int(models.self_filter.n_spheres),
            "n_constraint_spheres": int(models.constraint.n_spheres),
            "constraint_links": sorted(set(models.constraint.sphere_link_names)),
            "attached_parent_links": list(parents),
            "held_body_cover": bool(ag3s.held_body_cover),
            "held_free_observed_pad": ag3s.held_free_observed_pad,
            "reach_at_capture": bool(getattr(ag3s, "reach_at_capture", False)),
            "gripper_max_opening": ag3s.max_opening.record() if hasattr(ag3s.max_opening, "record") else None}
    # SL6-A — keys only when on (the V11AB / pb panda_server_config.json is unchanged)
    if target_exclusion:
        info["target_exclusion"] = _jsonable(target_exclusion)
    if contact:   # SL15 -- key only when a flag gave a contact option
        info["contact"] = _jsonable(contact)
    if self_filter:   # SL15 sfhand -- key only when a flag gave a self-filter option: the table the filter actually applies
        info["self_filter"] = _jsonable({"asked": self_filter, "applied": ag3s.self_filter_settings()})
    if ag3s.surface_pad_missing_cameras:   # SL7 -- key only with the attention-surface extent
        info["surface_pad_missing_cameras"] = True
    if ag3s.held_free_query_spheres:   # SL10d -- key only when on
        info["held_free_query_spheres"] = True
    if ag3s.hand_voxel_carve:   # SL16 HVC -- key only when on
        info["hand_voxel_carve"] = _jsonable(ag3s.hand_voxel_carve)
    if getattr(ag3s, "held_point_model_m", None) is not None:   # SL-GTTO17 (B) -- key only when on
        info["held_point_model_m"] = float(ag3s.held_point_model_m)
    if getattr(models.constraint, "palm_shell_report", None) is not None:
        info["palm_shell"] = _jsonable(models.constraint.palm_shell_report)
    return ag3s, info


def manifest_entry(arm: str = V11AB_ARM, *, esdf_bounds: str = "panda") -> dict[str, Any]:
    """Run-manifest record: sources (paths + SHA-256), parsed flags, config diff, constructor differences."""
    args = parse_v11ab_args(arm)
    kw = v11ab_build_kwargs(args)
    v11ab = v11ab_ag3s_config(kw)
    panda = panda_ag3s_config(v11ab, esdf_bounds=esdf_bounds)
    return {
        "arm": arm,
        "sources": {str(p.relative_to(PROJECT_ROOT)): _sha256(p) for p in (START_SERVER, ENV_SH, RUN_CLIENT, SERVE_SAFE)},
        "server_flags": v11ab_server_flags(arm),
        "client_phase": v11ab_client_phase(),
        "build_ag3s_call": v11ab_build_call_source(),
        "build_ag3s_kwargs": _jsonable({k: v for k, v in kw.items()}),
        "esdf_bounds": esdf_bounds,
        "esdf_bounds_effective": {"lower": effective_config_dict(panda)["esdf.bounds_lower"],
                                  "upper": effective_config_dict(panda)["esdf.bounds_upper"]},
        "config_diff_against_v11ab": diff_against_v11ab(panda, v11ab),
        "constructor_differences": constructor_differences(kw),
        "override_reasons": dict(OVERRIDE_REASONS),
        "panda_config": _jsonable(panda.to_dict()),
        # This string is kept byte for byte (it is in every run manifest). "offline" = this profile / the perception
        # replay; the closed-loop server applies the TO flags (SL-5) and the recording flags (SL5j,
        # server.constraint_recorder).
        "not_ag3s_settings": "TO / recording flags (esdf-margin, obstacle-margin, plan-horizon, lookahead*, qp-eps, "
                             "w-deflection-rate, record-constraints) are parsed but not used offline (SL-5)",
    }


__all__ = [
    "ESDF_BOUNDS_CHOICES", "LOG_ONLY_BUILD_KEYWORDS", "OVERRIDE_REASONS", "PANDA_ESDF_BOUNDS_LOWER_Z", "PANDA_GRASP_LINKS",
    "PANDA_HOLD_LINKS", "V11AB_ARM", "build_panda_ag3s", "constructor_differences", "default_esdf_bounds",
    "diff_against_v11ab", "effective_config_dict", "manifest_entry", "panda_ag3s_config", "panda_attached_parent_links",
    "panda_esdf_bounds", "panda_overrides", "parse_v11ab_args", "ta_flags", "v11ab_ag3s_config", "v11ab_build_call_source",
    "v11ab_build_kwargs", "v11ab_client_phase", "v11ab_config_dict", "v11ab_required_args", "v11ab_server_argv",
    "v11ab_server_flags", "match_ag3s_version", "version_flag_values",
    "FINAL_CONFIG_FLAGS", "FINAL_CONFIG_VERSIONS", "G1A4_FLAGS", "P0M_FLAGS", "P0M_NBC_FLAGS", "P0M_NOHULL_FLAGS",
    "P0M_VERSIONS", "PANDA_HSR6_PROBE", "PANDA_HSR6_PROBE_REL", "PANDA_HSR6_PROBE_SHA256", "is_final_version", "version_server_argv",
    "version_server_flags",
]
