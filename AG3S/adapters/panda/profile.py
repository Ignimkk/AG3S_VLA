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
UNPORTED_BUILD_KEYWORDS = ("palm_cover", "wrist_cover", "contact", "confidence_gate")

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
    return (str(version) in FINAL_CONFIG_VERSIONS or str(version) in P0M_VERSIONS or str(version) in P0M_SL9_VERSIONS
            or str(version) in PC_LATCH_VERSIONS or str(version) in PC_NOPROBE_VERSIONS)   # SL7d / SL7e / SL9


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
    if version in P0M_VERSIONS or version in P0M_SL9_VERSIONS or version in PC_NOPROBE_VERSIONS:   # SL7 / SL7e / SL9: no HSR6 rule -> no probe
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
    if kw.get("held_body_cover"):
        ag3s.held_body_cover = True
    if kw.get("held_free_observed_pad") is not None:
        ag3s.held_free_observed_pad = float(kw["held_free_observed_pad"])
    if kw.get("reach_at_capture"):
        ag3s.reach_at_capture = True
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
    if ag3s.surface_pad_missing_cameras:   # SL7 -- key only with the attention-surface extent
        info["surface_pad_missing_cameras"] = True
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
