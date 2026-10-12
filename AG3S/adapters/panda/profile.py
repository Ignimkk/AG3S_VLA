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
and kept in :func:`manifest_entry` but play no part here (closed loop = SL-5).
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


def v11ab_server_argv(arm: str = V11AB_ARM) -> list[str]:
    req = v11ab_required_args()
    return ["--config", req["config"], "--checkpoint", "<not loaded: perception only>",
            "--model-xml", req["model_xml"], "--port", "0", *v11ab_server_flags(arm)]


def parse_v11ab_args(arm: str = V11AB_ARM, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """serve_safe's parser + its own flag checks + its subtask-gate resolution, as ``main`` runs them."""
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    args = ap.parse_args(list(argv) if argv is not None else v11ab_server_argv(arm))
    ss.reject_bad_flag_combinations(ap, args)
    ss.resolve_subtask_gate(args)
    return args


def _main_build_call() -> ast.Call:
    tree = ast.parse(SERVE_SAFE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "build_ag3s"]
    if len(calls) != 1:
        raise ValueError(f"serve_safe.main has {len(calls)} build_ag3s calls; expected 1")
    return calls[0]


def v11ab_build_call_source() -> dict[str, str]:
    """``{keyword: source expression}`` of ``serve_safe.main``'s ``build_ag3s(...)`` call."""
    return {k.arg: ast.unparse(k.value) for k in _main_build_call().keywords}


def v11ab_build_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The keywords ``serve_safe.main`` passes to ``build_ag3s`` for ``args`` (log-only ones omitted).

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
        out[keyword.arg] = eval(compile(ast.Expression(keyword.value), str(SERVE_SAFE), "eval"), namespace)  # noqa: S307
    return out


# --- config ---------------------------------------------------------------------------------------------
def v11ab_config_dict(kw: Mapping[str, Any]) -> tuple[dict, Optional[dict]]:
    """``(AG3SConfig.from_dict input, identity overrides)`` -- the ``build_ag3s`` assembly, verbatim.

    Copy of ``serve_safe.build_ag3s`` (config part). The test compares the resulting config with the one
    the real ``build_ag3s`` hands to ``AG3S`` for the same keywords.
    """
    from AG3S.trajopt import serve_safe as ss

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
    models = models or pm.build_panda_models(kw.get("links", "gripper"))
    parents = panda_attached_parent_links()
    ag3s = AG3S(config, robot_model=models.self_filter, constraint_robot_model=models.constraint,
                attached_parent_links=parents)
    if not extra_overrides:
        _, identity = v11ab_config_dict(kw)
        ss.assert_ag3s_config(ag3s.config, collision_backend="esdf", esdf_backend=kw.get("backend", "legacy"),
                              clustering=dict(identity or {}))
    if kw.get("held_body_cover"):
        ag3s.held_body_cover = True
    if kw.get("held_free_observed_pad") is not None:
        ag3s.held_free_observed_pad = float(kw["held_free_observed_pad"])
    info = {"esdf_bounds": esdf_bounds, "extra_overrides": _jsonable(dict(extra_overrides or {})),
            "n_self_filter_spheres": int(models.self_filter.n_spheres),
            "n_constraint_spheres": int(models.constraint.n_spheres),
            "constraint_links": sorted(set(models.constraint.sphere_link_names)),
            "attached_parent_links": list(parents),
            "held_body_cover": bool(ag3s.held_body_cover),
            "held_free_observed_pad": ag3s.held_free_observed_pad,
            "gripper_max_opening": ag3s.max_opening.record() if hasattr(ag3s.max_opening, "record") else None}
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
        "not_ag3s_settings": "TO / recording flags (esdf-margin, obstacle-margin, plan-horizon, lookahead*, qp-eps, "
                             "w-deflection-rate, record-constraints) are parsed but not used offline (SL-5)",
    }


__all__ = [
    "ESDF_BOUNDS_CHOICES", "LOG_ONLY_BUILD_KEYWORDS", "OVERRIDE_REASONS", "PANDA_ESDF_BOUNDS_LOWER_Z", "PANDA_GRASP_LINKS",
    "PANDA_HOLD_LINKS", "V11AB_ARM", "build_panda_ag3s", "constructor_differences", "default_esdf_bounds",
    "diff_against_v11ab", "effective_config_dict", "manifest_entry", "panda_ag3s_config", "panda_attached_parent_links",
    "panda_esdf_bounds", "panda_overrides", "parse_v11ab_args", "ta_flags", "v11ab_ag3s_config", "v11ab_build_call_source",
    "v11ab_build_kwargs", "v11ab_client_phase", "v11ab_config_dict", "v11ab_required_args", "v11ab_server_argv",
    "v11ab_server_flags",
]
