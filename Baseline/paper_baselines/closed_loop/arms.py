"""Closed-loop arms: which method, which family, which (non-)privileged inputs, how it differs from the paper.

An arm = ``--method`` + ``--obstacles`` + ``--target``. Everything here is **non-privileged** (the SafeLIBERO wire carries
RGB / depth / K / T / q only: no GT segmentation, no obstacle names). Privileged variants need a client change and are
left to the user (task "되돌아올 지점").

| family | where the method acts | process |
|---|---|---|
| ``none``     | nothing -- the VLA-only server path (``PandaVLAOnly(SeededPolicy(policy))``)            | server |
| ``filter``   | after the policy: the 10-row EE-delta chunk is edited, the client executes rows 0..4      | ``.venv-baselines`` sidecar |
| ``guidance`` | inside flow-matching sampling (``pi0.sample_actions(guidance_fn=...)``, post-Euler hook) | server (JAX) |
"""
from __future__ import annotations

import dataclasses

FAMILIES = ("none", "filter", "guidance")
OBSTACLE_SOURCES = ("detector", "depth")  # filters; guidance always uses the depth lift
TARGET_SOURCES = ("detector", "none")


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    family: str
    method: str | None
    cite: str
    default_obstacles: str | None
    default_target: str | None
    deviation: tuple[str, ...]


#: Deviations every filter / guidance arm shares in this closed loop (on top of each method's own README).
COMMON_FILTER_DEVIATIONS = (
    "chunk-level filtering at the policy call (10 rows, later rows on the method's predicted EE pose; the client executes "
    "rows 0-4) instead of the original per-env-step call",
    "perception from the SafeLIBERO wire only (head + wrist 256 px depth, 224 px policy RGB re-mirrored and resized to "
    "256 px); no VLM, no external API",
)
COMMON_GUIDANCE_DEVIATIONS = (
    "post-Euler openpi hook; x0_hat from the sampling noise (guidance_common x0_mode='auto' -> 'noise')",
    "obstacles = depth lift of head + wrist minus robot spheres (Panda kinematics, +2 cm), minus the table plane, minus "
    "the detected target box (non-privileged); padded to a fixed size",
    "guidance evaluated on the JAX CPU backend through a host callback inside the sampling loop (--guidance-exec "
    "callback, default) or inlined in the jit (--guidance-exec inline)",
)

ARMS: dict[str, Arm] = {a.name: a for a in (
    Arm("none", "none", None, "[18] pi0.5 (VLA only)", None, None,
        ("identical server path to `serve_safe --robot panda --no-safe` (PandaVLAOnly(SeededPolicy(policy)))",)),
    Arm("knows", "filter", "knows", "[17] KNOWS", "detector", None,
        COMMON_FILTER_DEVIATIONS + ("target exclusion by the Grounding DINO target box (paper: VLA attention, Eq. 4)",
                                    "obstacles = Grounding DINO 'object' boxes minus the target box, box-cropped depth")),
    Arm("aegis", "filter", "aegis", "[10] VLSA / AEGIS", "detector", None,
        COMMON_FILTER_DEVIATIONS + ("GLM-4.5V obstacle naming replaced by a generic Grounding DINO prompt minus the "
                                    "target box (= offline arm aegis:detector_all)",)),
    Arm("multilink_cbf", "filter", "multilink_cbf", "[1] Multi-link safety filter", "detector", None,
        COMMON_FILTER_DEVIATIONS + ("same obstacle source as aegis; moving-hazard optical flow omitted",)),
    Arm("omniguide", "guidance", "omniguide", "[19] OmniGuide", None, "detector", COMMON_GUIDANCE_DEVIATIONS),
    Arm("constrained_flow_matching", "guidance", "constrained_flow_matching", "[5] Constrained flow matching", None,
        "detector", COMMON_GUIDANCE_DEVIATIONS + ("paper uses SafeLIBERO GT spheres (privileged); here the depth cloud",)),
    Arm("lan_o3dp_guidance_adapted", "guidance", "lan_o3dp_guidance_adapted", "[15] Lan-o3dp (guidance adapted to pi0.5)",
        None, "detector", COMMON_GUIDANCE_DEVIATIONS + ("not the original policy's result",)),
    Arm("nudge", "guidance", "nudge", "[9] NUDGE", None, "detector", COMMON_GUIDANCE_DEVIATIONS),
)}


def resolve(name: str, obstacles: str | None = None, target: str | None = None) -> dict:
    """Arm record for the manifest (method, family, effective perception, privileged, deviations)."""
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; available: {sorted(ARMS)}")
    a = ARMS[name]
    obst = obstacles if obstacles is not None else a.default_obstacles
    tgt = target if target is not None else a.default_target
    if a.family == "filter" and obst not in OBSTACLE_SOURCES:
        raise ValueError(f"--obstacles must be one of {OBSTACLE_SOURCES} for a filter arm")
    if a.family != "filter" and obstacles is not None:
        raise ValueError("--obstacles applies to filter arms only (guidance uses the depth lift)")
    if a.family == "none" and target is not None:
        raise ValueError("--target does not apply to --method none")
    if a.family == "filter" and obst == "detector" and target not in (None, "detector"):
        raise ValueError("the detector obstacle source removes the target box itself (target=detector)")
    if a.family == "filter":
        tgt = "detector" if obst == "detector" else (tgt or "detector")
    if tgt is not None and tgt not in TARGET_SOURCES:
        raise ValueError(f"--target must be one of {TARGET_SOURCES}")
    variant = name if a.family == "none" else f"{name}:obst-{obst or 'depth'}:target-{tgt}"
    extra = []
    if tgt == "none":
        extra.append("no target exclusion (--target none): the target is part of the obstacle set")
    if a.family == "filter" and obst == "depth":
        extra.append("obstacles = connected components of the depth lift (robot / table / target box removed), "
                     "passed as obstacle_source='given'")
    return {"arm": name, "variant": variant, "family": a.family, "method": a.method, "cite": a.cite,
            "obstacles": obst if a.family == "filter" else ("depth_lift" if a.family == "guidance" else None),
            "target": tgt, "privileged": False, "privileged_inputs": [],
            "deviation_from_paper": list(a.deviation) + extra}


__all__ = ["ARMS", "Arm", "FAMILIES", "OBSTACLE_SOURCES", "TARGET_SOURCES", "resolve"]
