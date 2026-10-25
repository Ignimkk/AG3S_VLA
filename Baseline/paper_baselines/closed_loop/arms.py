"""Closed-loop arms: which method, which family, which (non-)privileged inputs, how it differs from the paper.

An arm = ``--method`` + ``--obstacles`` + ``--target`` (+ ``--gt``). Without ``--gt`` everything is **non-privileged**
(the SafeLIBERO wire carries RGB / depth / K / T / q only). ``--gt [SOURCE]`` (BASELINE-BENCH-READY) is the privileged
variant: simulator ground-truth obstacles from ``libero_safe_client.py --emit-privileged-obstacles`` (``privileged.py``),
the original papers' conditions (AEGIS: obstacle names given to the detector in place of the VLM; CFM: SafeLIBERO GT
spheres); the manifest records ``privileged: true`` and the privileged inputs.

| ``--gt`` source | family | obstacles | client flags |
|---|---|---|---|
| ``names``  | filter   | Grounding DINO prompted with the active SafeLIBERO obstacle names (AEGIS's VLM output replaced by GT names), depth crop as the non-privileged arm | ``--emit-privileged-obstacles`` |
| ``seg``    | filter   | GT segmentation of the active obstacle objects (``ObstacleCloudSource(gt_seg)``) | ``--emit-privileged-obstacles --capture-segmentation`` |
| ``geoms``  | guidance | the active obstacles' collision geoms as primitives (``privileged/obstacle_geoms``) | ``--emit-privileged-obstacles`` |
| ``repo``   | guidance | the CFM repo rule: every geom named ``*obstacle*``, a mesh as the box of its ``geom_size`` | ``--emit-privileged-obstacles`` |
| ``spheres``| guidance | one enclosing sphere per active obstacle object (CFM paper: "GT sphere") | ``--emit-privileged-obstacles`` |

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
GT_SOURCES = {"filter": ("names", "seg"), "guidance": ("geoms", "repo", "spheres")}
#: ``--gt`` without a value: the source closest to each original paper
GT_DEFAULT = {"aegis": "names", "knows": "seg", "multilink_cbf": "seg", "constrained_flow_matching": "spheres",
              "nudge": "geoms", "omniguide": "geoms", "lan_o3dp_guidance_adapted": "geoms"}
#: ``privileged.obstacle_geoms`` shape of a guidance ``--gt`` source
GT_GEOM_SHAPE = {"geoms": "collision", "repo": "repo", "spheres": "sphere"}
#: guidance arms whose method reads the target (OmniGuide's attractor); the others get ``--target none`` under ``--gt``
TARGET_READERS = ("omniguide",)
#: arm-specific deviations that remain under ``--gt``
GT_EXTRA_DEVIATIONS = {"multilink_cbf": ("moving-hazard optical flow omitted",),
                       "lan_o3dp_guidance_adapted": ("not the original policy's result",),
                       "knows": ("no target exclusion step: only the SafeLIBERO obstacles are instances (paper: VLA "
                                 "attention, Eq. 4)",)}
GT_DESCRIPTION = {
    "names": "obstacle names = the active SafeLIBERO obstacle objects (privileged/obstacle_names) -> Grounding DINO "
             "prompt (detector_prompt='obstacle_names'), depth crop; replaces the paper's VLM naming",
    "seg": "obstacles = GT segmentation of the active SafeLIBERO obstacle objects (privileged/segmentation_geom + "
           "instances_geom_ids, obstacle_source='gt_seg')",
    "geoms": "obstacles = collision geoms of the active SafeLIBERO obstacle objects as primitives "
             "(privileged/obstacle_geoms, base frame)",
    "repo": "obstacles = CFM repo rule (every geom named *obstacle*, mesh -> box of geom_size; "
            "live_viewer.GroundTruthObstacleTracker)",
    "spheres": "obstacles = one enclosing sphere per active SafeLIBERO obstacle object (CFM paper Sec. 5 GT sphere)"}


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


def resolve_gt(name: str, gt: str | None, obstacles: str | None = None, target: str | None = None) -> dict:
    """Privileged (``--gt``) arm record. ``gt="default"`` = :data:`GT_DEFAULT`."""
    a = ARMS[name]
    if a.family == "none":
        raise ValueError("--gt does not apply to --method none")
    src = GT_DEFAULT[name] if gt == "default" else gt
    if src not in GT_SOURCES[a.family]:
        raise ValueError(f"--gt for a {a.family} arm must be one of {GT_SOURCES[a.family]}, got {src!r}")
    if obstacles is not None:
        raise ValueError("--obstacles and --gt are exclusive (--gt sets the obstacle source)")
    tgt = target if target is not None else ("detector" if name in TARGET_READERS else "none")
    if a.family == "filter" and tgt != "none":
        raise ValueError("a --gt filter arm has no target exclusion (only the SafeLIBERO obstacles are obstacles)")
    if tgt not in TARGET_SOURCES:
        raise ValueError(f"--target must be one of {TARGET_SOURCES}")
    inputs = {"names": ["obstacle_names"], "seg": ["segmentation_geom", "instances_geom_ids"],
              "geoms": ["obstacle_geoms"], "repo": ["obstacle_geoms"], "spheres": ["obstacle_geoms"]}[src]
    if a.family == "filter":
        dev = [COMMON_FILTER_DEVIATIONS[0]]
        if src == "names":
            dev.append("RGB / depth from the SafeLIBERO wire (head + wrist 256 px depth, 224 px policy RGB "
                       "re-mirrored and resized to 256 px)")
    else:
        dev = [COMMON_GUIDANCE_DEVIATIONS[0], COMMON_GUIDANCE_DEVIATIONS[2]]
    dev.append("PRIVILEGED: " + GT_DESCRIPTION[src])
    dev += list(GT_EXTRA_DEVIATIONS.get(name, ()))
    if tgt == "detector":
        dev.append("target (attractor) = Grounding DINO target box of the instruction phrase (non-privileged)")
    return {"arm": name, "variant": f"{name}:gt-{src}", "family": a.family, "method": a.method, "cite": a.cite,
            "obstacles": f"gt-{src}", "target": tgt, "privileged": True, "gt_source": src,
            "gt_geom_shape": GT_GEOM_SHAPE.get(src), "privileged_inputs": inputs,
            "client_flags": ["--emit-privileged-obstacles"] + (["--capture-segmentation"] if src == "seg" else []),
            "deviation_from_paper": dev}


def resolve(name: str, obstacles: str | None = None, target: str | None = None, gt: str | None = None) -> dict:
    """Arm record for the manifest (method, family, effective perception, privileged, deviations)."""
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; available: {sorted(ARMS)}")
    if gt is not None:
        return resolve_gt(name, gt, obstacles, target)
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


def arm_spec(spec: str) -> dict:
    """``jobs.py`` arm spec -> launcher arguments: ``<arm>`` | ``<arm>:gt`` | ``<arm>:gt-<source>``."""
    name, _, rest = spec.partition(":")
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r} in {spec!r}; available: {sorted(ARMS)}")
    if not rest:
        return {"method": name, "gt": None, "rec": resolve(name)}
    if rest == "gt":
        return {"method": name, "gt": "default", "rec": resolve(name, gt="default")}
    if rest.startswith("gt-"):
        return {"method": name, "gt": rest[3:], "rec": resolve(name, gt=rest[3:])}
    raise ValueError(f"arm spec {spec!r}: expected <arm>, <arm>:gt or <arm>:gt-<source>")


__all__ = ["ARMS", "Arm", "FAMILIES", "GT_DEFAULT", "GT_SOURCES", "OBSTACLE_SOURCES", "TARGET_SOURCES", "arm_spec",
           "resolve", "resolve_gt"]
