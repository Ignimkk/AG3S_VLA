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

``filter_rate`` (F2, 2026-11-10): ``chunk`` (default) = the row above; ``step`` = the policy still runs every 5 env steps,
and every env step one row of the cached chunk is filtered on that step's measured EE pose and observation
(``--filter-rate step``; :data:`STEP_FILTER_DEVIATION` replaces the chunk-level deviation).
``knows`` detector arm target (F4a, 2026-11-10): ``attention`` (default) = paper Eq. 2-4 on the pi0.5 attention
(layer 12 / head 3); ``--target detector`` = the port's Grounding DINO target box (before F4a).
| ``guidance`` | inside flow-matching sampling (``pi0.sample_actions(guidance_fn=...)``, post-Euler hook) | server (JAX) |
"""
from __future__ import annotations

import dataclasses

FAMILIES = ("none", "filter", "guidance")
OBSTACLE_SOURCES = ("detector", "depth")  # filters; guidance always uses the depth lift
TARGET_SOURCES = ("detector", "none", "attention")
#: arms whose detector variant can exclude the target by the policy's attention (KNOWS Eq. 2-4, F4a); their default
ATTENTION_TARGET_ARMS = ("knows",)
FILTER_RATES = ("chunk", "step")
GT_SOURCES = {"filter": ("names", "seg", "exact"), "guidance": ("geoms", "repo", "spheres")}
#: ``--gt`` without a value: the source closest to each original paper
#: multi-link CBF: "exact" since 2026-11-10 (its paper's privileged condition, Table III; doc §4.12); before: "seg"
GT_DEFAULT = {"aegis": "names", "knows": "seg", "multilink_cbf": "exact", "constrained_flow_matching": "spheres",
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
#: F2 ``--filter-rate step``: replaces ``COMMON_FILTER_DEVIATIONS[0]`` (what remains different from the papers)
STEP_FILTER_DEVIATION = ("per-env-step filtering on the measured EE pose and the current observation (as the original "
                         "methods); the policy chunk is still queried every 5 env steps (H=10, rows 0-4 executed, "
                         "replan 5) and row k of it is filtered at the k-th step after the call")
#: F4a: KNOWS detector arm with the paper's attention target exclusion (``--target attention``, default)
KNOWS_ATTENTION_DEVIATION = ("target exclusion by pi0.5 attention (paper Eq. 2-4, layer 12 / head 3, agent view) over "
                             "the Grounding DINO 'object' instances; masks = projected tracked ellipsoids; K = 5 policy "
                             "calls and delta = 7.3e-6 from our offline P0b (not in the paper); attention recomputed "
                             "at denoising step 0 of the served sample (same obs / noise / weights)")
GT_DESCRIPTION = {
    "names": "obstacle names = the active SafeLIBERO obstacle objects (privileged/obstacle_names) -> Grounding DINO "
             "prompt (detector_prompt='obstacle_names'), depth crop; replaces the paper's VLM naming",
    "seg": "obstacles = GT segmentation of the active SafeLIBERO obstacle objects (privileged/segmentation_geom + "
           "instances_geom_ids, obstacle_source='gt_seg')",
    "geoms": "obstacles = collision geoms of the active SafeLIBERO obstacle objects as primitives "
             "(privileged/obstacle_geoms, base frame)",
    "repo": "obstacles = CFM repo rule (every geom named *obstacle*, mesh -> box of geom_size; "
            "live_viewer.GroundTruthObstacleTracker)",
    "spheres": "obstacles = one enclosing sphere per active SafeLIBERO obstacle object (CFM paper Sec. 5 GT sphere)",
    "exact": "obstacles = exact geometry: MVEE of the active SafeLIBERO obstacle's collision geoms (privileged/obstacles, "
             "base frame), fitted at the first call, no perception / trim (multi-link CBF paper Table III privileged "
             "condition; obstacle_source='ellipsoids')"}


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
        COMMON_FILTER_DEVIATIONS + ("GLM-4.5V obstacle naming (one obstacle, 'preferably from prefer_list') replaced by "
                                    "the official prompt answered by a local open VLM (Qwen2.5-VL-7B-Instruct, "
                                    "greedy) -> one Grounding DINO phrase (detector_prompt=vlm_name, 2026-11-10; before: a "
                                    "generic Grounding DINO prompt minus the "
                                    "target box (= offline arm aegis:detector_all)",)),
    Arm("multilink_cbf", "filter", "multilink_cbf", "[1] Multi-link safety filter", "detector", None,
        COMMON_FILTER_DEVIATIONS + ("same obstacle source as aegis (paper: Qwen3-VL names one hazard -> here Qwen2.5-VL-7B "
                                    "with the official AEGIS prompt, vlm_name, 2026-11-10); moving-hazard "
                                    "optical flow omitted",)),
    Arm("omniguide", "guidance", "omniguide", "[19] OmniGuide", None, "detector", COMMON_GUIDANCE_DEVIATIONS),
    Arm("constrained_flow_matching", "guidance", "constrained_flow_matching", "[5] Constrained flow matching", None,
        "detector", COMMON_GUIDANCE_DEVIATIONS + ("paper uses SafeLIBERO GT spheres (privileged); here the depth cloud",)),
    Arm("lan_o3dp_guidance_adapted", "guidance", "lan_o3dp_guidance_adapted", "[15] Lan-o3dp (guidance adapted to pi0.5)",
        None, "detector", COMMON_GUIDANCE_DEVIATIONS + ("not the original policy's result",)),
    Arm("nudge", "guidance", "nudge", "[9] NUDGE", None, "detector", COMMON_GUIDANCE_DEVIATIONS),
)}


def _rate_deviations(dev: list, filter_rate: str) -> list:
    if filter_rate not in FILTER_RATES:
        raise ValueError(f"filter_rate must be one of {FILTER_RATES}")
    if filter_rate == "chunk":
        return dev
    return [STEP_FILTER_DEVIATION if d == COMMON_FILTER_DEVIATIONS[0] else d for d in dev]


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
    if tgt == "attention":
        raise ValueError("--target attention applies to the non-privileged detector arm (a --gt filter arm has no "
                         "target exclusion)")
    if a.family == "filter" and tgt != "none":
        raise ValueError("a --gt filter arm has no target exclusion (only the SafeLIBERO obstacles are obstacles)")
    if tgt not in TARGET_SOURCES:
        raise ValueError(f"--target must be one of {TARGET_SOURCES}")
    inputs = {"names": ["obstacle_names"], "seg": ["segmentation_geom", "instances_geom_ids"],
              "geoms": ["obstacle_geoms"], "repo": ["obstacle_geoms"], "spheres": ["obstacle_geoms"],
              "exact": ["obstacles (collision geoms)"]}[src]
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


def resolve(name: str, obstacles: str | None = None, target: str | None = None, gt: str | None = None,
            filter_rate: str = "chunk") -> dict:
    """Arm record for the manifest (method, family, effective perception, privileged, deviations).

    ``filter_rate`` (F2): ``chunk`` (default) = the record as before; ``step`` (filter arms only) swaps the chunk-level
    deviation for :data:`STEP_FILTER_DEVIATION` and adds ``filter_rate: "step"``."""
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; available: {sorted(ARMS)}")
    if filter_rate != "chunk" and ARMS[name].family != "filter":
        raise ValueError("--filter-rate step applies to filter arms only")
    if gt is not None:
        rec = resolve_gt(name, gt, obstacles, target)
        if filter_rate != "chunk":
            rec["deviation_from_paper"] = _rate_deviations(rec["deviation_from_paper"], filter_rate)
            rec["filter_rate"] = filter_rate
        return rec
    a = ARMS[name]
    obst = obstacles if obstacles is not None else a.default_obstacles
    tgt = target if target is not None else a.default_target
    if a.family == "filter" and obst not in OBSTACLE_SOURCES:
        raise ValueError(f"--obstacles must be one of {OBSTACLE_SOURCES} for a filter arm")
    if a.family != "filter" and obstacles is not None:
        raise ValueError("--obstacles applies to filter arms only (guidance uses the depth lift)")
    if a.family == "none" and target is not None:
        raise ValueError("--target does not apply to --method none")
    if target == "attention" and not (name in ATTENTION_TARGET_ARMS and obst == "detector"):
        raise ValueError(f"--target attention applies to {ATTENTION_TARGET_ARMS} with --obstacles detector only")
    if a.family == "filter" and obst == "detector" and target not in (None, "detector", "attention"):
        raise ValueError("the detector obstacle source removes the target box itself (target=detector)")
    if a.family == "filter":
        if obst == "detector":
            tgt = target or ("attention" if name in ATTENTION_TARGET_ARMS else "detector")
        else:
            tgt = tgt or "detector"
    if tgt is not None and tgt not in TARGET_SOURCES:
        raise ValueError(f"--target must be one of {TARGET_SOURCES}")
    variant = name if a.family == "none" else f"{name}:obst-{obst or 'depth'}:target-{tgt}"
    extra = []
    if tgt == "none":
        extra.append("no target exclusion (--target none): the target is part of the obstacle set")
    if a.family == "filter" and obst == "depth":
        extra.append("obstacles = connected components of the depth lift (robot / table / target box removed), "
                     "passed as obstacle_source='given'")
    dev = list(a.deviation)
    if tgt == "attention":  # F4a: the GDINO target-box deviation of the knows record is replaced
        dev = [KNOWS_ATTENTION_DEVIATION if d.startswith("target exclusion by the Grounding DINO target box") else
               ("obstacles = Grounding DINO 'object' boxes (target not removed in the detector; attention excludes "
                "it), box-cropped depth" if d.startswith("obstacles = Grounding DINO 'object' boxes minus") else d)
               for d in dev]
    rec = {"arm": name, "variant": variant, "family": a.family, "method": a.method, "cite": a.cite,
           "obstacles": obst if a.family == "filter" else ("depth_lift" if a.family == "guidance" else None),
           "target": tgt, "privileged": False, "privileged_inputs": [],
           "deviation_from_paper": _rate_deviations(dev, filter_rate) + extra}
    if filter_rate != "chunk":
        rec["filter_rate"] = filter_rate
    return rec


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


__all__ = ["ATTENTION_TARGET_ARMS", "FILTER_RATES", "STEP_FILTER_DEVIATION", "ARMS", "Arm", "FAMILIES", "GT_DEFAULT", "GT_SOURCES", "OBSTACLE_SOURCES", "TARGET_SOURCES", "arm_spec",
           "resolve", "resolve_gt"]
