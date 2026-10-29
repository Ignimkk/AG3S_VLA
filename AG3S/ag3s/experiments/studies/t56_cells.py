"""T56 — the candidates whose self score is evaluated: T51 / T51b 3D cells and T54's 8 arms.

`T56.task.md` 사전 고정 1. Two candidate kinds, evaluated **separately**:

``cell``  a single attention cell fed to the serving grounding as its only map (primary = split =
          choice = that cell; the T51 / T51b replay). The set is the T51 / T51b **3D candidate sets**,
          read (never recomputed) from their scene-aggregate ``score2d__scene.json``:
          action Ω (T51, forced cells included — the T51 ``candidates_3d`` already contains them) and
          noun Ω (T51b ``noun``). Because the 3D candidate set of T51 / T51b depended on the label
          used for the 2D pre-selection, the default here is the **union of the GT-label and the
          self-label candidate sets** (`CELL_SETS`), so the set itself is not chosen by either label
          alone; ``gt`` / ``self`` restrict it. Groups reported: ``cells`` (all), ``cells_action``,
          ``cells_noun``.
          Maps: action ``<agg>:L<l>H<h>`` = denoise 0 of the T47-format attention npz (the
          `t54_arms.action_map` expression, float64 → float32 — the T51 `cell_map` values); noun
          ``noun_target:L<l>H<h>`` = the ``target`` query block of the T49-format noun extract at that
          (layer, head), raw σ0 (the T51b `noun_block` values).
``arm``   T54's 8 arms (`t54_arms.ARMS`, unchanged — imported, never redefined).

CPU, no MuJoCo.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Mapping, Optional, Sequence

import numpy as np

#: The T51 / T51b run whose 3D candidate sets T56 reads (`T51.verify.json` / `T51b.verify.json`).
T51_RUN = ("/mnt/dev/AG3S_VLA-storage/results/development/"
           "20261012T180318+0900_28b981600bf7_852011fdcae0")
#: score2d files (scene aggregate = the final T51 / T51b protocol) per (Ω, label).
SCORE2D = {
    ("action", "gt"): "verify/T51/gt/score2d/score2d__scene.json",
    ("action", "self"): "verify/T51/self/score2d/score2d__scene.json",
    ("noun", "gt"): "verify/T51b/noun/gt/score2d/score2d__scene.json",
    ("noun", "self"): "verify/T51b/noun/self/score2d/score2d__scene.json",
}
CELL_SETS = ("union", "gt", "self")
DEFAULT_CELL_SET = "union"
ACTION_AGGS = ("last", "mean", "first")
NOUN_QUERY = "noun_target"
_CELL_RE = re.compile(r"^([A-Za-z0-9_]+):L(\d+)H(\d+)$")
#: Candidate groups of the cell kind (selection is made inside each group).
CELL_GROUPS = ("cells", "cells_action", "cells_noun")
ARM_GROUP = "arms"


def parse_cell(cid: str) -> tuple[str, int, int]:
    """``(query, layer, head)`` of ``"<query>:L<l>H<h>"``; query ∈ action aggs or ``noun_target``."""
    m = _CELL_RE.match(str(cid).strip())
    if not m:
        raise ValueError(f"not a cell id {cid!r}")
    q, l, h = m.group(1), int(m.group(2)), int(m.group(3))
    if q not in ACTION_AGGS and q != NOUN_QUERY:
        raise ValueError(f"cell {cid!r}: query {q!r} is neither an action aggregation nor {NOUN_QUERY}")
    return q, l, h


def family(cid: str) -> str:
    return "noun" if parse_cell(cid)[0] == NOUN_QUERY else "action"


def read_candidates(path: pathlib.Path) -> dict:
    """``{"candidates_3d": [...], "forced": [...]}`` of one score2d file (read only)."""
    d = json.loads(pathlib.Path(path).read_text())
    if "candidates_3d" not in d:
        raise SystemExit(f"{path}: no candidates_3d")
    return {"candidates_3d": [str(c) for c in d["candidates_3d"]],
            "forced": [str(c) for c in d.get("forced", [])], "label": d.get("label")}


def cell_candidates(t51_run: str | pathlib.Path = T51_RUN, which: str = DEFAULT_CELL_SET) -> dict:
    """The cell candidate list (sorted by name, which is also the final tie-break order) + provenance.

    Action: T51 ``candidates_3d`` (forced cells included by T51's rule — checked here); noun: T51b noun
    ``candidates_3d``. `which` = ``union`` (GT ∪ self label sets), ``gt`` or ``self``.
    """
    if which not in CELL_SETS:
        raise ValueError(f"cell set {which!r} not in {CELL_SETS}")
    root = pathlib.Path(t51_run)
    src: dict = {}
    per: dict[str, set] = {"action": set(), "noun": set()}
    for (om, lab), rel in SCORE2D.items():
        if which != "union" and lab != which:
            continue
        p = root / rel
        if not p.is_file():
            raise SystemExit(f"missing T51/T51b candidate file {p}")
        c = read_candidates(p)
        if om == "action":
            missing = [f for f in c["forced"] if f not in c["candidates_3d"]]
            if missing:
                raise SystemExit(f"{p}: forced cells {missing} are not in candidates_3d")
        bad = [x for x in c["candidates_3d"] if family(x) != om]
        if bad:
            raise SystemExit(f"{p}: {om} candidate file lists other-family cells {bad}")
        per[om].update(c["candidates_3d"])
        src[f"{om}:{lab}"] = {"path": str(p), "n": len(c["candidates_3d"]),
                              "forced": c["forced"] if om == "action" else []}
    cells = sorted(per["action"] | per["noun"])
    return {"cells": cells, "action": sorted(per["action"]), "noun": sorted(per["noun"]),
            "which": which, "sources": src}


def groups_of(cells: Sequence[str]) -> dict[str, list[str]]:
    """Cell groups (`CELL_GROUPS`), each sorted by name."""
    cs = sorted(cells)
    return {"cells": cs, "cells_action": [c for c in cs if family(c) == "action"],
            "cells_noun": [c for c in cs if family(c) == "noun"]}


def arm_names() -> list[str]:
    from AG3S.ag3s.experiments.studies import t54_arms as arms_mod

    return list(arms_mod.ARMS)


# ------------------------------------------------------------------------------------ maps


def cell_maps(cells: Sequence[str], *, action: Optional[tuple] = None,
              noun_ex: Optional[dict] = None) -> dict[str, np.ndarray]:
    """``{cell: (F, 3, 16, 16) float32}``. `action` = `t54_study.load_action` output
    ``(att, denoise, aggs, t_step)``; `noun_ex` = `t49_noun_query_extract.load_extract` output."""
    from AG3S.ag3s.experiments.studies import t54_arms as arms_mod

    out: dict[str, np.ndarray] = {}
    block = None
    for c in cells:
        q, l, h = parse_cell(c)
        if q == NOUN_QUERY:
            if noun_ex is None:
                raise SystemExit(f"cell {c} needs the noun extract")
            if block is None:
                from AG3S.ag3s.experiments.studies.t49_noun_query_extract import query_block

                ok = np.asarray(noun_ex["ok"], bool)
                if not ok.all():
                    raise SystemExit(f"noun extract has failed frames {np.flatnonzero(~ok).tolist()}")
                block = query_block(noun_ex, "target")  # (F, L, H, C, 16, 16) float64
            out[c] = np.asarray(block[:, l, h], np.float64).astype(np.float32)
        else:
            if action is None:
                raise SystemExit(f"cell {c} needs the action attention npz")
            att, dn, agl, _ = action
            spec = {"denoise": 0, "agg": q, "layer": l, "head": h, "sigma": 0.0}
            out[c] = arms_mod.action_map(att, dn, agl, spec)
    return out


def arm_specs_for_cells(cells: Sequence[str]):
    """A `t54_arms.ArmSpec` per cell (primary = split = choice = the cell): the plain serving call,
    run through `t54_replay.run_frames` exactly as the T54 arms whose roles equal their primary."""
    from AG3S.ag3s.experiments.studies.t54_arms import ArmSpec

    return [ArmSpec(c, c, role="T56 cell candidate") for c in cells]


def candidate_maps(kind: str, names: Sequence[str], *, action: Optional[tuple],
                   noun_ex: Optional[dict]) -> tuple[list, dict[str, np.ndarray]]:
    """``(ArmSpec list, maps by source)`` for a candidate batch of one kind."""
    from AG3S.ag3s.experiments.studies import t54_arms as arms_mod
    from AG3S.ag3s.experiments.studies import t54_study as st54

    if kind == "cell":
        return arm_specs_for_cells(names), cell_maps(names, action=action, noun_ex=noun_ex)
    if kind == "arm":
        arms = arms_mod.resolve_arms(list(names))
        need = arms_mod.sources_needed(arms)
        if action is None:
            raise SystemExit("arms need the action attention npz")
        att, dn, agl, _ = action
        return arms, st54.build_maps(att, dn, agl, noun_ex, need)
    raise ValueError(f"candidate kind {kind!r}")


def needs_noun(kind: str, names: Sequence[str]) -> bool:
    if kind == "cell":
        return any(family(c) == "noun" for c in names)
    from AG3S.ag3s.experiments.studies import t54_arms as arms_mod

    need = arms_mod.sources_needed(arms_mod.resolve_arms(list(names)))
    return any(arms_mod.SOURCES[s]["kind"] == "noun" for s in need)


def batches(names: Sequence[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("batch size must be >= 1")
    names = list(names)
    return [names[i:i + size] for i in range(0, len(names), size)]


__all__ = ["ARM_GROUP", "CELL_GROUPS", "CELL_SETS", "DEFAULT_CELL_SET", "NOUN_QUERY", "SCORE2D",
           "T51_RUN", "arm_names", "arm_specs_for_cells", "batches", "candidate_maps",
           "cell_candidates", "cell_maps", "family", "groups_of", "needs_noun", "parse_cell",
           "read_candidates"]
