"""GT oracle: the simulator's segmentation of the BDDL target (upper bound; privileged).

Mask = pixels whose ``segmentation_geom/<cam>`` geom id belongs to the target instance. The target instance comes
from the task's BDDL goal (``benchmark.adapters.libero_panda.task_target``: target = first argument of the first
goal predicate acting on an ``obj_of_interest``; geom ids via ``scene_instances``) -- the same resolution SL3/SL4
and ``benchmark/evaluators/libero_scene_gt.py`` (``target_json``) use. Nothing is re-derived here.

``aux`` (required): ``{"segmentation": {cam: int (H, W) geom ids, OpenCV rows}, "target_geom_ids": [int, ...]}``.

Note (conventions_json ``segmentation``): the recorded segmentation is rendered at ``env.step`` return, one physics
substep after the RGB / depth image instant. The oracle therefore carries that sub-step offset; it is reported, not
corrected.
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines.interface import TargetMaskResult, TargetQuery, Timer

SEGMENTATION_NOTE = ("segmentation_geom is rendered at env.step return (sim_state_at_return), one substep after the "
                     "RGB/depth image instant (geom npz conventions_json 'segmentation')")


class GTOracleProvider:
    name = "gt_oracle"

    def provenance(self) -> dict:
        return {"provider": self.name, "privileged": True, "source": "simulator segmentation_geom + BDDL target",
                "target_rule": "benchmark.adapters.libero_panda.task_target.task_target + scene_instances",
                "selection_rule": "n/a (GT)", "note": SEGMENTATION_NOTE}

    def reset(self) -> None:
        pass

    def predict(self, rgb: dict[str, np.ndarray], query: TargetQuery, *, aux: dict | None = None
                ) -> dict[str, TargetMaskResult]:
        if not aux or "segmentation" not in aux or "target_geom_ids" not in aux:
            raise ValueError("gt_oracle needs aux={'segmentation': {cam: geom-id image}, 'target_geom_ids': [...]} ")
        ids = np.asarray(list(aux["target_geom_ids"]), np.int64)
        out = {}
        for cam, img in rgb.items():
            with Timer() as t:
                seg = np.asarray(aux["segmentation"][cam])
                if seg.shape != np.asarray(img).shape[:2]:
                    raise ValueError(f"{cam}: segmentation {seg.shape} vs rgb {np.asarray(img).shape[:2]}")
                mask = np.isin(seg, ids)
            if mask.any():
                out[cam] = TargetMaskResult(cam, mask, True, 1.0, None, [query.target_phrase], t.ms,
                                            {"target_geom_ids": int(len(ids))})
            else:
                out[cam] = TargetMaskResult.not_found(cam, seg.shape, t.ms, reason="target_not_visible")
        return out


def libero_target(suite: str, task_id: int, geom_tables, scene_target_json: dict | None = None) -> dict:
    """``{"target", "destination", "instances_geom_ids", "objects", "fixtures", "language"}`` of a LIBERO task.

    ``scene_target_json`` (``SceneGT.target_info``) is used as is when given; otherwise the BDDL is parsed with
    ``task_target`` and instances resolved on the episode's ``_geom.npz`` tables.
    """
    if scene_target_json is not None:
        return dict(scene_target_json)
    from benchmark.adapters.libero_panda import task_target as tt

    target = tt.task_target(suite, task_id)
    target["instances_geom_ids"] = tt.scene_instances(geom_tables, target)
    return target


def phrase_from_object(name: str) -> str:
    """BDDL object name -> noun phrase (``akita_black_bowl_1`` -> ``akita black bowl``)."""
    parts = str(name).split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)


__all__ = ["GTOracleProvider", "SEGMENTATION_NOTE", "libero_target", "phrase_from_object"]
