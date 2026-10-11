"""T43 T — re-ground recorded chunks with a fresh TargetConfirm (the T30b replay method, whole run).
Inputs per chunk: attention cloud/values/raw, support mask, freeze (exclusion.grasp_active or attached),
subtask p, latch phase (summary grasp.state), hand spheres (recorded constraint spheres on hand links)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, dataclasses
import numpy as np

S = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
IX = {**json.load(open(S / "chunk_index_t42.json")), **json.load(open(S / "chunk_index_t40t41.json"))}
HAND = ("ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")


def chunks(run_dir):
    ix = IX[str(run_dir)]
    return {int(k): v["npz"] for k, v in ix["chunks"].items()}


def replay(run_dir, *, overrides=None, seqs=None, gate_record=True):
    """Yields (seq, recorded summary, TargetConfirm after the call, confirm.last, gate refusal)."""
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.stages.admissibility import HandSpheres, destination_overlap
    from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target
    from AG3S.ag3s.types import AttentionPointCloud, PointCloud

    cfg = AG3SConfig()
    if overrides:
        cfg = cfg.with_overrides(overrides)
    confirm = TargetConfirm(cfg.clustering, max_opening=0.07146,
                            overlap_distance=float(cfg.pointcloud.voxel_size) or cfg.clustering.eps)
    files = chunks(run_dir)
    out = []
    for seq in sorted(files):
        if seqs is not None and seq > max(seqs):
            break
        with np.load(files[seq], allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
            excl = s.get("exclusion") or {}
            g = s.get("grasp") or {}
            confirm.freeze(bool(excl.get("grasp_active")) or bool(g.get("attached")))
            if hasattr(confirm, "note_latched"):
                confirm.note_latched(g.get("state") in ("latched", "closing"))
            confirm.note_subtask((s.get("subtask") or {}).get("p"))
            confirm.note_placed(g.get("state") == "placed")
            names = s.get("sphere_link_names") or []
            keep = np.array([n in HAND for n in names], bool)
            if keep.any():
                confirm.note_hand(HandSpheres(np.asarray(z["sphere_centres"], float)[keep],
                                              np.asarray(z["sphere_radii"], float)[keep],
                                              tuple(n for n, k in zip(names, keep) if k)))
            if getattr(confirm, "supported_max_bottom", None) is not None:
                from AG3S.ag3s.stages.support_surface import fit_support_surfaces
                surf, _ = fit_support_surfaces(PointCloud(points=np.asarray(z["filtered_cloud"], np.float64)),
                                               cfg.support_surface, timestamp=0.0)
                confirm.note_support(np.asarray([[*np.asarray(f.normal, float), float(f.offset)] for f in surf],
                                                float).reshape(-1, 4) if surf else None)
            apc = AttentionPointCloud(PointCloud(z["attention_cloud"].astype(np.float64)),
                                      z["attention_values"], raw_attention=z["attention_raw"])
            ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                          seed_threshold=cfg.attention.seed_threshold, exclude_mask=z["support_mask"],
                          min_radius=cfg.geometry.min_radius, timestamp=float(z["t_step"]),
                          confirm=confirm)
        man = confirm.manipulated
        refused = None
        if man is not None and man.usable and confirm.destination is not None:
            ov = destination_overlap(man.points, confirm.destination, confirm.overlap_distance)
            refused = bool(ov >= float(cfg.clustering.destination_overlap_fraction))
        if hasattr(confirm, "note_exclusion_refused"):
            confirm.note_exclusion_refused(bool(refused))
        if seqs is None or seq in seqs:
            out.append((seq, s, man, confirm.last, refused))
    return out
