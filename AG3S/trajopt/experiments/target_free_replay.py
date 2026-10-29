"""T46 — offline replay of a recorded chunk's **fine target-free layer** under the T46 options F1 / F2.

The verifier compares `off` / `F1` / `F2` / `F1F2` on recorded chunks (`T46.task.md` V-T46) with one harness:
this module rebuilds the layer the server would have built from what the chunk record holds, and
`ta_offline` swaps it into the reconstructed scene (variants `TF0` · `F1` · `F2` · `F1F2`). Nothing here reads
ground truth.

    from AG3S.trajopt.experiments.target_free_replay import rebuild_target_free
    rep = rebuild_target_free(z, summary, nontarget_evidence="surface", owned_ambiguous="remove")
    rep["values"]   # (nx, ny, nz) float32, on the recorded fine grid (`esdf_free_origin` / `_voxel_size`)

## Model of the server's build (`CuroboFieldBuilder.update` → `_build_tiers(only="fine", tier_name="fine_no_target")`)

| step | server | replay |
|---|---|---|
| seeds | cuRobo PBA seeds of the fine TSDF (same TSDF / origin as the main fine layer) | voxels of the recorded main fine layer `esdf_distance` with value exactly 0 |
| non-target evidence | `AG3S._nontarget_points(cloud, support_mask, geometry)` | **the same method** on `filtered_cloud` · `support_mask` · `target_points` · `target_point_indices` (config stub) |
| F1 `surface` | `nontarget_surface_evidence(evidence, target points, ball, shell_m, tie_m)` | the same function on the recorded ball (`stats["target_free"]["ball"]`: centre, radius, z_min, `centre_method`, `fit`) and the recorded tie |
| split | `_ball_owned_split(centres(cand), seed, ball, OwnedEvidence(...), vs)` | the same function; tie / max-dist / unowned from the record's `owned` |
| distance | PBA propagate from the remaining seeds | exact EDT from the remaining seed voxel centres (`scipy.ndimage.distance_transform_edt`) |
| sign | the TSDF sign at the voxel | main < 0 → −; a removed seed: the recorded free layer's sign (where the record removed it too) or `new_seed_sign` (the TSDF sign of a seed the record kept is not recorded) |
| sign forcing | `_force_sign_on_pure_voxels(values, split["sign"], vs, threshold 1.5)` | the same rule (float32 `|d|` > float32 threshold) |
| clamp | cuRobo max distance | `±max|main|` of this chunk |

D-TF (`T45TF.verify.json` `reconstruction`) validated this reconstruction against the record: the record's trace sets
equal 8/8 (1 seed differs in 1/57), full grid 1 voxel > 0.6 mm in 162 chunks. The `TF0` control (`ta_offline`) is
this replay with the record's own settings, so a variant minus `TF0` is the option's effect alone.

**Not modelled:** the target-free *coarse* tier (`esdf_free_coarse_*`, T44c) — F1 / F2 never touch it (the server
builds it from the original evidence and `ambiguous = keep`), so a caller keeps the recorded coarse free layer.
"""

from __future__ import annotations

import ast
import json
import types
from typing import Any, Optional

import numpy as np

#: `new_seed_sign` choices — the sign of a seed the record kept but a variant removes.
NEW_SEED_SIGNS = ("negative", "positive")


def _lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def target_free_stats(summary: dict) -> dict:
    """`summary_json["esdf_stats"]["target_free"]` as a dict (`{}` if absent)."""
    e = _lit(summary.get("esdf_stats")) or {}
    tf = _lit(e.get("target_free")) if isinstance(e, dict) else None
    return tf if isinstance(tf, dict) else {}


def recorded_ball(tf: dict):
    """The record's `TargetBall` (with the provenance F1 reads), or None."""
    from AG3S.ag3s.fields.curobo_builder import TargetBall

    b = tf.get("ball")
    if not isinstance(b, dict) or b.get("centre_m") is None:
        return None
    prov = {k: v for k, v in b.items() if k not in ("centre_m", "radius_mm", "z_min_m")}
    return TargetBall(centre=np.asarray(b["centre_m"], np.float64), radius=float(b["radius_mm"]) / 1000.0,
                      z_min=float(b["z_min_m"]), provenance=prov)


class _EvidenceStub:
    """Just enough of `AG3S` for `AG3S._nontarget_points` (+ `_proximity_reason`) on a recorded cloud."""

    def __init__(self, esdf_cfg):
        self.config = types.SimpleNamespace(esdf=esdf_cfg)

    @staticmethod
    def _bind():
        from AG3S.ag3s.runtime.pipeline import AG3S

        return AG3S._nontarget_points, AG3S._proximity_reason

    def nontarget(self, cloud, support_mask, geometry) -> dict:
        nontarget_points, reason = self._bind()
        self._proximity_reason = types.MethodType(reason, self)
        return nontarget_points(self, types.SimpleNamespace(points=cloud), support_mask, geometry)


def nontarget_from_record(z, *, fine_voxel: float, nontarget_evidence: str = "proximity") -> dict:
    """`AG3S._nontarget_points` on the record (the server's own method, config stub)."""
    esdf = types.SimpleNamespace(fine_voxel_size=float(fine_voxel), voxel_size=float(fine_voxel),
                                 target_exclusion="ball_owned", nontarget_evidence=str(nontarget_evidence))
    cloud = np.asarray(z["filtered_cloud"], np.float64).reshape(-1, 3)
    support = (np.asarray(z["support_mask"], bool).reshape(-1) if "support_mask" in z.files else None)
    tp = (np.asarray(z["target_points"], np.float64).reshape(-1, 3) if "target_points" in z.files
          else np.zeros((0, 3)))
    idx = (np.asarray(z["target_point_indices"], np.int64).reshape(-1)
           if "target_point_indices" in z.files else np.zeros(0, np.int64))
    geom = types.SimpleNamespace(points=tp, point_indices=idx, metrics={})
    return _EvidenceStub(esdf).nontarget(cloud, support, geom)


def _replay_layer(z, main, rec_free, removed, sign_set, vs: float, *, sign_threshold_voxels: float,
                  new_seed_sign: str):
    """`(values, new_removed, lost)` — the server's tier model (module table) from the removed seeds and the sign
    candidates: EDT from the remaining seeds, the sign (record / `new_seed_sign`), the pure-voxel sign forcing, the clamp.
    Shared by `rebuild_target_free` (T46) and `rebuild_oracle_free` (T46e)."""
    from scipy.ndimage import distance_transform_edt

    shape = main.shape
    seed_all = main == 0
    seeds = seed_all.copy()
    if len(removed):
        seeds[removed[:, 0], removed[:, 1], removed[:, 2]] = False
    d = distance_transform_edt(~seeds, sampling=vs).astype(np.float32)
    rec_removed = np.zeros(shape, bool)
    if "target_free_removed_ijk" in z.files:
        r = np.asarray(z["target_free_removed_ijk"], np.int64).reshape(-1, 3)
        if len(r):
            rec_removed[r[:, 0], r[:, 1], r[:, 2]] = True
    new_removed = np.zeros(shape, bool)
    if len(removed):
        new_removed[removed[:, 0], removed[:, 1], removed[:, 2]] = True
    lost = rec_removed & ~new_removed            # removed in the record, kept here (seed again)
    new_removed &= ~rec_removed
    seed_neg = rec_free < 0                      # the TSDF sign the record exposes at its removed seeds
    seed_neg = np.where(new_removed, new_seed_sign == "negative", seed_neg)
    neg = (main < 0) | (seed_all & seed_neg)
    values = np.where(neg, -d, d).astype(np.float32)
    if len(sign_set):
        thr = np.float32(float(sign_threshold_voxels) * vs)
        vals = values[sign_set[:, 0], sign_set[:, 1], sign_set[:, 2]]
        pure = np.abs(vals) > thr
        s = sign_set[pure]
        values[s[:, 0], s[:, 1], s[:, 2]] = np.abs(vals[pure])
    mx = float(np.abs(main[np.isfinite(main)]).max()) if np.isfinite(main).any() else 0.0
    values = np.clip(values, -mx, mx).astype(np.float32)
    return values, new_removed, lost


def rebuild_target_free(z, summary: dict, *, nontarget_evidence: str = "proximity",
                        owned_ambiguous: str = "keep", nontarget_shell_voxels: float = 2.0,
                        sign_threshold_voxels: float = 1.5, new_seed_sign: str = "negative",
                        owned: Optional[dict] = None) -> Optional[dict]:
    """The fine target-free layer of one recorded chunk under F1 / F2, or None when the chunk has no
    `ball_owned` fine target-free layer (no target, held, `ball` mode, points fallback).

    `owned` overrides the record's `{"tie_voxels", "max_dist_voxels", "unowned"}` (default: as recorded).
    Returns `{"values" (float32 grid), "origin", "voxel_size", "counts", "evidence", "removed_ijk",
    "ambiguous_ijk", "nontarget_kept_ijk", "n_removed_not_in_record", "n_record_removed_not_removed",
    "settings"}`.
    """
    from AG3S.ag3s.fields.curobo_builder import (
        CuroboFieldBuilder,
        OwnedEvidence,
        _ball_owned_split,
        nontarget_surface_evidence,
    )

    if new_seed_sign not in NEW_SEED_SIGNS:
        raise ValueError(f"new_seed_sign ∈ {NEW_SEED_SIGNS}: {new_seed_sign!r}")
    if nontarget_evidence not in ("proximity", "surface"):
        raise ValueError(f"nontarget_evidence ∈ ('proximity', 'surface'): {nontarget_evidence!r}")
    if owned_ambiguous not in ("keep", "remove"):
        raise ValueError(f"owned_ambiguous ∈ ('keep', 'remove'): {owned_ambiguous!r}")
    tf = target_free_stats(summary)
    if tf.get("exclusion_method") != "ball_owned" or "esdf_free_distance" not in z.files:
        return None
    ball = recorded_ball(tf)
    if ball is None:
        return None
    main = np.asarray(z["esdf_distance"], np.float32)
    rec_free = np.asarray(z["esdf_free_distance"], np.float32)
    origin = np.asarray(z["esdf_origin"], np.float64).reshape(3)
    vs = float(z["esdf_voxel_size"])
    if (main.shape != rec_free.shape
            or not np.allclose(np.asarray(z["esdf_free_origin"], np.float64).reshape(3), origin)
            or abs(float(z["esdf_free_voxel_size"]) - vs) > 1e-12):
        raise ValueError("the recorded free layer is not on the main fine grid")
    shape = main.shape
    rec_owned = dict(tf.get("owned") or {})
    ow = {"tie_voxels": float(rec_owned.get("tie_voxels", 1.0)),
          "max_dist_voxels": float(rec_owned.get("max_dist_voxels", 2.0)),
          "unowned": str(rec_owned.get("unowned", "remove")), **(owned or {})}
    tp = np.asarray(z["target_points"], np.float64).reshape(-1, 3)

    # non-target evidence: the server's method, then F1
    nt = nontarget_from_record(z, fine_voxel=vs, nontarget_evidence=nontarget_evidence)
    evidence = {k: v for k, v in nt.items() if k != "points"} | {"n_points": int(len(nt["points"]))}
    nt_points = np.asarray(nt["points"], np.float64).reshape(-1, 3)
    if nontarget_evidence == "surface":
        drop, f1 = nontarget_surface_evidence(nt_points, tp, ball,
                                              shell_m=float(nontarget_shell_voxels) * vs,
                                              tie_m=ow["tie_voxels"] * vs)
        nt_points = nt_points[~drop]
        evidence["evidence"] = f1
        evidence["n_points_owned"] = int(len(nt_points))

    # split on the recorded seeds (value 0 of the main fine layer)
    cand = CuroboFieldBuilder._ball_candidates(ball, origin, vs, shape)
    if cand is None:
        return None
    seed_all = main == 0
    seed = seed_all[cand[:, 0], cand[:, 1], cand[:, 2]]
    centres = origin[None, :] + cand.astype(np.float64) * vs
    owned_ev = OwnedEvidence(target_points=tp, nontarget_points=nt_points,
                             tie_voxels=ow["tie_voxels"], max_dist_voxels=ow["max_dist_voxels"],
                             unowned=ow["unowned"], ambiguous=owned_ambiguous)
    split = _ball_owned_split(centres, seed, ball, owned_ev, vs)
    removed = cand[split["removed"]]
    sign_set = cand[split["sign"]]

    # distance from the remaining seeds, the sign, the sign forcing, the clamp
    values, new_removed, lost = _replay_layer(z, main, rec_free, removed, sign_set, vs,
                                              sign_threshold_voxels=sign_threshold_voxels,
                                              new_seed_sign=new_seed_sign)

    keep_unowned = ow["unowned"] == "keep"
    counts = {
        "exclusion_method": "ball_owned",
        "n_seeds_in_ball": int(seed.sum()),
        "n_removed": int(len(removed)),
        "n_target_owned": int(split["target"].sum()),
        "n_kept_nontarget": int(split["nontarget"].sum()),
        "n_ambiguous": int(split["ambiguous"].sum()),
        "n_unowned_removed": 0 if keep_unowned else int(split["unowned"].sum()),
        "n_unowned_kept": int(split["unowned"].sum()) if keep_unowned else 0,
        "n_sign_candidates": int(len(sign_set)),
        **({"n_ambiguous_removed": int(split["ambiguous"].sum())} if owned_ambiguous == "remove" else {}),
    }
    return {
        "values": values, "origin": origin.copy(), "voxel_size": vs, "counts": counts,
        "evidence": evidence,
        "removed_ijk": removed, "ambiguous_ijk": cand[split["ambiguous"]],
        "nontarget_kept_ijk": cand[split["nontarget"]],
        "n_removed_not_in_record": int(new_removed.sum()),
        "n_record_removed_not_removed": int(lost.sum()),
        "settings": {"nontarget_evidence": nontarget_evidence, "owned_ambiguous": owned_ambiguous,
                     "nontarget_shell_voxels": float(nontarget_shell_voxels),
                     "sign_threshold_voxels": float(sign_threshold_voxels),
                     "new_seed_sign": new_seed_sign, "owned": ow},
    }


def summary_of(z) -> dict:
    return json.loads(str(z["summary_json"]))


def recorded_split_equal(z, rep: dict) -> dict[str, Any]:
    """Diagnostic: the replay's trace sets vs the record's (`target_free_*_ijk`), per set — True when equal."""
    out = {}
    for key, rk in (("removed", "removed_ijk"), ("ambiguous", "ambiguous_ijk"),
                    ("nontarget_kept", "nontarget_kept_ijk")):
        f = f"target_free_{key}_ijk"
        if f not in z.files:
            out[key] = None
            continue
        a = {tuple(v) for v in np.asarray(z[f], np.int64).reshape(-1, 3).tolist()}
        b = {tuple(v) for v in np.asarray(rep[rk], np.int64).reshape(-1, 3).tolist()}
        out[key] = a == b
    return out


# ----------------------------------------------------------------------------------------- T46e A13 oracle


def rebuild_oracle_free(z, summary: dict, oracle, *, sign_threshold_voxels: float = 1.5,
                        new_seed_sign: str = "negative") -> Optional[dict]:
    """T46e — the fine target-free layer of one recorded chunk under `esdf.target_extent = "oracle_visible"`.

    `oracle` = the `OracleVisibleEvidence` the server would have built this request (`oracle_evidence_for_chunk`).
    Same seed / distance / sign model as `rebuild_target_free` (module table); the split is the server's own
    `oracle_visible_split` on `oracle_candidates`. None when the chunk has no fine target-free layer (no target, held).
    Returns `{"values", "origin", "voxel_size", "counts", "removed_ijk", "nontarget_kept_ijk", "ambiguous_ijk" (empty),
    "n_removed_not_in_record", "n_record_removed_not_removed", "settings"}`.
    """
    from AG3S.ag3s.fields.curobo_builder import oracle_candidates, oracle_visible_split

    if new_seed_sign not in NEW_SEED_SIGNS:
        raise ValueError(f"new_seed_sign ∈ {NEW_SEED_SIGNS}: {new_seed_sign!r}")
    tf = target_free_stats(summary)
    if "esdf_free_distance" not in z.files or not int(tf.get("n_layers", 0) or 0):
        return None
    main = np.asarray(z["esdf_distance"], np.float32)
    rec_free = np.asarray(z["esdf_free_distance"], np.float32)
    origin = np.asarray(z["esdf_origin"], np.float64).reshape(3)
    vs = float(z["esdf_voxel_size"])
    shape = main.shape
    max_d = float(oracle.max_dist_voxels) * vs
    cand = oracle_candidates(oracle.target_points, origin, vs, shape, max_d)
    empty = np.zeros((0, 3), np.int64)
    if cand is None:
        removed, sign_set, kept = empty, empty, empty
        seed = np.zeros(0, bool)
        n_other_near = n_surface = n_band = 0
    else:
        seed = (main == 0)[cand[:, 0], cand[:, 1], cand[:, 2]]
        centres = origin[None, :] + cand.astype(np.float64) * vs
        split = oracle_visible_split(centres, seed, oracle, vs)
        removed, sign_set, kept = cand[split["removed"]], cand[split["sign"]], cand[split["kept_other"]]
        n_other_near = split["n_other_near"]
        n_surface, n_band = int(split["surface"].sum()), int(split["band"].sum())
    values, new_removed, lost = _replay_layer(z, main, rec_free, removed, sign_set, vs,
                                              sign_threshold_voxels=sign_threshold_voxels,
                                              new_seed_sign=new_seed_sign)
    return {
        "values": values, "origin": origin.copy(), "voxel_size": vs,
        "counts": {"exclusion_method": "oracle_visible", "n_seeds_in_ball": int(seed.sum()),
                   "n_removed": int(len(removed)), "n_kept_nontarget": int(len(kept)),
                   "n_removed_surface": n_surface, "n_removed_band": n_band,
                   "n_sign_candidates": int(len(sign_set)), "n_candidates": 0 if cand is None else int(len(cand)),
                   "n_target_points": int(len(oracle.target_points)), "n_other_points": int(len(oracle.other_points)),
                   "n_other_near": int(n_other_near)},
        "removed_ijk": removed, "nontarget_kept_ijk": kept, "ambiguous_ijk": empty, "sign_ijk": sign_set,
        "n_removed_not_in_record": int(new_removed.sum()), "n_record_removed_not_removed": int(lost.sum()),
        "settings": {"max_dist_voxels": float(oracle.max_dist_voxels),
                     "sign_threshold_voxels": float(sign_threshold_voxels), "new_seed_sign": new_seed_sign},
    }


class OracleRenderer:
    """T46e — re-render a recorded run's requests (planning-row `qpos`) into the server's `OracleMemory`.

    The same cameras and the same wire quantisation (uint16 mm depth) as the client; the code images from the same
    `code_table` / `label_image` the client uses. `memory_at(seq)` folds every planning row up to `seq` (the TSDF of
    request `seq` holds them all) and returns the memory — rows are folded once, in order, and cached.

        r = OracleRenderer(model_xml, target="orange", rows=planning_rows)
        mem = r.memory_at(15)
        ev, rec = oracle_evidence_for_chunk(mem, target_points, fusion_voxel=0.01, fine_voxel=0.005)
    """

    def __init__(self, model_xml: str, *, target: str, rows: list, cameras=("zed_left", "wrist_cam_l", "wrist_cam_r"),
                 fine_voxel: float = 0.005, max_dist_voxels: float = 6.0, depth_min: float = 0.05,
                 depth_max: float = 3.0, fine_truncation: float = 0.030):
        from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
        from AG3S.ag3s.runtime.oracle_segmentation import OracleMemory, code_table

        self.scene = TransportScene(model_xml, settle_steps=0)
        root = {int(i): int(r) for i, r in enumerate(np.asarray(self.scene.model.body_rootid))}
        self.lut, self.codes = code_table(self.scene._body_names, target, root_of=root)
        self.cameras = tuple(cameras)
        self.rows = sorted((r for r in rows if r.get("qpos") is not None), key=lambda r: int(r["seq"]))
        max_d = float(max_dist_voxels) * float(fine_voxel)
        # the server's construction (`AG3S._oracle_update`)
        self.memory = OracleMemory(cell=0.5 * fine_voxel, roi_margin=2.0 * max_d + fine_voxel,
                                   carve_tol=2.0 * fine_voxel, depth_min=depth_min, depth_max=depth_max,
                                   band_depth=fine_truncation)
        self._next = 0
        self.max_pose_error_m = 0.0

    def views(self, row) -> list:
        from AG3S.ag3s.runtime.oracle_segmentation import OracleView, label_image

        sc = self.scene
        q = np.asarray(row["qpos"], np.float64)
        if q.shape[0] != int(sc.model.nq):
            raise ValueError(f"planning row qpos has {q.shape[0]} entries, model nq={sc.model.nq}")
        sc.data.qpos[:] = q
        sc.data.qvel[:] = 0.0
        sc.mujoco.mj_forward(sc.model, sc.data)
        for name, p in (row.get("object_poses") or {}).items():
            try:
                got = sc.body_position(name)
            except KeyError:
                continue
            self.max_pose_error_m = max(self.max_pose_error_m,
                                        float(np.linalg.norm(got - np.asarray(p["pos"], np.float64))))
        out = []
        for cam in self.cameras:
            f = sc.capture(cam)
            mm = np.clip(np.rint(np.asarray(f.depth, np.float64) * 1000.0), 0, 65535).astype(np.uint16)
            out.append(OracleView(depth_m=mm.astype(np.float64) / 1000.0, K=np.asarray(f.camera_intrinsics),
                                  T_base_cam=np.asarray(f.T_base_cam), labels=label_image(f.body_ids, self.lut),
                                  name=cam))
        return out

    def memory_at(self, seq: int):
        while self._next < len(self.rows) and int(self.rows[self._next]["seq"]) <= int(seq):
            self.memory.update(self.views(self.rows[self._next]), legend=self.codes)
            self._next += 1
        return self.memory


def oracle_evidence_for_chunk(memory, target_points, *, fusion_voxel: float = 0.01, identity: str = "pipeline",
                              max_dist_voxels: float = 6.0, z_floor=None):
    """T46e — `(OracleVisibleEvidence | None, record)` as `AG3S._oracle_evidence` builds it (the same function,
    `oracle_segmentation.oracle_evidence`) from the memory and the chunk's manipulated geometry (`z["target_points"]`).
    `z_floor` = the recorded ball's `z_min` (`recorded_ball(target_free_stats(summary)).z_min`)."""
    from AG3S.ag3s.runtime.oracle_segmentation import oracle_evidence

    return oracle_evidence(memory, target_points, radius=float(fusion_voxel), identity=identity,
                           max_dist_voxels=float(max_dist_voxels),
                           z_floor=None if z_floor is None else float(z_floor))


__all__ = ["rebuild_target_free", "nontarget_from_record", "recorded_ball", "target_free_stats",
           "recorded_split_equal", "summary_of", "NEW_SEED_SIGNS",
           "rebuild_oracle_free", "OracleRenderer", "oracle_evidence_for_chunk"]
