"""offline_grounding wiring: label LUT, ATS slicing, arm parsing (synthetic); one real recorded frame if present."""
import json
import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.eval import metrics as M
from Baseline.paper_baselines.eval import offline_grounding as OG

SL3 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0")
SCENE = pathlib.Path(__file__).resolve().parents[3] / "outputs/verify/SL4/raw/scene"


def test_label_lut_roles():
    bodyid = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    names = ["world", "table", "robot0_link0", "mount0_base", "bowl_1_main", "plate_1_main", "cup_1_main", "cab_1_main"]
    target = {"target": "bowl_1", "destination": "plate_1", "fixtures": ["main_table", "cab_1"],
              "instances_geom_ids": {"bowl_1": [4], "plate_1": [5], "cup_1": [6], "main_table": [1], "cab_1": [7]}}
    lut = OG.label_lut(target, bodyid, names)
    C = M.CLASSES
    assert [C[i] for i in lut[:8]] == ["background", "support", "robot", "mount", "target", "destination", "other_object", "fixture"]
    seg = np.array([[-1, 4], [2, 7]])
    assert [C[i] for i in OG.seg_labels(seg, lut).ravel()] == ["background", "target", "robot", "fixture"]


def test_parse_arm():
    assert OG.parse_arm("grounded_sam2:track") == ("grounded_sam2", {"mode": "track"})
    assert OG.parse_arm("yoloe") == ("yoloe", {})


def test_ats_points_slice(tmp_path):
    run = tmp_path / "run"
    (run / "perception").mkdir(parents=True)
    np.savez(run / "perception" / "s_perception.npz", obs_index=np.array([10, 15]), has_target=np.array([True, False]),
             target_points=np.arange(15, dtype=np.float32).reshape(5, 3), target_points_offsets=np.array([0, 3, 5]),
             t_process_s=np.array([0.1, 0.2]))
    a = OG.ATSRun(run)
    p = a.points("s", 10)
    assert p["found"] and p["points"].shape == (3, 3) and p["latency_ms"] == pytest.approx(100.0)
    assert a.points("s", 15)["points"].shape == (2, 3) and not a.points("s", 15)["found"]
    assert a.points("s", 99) is None and a.points("missing", 10) is None


@pytest.mark.skipif(not (SL3 / "episodes/libero_spatial_t0_e0_geom.npz").exists() or not SCENE.exists(),
                    reason="recorded SL3 dev episode / SL4 scene GT not present")
def test_real_recorded_frame_wiring():
    ep_meta = next(e for e in OG.load_episodes(SL3) if e["stem"] == "libero_spatial_t0_e0")
    ep = OG.Episode(ep_meta, OG.CAMERAS, SCENE)
    assert ep.target["target"] == "akita_black_bowl_1" and ep.eval_frames()[:2] == [10, 15]
    fr = ep.frame(10)
    for c in OG.CAMERAS:
        assert fr["rgb"][c].shape == fr["depth_m"][c].shape + (3,) == fr["seg"][c].shape + (3,)
    q = OG.query_for(ep, "phrase", {})
    assert q.target_phrase == "akita black bowl" and "black bowl" in q.instruction
    from Baseline.paper_baselines.interface import get_provider

    res = get_provider("gt_oracle").predict(fr["rgb"], q, aux={"segmentation": fr["seg"], "target_geom_ids": ep.target_geoms.tolist()})
    grid = M.build_eval_grid([{"depth_m": fr["depth_m"][c], "K": fr["K"][c], "T_base_cam": fr["T"][c], "labels": fr["labels"][c]}
                              for c in OG.CAMERAS])
    row = OG.evaluate_masks(res, fr, grid, ep, "none", OG.CAMERAS)
    assert row["cameras"]["head"]["2d_iou"] == 1.0  # oracle mask == GT label mask by construction
    assert row["3d_recall"] == pytest.approx(1.0)
    assert 0 < row["lift_within_10mm"] <= 1 and row["found_any"]
    json.dumps(row, default=float)


# --- PAPER-BASELINES-2 (Implementer E): --attention-run and VLA-internal provider modes ---------------------------
SL3_AUDIT = SL3.parent / "20261012T202459_309e4c6da477_sl3-attention-audit-dev"


def test_parse_arm_vla_internal_modes():
    assert OG.parse_arm("localization_heads:threshold") == ("localization_heads", {"mask_mode": "threshold"})
    assert OG.parse_arm("localization_heads:sam_point") == ("localization_heads", {"mask_mode": "sam_point"})
    assert OG.parse_arm("tokencut:foreground") == ("tokencut", {"target_select": "foreground"})
    assert OG.parse_arm("tokencut:attention_overlap") == ("tokencut", {"target_select": "attention_overlap"})
    assert OG.needs_attention("localization_heads") and OG.needs_attention("localization_heads:sam_point")
    assert OG.needs_attention("tokencut:attention_overlap")
    assert not OG.needs_attention("tokencut:foreground") and not OG.needs_attention("tokencut")
    assert not OG.needs_attention("grounded_sam2:track")


def test_provider_kwargs_per_provider():
    name, kw = OG.provider_kwargs("localization_heads:threshold", "default", None)
    assert name == "localization_heads" and kw == {"mask_mode": "threshold", "selection": str(OG.DEFAULT_LH_SELECTION)}
    assert OG.DEFAULT_LH_SELECTION.exists()
    _, kw = OG.provider_kwargs("localization_heads:sam_point", "default", "cuda", "x.json")
    assert kw == {"mask_mode": "sam_point", "selection": "x.json", "_sam_weight": "sam-vit-huge", "_sam_device": "cuda"}
    _, kw = OG.provider_kwargs("localization_heads:sam_point", "smoke", None)
    assert kw["_sam_weight"] == "sam-vit-base" and "size" not in kw  # same SAM as grounded_sam at that size
    assert OG.provider_kwargs("tokencut:attention_overlap", "default", None) == ("tokencut", {"target_select": "attention_overlap"})
    assert OG.provider_kwargs("tokencut:foreground", "smoke", "cuda")[1] == {"target_select": "foreground", "device": "cuda"}
    assert OG.provider_kwargs("grounded_sam2:track", "smoke", None)[1] == {"mode": "track", "size": "smoke"}
    assert OG.provider_kwargs("gt_oracle", "default", "cuda")[1] == {}


def _fake_audit(tmp_path, source_name="src", npz_sha="abc"):
    run = tmp_path / "audit"
    (run / "attention").mkdir(parents=True)
    rng = np.random.default_rng(0)
    pooled = rng.random((2, 18, 8, 768)).astype(np.float16)
    pol = rng.random((2, 2, 16, 16)).astype(np.float32)
    np.savez(run / "attention" / "s_t0_e0_attn.npz", obs_index=np.array([10, 15]), pooled_f16=pooled,
             cell_grid_policy=pol, cell_grid_opencv=pol[..., ::-1], cameras=np.array(["head", "left_wrist"]))
    row = {"source_run": source_name, "suite": "s", "task_id": 0, "episode_idx": 0, "npz_sha256": npz_sha,
           "attn_npz": "attention/s_t0_e0_attn.npz"}
    (run / "attention_index.jsonl").write_text(json.dumps(row) + "\n")
    return run, pooled, pol


def test_attention_run_aux_and_join(tmp_path):
    run, pooled, pol = _fake_audit(tmp_path)
    a = OG.AttentionRun(run, tmp_path / "src")
    a.check_episode("s_t0_e0", "abc")
    with pytest.raises(ValueError, match="sha256"):
        a.check_episode("s_t0_e0", "other")
    with pytest.raises(ValueError, match="not --source-run"):
        OG.AttentionRun(run, tmp_path / "another_source").check_episode("s_t0_e0", "abc")
    aux = a.aux("s_t0_e0", 15)
    assert aux["attention_call_index"] == 1 and aux["attention_orientation"] == "libero_policy"
    assert aux["attention_pooled"].dtype == np.float32 and aux["attention_pooled"].shape == (18, 8, 768)
    np.testing.assert_array_equal(aux["attention_pooled"], pooled[1].astype(np.float32))
    np.testing.assert_array_equal(aux["attention_map"]["left_wrist"], pol[1, 1])
    assert a.aux("s_t0_e0", 12) is None and a.aux("missing", 10) is None
    d = a.describe("s_t0_e0", 10)
    assert d["attention_call"] == 0 and d["head_cell_opencv_is_unmirrored"] and d["left_wrist_cell_grid_shape"] == [16, 16]
    assert a.describe("s_t0_e0", 11) == {"attention_call": None}
    assert set(a.provenance()["files"]) == {"s_t0_e0", "missing"}


def test_aux_feeds_both_vla_internal_providers(tmp_path):
    """Synthetic attention through the real providers: localization_heads threshold and tokencut attention_overlap."""
    from Baseline.paper_baselines.interface import TargetQuery, get_provider, validate_results

    run, _, _ = _fake_audit(tmp_path)
    aux = OG.AttentionRun(run, tmp_path / "src").aux("s_t0_e0", 10)
    rgb = {c: np.zeros((64, 64, 3), np.uint8) for c in OG.CAMERAS}
    q = TargetQuery("pick the bowl", "bowl", "phrase", "s_t0_e0", 10)
    _, kw = OG.provider_kwargs("localization_heads:threshold", "default", None)
    res = get_provider("localization_heads", **kw).predict(rgb, q, aux=aux)
    validate_results(res, rgb)
    assert all(r.found and r.extras["attention_orientation"] == "libero_policy" for r in res.values())

    class _Feats:  # stand-in DINO extractor: two clusters of 4x4 patch features
        def __call__(self, image):
            f = np.zeros((4, 4, 8), np.float32)
            f[:, :2, 0], f[:, 2:, 1] = 1.0, 1.0
            return f.reshape(16, 8) + 0.01 * np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 128, (4, 4)

    tc = get_provider("tokencut", target_select="attention_overlap", extractor=_Feats())
    res = tc.predict({c: np.zeros((64, 64, 3), np.uint8) for c in OG.CAMERAS}, q, aux=aux)
    assert all(r.found and len(r.extras["candidates"]) == 2 and r.score == r.extras["overlap_frac"] for r in res.values())


@pytest.mark.skipif(not (SL3_AUDIT / "attention_index.jsonl").exists() or not SCENE.exists()
                    or not (SL3 / "episodes/libero_spatial_t0_e0_geom.npz").exists(),
                    reason="recorded SL3 dev episode / SL3 audit / SL4 scene GT not present")
def test_real_attention_frame_wiring():
    ep_meta = next(e for e in OG.load_episodes(SL3) if e["stem"] == "libero_spatial_t0_e0")
    ep = OG.Episode(ep_meta, OG.CAMERAS, SCENE)
    a = OG.AttentionRun(SL3_AUDIT, SL3)
    from Baseline.paper_baselines.interface import sha256_file, get_provider

    a.check_episode(ep.stem, sha256_file(ep_meta["npz_path"]))
    t = ep.eval_frames()[0]
    d = a.describe(ep.stem, t)
    assert d["attention_call"] is not None and d["attention_pooled_shape"] == [18, 8, 768]
    assert d["head_cell_opencv_is_unmirrored"] and d["left_wrist_cell_opencv_is_unmirrored"]
    # cell grid = pooled[8, 2] block (pooled is float16, values < 1 -> rounding <= 2^-11)
    assert d["head_cell_vs_pooled_L8H2_maxabs"] <= 1e-3 and d["left_wrist_cell_vs_pooled_L8H2_maxabs"] <= 1e-3
    fr = ep.frame(t)
    _, kw = OG.provider_kwargs("localization_heads:threshold", "default", None)
    res = get_provider("localization_heads", **kw).predict(fr["rgb"], OG.query_for(ep, "phrase", {}), aux=a.aux(ep.stem, t))
    grid = M.build_eval_grid([{"depth_m": fr["depth_m"][c], "K": fr["K"][c], "T_base_cam": fr["T"][c], "labels": fr["labels"][c]}
                              for c in OG.CAMERAS])
    row = OG.evaluate_masks(res, fr, grid, ep, "none", OG.CAMERAS)
    assert row["found_any"] and 0.0 <= row["3d_iou"] <= 1.0
