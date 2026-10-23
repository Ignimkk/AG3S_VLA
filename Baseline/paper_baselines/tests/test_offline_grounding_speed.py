"""BASELINE-EVAL (G): the faster evaluation path gives bit-identical numbers; provider failures stay inside their arm.

* pruned lift-check distance == ``benchmark.evaluators.libero_scene_gt.point_triangles_distance_near`` bit for bit
  (synthetic soups with slivers / big triangles, and every valid depth pixel of recorded frames);
* one recorded SL3 episode re-scored with the new code reproduces the stored ``frames.jsonl`` lines of the
  PAPER-BASELINES-2 run byte for byte: ``gt_oracle`` (re-run; its wall-clock latency excluded) and ATS directly, the
  GPU providers by replaying the masks that run saved (``--save-masks``) with their recorded scores / latencies;
* a provider exception (at build or at a frame) becomes ``provider_error`` rows of that arm only.
"""
import json
import pathlib
import shutil
import uuid

import numpy as np
import pytest

from Baseline.paper_baselines.eval import offline_grounding as OG
from Baseline.paper_baselines.interface import TargetMaskResult, register_provider

ROOT = pathlib.Path(__file__).resolve().parents[3]
STORE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
SL3 = STORE / "20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0"
SL3_RUN = STORE / "20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack"  # PAPER-BASELINES-2, old code
ATS = STORE / "20261012T225820_46a08d6ff644_sl4-perception-dev"
AUDIT = STORE / "20261012T202459_309e4c6da477_sl3-attention-audit-dev"
SCENE = ROOT / "outputs/verify/SL4/raw/scene"
STEM = "libero_spatial_t0_e0"
HAVE_SL3 = (SL3 / f"episodes/{STEM}_geom.npz").exists() and SCENE.exists()
HAVE_RUN = HAVE_SL3 and (SL3_RUN / "frames.jsonl").exists() and ATS.exists() and AUDIT.exists()
#: numpy of the venv that wrote SL3_RUN (.venv-baselines, PAPER-BASELINES-2.verify.json code_state). Under another numpy
#: (e.g. .venv-ag3s 2.4.6) float sums such as 3d_target_voxels differ in the last bit -- for the original code too.
RUN_NUMPY = "2.2.6"


def _reference():
    from benchmark.evaluators.libero_scene_gt import point_triangles_distance_near

    return point_triangles_distance_near


def _same_bits(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# --- pruned distance ------------------------------------------------------------------------------------------
def test_pruned_distance_bit_identical_synthetic():
    ref = _reference()
    rng = np.random.default_rng(7)
    for trial in range(12):
        n_tri = int(rng.integers(50, 600))
        c = rng.uniform(-0.05, 0.05, (n_tri, 1, 3))
        size = np.where(rng.random((n_tri, 1, 1)) < 0.1, 0.02, 0.002)  # mostly small, some large triangles
        tri = c + rng.normal(0, 1, (n_tri, 3, 3)) * size
        tri[:5, 2] = tri[:5, 0] + 0.5 * (tri[:5, 1] - tri[:5, 0])  # degenerate (collinear) slivers
        tri[5:8, 1] = tri[5:8, 0]  # zero-length edge
        # thin triangles around the sliver threshold (|ab x ac| / longest^2 ~ 1e-4 .. 1e-2)
        lam = rng.uniform(0.2, 0.8, (24, 1))
        off = rng.normal(0, 1, (24, 3)) * np.linalg.norm(tri[8:32, 1] - tri[8:32, 0], axis=1, keepdims=True) \
            * 10 ** rng.uniform(-4, -2, (24, 1))
        tri[8:32, 2] = tri[8:32, 0] + lam * (tri[8:32, 1] - tri[8:32, 0]) + off
        surf = tri[rng.integers(0, n_tri, 3000)].mean(1)
        p = np.concatenate([surf + rng.normal(0, 0.003, surf.shape), rng.uniform(-0.1, 0.1, (2000, 3))])
        for md in (0.01, 0.003):
            a, b = ref(p, tri, md), OG.point_triangles_distance_near_pruned(p, tri, md)
            assert _same_bits(a, b), f"trial {trial} md {md}: {np.flatnonzero(a != b)[:5]}"
            assert np.isfinite(a).sum() > 100
    assert _same_bits(OG.point_triangles_distance_near_pruned(np.zeros((0, 3)), tri, 0.01), ref(np.zeros((0, 3)), tri, 0.01))
    far = np.full((4, 3), 5.0)
    assert np.isinf(OG.point_triangles_distance_near_pruned(far, tri, 0.01)).all()


@pytest.mark.skipif(not HAVE_SL3, reason="recorded SL3 dev episode / SL4 scene GT not present")
def test_pruned_distance_bit_identical_recorded_pixels():
    ref = _reference()
    eps = {e["stem"]: e for e in OG.load_episodes(SL3)}
    for stem, t in ((STEM, 10), ("libero_spatial_t1_e0", 20)):
        ep = OG.Episode(eps[stem], OG.CAMERAS, SCENE)
        fr = ep.frame(t)
        geo = ep.geometry(fr)
        tri = ep.target_triangles(t)
        for c in OG.CAMERAS:
            a, b = ref(geo.points(c), tri, OG.LIFT_CHECK_MM / 1000.0), geo.pixel_distance(c)
            assert _same_bits(a, b) and np.isfinite(a).sum() > 100
        # per-pixel distances indexed by a mask == reference on the lifted points of that mask
        gt = {c: fr["labels"][c] == OG.M.CLASSES.index("target") for c in OG.CAMERAS}
        pts = np.concatenate([OG.L.lift_mask(gt[c], fr["depth_m"][c], fr["K"][c], fr["T"][c]) for c in OG.CAMERAS])
        old = OG.lift_check_from_distances(ref(pts, tri, OG.LIFT_CHECK_MM / 1000.0))
        new = geo.lift_check({c: geo.selection(c, gt[c], "none", fr["robot"][c]) for c in OG.CAMERAS})
        assert json.dumps(old) == json.dumps(new)


# --- recorded episode, byte-identical rows ----------------------------------------------------------------------
class SavedMaskReplay:
    """Replays the masks a run saved (``masks/<spec>__<pm>/<stem>_<t>.npz``) with the run's recorded per-camera
    found / score / boxes / phrases / latency / extras, so a re-scored row must equal the stored one byte for byte."""

    name = "saved_mask_replay"

    def __init__(self, run: pathlib.Path, spec: str, pm: str, ref_rows: dict):
        self.dir = run / "masks" / f"{spec.replace(':', '-')}__{pm}"
        self.ref = ref_rows  # obs_index -> stored row

    def provenance(self):
        return {"provider": self.name, "dir": str(self.dir)}

    def reset(self):
        pass

    def predict(self, rgb, query, *, aux=None):
        t = int(query.frame_index)
        out = {}
        with np.load(self.dir / f"{query.episode_id}_{t:04d}.npz") as z:
            for c, img in rgb.items():
                shape = tuple(int(x) for x in z[f"{c}__shape"])
                mask = np.unpackbits(z[c], count=int(np.prod(shape))).reshape(shape).astype(bool)
                s = self.ref[t]["cameras"][c]
                boxes = None if s["boxes"] is None else np.asarray(s["boxes"], np.float64).reshape(-1, 4)
                out[c] = TargetMaskResult(c, mask, s["found"], s["score"], boxes, list(s["phrases"]), s["latency_ms"],
                                          dict(s["extras"]))
        return out


def _strip_latency(row: dict) -> dict:
    row = json.loads(json.dumps(row))
    row.pop("latency_ms_total", None)
    for c in row.get("cameras", {}).values():
        c.pop("latency_ms", None)
    return row


def rescore_episode(stem: str, run: pathlib.Path = SL3_RUN) -> dict:
    """Re-score one stored episode with the current code: ``{"n_rows", "n_identical", "mismatch": [...]}``."""
    from Baseline.paper_baselines.interface import _jsonable, get_provider

    stored = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines() if line.strip()]
    lines = {(r["arm"], r["obs_index"]): line for r, line in zip(stored, (run / "frames.jsonl").read_text().splitlines())
             if r["episode_id"] == stem}
    rows = {k: json.loads(v) for k, v in lines.items()}
    meta = next(e for e in OG.load_episodes(SL3) if e["stem"] == stem)
    ep = OG.Episode(meta, OG.CAMERAS, SCENE)
    frames = ep.eval_frames()
    frs = {t: ep.frame(t) for t in frames}
    grids = {t: OG.eval_grid(ep, frs[t], OG.CAMERAS) for t in frames}
    any_row = next(iter(rows.values()))
    base = {k: any_row[k] for k in ("episode_id", "suite", "task_id", "episode_idx", "rollout_status", "robot_removal")}
    attn = OG.AttentionRun(AUDIT, SL3)
    arms = sorted({r["arm"] for r in rows.values()} - {"ats_v11ab|noun_attention"})
    new: dict = {}
    for arm in arms:
        spec, pm = arm.split("|")
        if spec == "gt_oracle":
            prov = get_provider("gt_oracle")
        else:
            prov = SavedMaskReplay(run, spec, pm, {t: r for (a, t), r in rows.items() if a == arm})
        for r in OG.arm_rows(ep, spec, pm, prov, frames, frs, grids, base, cameras=OG.CAMERAS, phrase_map={}, attn=attn):
            new[(arm, r["obs_index"])] = r
    for r in OG.ats_rows(OG.ATSRun(ATS), ep, frames, grids, base):
        new[(r["arm"], r["obs_index"])] = r
    mismatch = []
    for k, r in new.items():
        line = json.dumps(_jsonable(r))
        if k[0].startswith("gt_oracle|"):  # re-run provider: its wall-clock latency is the only free field
            same = json.dumps(_strip_latency(_jsonable(r))) == json.dumps(_strip_latency(rows[k]))
        else:
            same = line == lines[k]
        if not same:
            mismatch.append(k)
    return {"n_rows": len(new), "n_stored": len(lines), "n_identical": len(new) - len(mismatch), "mismatch": mismatch,
            "arms": arms + ["ats_v11ab|noun_attention"]}


@pytest.mark.skipif(not HAVE_RUN, reason="PAPER-BASELINES-2 SL3 run / ATS / attention audit not present")
@pytest.mark.skipif(np.__version__ != RUN_NUMPY, reason=f"stored run was computed with numpy {RUN_NUMPY} (.venv-baselines)")
def test_recorded_episode_rows_byte_identical():
    res = rescore_episode(STEM)
    assert res["n_rows"] == res["n_stored"] == 13 * 21
    assert res["mismatch"] == [], res["mismatch"][:5]


# --- provider failures -------------------------------------------------------------------------------------------
class _Boom:
    name = "boom"

    def __init__(self, **kw):
        self.calls = 0

    def provenance(self):
        return {"provider": "boom"}

    def reset(self):
        pass

    def predict(self, rgb, query, *, aux=None):
        self.calls += 1
        if self.calls >= 2:
            raise RuntimeError("CUDA out of memory (simulated)")
        return {c: TargetMaskResult.not_found(c, img.shape, 0.0) for c, img in rgb.items()}


class _BoomBuild:
    def __init__(self, **kw):
        raise OSError("weights missing (simulated)")


@pytest.mark.skipif(not HAVE_SL3, reason="recorded SL3 dev episode / SL4 scene GT not present")
def test_provider_failure_is_isolated_to_its_arm():
    register_provider("boom")(_Boom)
    register_provider("boom_build")(_BoomBuild)
    out = ROOT / "outputs" / f"paper_baselines_smoke/_test_provider_failure_{uuid.uuid4().hex[:8]}"
    try:
        args = OG.build_parser().parse_args([
            "--source-run", str(SL3), "--scene-gt", str(SCENE), "--providers", "boom", "boom_build", "gt_oracle",
            "--prompt-modes", "phrase", "--episodes", STEM, "--limit-frames", "3", "--out-dir", str(out)])
        OG.run(args)
        rows = [json.loads(x) for x in (out / "frames.jsonl").read_text().splitlines()]
        by = {}
        for r in rows:
            by.setdefault(r["arm"], []).append(r)
        assert [r["obs_index"] for r in by["boom|phrase"]] == [10, 15, 20]
        assert "missing_reason" not in by["boom|phrase"][0]
        assert [r.get("missing_reason") for r in by["boom|phrase"][1:]] == ["provider_error"] * 2
        assert by["boom|phrase"][1]["provider_error_obs_index"] == 15
        assert "CUDA out of memory" in by["boom|phrase"][1]["provider_error"]
        assert all(r["missing_reason"] == "provider_error" and "not built" in r["provider_error"]
                   for r in by["boom_build|phrase"]) and len(by["boom_build|phrase"]) == 3
        assert all("missing_reason" not in r and r["3d_iou"] > 0.9 for r in by["gt_oracle|phrase"])  # later arm ran
        s = json.loads((out / "summary.json").read_text())
        assert s["arms"]["boom|phrase"]["n_provider_error"] == 2 and s["arms"]["gt_oracle|phrase"]["n_provider_error"] == 0
        assert {(e["arm"], e["n_frames"]) for e in s["provider_errors"]} == {("boom|phrase", 2), ("boom_build|phrase", 3)}
        prov = json.loads((out / "provenance.json").read_text())
        assert "boom_build" in prov["provider_build_errors"] and "boom" in prov["providers"]
        with pytest.raises(RuntimeError):  # --fail-fast keeps the old behaviour
            OG.run(OG.build_parser().parse_args([
                "--source-run", str(SL3), "--scene-gt", str(SCENE), "--providers", "boom", "--episodes", STEM,
                "--limit-frames", "3", "--fail-fast", "--out-dir", str(out / "ff")]))
    finally:
        shutil.rmtree(out, ignore_errors=True)
