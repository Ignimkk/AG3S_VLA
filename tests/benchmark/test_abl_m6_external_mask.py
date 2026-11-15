"""ABL-M6 (main ablation row 6 "AG3S with external segmenter", plan §9.45) -- ``--external-target-mask``. CPU only.

What this file pins:

(1) the prompt = the baselines' noun rule (``wire_obs.target_phrase_from_instruction``) on every in-scope LIBERO /
    SafeLIBERO instruction of the task registry (``benchmark/configs/suites.json``, language = LIBERO's
    ``grab_language_from_filename``);
(2) the code image (1 mask, 2 other depth, 0 no depth; robot -> 0 inside AG3S from the TSDF robot mask) and the server
    path: the REAL sidecar process (``closed_loop/sidecar.py``, ops ``segmenter_init`` / ``target_mask`` / ``reset``)
    with a stub provider of known mask, ``ExternalTargetMask`` in the REAL ``PandaSafeServer`` built by
    ``build_safe_server`` for the -m6 version, a recorded Tab. 1 frame; the request ``SafePolicy`` gets carries the code
    images, the real AG3S (``wire.unpack_oracle_segmentation`` -> ``set_oracle_segmentation`` -> ``process_multi_debug``)
    folds them into the episode's ``OracleMemory`` and ``oracle_evidence`` removes the mask's samples; the call record
    ``panda/oracle.external_mask`` appears;
(4) the version: X-fwd minus ``--nontarget-evidence surface`` plus the three oracle / mask flags, parse +
    ``reject_bad_flag_combinations`` + ``check_unported_flags`` + config validate, launcher argv (no client
    segmentation flag), refusals;
(5) off: no new record key on any earlier version (``TAB3_CODE_DESTS``), metadata / request untouched.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: F401 (fixture)

ROOT = pathlib.Path(__file__).resolve().parents[2]
STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", ROOT.parent / "AG3S_VLA-storage")) / "results/benchmark"
TAB1_SPATIAL_LI = ("20261114T184202_934788904264_tab1sdm-ag3s-sl-spatial-LI-LI-vla_v11ab-AG3S-commonto-att-termsoff-pass-zr-"
                   "ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs-gh156a-"
                   "fwd-tro-C")
STEM = "safelibero_spatial_t0_e13"
STUB = "tests.benchmark.abl_m6_stub_segmenter:StubSegmenter"


def grab_language_from_filename(x: str) -> str:
    """LIBERO ``libero/libero/benchmark/__init__.py:44`` (the ``task.language`` the clients send as ``prompt``)."""
    if x[0].isupper():
        language = " ".join(x[x.find("SCENE") + (8 if "SCENE10" in x else 7):].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language if en < 0 else language[:en]


# ------------------------------------------------------------------------------------------------ (1) the prompt rule
#: expected noun phrase per SafeLIBERO task (Tab. 1 / 3 suites, both levels share the names)
SAFELIBERO_PROMPTS = {
    "pick up the black bowl between the plate and the ramekin and place it on the plate": "black bowl",
    "pick up the black bowl on the ramekin and place it on the plate": "black bowl",
    "pick up the black bowl on the stove and place it on the plate": "black bowl",
    "pick up the black bowl on the wooden cabinet and place it on the plate": "black bowl",
    "pick up the orange juice and place it in the basket": "orange juice",
    "pick up the chocolate pudding and place it in the basket": "chocolate pudding",
    "pick up the milk and place it in the basket": "milk",
    "pick up the bbq sauce and place it in the basket": "bbq sauce",
    "put the bowl on the plate": "bowl",
    "put the bowl on top of the cabinet": "bowl",
    "put the bowl on the stove": "bowl",
    "open the top drawer and put the bowl inside": "drawer",
    "put the cream cheese in the bowl": "cream cheese",
    "put both the alphabet soup and the cream cheese box in the basket": "alphabet soup",
    "put both the alphabet soup and the tomato sauce in the basket": "alphabet soup",
    "put the white mug on the left plate and put the yellow and white mug on the right plate": "white mug",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": "white mug",
}


def _registry_instructions(in_scope=True):
    reg = json.loads((ROOT / "benchmark/configs/suites.json").read_text())
    out = []
    for name, s in reg["suites"].items():
        if in_scope and not s["in_scope"]:
            continue
        for _lvl, tasks in s["tasks"].items():
            out += [(name, grab_language_from_filename(t)) for t in tasks]
    return out


def test_prompt_rule_on_every_registry_instruction():
    from AG3S.adapters.panda import external_mask as em
    from Baseline.paper_baselines.closed_loop.wire_obs import target_phrase_from_instruction

    rows = _registry_instructions()
    assert len(rows) == 72 and {n for n, _ in rows} == {"libero_spatial", "libero_object", "libero_goal", "libero_10",
                                                        "safelibero_spatial", "safelibero_object", "safelibero_goal",
                                                        "safelibero_long"}
    for suite, instr in rows:
        p = em.prompt_from_instruction(instr)
        assert p and p == target_phrase_from_instruction(instr), (suite, instr)   # the baselines' rule, verbatim
        assert p in instr.lower(), (suite, instr, p)                                # a span of the instruction
    for suite, instr in rows:
        if suite.startswith("safelibero"):
            assert em.prompt_from_instruction(instr) == SAFELIBERO_PROMPTS[instr], instr
    # case and the trailing period as the client may send them; no rule -> None (no sidecar call)
    assert em.prompt_from_instruction("Pick up the black bowl on the stove and place it on the plate.") == "black bowl"
    assert em.prompt_from_instruction("") is None and em.prompt_from_instruction(None) is None


# ------------------------------------------------------------------------------------------------ (2) code image
def test_code_image_convention():
    from AG3S.adapters.panda import external_mask as em
    from AG3S.ag3s.runtime import oracle_segmentation as osg

    depth = np.array([[0, 500, 600], [700, 0, 800]], np.uint16)
    mask = np.array([[True, True, False], [False, True, False]])
    lab = em.code_image(mask, depth)
    assert lab.dtype == np.uint8
    assert lab.tolist() == [[osg.CODE_NONE, osg.CODE_TARGET, osg.CODE_STATIC], [osg.CODE_STATIC, osg.CODE_NONE, osg.CODE_STATIC]]
    assert em.code_image(None, depth).tolist() == [[0, 2, 2], [2, 0, 2]]           # not found: no code 1
    with pytest.raises(ValueError):
        em.code_image(np.zeros((3, 3), bool), depth)


def test_pipeline_zeroes_robot_pixels_only_for_server_code_images():
    from AG3S.ag3s.runtime.pipeline import AG3S

    class Cam:
        def __init__(self, name, mask):
            self.name, self.robot_mask = name, mask

    lab = np.full((2, 3), 2, np.uint8)
    lab[0, 0] = 1
    robot = np.zeros((2, 3), bool)
    robot[0, :2] = True
    out, n = AG3S._oracle_server_robot_codes({"head": lab, "left_wrist": lab}, [Cam("head", robot), Cam("left_wrist", None)])
    assert out["head"].tolist() == [[0, 0, 2], [2, 2, 2]] and n == {"head": 2, "left_wrist": None}
    assert out["left_wrist"].tolist() == lab.tolist() and lab[0, 0] == 1             # input not changed


# ------------------------------------------------------------------------------------------------ sidecar (real process, stub)
def _sidecar(tmp_path, boxes=None):
    from AG3S.adapters.panda import external_mask as em
    from Baseline.paper_baselines.closed_loop.sidecar import SidecarClient

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    client = SidecarClient(sys.executable, log_path=tmp_path / "sidecar.log", env=env)
    ext = em.ExternalTargetMask("grounded-sam2", None, client=client, sidecar_provider=STUB)
    if boxes is not None:   # re-init with other boxes (the stub's kwargs)
        ext.provenance = client.call("segmenter_init", provider=STUB, kwargs={"boxes": boxes})["provenance"]
    return ext


def test_sidecar_ops_round_trip(tmp_path):
    ext = _sidecar(tmp_path)
    try:
        assert ext.mode == "per_frame" and ext.provider_kwargs == {"mode": "per_frame", "size": "default"}
        assert ext.provenance["provider"] == "stub" and ext.versions["numpy"]
        rgb = {"head": np.zeros((256, 256, 3), np.uint8), "left_wrist": np.zeros((256, 256, 3), np.uint8)}
        r = ext.client.call("target_mask", rgb=rgb, prompt="black bowl")
        assert set(r["masks"]) == {"head", "left_wrist"} and r["prompt"] == "black bowl"
        assert r["masks"]["head"].dtype == bool and r["masks"]["head"].shape == (256, 256)
        assert int(r["masks"]["head"].sum()) == 64 * 70 and not r["masks"]["left_wrist"].any()
        assert r["results"]["head"]["found"] is True and r["results"]["head"]["mask_pixels"] == 64 * 70
        assert r["results"]["left_wrist"]["found"] is False
        ext.reset()
        r = ext.client.call("target_mask", rgb=rgb, prompt="milk")
        assert r["results"]["head"]["extras"]["n_reset"] == 1
    finally:
        assert ext.close() == 0


# ------------------------------------------------------------------------------------------------ (4) version
def _m6():
    from AG3S.adapters.panda import profile as pf

    return pf.SOTA_ABL_M6_VERSIONS[0]


def _parse(v):
    from AG3S.adapters.panda import profile as pf

    return pf.parse_v11ab_args(argv=pf.version_server_argv(v) + list(pf.AG3S_VERSIONS[v]))


def _server_args(v, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(v)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + list(pf.version_server_flags(v)))


def test_version_is_x_fwd_with_only_the_target_surface_source_replaced(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv

    xf, v = pf.SOTA_FINAL_X_FWD, _m6()
    assert v == xf[:-len("-tro")] + "-m6-tro" and list(pf.AG3S_VERSIONS)[-1] == v and v in pf.GT15_VERSIONS
    base = list(pf.AG3S_VERSIONS[xf])
    i = base.index("--nontarget-evidence")
    assert base[i + 1] == "surface" and base.count("--nontarget-evidence") == 1
    assert list(pf.AG3S_VERSIONS[v]) == base[:i] + base[i + 2:] + [
        "--target-extent", "oracle-visible", "--oracle-extent-identity", "pipeline", "--external-target-mask", "grounded-sam2"]
    assert "--target-extent" not in base and "--oracle-extent-identity" not in base
    a, x = _parse(v), _parse(xf)                                                   # reject_bad_flag_combinations inside
    assert (a.target_extent, a.oracle_extent_identity, a.external_target_mask, a.external_target_mask_mode) == (
        "oracle-visible", "pipeline", "grounded-sam2", None)
    va, vx = pf.version_flag_values(a), pf.version_flag_values(x)
    assert set(va) - set(vx) == {"external_target_mask"} and not set(vx) - set(va)
    assert {k for k in vx if va[k] != vx[k]} == {"nontarget_evidence", "target_extent", "oracle_extent_identity"}
    assert set(srv.final_options_on(a)) - set(srv.final_options_on(x)) == {"external_target_mask", "target_extent"}
    assert pf.is_final_version(v) and pf.is_gtto_version(v)
    assert not pf.version_requires_oracle_segmentation(v) and not pf.needs_oracle_segmentation(v)   # no client GT


def test_version_passes_the_panda_checks_and_validates_its_config(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    a = _server_args(_m6(), monkeypatch)
    ap = ss.build_parser()
    ss.reject_bad_flag_combinations(ap, a)
    ss.reject_panda_only_flags(ap, a)
    srv.check_unported_flags(a, None)
    srv.check_inpass_flags(a, None)
    srv.check_record_flags(a)
    to_config = ss.trajopt_config_from_args(a)
    to_config.validate() if hasattr(to_config, "validate") else None
    kw = pf.v11ab_build_kwargs(a)
    ag3s, info = pf.build_panda_ag3s(kw=kw)
    ag3s.config.validate()
    assert ag3s.config.esdf.target_extent == "oracle_visible" and ag3s.config.esdf.oracle_extent_identity == "pipeline"
    assert ag3s.config.esdf.nontarget_evidence == "proximity"          # the config default once the flag is dropped


@pytest.mark.parametrize("extra, msg", [
    (["--external-target-mask-mode", "track"], "--external-target-mask 의 분할기 mode"),
    (["--external-target-mask", "grounded-sam2"], "--target-extent oracle-visible 의 code 영상"),
    (["--external-target-mask", "grounded-sam2", "--target-extent", "oracle-visible", "--grounding-source", "oracle"],
     "--grounding-source oracle"),
    (["--external-target-mask", "grounded-sam2", "--target-extent", "oracle-visible", "--oracle-target-volume"],
     "--oracle-target-volume"),
])
def test_refusals(extra, msg, capsys):
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    a = ap.parse_args(["--config", "x", "--checkpoint", "y", "--model-xml", "z"] + extra)
    with pytest.raises(SystemExit):
        ss.reject_external_mask_flags(ap, a)
    assert msg in capsys.readouterr().err


def test_rby1_server_refuses_and_mode_choices(capsys):
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    a = ap.parse_args(["--config", "x", "--checkpoint", "y", "--model-xml", "z", "--external-target-mask", "grounded-sam2",
                       "--target-extent", "oracle-visible"])
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, a)
    assert "--robot panda 만" in capsys.readouterr().err
    a.robot = "panda"
    ss.reject_panda_only_flags(ap, a)
    ss.reject_external_mask_flags(ap, a)                                      # oracle-visible + mask: fine
    with pytest.raises(SystemExit):
        ap.parse_args(["--config", "x", "--checkpoint", "y", "--model-xml", "z", "--external-target-mask", "grounded-sam"])
    d = ap.parse_args(["--config", "x", "--checkpoint", "y", "--model-xml", "z"])
    assert d.external_target_mask is None and d.external_target_mask_mode is None


def test_launcher_dry_run_has_no_client_segmentation_flag(gpu_ok, monkeypatch, capsys):
    from AG3S.adapters.panda import profile as pf
    from benchmark.runners import launch_libero_paired as lp

    v = _m6()
    got = lp.server_argv("vla_v11ab", "/ck", 9, v)   # GT15 family: no probe (as -ab7)
    assert got[got.index("--external-target-mask") + 1] == "grounded-sam2"
    assert got[got.index("--target-extent") + 1] == "oracle-visible" and "--nontarget-evidence" not in got
    a = lp.build_parser().parse_args(["--task-ids", "0", "--episode-ids", "1", "--run-id", "x", "--conversion", "C",
                                      "--ag3s-version", v])
    assert not lp.oracle_segmentation_required("vla_v11ab", a) and lp.oracle_client_argv("vla_v11ab", a) == []
    cl = lp.client_argv("/run", 9, a)
    assert lp.gt_oracle_client_argv("vla_v11ab", a, cl) == cl
    lp.check_oracle_client_argv("vla_v11ab", a, cl)
    assert pf.version_server_flags(v, **lp.version_options(a))


# ------------------------------------------------------------------------------------------------ (2) the server path
class _Stop(Exception):
    pass


class _StubVLA:
    def infer(self, obs, **kw):
        return {"actions": np.zeros((10, 7), np.float32)}


class _StubInpass:
    """``AttentionPolicy`` in-pass sampler stand-in (no jax): the stub VLA's chunk, an attention block of zeros."""

    def __init__(self, policy, model_config, source):
        self.policy = policy

    def infer(self, obs, **kw):
        return dict(self.policy.infer(obs)), np.zeros((1, 768), np.float32)


def _recorded_request(call: int):
    """One request of a Tab. 1 AG3S X-fwd call, rebuilt from its records (policy images from the raw renders)."""
    from PIL import Image

    from benchmark.adapters.libero_panda import wire_pack

    ep = STORAGE / TAB1_SPATIAL_LI / "episodes"
    rows = [json.loads(line) for line in open(ep / f"{STEM}_calls.jsonl")]
    with np.load(ep / f"{STEM}_geom.npz", allow_pickle=False) as z:
        geom = {k: z[k] for k in z.files}
    with np.load(ep / f"{STEM}.npz", allow_pickle=False) as z:
        oi = int(rows[call]["obs_index"])
        raw = {"observation/image": z["obs_agentview_image"][oi], "observation/wrist_image": z["obs_robot0_eye_in_hand_image"][oi]}
    gi = list(geom["obs_index"]).index(oi)
    req = dict(wire_pack.wire_from_geom(geom, gi))
    for key, img in raw.items():   # the client: raw[::-1, ::-1] then resize_with_pad 256 -> 224
        req[key] = np.asarray(Image.fromarray(np.ascontiguousarray(img[::-1, ::-1])).resize((224, 224), Image.BILINEAR))
    req["observation/state"] = np.zeros(8, np.float32)
    req["prompt"] = grab_language_from_filename(
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate")
    req["panda/robot_state"] = np.asarray(geom["robot_state/head"][gi], np.float64)
    req["panda/T_world_base"] = np.asarray(geom["T_world_base"][gi], np.float64)
    req["ag3s/seq"] = call
    if call == 0:
        req["ag3s/reset"] = True
    return req


@pytest.mark.skipif(not (STORAGE / TAB1_SPATIAL_LI / "episodes" / f"{STEM}_geom.npz").exists(),
                    reason="Tab. 1 SafeLIBERO Spatial L-I record not in this storage")
def test_server_path_feeds_the_oracle_extent(gpu_ok, monkeypatch, tmp_path):
    import dataclasses

    from AG3S.adapters.panda import model as pm
    from AG3S.adapters.panda import server as srv
    from AG3S.ag3s.runtime import oracle_segmentation as osg
    from AG3S.trajopt import attention_policy as ap_mod
    from AG3S.trajopt import wire

    monkeypatch.setattr(ap_mod, "_build_inpass", _StubInpass)   # no jax
    args = _server_args(_m6(), monkeypatch)
    ext = _sidecar(tmp_path)
    served, info = srv.build_safe_server(_StubVLA(), args, gpu_warm_up=False, model_config=object(), external_mask=ext)
    try:
        assert served.external_mask is ext and info["external_target_mask"]["provider"] == "grounded-sam2"
        assert info["final_config"]["oracle_visible"]["needs_client_flag"].startswith("none: the server makes")
        meta = served.metadata
        assert wire.ORACLE_SEG_REQUIRED not in meta and meta["external_target_mask"]["client_segmentation"] is False
        assert meta["target_extent"] == "oracle_visible"

        safe = served._policy
        ag = safe.ag3s
        seen = {}

        def _no_esdf(depth_cameras, manipulated_geometry, *_a, **_k):
            seen["cameras"] = list(depth_cameras)
            return None, ["ABL-M6 test: no ESDF on CPU"]
        ag._build_esdf = _no_esdf

        def _infer(obs, **kw):   # what SafePolicy gets: run the real AG3S on it, then the real Panda conversion
            seen["obs"] = obs
            if obs.get("ag3s/reset"):   # SafePolicy.infer resets AG3S on the episode's first request
                ag.reset()
            _, scene = wire.strip_request(dict(obs))
            seg = wire.unpack_oracle_segmentation(scene)
            ag.set_oracle_segmentation(seg)
            o = wire.unpack_camera_observations(scene, robot_model=ag.robot_model,
                                                attention={c: np.zeros((16, 16), np.float32) for c in scene["cameras"]})
            q7, fingers = pm.split_robot_state(np.asarray(obs["panda/robot_state"], np.float64))
            o = [dataclasses.replace(x, robot_state=np.asarray(x.robot_state, np.float64)[:7].copy()) for x in o]
            ag.set_finger_joints(fingers)
            ag.process_multi_debug(o, phase="approach", active_manipulators=["left"])
            res = served.chunk_policy.infer(obs)
            return {"actions": res["actions"], "actions_reference": res["actions"]}
        monkeypatch.setattr(safe, "infer", _infer)
        monkeypatch.setattr(safe, "reset", lambda: None)

        req = _recorded_request(0)
        resp = served.infer(req)
        obs = seen["obs"]
        for cam in ("head", "left_wrist"):
            lab = obs[f"ag3s/oracle_seg/{cam}"]
            assert lab.dtype == np.uint8 and lab.shape == np.asarray(req[f"ag3s/depth/{cam}"]).shape
        head = obs["ag3s/oracle_seg/head"]
        assert int((head == osg.CODE_TARGET).sum()) > 0 and not (obs["ag3s/oracle_seg/left_wrist"] == osg.CODE_TARGET).any()
        assert set(np.unique(head)) <= {0, 1, 2}
        assert obs["ag3s/oracle_seg_meta"]["target"] == "black bowl" and obs["ag3s/oracle_seg_meta"]["robot_mask"] == "server"
        assert "ag3s/oracle_seg/head" not in req                                     # the client's request is not changed
        # the extent stage: the code images went into the episode's memory with this request's depth
        fr = ag._oracle_frame
        assert fr["segmentation"] is True and fr["target"] == "black bowl" and set(fr["cameras"]) == {"head", "left_wrist"}
        assert set(fr["robot_masked_px"]) == {"head", "left_wrist"} and fr["robot_masked_px"]["left_wrist"] > 0
        mem = ag._oracle_memory
        tp, op = mem.evidence(osg.CODE_TARGET)
        assert len(tp) > 0 and len(op) > 0 and mem.legend == {"black bowl": 1}
        # robot pixels never became samples: every sample is code 1 or 2
        assert set(np.unique(mem.codes)) <= {osg.CODE_TARGET, osg.CODE_STATIC}
        # oracle_evidence (what the target-free layer reads): AG3S identity on the mask -> its samples removed
        ev, rec = osg.oracle_evidence(mem, tp[:50], radius=float(ag.config.timing.fusion_voxel_size), identity="pipeline")
        assert rec["applied"] is True and rec["code"] == 1 and rec["object"] == "black bowl" and len(ev.target_points) == len(tp)
        far = op[np.linalg.norm(op - tp.mean(0), axis=1) > 0.15][:50]
        ev2, rec2 = osg.oracle_evidence(mem, far, radius=float(ag.config.timing.fusion_voxel_size), identity="pipeline")
        assert ev2 is None and rec2["why"].startswith("identity_unresolved")        # identity elsewhere: estimated extent
        # the call record
        em = resp["panda/oracle"]["external_mask"]
        assert em["prompt"] == "black bowl" and em["found"] is True and em["mode"] == "per_frame"
        assert em["cameras"]["head"]["found"] is True and em["cameras"]["head"]["score"] == pytest.approx(0.77)
        assert em["cameras"]["head"]["mask_pixels"] == 64 * 70 and 0 < em["cameras"]["head"]["code1_pixels"] <= 64 * 70
        assert em["cameras"]["left_wrist"] == {"found": False, "score": None, "mask_pixels": 0, "code1_pixels": 0,
                                                "latency_ms": 1.5, "why": "no_box_above_threshold", "source": "detect"}
        assert em["total_ms"] >= em["sidecar_ms"] >= 0.0 and em["robot_masked_px"] == fr["robot_masked_px"]
        assert resp["panda/oracle"]["segmentation"] is True
        json.dumps(resp["panda/oracle"])
        # a client that sends a GT code image is refused
        bad = _recorded_request(1)
        bad["ag3s/oracle_seg/head"] = head
        with pytest.raises(RuntimeError, match="client GT code image"):
            served.infer(bad)
    finally:
        assert ext.close() == 0


def test_flag_off_server_has_no_mask_and_refuses_one(gpu_ok, monkeypatch, tmp_path):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import attention_policy as ap_mod

    monkeypatch.setattr(ap_mod, "_build_inpass", _StubInpass)
    args = _server_args(pf.SOTA_FINAL_X_FWD, monkeypatch)
    served, info = srv.build_safe_server(_StubVLA(), args, gpu_warm_up=False, model_config=object())
    assert served.external_mask is None and "external_target_mask" not in info
    assert served.metadata is served._policy.metadata or served.metadata == served._policy.metadata
    with pytest.raises(ValueError, match="without --external-target-mask"):
        srv.build_safe_server(_StubVLA(), args, gpu_warm_up=False, model_config=object(), external_mask=object())


# ------------------------------------------------------------------------------------------------ (5) off: record keys
def test_earlier_versions_keep_their_record_keys(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert {"external_target_mask", "external_target_mask_mode"} <= set(pf.TAB3_CODE_DESTS)
    for v in (pf.SOTA_FINAL_X_FWD,) + pf.SOTA_TAB3_CODE_VERSIONS + pf.SOTA_ABL_MAIN_VERSIONS + pf.SOTA_TAB3_VERSIONS:
        vals = pf.version_flag_values(_parse(v))
        assert "external_target_mask" not in vals and "external_target_mask_mode" not in vals, v
    assert "external_target_mask" not in pf.version_flag_values(pf.parse_v11ab_args())
