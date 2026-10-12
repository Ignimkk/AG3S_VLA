"""SL3: pi05_libero attention binding (token blocks, un-mirror mapping), BDDL target, audit runner joins,
py3.8 environment sanitize (CPU)."""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from AG3S.adapters.panda import attention as pa
from benchmark.adapters.libero_panda import camera, client_env
from benchmark.adapters.libero_panda import task_target as tt
from benchmark.runners import launch_libero_pi05_only as launcher
from common.workspace import PROJECT_ROOT

LIBERO_PYTHON = PROJECT_ROOT / ".venv-libero/bin/python"
ROBOT_GEOMETRY = PROJECT_ROOT / "benchmark/adapters/libero_panda/robot_geometry.py"
FOREIGN_TORCH = "/usr/local/lib/python3.10/dist-packages/torch/lib"


def _has(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


# --- token blocks ----------------------------------------------------------------------------------------
def test_token_blocks_match_v11ab_bindings_and_the_camera_adapter():
    from AG3S.ag3s.experiments.sources.policy_record import CAMERA_BINDINGS, POLICY_CAMERA_NAMES

    blocks = pa.PI05_LIBERO_TOKEN_BLOCKS
    assert [b.model_key for b in blocks] == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    assert [b.tokens for b in blocks] == [tuple(CAMERA_BINDINGS[k][2]) for k in POLICY_CAMERA_NAMES]
    assert [b.v11ab_policy_camera for b in blocks] == list(POLICY_CAMERA_NAMES)
    assert [b.image_mask for b in blocks] == [True, True, False] and [b.used for b in blocks] == [True, True, False]
    # The adapter's camera table: agentview -> head, robot0_eye_in_hand -> left_wrist (SL1).
    assert [(b.robosuite_camera, b.wire_camera) for b in blocks if b.used] == list(camera.CAMERA_TABLE)
    assert pa.WIRE_CAMERAS == ("head", "left_wrist") and pa.block_for("left_wrist").tokens == (256, 512)
    with pytest.raises(ValueError):
        pa.block_for("right_wrist")
    layout = {"blocks": [{"model_key": b.model_key, "start": b.tokens[0], "stop": b.tokens[1], "image_mask": b.image_mask,
                          "prefix_mask_all": b.image_mask, "prefix_mask_any": b.image_mask} for b in blocks],
              "n_image_tokens": 768}
    assert pa.check_token_layout(layout) == []
    swapped = json.loads(json.dumps(layout))
    swapped["blocks"][2]["image_mask"] = True
    swapped["blocks"][0]["model_key"], swapped["blocks"][1]["model_key"] = "left_wrist_0_rgb", "base_0_rgb"
    problems = pa.check_token_layout(swapped)
    assert len(problems) == 2 and problems[0].startswith("order") and "image mask True" in problems[1]
    leaked = json.loads(json.dumps(layout))
    leaked["blocks"][2]["prefix_mask_any"] = True  # the model attends to the padding block
    assert pa.check_token_layout(leaked) == ["right_wrist_0_rgb: prefix_mask_any True != image mask False"]


def test_v11ab_cell_is_read_from_attention_policy():
    from AG3S.trajopt import attention_policy as ap

    cell = pa.v11ab_cell()
    assert (cell["denoise_step"], cell["aggregation"], cell["layer"], cell["head"]) == (0, "last", 8, 2)
    assert (ap._DENOISE_STEP, ap._AGG, ap._LAYER, ap._HEAD) == (0, "last", 8, 2)


# --- mapping -------------------------------------------------------------------------------------------------
def test_resize_with_pad_geometry_is_openpi_clients():
    from openpi_client import image_tools

    for in_hw, out_hw in (((256, 256), (224, 224)), ((480, 640), (224, 224)), ((300, 200), (224, 224)),
                          ((224, 224), (224, 224))):
        geom = pa.resize_with_pad_geometry(in_hw, out_hw)
        out = image_tools.resize_with_pad(np.full((*in_hw, 3), 255, np.uint8), *out_hw)
        rows, cols = np.nonzero(out[..., 0] > 0)
        assert (rows.min(), cols.min()) == (geom["pad_top"], geom["pad_left"]), (in_hw, geom)
        assert (rows.max() - rows.min() + 1, cols.max() - cols.min() + 1) == geom["resized_hw"], (in_hw, geom)
    geom = pa.check_square_no_padding()
    assert geom["resized_hw"] == (224, 224) and geom["scale_x"] == geom["scale_y"] == 224 / 256
    assert pa.resize_with_pad_geometry((480, 640), (224, 224))["pad_top"] > 0  # a 4:3 frame would pad


def _blob(h, w, cy, cx, sigma=3.0):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def _centroid(img):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    w = img.astype(np.float64)
    return float((yy * w).sum() / w.sum()), float((xx * w).sum() / w.sum())


@pytest.mark.parametrize("cy, cx", [(40.0, 30.0), (128.0, 200.5), (210.25, 90.75)])
def test_unmirror_and_scale_invert_the_client_preprocessing(cy, cx):
    """An OpenCV blob at (cy, cx) -> raw robosuite frame -> the client's policy image -> back to OpenCV."""
    from openpi_client import image_tools

    opencv = (_blob(256, 256, cy, cx) * 250).astype(np.float64)
    raw = opencv[::-1]  # adapter: OpenCV = raw[::-1]
    policy_in = np.repeat(np.ascontiguousarray(raw[::-1, ::-1])[..., None], 3, axis=2)
    policy = image_tools.resize_with_pad(np.round(policy_in).astype(np.uint8), 224, 224)[..., 0]
    py, px = _centroid(policy)
    u, v = pa.policy_pixel_to_opencv_uv(px, py)
    assert abs(u - cx) < 0.06 and abs(v - cy) < 0.06, (u, v, cx, cy)
    u_mirrored, _ = pa.policy_pixel_to_opencv_uv(px, py, unmirror=False)
    assert abs(u_mirrored - (255 - cx)) < 0.06  # the comparison mapping lands on the mirror image column
    # Same through the attention grid: the patch holding the blob, and the un-mirrored pixel map's peak.
    row, col = pa.policy_patch_of_uv(cx, cy)
    assert (int((py + 0.5) // 14), int((px + 0.5) // 14)) == (int(row), int(col))  # index -> edge convention
    grid = np.zeros((16, 16), np.float32)
    grid[int(row), int(col)] = 1.0
    peak = np.unravel_index(np.argmax(pa.attention_pixel_map(grid)), (256, 256))
    rs, cs = pa.patch_footprint(int(row), int(col))
    assert rs.start <= peak[0] < rs.stop and cs.start <= peak[1] < cs.stop and rs.start <= cy < rs.stop


def test_patch_centres_footprints_and_masks_are_one_convention():
    from AG3S.ag3s.experiments.sources.policy_record import patch_to_uv

    for r in range(16):
        for c in range(16):
            u, v = pa.patch_centre_uv(r, c)
            ue, ve = pa.patch_centre_uv(r, c, convention="edge")
            assert (ue, ve) == patch_to_uv(r, 15 - c, 256, 256) and (u, v) == (ue - 0.5, ve - 0.5)
            assert (u, v) == (16 * (15 - c) + 7.5, 16 * r + 7.5)
            rs, cs = pa.patch_footprint(r, c)
            assert rs.start <= v < rs.stop and cs.start <= u < cs.stop
            assert tuple(int(x) for x in pa.policy_patch_of_uv(u, v)) == (r, c)
            assert tuple(int(x) for x in pa.policy_patch_of_uv(u, v, unmirror=False)) == (r, 15 - c)
            mask = np.zeros((256, 256), bool)
            mask[rs, cs] = True
            cov = pa.opencv_mask_to_policy_patches(mask)
            assert cov[r, c] == 1.0 and cov.sum() == 1.0
    assert pa.patch_centre_uv(0, 0, unmirror=False) == (7.5, 7.5)
    with pytest.raises(ValueError):
        pa.patch_centre_uv(0, 0, convention="pixels")


def test_attention_pixel_map_is_v11ab_resampling_of_the_unmirrored_grid():
    from AG3S.ag3s.stages.attention_lifting import GridAttentionAdapter

    rng = np.random.default_rng(0)
    grid = rng.random((16, 16)).astype(np.float32)
    adapter = GridAttentionAdapter(interpolation="bilinear")
    mine = pa.attention_pixel_map(grid)
    np.testing.assert_array_equal(mine, adapter.to_pixel_map(np.ascontiguousarray(grid[:, ::-1]), (256, 256)))
    # pixel-centre bilinear is mirror-symmetric: un-mirroring the grid = mirroring the map
    np.testing.assert_allclose(mine, adapter.to_pixel_map(grid, (256, 256))[:, ::-1], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(pa.attention_pixel_map(grid, unmirror=False), adapter.to_pixel_map(grid, (256, 256)))
    assert mine.dtype == np.float32 and mine.shape == (256, 256)
    np.testing.assert_array_equal(pa.unmirror_grid(np.stack([grid, grid]))[1], grid[:, ::-1])
    conv = pa.mapping_conventions()
    json.dumps(conv)
    assert conv["resize_with_pad_256_to_224"]["pad_left"] == 0


# --- BDDL target ---------------------------------------------------------------------------------------------
def test_bddl_target_of_every_libero_spatial_task():
    names = tt.suite_task_names("libero_spatial")
    assert len(names) == 10 and names[0].startswith("pick_up_the_black_bowl_between_the_plate_and_the_ramekin")
    assert names[7] == "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
    for task_id in range(10):
        target = tt.task_target("libero_spatial", task_id)
        assert (target["target"], target["destination"]) == ("akita_black_bowl_1", "plate_1"), task_id
        assert target["obj_of_interest"] == ["akita_black_bowl_1", "plate_1"]
        assert target["goal"] == [["on", "akita_black_bowl_1", "plate_1"]]
        assert "main_table" in target["fixtures"] and "akita_black_bowl_1" in target["objects"]
        assert (PROJECT_ROOT / target["bddl_file"]).read_bytes() and len(target["bddl_sha256"]) == 64
        json.dumps(target)


def test_bddl_parser_rules():
    text = """(define (problem P) ; comment (with parens)
      (:domain robosuite)
      (:language Open THE Drawer)
      (:fixtures main_table - table cab_1 - cabinet)
      (:objects bowl_1 bowl_2 - bowl loose_1)
      (:obj_of_interest cab_1_top_region bowl_1)
      (:goal (And (Open cab_1_top_region) (On bowl_2 main_table))))"""
    problem = tt.parse_bddl(text)
    assert problem["language"] == "open the drawer"  # LIBERO's tokenizer lower-cases
    assert problem["objects"] == {"bowl": ["bowl_1", "bowl_2"], "object": ["loose_1"]}
    assert problem["fixtures"] == {"table": ["main_table"], "cabinet": ["cab_1"]}
    assert tt.goal_roles(problem) == ("cab_1_top_region", None)
    single = tt.parse_bddl("(define (problem P) (:obj_of_interest x_1) (:goal (In x_1 basket_1)))")
    assert single["goal"] == [["in", "x_1", "basket_1"]] and tt.goal_roles(single) == ("x_1", "basket_1")
    with pytest.raises(ValueError, match="no goal predicate"):
        tt.goal_roles(tt.parse_bddl("(define (problem P) (:obj_of_interest a_1) (:goal (On b_1 c_1)))"))
    with pytest.raises(ValueError):
        tt.scan_tokens("(define (a)")


def test_segmentation_instances_by_body_name():
    bodies = ["world", "table", "robot0_base", "akita_black_bowl_1_main", "akita_black_bowl_10_main",
              "wooden_cabinet_1_main", "wooden_cabinet_1_base", "wooden_cabinet_1_cabinet_top", "plate_1_main"]
    geom_bodyid = [0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8]
    assert tt.instance_body_ids("akita_black_bowl_1", bodies) == [3]  # not _10
    assert tt.instance_body_ids("wooden_cabinet_1", bodies) == [5, 6, 7]
    assert tt.instance_body_ids("main_table", bodies) == [1]
    assert tt.instance_geom_ids("akita_black_bowl_1", geom_bodyid, bodies) == [4, 5]
    assert tt.instance_geom_ids("wooden_cabinet_1", geom_bodyid, bodies) == [7, 8, 9, 10]
    with pytest.raises(ValueError, match="no MuJoCo body"):
        tt.instance_body_ids("cookies_1", bodies)
    seg = np.array([[-1, 4, 5], [11, 8, 4]], np.int16)
    np.testing.assert_array_equal(tt.instance_mask(seg, [4, 5]), [[False, True, True], [False, False, True]])
    tables = {"model_geom_bodyid": np.array(geom_bodyid), "model_body_names": np.array(bodies)}
    inst = tt.scene_instances(tables, {"objects": ["akita_black_bowl_1", "plate_1"], "fixtures": ["main_table"]})
    assert inst == {"akita_black_bowl_1": [4, 5], "plate_1": [11], "main_table": [1, 2]}


# --- py3.8 environment (SL3 0.2) -----------------------------------------------------------------------------
def test_launcher_and_client_env_are_one_rule():
    value = f"{FOREIGN_TORCH}:/usr/local/cuda/compat/lib:{PROJECT_ROOT}/.venv-libero/lib/python3.8/site-packages/x"
    assert launcher.sanitize_library_path(value) == client_env.sanitize_library_path(value, PROJECT_ROOT) == (
        f"/usr/local/cuda/compat/lib:{PROJECT_ROOT}/.venv-libero/lib/python3.8/site-packages/x")
    base = {"PYTHONPATH": str(PROJECT_ROOT), "LD_LIBRARY_PATH": value}
    assert launcher.client_environment(base) == client_env.client_environment(base, PROJECT_ROOT)
    assert launcher.THREAD_LIMIT_KEYS == client_env.THREAD_LIMIT_KEYS and launcher.CLIENT_THREADS == 4
    clean = client_env.client_environment(base, PROJECT_ROOT)
    assert client_env.needs_reexec(base, PROJECT_ROOT) and not client_env.needs_reexec(clean, PROJECT_ROOT)
    assert not client_env.needs_reexec({**base, client_env.REEXEC_SENTINEL: "1"}, PROJECT_ROOT)
    assert client_env.PROJECT_ROOT == PROJECT_ROOT


def test_robot_geometry_cli_runs_from_the_env_sh_shell(tmp_path):
    """SL2 verifier: the CLI failed under the pod LD_LIBRARY_PATH (foreign libtorch). It re-execs itself now."""
    if not LIBERO_PYTHON.exists():
        pytest.skip(".venv-libero not installed")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = FOREIGN_TORCH + ":" + env.get("LD_LIBRARY_PATH", "/usr/local/nvidia/lib64")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/third_party/openpi/src"
    env.pop(client_env.REEXEC_SENTINEL, None)
    # --source libero imports LIBERO (torch); --source robosuite does not and never failed.
    cmd = [str(LIBERO_PYTHON), str(ROBOT_GEOMETRY), "fk", "--source", "libero", "--out"]
    out = tmp_path / "fk.npz"
    done = subprocess.run(cmd + [str(out)], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and out.exists(), done.stdout[-2000:] + done.stderr[-3000:]
    assert json.loads(done.stdout.strip().splitlines()[-1])["out"] == str(out)
    if not os.path.isdir(FOREIGN_TORCH):
        return  # not this pod: nothing to reproduce
    # Without the re-exec (sentinel pre-set) the same shell reproduces the SL2 failure.
    failed = subprocess.run(cmd + [str(tmp_path / "fk_no_reexec.npz")], cwd=str(tmp_path),
                            env={**env, client_env.REEXEC_SENTINEL: "1"}, capture_output=True, text=True, timeout=300)
    assert failed.returncode != 0 and "libtorch_python.so" in failed.stderr, failed.stderr[-2000:]


# --- audit runner (CPU parts) --------------------------------------------------------------------------------
@pytest.fixture
def fake_source_run(tmp_path, monkeypatch):
    """A two-episode source run laid out like launch_libero_pi05_only's, below a temporary storage root."""
    from openpi_client import image_tools

    root = tmp_path / "storage"
    run = root / "results/benchmark/20261012T000000_000000000000_fake"
    (run / "episodes").mkdir(parents=True)
    (run / "policy_records").mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"commit": "0" * 40, "settings": {
        "condition": "pi0.5-only", "policy_config": "pi05_libero", "checkpoint": {"path": "/nowhere"}}}))
    rng = np.random.default_rng(1)
    rows, step = [], 0
    for task_id, calls in ((0, 2), (3, 3)):
        n_obs = 10 + 5 * calls + 1
        agent = rng.integers(0, 256, (n_obs, 256, 256, 3), dtype=np.uint8)
        wrist = rng.integers(0, 256, (n_obs, 256, 256, 3), dtype=np.uint8)
        chunk_t = np.array([10 + 5 * k for k in range(calls)])
        state = rng.normal(size=(calls, 8))
        stem = f"libero_spatial_t{task_id}_e0"
        np.savez_compressed(run / "episodes" / f"{stem}.npz", obs_agentview_image=agent, obs_robot0_eye_in_hand_image=wrist,
                            chunk_t=chunk_t, policy_state=state)
        description = f"task {task_id}"
        for k in range(calls):
            t = int(chunk_t[k])
            img = image_tools.resize_with_pad(np.ascontiguousarray(agent[t][::-1, ::-1]), 224, 224)
            wimg = image_tools.resize_with_pad(np.ascontiguousarray(wrist[t][::-1, ::-1]), 224, 224)
            record = {"inputs/observation/image": img, "inputs/observation/wrist_image": wimg,
                      "inputs/observation/state": state[k].copy(), "inputs/prompt": description,
                      "outputs/actions": np.zeros((10, 7))}
            np.save(run / "policy_records" / f"step_{step}.npy", np.asarray(record))
            step += 1
        rows.append({"suite": "libero_spatial", "task_id": task_id, "episode_idx": 0, "init_state_index": 0,
                     "task_description": description, "status": "success", "official_success": True,
                     "policy_calls": calls, "npz": f"episodes/{stem}.npz"})
    (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("AG3S_STORAGE_ROOT", str(root))
    return run


def test_audit_runner_indexes_and_joins_policy_records(fake_source_run):
    from benchmark.runners import attention_audit as audit

    run = audit.resolve_source_run(fake_source_run.name)
    index = audit.index_source_run(run)
    assert index["policy_records"] == 5 and [e["global_steps"] for e in index["episodes"]] == [[0, 1], [2, 3, 4]]
    assert index["episodes"][1]["task_id"] == 3 and index["episodes"][0]["geom_npz"] is None
    with np.load(run / index["episodes"][1]["npz"]) as ep:
        ep = dict(ep)
    for k, step in enumerate(index["episodes"][1]["global_steps"]):
        record = audit.load_policy_record(run / "policy_records" / f"step_{step}.npy")
        join = audit.check_join(record, ep, k, "task 3", 224)
        assert join == {"chunk_t": 10 + 5 * k, "obs_index": 10 + 5 * k, "state": True, "prompt": True,
                        "image": True, "wrist_image": True}
        assert sorted(audit.request_of(record)) == sorted(audit.REQUEST_KEYS)
    record = audit.load_policy_record(run / "policy_records" / "step_2.npy")
    with pytest.raises(ValueError, match="does not join"):
        audit.check_join(record, ep, 1, "task 3", 224)  # the next call's frame
    (run / "policy_records" / "step_5.npy").write_bytes((run / "policy_records" / "step_4.npy").read_bytes())
    with pytest.raises(ValueError, match="policy calls"):
        audit.index_source_run(run)
    with pytest.raises(ValueError, match="below"):
        audit.resolve_source_run(str(PROJECT_ROOT))


def test_audit_runner_dry_run_settings(fake_source_run, capsys):
    from benchmark.runners import attention_audit as audit

    assert audit.main(["--source-run", fake_source_run.name, "--dry-run", "--task-ids", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    settings = out["settings"]
    assert out["calls"] == 3 and settings["step"] == "SL3" and settings["policy_config"] == "pi05_libero"
    assert settings["v11ab_cell"]["layer"] == 8 and settings["v11ab_cell"]["head"] == 2
    assert [b["model_key"] for b in settings["token_blocks"]] == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    source = settings["attention_sources"][0]
    assert source["run_id"] == fake_source_run.name and len(source["manifest_sha256"]) == 64
    assert [e["task_id"] for e in source["episodes"]] == [3]
    assert settings["attention"]["full_grid"] and settings["attention"]["subtask_feature"]


# --- the read-out itself, on a tiny pi05_libero model (subprocess, JAX on CPU) --------------------------------
@pytest.mark.skipif(not _has("jax", "flax", "openpi"), reason="needs jax/flax/openpi (.venv-openpi-live)")
def test_attention_reader_on_a_tiny_pi05_libero_model():
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    done = subprocess.run([sys.executable, str(PROJECT_ROOT / "tests/benchmark/sl3_tiny_pi05.py")], cwd=str(PROJECT_ROOT),
                          env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    # Token layout read from the model input: order, 256 tokens each, right wrist masked out.
    assert out["problems"] == [], out["problems"]
    assert [(b["model_key"], b["start"], b["stop"], b["image_mask"], b["prefix_mask_any"])
            for b in out["layout"]["blocks"]] == [("base_0_rgb", 0, 256, True, True),
                                                   ("left_wrist_0_rgb", 256, 512, True, True),
                                                   ("right_wrist_0_rgb", 512, 768, False, False)]
    # The read-out is V11AB's AttentionPolicy cell, bit for bit; the padding block gets no attention.
    assert out["diffs"] == {"base_0_rgb": 0.0, "left_wrist_0_rgb": 0.0, "right_wrist_0_rgb": 0.0}
    assert out["max_block_mass_over_cells"]["right_wrist_0_rgb"] == 0.0
    assert out["max_block_mass_over_cells"]["base_0_rgb"] > 0.0
    assert out["pooled_shape"][0] == 10 and out["pooled_shape"][2] == 768 and out["grids_equal"]
    assert out["same_without_feature"] and out["cameras"] == ["head", "left_wrist"]
