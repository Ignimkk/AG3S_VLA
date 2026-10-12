"""SL1 adapter checks that need LIBERO/robosuite -- run by ``.venv-libero`` (Python 3.8, no pytest).

Driven by ``tests/benchmark/test_libero_sl1.py::test_py38_adapter_checks`` via subprocess::

    .venv-libero/bin/python tests/benchmark/libero_py38_checks.py --out DIR

Writes ``DIR/checks.json`` ({name: {"ok": bool, ...numbers}}) and the fixtures
``DIR/wire_t0.npz`` (adapter -> V11AB wire, initial observation) and ``DIR/reference_t0.npz``
(float depth / K / T of the same frame) for the 3.11 side. Exit code 0 even if a check fails --
the pytest side asserts each check so the failure names the check.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import pathlib
import sys
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "benchmark/runners"))

from benchmark.adapters.libero_panda import camera, wire_pack  # noqa: E402
import libero_pi05_client as client_mod  # noqa: E402

TABLE_TOP_GEOMS = ("table_visual", "table_collision")


def robosuite_reference():
    """robosuite camera_utils formulas, exec'd from its source (the module itself imports h5py)."""
    import robosuite.utils.transform_utils as T

    spec = importlib.util.find_spec("robosuite")
    path = pathlib.Path(list(spec.submodule_search_locations)[0]) / "utils/camera_utils.py"
    tree = ast.parse(path.read_text())
    wanted = ("get_camera_intrinsic_matrix", "get_camera_extrinsic_matrix", "get_real_depth_map")
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {"np": np, "T": T}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, str(path)


def make_env():
    from libero.libero import benchmark

    np.random.seed(client_mod.DEFAULT_SEED)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    init = suite.get_task_init_states(0)
    env, description = client_mod._get_libero_env_with_depth(task, client_mod.LIBERO_ENV_RESOLUTION,
                                                               client_mod.DEFAULT_SEED)
    camera.install_capture_observable(env)
    env.reset()
    obs = env.set_init_state(init[0])
    return env, obs, description


def check(results, name):
    def wrap(fn):
        try:
            out = fn() or {}
            out.setdefault("ok", True)
        except Exception as error:  # report, do not stop the other checks
            out = {"ok": False, "error": "{}: {}".format(type(error).__name__, error), "trace": traceback.format_exc()}
        results[name] = out
        return fn
    return wrap


def table_errors(frame, depth_m, seg, sim, table_ids):
    """Back-project table pixels of ``depth_m`` (rows as given) -> world z minus table-top z (m)."""
    mask = np.isin(seg, table_ids)
    rows, cols = np.nonzero(mask)
    z = depth_m[rows, cols].astype(np.float64)
    K = frame["K"]
    pts_cam = np.stack([(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z], axis=1)
    T_world_cam = frame["T_world_base"] @ frame["T_base_cam"]
    pts_world = pts_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    gid = int(table_ids[0])
    top = float(sim.data.geom_xpos[gid][2] + sim.model.geom_size[gid][2])
    return pts_world[:, 2] - top, rows, z


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"_python": sys.version.split()[0], "_numpy": np.__version__}

    env, obs, _ = make_env()
    sim = env.sim
    frames = camera.capture(env, obs, segmentation=True)
    reference, reference_path = robosuite_reference()
    results["_robosuite_camera_utils"] = reference_path

    @check(results, "camera_table_and_keys")
    def _():
        assert list(frames) == ["head", "left_wrist"], list(frames)
        for wire, frame in frames.items():
            assert frame["depth_m"].dtype == np.float32 and frame["depth_m"].shape == (256, 256)
            assert frame["rgb"].dtype == np.uint8 and frame["rgb"].shape == (256, 256, 3)
            assert frame["robot_state"].shape == (9,)
        return {"cameras": list(frames)}

    @check(results, "K_matches_fovy_formula_and_robosuite")
    def _():
        info = {}
        for wire, frame in frames.items():
            source = frame["source_camera"]
            cam_id = sim.model.camera_name2id(source)
            fovy = float(sim.model.cam_fovy[cam_id])
            f = 0.5 * 256 / np.tan(np.deg2rad(fovy) / 2)
            K = frame["K"]
            assert abs(K[0, 0] - f) < 1e-9 and abs(K[1, 1] - f) < 1e-9, (K, f)
            assert K[0, 2] == 127.5 and K[1, 2] == 127.5 and K[2, 2] == 1.0 and K[0, 1] == 0 and K[1, 0] == 0
            ref = np.asarray(reference["get_camera_intrinsic_matrix"](sim, source, 256, 256), np.float64)
            # SL2: pixel-centre principal point = robosuite's W/2, H/2 minus 0.5 px; focal length identical.
            assert np.array_equal(camera.intrinsic_matrix(fovy, 256, 256), ref), ref
            shifted = ref.copy()
            shifted[0, 2] -= 0.5
            shifted[1, 2] -= 0.5
            assert np.array_equal(K, shifted), (K, ref)
            info[wire] = {"fovy_deg": fovy, "f_px": float(K[0, 0])}
        return info

    @check(results, "T_world_cam_and_depth_match_robosuite")
    def _():
        info = {}
        for wire, frame in frames.items():
            source = frame["source_camera"]
            ref_T = reference["get_camera_extrinsic_matrix"](sim, source)
            dT = float(np.max(np.abs(frame["T_world_cam"] - ref_T)))
            assert dT < 1e-12, dT
            raw = obs[source + "_depth"]
            ref_depth = reference["get_real_depth_map"](sim, raw)[..., 0][::-1]
            dd = float(np.max(np.abs(frame["depth_m"].astype(np.float64) - ref_depth)))
            # robosuite computes in float32 (numpy 1.x value casting); this adapter in float64.
            assert dd < 1e-4, dd
            info[wire] = {"max_abs_T": dT, "max_abs_depth_m": dd}
        return info

    @check(results, "T_base_cam_orthonormal_over_steps")
    def _():
        worst = {"orth": 0.0, "det": 0.0, "compose": 0.0}
        local = camera.capture(env, obs)
        for step in range(8):
            for wire, frame in local.items():
                T = frame["T_base_cam"]
                R = T[:3, :3]
                worst["orth"] = max(worst["orth"], float(np.max(np.abs(R.T @ R - np.eye(3)))))
                worst["det"] = max(worst["det"], abs(float(np.linalg.det(R)) - 1.0))
                assert np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0])
                composed = np.linalg.inv(frame["T_world_base"]) @ frame["T_world_cam"]
                worst["compose"] = max(worst["compose"], float(np.max(np.abs(composed - T))))
            action = [0.3, -0.2, 0.2, 0.0, 0.0, 0.3, -1.0] if step >= 2 else client_mod.LIBERO_DUMMY_ACTION
            step_obs, _, _, _ = env.step(action)
            local = camera.capture(env, step_obs)
        assert worst["orth"] < 1e-9 and worst["det"] < 1e-9 and worst["compose"] < 1e-9, worst
        return worst

    @check(results, "robot_state_is_image_instant")
    def _():
        # SL2 item 0.2: FK of the recorded robot_state (9 robot qpos) reproduces the recorded wrist T_world_cam;
        # FK of obs robot0_joint_pos/gripper_qpos (data.qpos at the sampling substep) does not, while moving.
        import mujoco

        m = sim.model._model
        work = mujoco.MjData(m)
        cam_id = sim.model.camera_name2id("robot0_eye_in_hand")
        idx = camera.robot_qpos_indexes(env)
        worst_image, worst_data, sources = 0.0, [], []
        local_obs = env.set_init_state(client_mod_init_state())
        for step in range(10):
            frames_now = camera.capture(env, local_obs)
            frame = frames_now["left_wrist"]
            sources.append(int(frame["qpos_image_source"]))
            for label, q9 in (("image", frame["robot_state"]),
                              ("data", np.concatenate([local_obs["robot0_joint_pos"], local_obs["robot0_gripper_qpos"]]))):
                work.qpos[:] = frame["sim_state"][1:1 + m.nq]
                work.qpos[idx] = q9
                mujoco.mj_kinematics(m, work)
                mujoco.mj_comPos(m, work)
                mujoco.mj_camlight(m, work)  # cam_xpos / cam_xmat are not part of mj_kinematics
                T = camera.world_camera_pose_opencv(work.cam_xpos[cam_id], work.cam_xmat[cam_id])
                err = float(np.max(np.abs(T - frame["T_world_cam"])))
                if label == "image":
                    worst_image = max(worst_image, err)
                else:
                    worst_data.append(err)
            action = [0.5, -0.4, 0.3, 0.0, 0.0, 0.5, -1.0] if step >= 1 else client_mod.LIBERO_DUMMY_ACTION
            local_obs, _, _, _ = env.step(action)
        assert worst_image < 1e-12, worst_image
        assert max(worst_data) > 1e-7, worst_data
        return {"max_abs_T_world_cam_fk_robot_state": worst_image,
                "max_abs_T_world_cam_fk_obs_joint_pos": max(worst_data), "qpos_image_source": sources}

    @check(results, "depth_row_order_table_plane")
    def _():
        restored = env.set_init_state(client_mod_init_state())
        local = camera.capture(env, restored, segmentation=True)
        names = [sim.model.geom_id2name(i) for i in range(sim.model.ngeom)]
        table_ids = np.array([i for i, n in enumerate(names) if n in TABLE_TOP_GEOMS], dtype=np.int64)
        gid = int(table_ids[0])
        assert np.allclose(sim.data.geom_xmat[gid].reshape(3, 3), np.eye(3), atol=1e-9), "table not axis aligned"
        info = {"table_top_z_world": float(sim.data.geom_xpos[gid][2] + sim.model.geom_size[gid][2])}
        frame = local["head"]
        seg = frame["segmentation_geom"]
        err, rows, z = table_errors(frame, frame["depth_m"], seg, sim, table_ids)
        flipped, _, _ = table_errors(frame, frame["depth_m"][::-1], seg, sim, table_ids)
        top_half = rows < np.median(rows)
        info.update({
            "pixels": int(err.size),
            "opencv_rows": {"median_mm": float(np.median(err) * 1e3), "p95_abs_mm": float(np.percentile(np.abs(err), 95) * 1e3)},
            "flipped_rows": {"median_mm": float(np.median(flipped) * 1e3),
                             "p95_abs_mm": float(np.percentile(np.abs(flipped), 95) * 1e3)},
            "mean_depth_upper_rows_m": float(z[top_half].mean()), "mean_depth_lower_rows_m": float(z[~top_half].mean()),
        })
        assert err.size > 1000, err.size
        assert abs(info["opencv_rows"]["median_mm"]) < 5.0 and info["opencv_rows"]["p95_abs_mm"] < 10.0, info
        assert info["flipped_rows"]["p95_abs_mm"] > 50.0, info
        # agentview looks at the table from the front: lower picture rows are nearer table.
        assert info["mean_depth_lower_rows_m"] < info["mean_depth_upper_rows_m"], info
        return info

    @check(results, "client_episode_geometry_alignment")
    def _():
        class ZeroPolicy:
            def infer(self, element):
                return {"actions": np.zeros((10, 7))}

            def get_server_metadata(self):
                return {}

        run_args = client_mod.build_parser().parse_args(["--run-dir", "x", "--port", "1", "--capture-geometry"])
        geometry = client_mod._GeometryRecorder()
        summary, rec, _ = client_mod.run_episode(env, [client_mod_init_state()], 0, "t", ZeroPolicy(), run_args,
                                                 max_steps=3, observer=geometry.observe)
        arrays = geometry.arrays()
        n_obs = len(rec.obs["agentview_image"])
        assert summary["status"] == "max_steps" and len(geometry.frames) == n_obs == 14, (summary["status"], n_obs)
        for source, wire in camera.CAMERA_TABLE:
            raw = np.stack(rec.obs[source + "_image"])
            assert np.array_equal(np.stack([f[wire]["rgb"] for f in geometry.frames]), raw[:, ::-1])
            assert arrays["depth_mm/" + wire].shape == (n_obs, 256, 256) and arrays["depth_mm/" + wire].dtype == np.uint16
            # SL2: robot_state is the image-instant qpos, not obs robot0_joint_pos (= data.qpos, one substep later).
            idx = arrays["robot_qpos_indexes"]
            assert np.array_equal(arrays["robot_state/" + wire], arrays["sim_state_image"][:, 1:][:, idx])
            assert np.array_equal(arrays["robot_state/" + wire][0, :7], rec.obs["robot0_joint_pos"][0])
        dt = [float(r[0] - f["head"]["stamp"]) for f, r in zip(geometry.frames, geometry.sim_state_at_return)]
        dq = [float(np.max(np.abs(f["head"]["sim_state"][1:] - r[1:])))
              for f, r in zip(geometry.frames, geometry.sim_state_at_return)]
        assert dt[0] == 0.0 and dq[0] == 0.0  # set_init_state: forced sample at the current state
        source = arrays["qpos_image_source"].tolist()
        residual = arrays["qpos_image_residual_m"]
        # Initial obs: forced sample after mj_forward -> current qpos; every env.step obs: integrated back.
        assert source[0] == camera.QPOS_IMAGE_CURRENT and set(source[1:]) == {camera.QPOS_IMAGE_INTEGRATED_BACK}, source
        assert float(np.max(np.min(residual, axis=1))) <= camera.QPOS_IMAGE_TOLERANCE_M, residual
        inner = client_mod._geometry_adapter()[0]._inner(env)
        return {"frames": n_obs, "keys": sorted(arrays), "return_minus_capture_time_s": dt,
                "max_abs_state_return_minus_capture": dq, "control_timestep_s": float(inner.control_timestep),
                "model_timestep_s": float(inner.model_timestep), "qpos_image_source": source,
                "qpos_image_residual_chosen_max_m": float(np.max(np.min(residual, axis=1))),
                "qpos_image_residual_other_min_m": float(np.min(np.max(residual[1:], axis=1)))}

    # Fixture for the 3.11 side: initial observation of task 0 / init state 0.
    restored = env.set_init_state(client_mod_init_state())
    local = camera.capture(env, restored)
    wire = wire_pack.pack_cameras(local, stamp_mode=camera.STAMP_MODE)
    for name in ("wire_t0.npz", "reference_t0.npz"):
        if (out / name).exists():
            os.unlink(str(out / name))
    wire_pack.save_wire_npz(out / "wire_t0.npz", wire)
    reference_arrays = {}
    for name, frame in local.items():
        for key in ("depth_m", "K", "T_base_cam", "robot_state"):
            reference_arrays[key + "/" + name] = np.asarray(frame[key])
    np.savez_compressed(str(out / "reference_t0.npz"), **reference_arrays)
    results["_fixture_bytes"] = {name: (out / name).stat().st_size for name in ("wire_t0.npz", "reference_t0.npz")}

    (out / "checks.json").write_text(json.dumps(results, indent=1, default=str))
    env.close()
    return 0


_INIT = []


def client_mod_init_state():
    if not _INIT:
        from libero.libero import benchmark

        _INIT.append(benchmark.get_benchmark_dict()["libero_spatial"]().get_task_init_states(0)[0])
    return _INIT[0]


if __name__ == "__main__":
    raise SystemExit(main())
