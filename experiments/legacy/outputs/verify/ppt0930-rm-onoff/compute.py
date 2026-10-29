"""Robot mask (self-filter) ON vs OFF — ep1807 첫 정책 호출 (t_step 0), 16D, 현재 코드.

    cd ${AG3S_PROJECT_ROOT} && MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} \
        .venv-openpi-live/bin/python outputs/verify/ppt0930-rm-onoff/compute.py

- 씬: 기록 qpos 를 그대로 올린다 (`policy_record.replay_scene` + `pose_scene` — esdf_rollout 과 같은 길).
- AG3S: `serve_safe.build_parser()` 기본값 + `--esdf-backend curobo` 로 `serve_safe.build_ag3s` 를 그대로 부른다.
- ON  = 파이프라인 그대로 (`AG3S._robot_mask_for` 가 카메라마다 image mask 를 만든다).
- OFF = 같은 입력, 같은 설정, 인스턴스의 `_robot_mask_for` 만 None 을 돌려주게 (벤치마크 코드는 안 고친다).
       점군 경로의 self-filter(grounding 이 읽는 것)는 그대로 켜져 있다 — 거리장의 입력만 다르다.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np  # noqa: E402

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/ppt0930-rm-onoff'))
REC = _ag3s_workspace_path('outputs/live_test/20260925_ep1807/run_0000')
ATT = _ag3s_workspace_path('AG3S/ag3s/asset/data/attention_16d_ep1807.npz')
STEP1 = _ag3s_workspace_path('AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json')
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
POLICY_CAM = {"zed_left": "cam_high", "wrist_cam_l": "cam_left_wrist", "wrist_cam_r": "cam_right_wrist"}
MARGIN = 0.05
PHASE = "approach"          # client `--safe-phase` default (pi05_infer.py:742) and the E3 runs
MANIPULATORS = ["left"]     # E3 runs `--safe-manipulators left`


def eikonal(values, vs, free_above=0.03):
    from AG3S.ag3s.experiments.live.verify_env import eikonal as _e
    return _e(values, vs, free_above=free_above)


def main():
    import mujoco

    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import (
        TransportScene, camera_observation, set_finger_joints_from_scene)
    from AG3S.ag3s.experiments.sources.policy_record import (
        load_run, pose_scene, record_fingerprint, replay_scene)
    from AG3S.trajopt import serve_safe

    run = load_run(REC, limit=1)
    step = run.steps[0]
    assert step.t_step == 0 and step.qpos.shape == (66,) and step.state.shape == (16,)
    scene = replay_scene(run)
    pose_scene(scene, step)

    # --- serve_safe 기본값 그대로 (backend 만 curobo) ---------------------------------
    argv = ["--config", "pi05_rby1_randomized_pick_place_16d_lora", "--checkpoint", "/dev/null",
            "--model-xml", str(run.model_xml), "--esdf-backend", "curobo"]
    args = serve_safe.build_parser().parse_args(argv)
    cli = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()
           if isinstance(v, (int, float, str, bool, type(None), tuple, list))}

    def make_ag():
        return serve_safe.build_ag3s(
            args.model_xml, voxel=args.voxel, range_max=args.range_max, links=args.links,
            backend=args.esdf_backend, fine_voxel=args.fine_voxel, tsdf_voxel=args.tsdf_voxel,
            attached_sign_threshold=args.attached_sign_threshold,
            max_field_age_sec=args.max_field_age_sec, exclude_links=args.exclude_links,
            constraint_sphere_options=serve_safe.sphere_options(args),
            self_collision=not args.no_self_collision,
            target_field_policy=args.target_field_policy,
            self_filter=serve_safe.self_filter_options(args))

    # attention (live 배선: 카메라마다 자기 것) — 기록과 짝인 npz, step-1 best cell
    blob = np.load(ATT, allow_pickle=False)
    assert str(blob["records"]) == REC, blob["records"]
    cell = json.loads(pathlib.Path(STEP1).read_text())["best"]
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    cams = [str(c) for c in blob["cameras"]]
    fi = int(np.where(np.asarray(blob["t_step"]) == 0)[0][0])
    att = {c: np.asarray(blob["attention"][fi, di, ai, cell["layer"], cell["head"],
                                           cams.index(POLICY_CAM[c])], np.float32)
           for c in CAMERAS}

    results = {}
    frames_npz = {}
    fields = {}
    masks = {}
    for mode in ("on", "off"):
        ag = make_ag()
        # T29 — 촬영 시점 손가락 개도를 두 모델에 (SafePolicy._apply_finger_state 와 같은 값)
        vals = set_finger_joints_from_scene(scene, ag.robot_model)
        if vals is not None and hasattr(ag, "set_finger_joints"):
            ag.set_finger_joints(vals, record={"source": "mujoco_scene"})
        rec = []
        orig = ag._robot_mask_for

        def wrapped(d, K, T, q, _orig=orig, _rec=rec):
            m = _orig(d, K, T, q)
            _rec.append(None if m is None else np.asarray(m, bool).copy())
            return m

        if mode == "on":
            ag._robot_mask_for = wrapped
        else:
            ag._robot_mask_for = lambda d, K, T, q, _rec=rec: (_rec.append(None), None)[1]
        obs = []
        for i, c in enumerate(CAMERAS):
            o, fr = camera_observation(scene, c, ag.robot_model, timestamp=0.0,
                                       attention_map=att[c])
            obs.append(o)
            if mode == "on":
                T = np.asarray(o.resolve_T_base_cam(ag.robot_model), np.float64)
                frames_npz[f"depth_{c}"] = np.asarray(fr.depth, np.float32)
                frames_npz[f"K_{c}"] = np.asarray(fr.camera_intrinsics, np.float64)
                frames_npz[f"T_{c}"] = T
                frames_npz[f"T_mujoco_{c}"] = np.asarray(fr.T_base_cam, np.float64)
                frames_npz[f"robot_state_{c}"] = np.asarray(fr.robot_state, np.float64)
        cs, dbg = ag.process_multi_debug(obs, phase=PHASE, active_manipulators=MANIPULATORS)
        field = cs.esdf
        assert type(field).__name__ == "CuroboEsdfField", type(field)
        # 함정 4 — 다음 빌드 전에 값 격자를 복사해 둔다
        snap = [(np.asarray(L.distance_grid, np.float32).copy(),
                 np.asarray(L.grid.origin, np.float64).copy(), float(L.grid.voxel_size))
                for L in field.layers]
        fields[mode] = (field, snap, ag)
        masks[mode] = rec
        results[mode] = {"status": str(getattr(cs.status, "value", cs.status)),
                         "n_masks_recorded": len(rec)}

    # 버퍼 덮어쓰기 검사: ON 필드 객체의 격자가 OFF 빌드 뒤에도 스냅샷과 같은가
    on_field, on_snap, ag_on = fields["on"]
    same = [bool(np.array_equal(np.asarray(L.distance_grid, np.float32), s[0]))
            for L, s in zip(on_field.layers, on_snap)]

    # --- self-filter 설정 (실제로 들어간 값) --------------------------------------------
    sf = ag_on.self_filter_settings()
    n_filter = int(ag_on.robot_model.n_spheres)

    # --- 마스크 --------------------------------------------------------------------
    per_cam = {}
    for c, m in zip(CAMERAS, masks["on"]):
        d = frames_npz[f"depth_{c}"]
        valid = (d > 1e-6)
        frames_npz[f"mask_{c}"] = np.zeros(d.shape, bool) if m is None else m
        mm = frames_npz[f"mask_{c}"]
        per_cam[c] = {"mask_px": int(mm.sum()), "image_px": int(mm.size),
                      "erased_pct_of_image": 100.0 * float(mm.mean()),
                      "valid_depth_px": int(valid.sum()),
                      "erased_pct_of_valid_depth": 100.0 * float(mm[valid].mean())}

    # --- 제약 구 ------------------------------------------------------------------
    mj_model = mujoco.MjModel.from_xml_path(str(pathlib.Path(run.model_xml).resolve()))
    ts = TransportScene.attach(mj_model, mujoco.MjData(mj_model))
    q = np.asarray(frames_npz["robot_state_zed_left"], np.float64)
    spheres = {}
    for name, model in (("gripper", ag_on.constraint_robot_model),
                        ("arms", build_constraint_robot_model(
                            ts, link_filter=serve_safe.constraint_link_filter("arms")))):
        set_finger_joints_from_scene(scene, model)
        c_, r_ = model.sphere_centers_numeric(q)
        c_ = np.asarray(c_, np.float64); r_ = np.asarray(r_, np.float64)
        links = np.asarray(model.sphere_link_names)
        entry = {"n_spheres": int(len(r_)), "links": sorted(set(links.tolist()))}
        for mode in ("on", "off"):
            d = np.asarray(fields[mode][0].distance(c_), np.float64)
            clr = d - r_ - MARGIN
            k = int(np.argmin(clr))
            entry[mode] = {"n_violated": int((clr < 0).sum()),
                           "worst_clearance_mm": 1000.0 * float(clr[k]),
                           "worst_link": str(links[k]),
                           "median_clearance_mm": 1000.0 * float(np.median(clr))}
            frames_npz[f"clr_{name}_{mode}"] = clr
        frames_npz[f"centres_{name}"] = c_
        frames_npz[f"radii_{name}"] = r_
        frames_npz[f"links_{name}"] = links
        spheres[name] = entry

    # --- 필드 격자 · eikonal -------------------------------------------------------
    eik = {}
    for mode in ("on", "off"):
        eik[mode] = {}
        for tier, (v, o, vs) in zip(("coarse", "fine"), fields[mode][1]):
            e = eikonal(v, vs)
            eik[mode][tier] = {**e, "voxel_m": vs, "shape": list(v.shape),
                               "origin_voxel_centre_m": o.tolist(),
                               "negative_fraction": float((v < 0).mean())}
            frames_npz[f"{mode}_{tier}_values"] = v
            frames_npz[f"{mode}_{tier}_origin"] = o
            frames_npz[f"{mode}_{tier}_voxel"] = np.float64(vs)

    # --- 슬라이스: 왼팔을 지나는 수직 x–z 평면 (합성 필드 distance(), 5 mm 표본) --------
    arm = frames_npz["centres_arms"]
    lk = frames_npz["links_arms"]
    left = np.array([("left" in s) for s in lk])
    y0 = float(np.median(arm[left, 1]))
    xs = np.arange(-0.10, 1.10 + 1e-9, 0.005)
    zs = np.arange(0.40, 1.60 + 1e-9, 0.005)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    P = np.stack([X.ravel(), np.full(X.size, y0), Z.ravel()], 1)
    for mode in ("on", "off"):
        frames_npz[f"slice_{mode}"] = np.asarray(
            fields[mode][0].distance(P), np.float64).reshape(X.shape)
    frames_npz["slice_x"] = xs
    frames_npz["slice_z"] = zs
    frames_npz["slice_y"] = np.float64(y0)

    # RGB (같은 카메라·같은 해상도) — 그림용
    r = mujoco.Renderer(scene.model, height=480, width=640)
    for c in CAMERAS:
        r.update_scene(scene.data, camera=c)
        frames_npz[f"rgb_{c}"] = np.asarray(r.render(), np.uint8).copy()
    r.close()
    # 배치도용: 전 몸 자기필터 구 (x–y) 와 과일 위치
    fc, fr_ = ag_on.robot_model.sphere_centers_numeric(q)
    frames_npz["filter_centres"] = np.asarray(fc, np.float64)
    frames_npz["filter_radii"] = np.asarray(fr_, np.float64)
    objs = {}
    for b in ("apple", "crate", "banana", "orange", "pear"):
        try:
            objs[b] = np.asarray(scene.body_position_in_base(b), np.float64).tolist()
        except Exception as exc:  # noqa: BLE001
            objs[b] = repr(exc)

    np.savez_compressed(OUT / "rm_onoff_raw.npz", **frames_npz)
    commit = subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True).stdout.split("\n")
    cfg = ag_on.config
    out = {
        "record": {"run": REC, "step_file": "step_00000.npz", "t_step": 0,
                   "fingerprint": record_fingerprint(run), "model_xml": str(run.model_xml),
                   "reproduction": "recorded qpos(66)/qvel(61) posed with policy_record.replay_scene+pose_scene, depth rendered 480x640 osmesa via TransportScene.capture (camera_observation)"},
        "code": {"benchmark_head": commit, "working_tree_modified_tracked_files": [l for l in dirty if l.strip()]},
        "serve_safe_cli": {"argv": argv, "parsed": {k: cli[k] for k in sorted(cli)}},
        "ag3s_config": {"esdf": {k: getattr(cfg.esdf, k) for k in ("backend", "voxel_size", "fine_voxel_size", "tsdf_voxel_size", "max_distance", "exclude_support_surfaces", "attached_sign_threshold_voxels") if hasattr(cfg.esdf, k)},
                        "pointcloud": {k: getattr(cfg.pointcloud, k) for k in ("self_filter", "self_filter_inflation", "self_filter_inflation_by_link", "self_filter_target_guard_radius", "range_max", "depth_min", "depth_max")},
                        "constraint_target_field_policy": cfg.constraint.target_field_policy},
        "process_multi_kwargs": {"phase": PHASE, "active_manipulators": MANIPULATORS},
        "attention": {"npz": ATT, "cell": cell, "frame_index": fi},
        "self_filter": {"n_spheres": n_filter, "settings": json.loads(json.dumps(sf, default=str))},
        "robot_mask_per_camera": per_cam,
        "constraint_spheres": spheres,
        "margin_m": MARGIN,
        "eikonal": eik,
        "on_buffer_unchanged_after_off_build": same,
        "slice": {"plane": "x-z", "y_m": y0, "y_rule": "median y of left-arm constraint spheres (arms model)",
                  "x_range_m": [float(xs[0]), float(xs[-1])], "z_range_m": [float(zs[0]), float(zs[-1])],
                  "step_m": 0.005, "query": "CuroboEsdfField.distance (coarse+fine min composite)"},
        "objects_in_base_m": objs,
        "pipeline_status": results,
    }
    (OUT / "rm_onoff_numbers.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: out[k] for k in ("robot_mask_per_camera", "constraint_spheres", "eikonal",
                                          "on_buffer_unchanged_after_off_build", "pipeline_status")},
                     indent=1, default=str))
    print("self-filter spheres", n_filter, "inflation", cfg.pointcloud.self_filter_inflation)


if __name__ == "__main__":
    main()
