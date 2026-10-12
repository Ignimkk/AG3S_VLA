"""LIBERO/robosuite camera observation -> V11AB camera frame (SL1). Python 3.8 compatible.

What a frame is (one per camera, one instant)::

    depth_m      (H, W) float32  metric z-depth (distance along the optical axis), OpenCV row order
    rgb          (H, W, 3) uint8 the same render, OpenCV row order (separate array; the policy image
                                 is built from the *raw* observation by the client and is untouched)
    K            (3, 3) float64  pinhole intrinsics from the MuJoCo camera fovy
    T_base_cam   (4, 4) float64  camera (OpenCV axes) -> robot base frame
    T_world_cam  (4, 4) float64  camera (OpenCV axes) -> MuJoCo world
    T_world_base (4, 4) float64  robot base frame -> MuJoCo world
    robot_state  (9,) float64    Panda joint qpos (7) + gripper finger qpos (2) at the image instant
                                 (kinematic state the render shows; see "Image-instant q" below)
    stamp        float           MuJoCo ``data.time`` (s); identical for all cameras (stamp_mode sim_frozen)
    sim_state    (1+nq+nv,)      robosuite flattened ``[time, qpos, qvel]`` of the simulator at this substep
    sim_state_image (1+nq+nv,)   the same with ``qpos`` replaced by the image-instant ``qpos``

"This instant" is the physics substep at which robosuite rendered the images (see
:func:`install_capture_observable`), which for an ``env.step`` observation is *earlier* than the
simulator state at ``step`` return.

Conventions (also returned by :func:`conventions` for the run manifest):

* **Robot base frame** = the MuJoCo body ``robot0_base`` (robosuite Panda ``<body name="base">``; its
  child ``robot0_link0`` sits at the same pose). Pose read from ``data.xpos`` / ``data.xmat`` at
  capture time. ``T_base_cam = inv(T_world_base) @ T_world_cam``.
* **Camera axes** = OpenCV (+x right, +y down, +z forward). MuJoCo cameras look along -z with +y up,
  so ``R_world_cam = cam_xmat @ diag(1, -1, -1)`` -- the same correction as robosuite
  ``get_camera_extrinsic_matrix`` and AG3S ``mujoco_source.T_base_cam``.
* **Rows**: robosuite renders with ``macros.IMAGE_CONVENTION == "opengl"`` (first array row = bottom
  of the picture). OpenCV row order is ``raw[::-1]`` (row flip only, columns unchanged).
* **Depth**: robosuite gives the normalized z-buffer in [0, 1]; metric depth is robosuite
  ``get_real_depth_map``: ``near / (1 - d * (1 - near / far))`` with ``near = znear * extent``,
  ``far = zfar * extent``. This is z-depth, the quantity AG3S ``backproject`` expects.
* **K** (SL2, user decision 2026-10-12: pixel-centre principal point) =
  ``f = 0.5 * H / tan(fovy / 2)``, ``cx = (W - 1) / 2``, ``cy = (H - 1) / 2``. Pixel coordinates
  here are indices with pixel centres at integers (AG3S ``backproject`` uses ``u = column index``),
  and in that index convention the optical axis of the symmetric MuJoCo frustum lies at
  ``(W - 1) / 2``. robosuite ``get_camera_intrinsic_matrix`` (:func:`intrinsic_matrix`, kept as the
  restated reference) uses ``W / 2``, i.e. +0.5 px; SL1 measured that offset (head table plane
  +2.0 mm -> +0.8 mm with the pixel-centre K). Convention name: ``K_CONVENTION``.
* **Capture instant**: robosuite samples observations (renders included) inside the physics substep
  loop of ``env.step`` at a substep set by the observables' sampling timers, not at ``step`` return
  (SL1 smoke: one 2 ms substep earlier); poses / robot_state / stamp are therefore taken from a
  read-only observable sampled at the same substep (:func:`install_capture_observable`).
* **Image-instant q** (SL2): inside that substep robosuite calls ``sim.step()`` (= ``mj_step``:
  kinematics from ``qpos_old``, then integration to ``qpos_new``) and then samples. The render and
  every ``xpos`` / ``cam_xpos`` therefore show ``qpos_old`` while ``data.qpos`` already holds
  ``qpos_new`` (SL1 verifier: re-render at ``mj_integratePos(qpos, qvel, -dt)`` is bit-identical
  to the record for 114/114 frames, at ``qpos`` it is not). The observable reconstructs
  ``qpos_image = mj_integratePos(qpos, qvel, -dt)`` and keeps it only if its kinematics reproduce
  ``data.xpos``; otherwise (a forced sample right after ``mj_forward``, e.g. ``reset`` /
  ``set_init_state``) the current ``qpos`` already is the image instant. ``robot_state`` is taken
  from ``qpos_image``; ``sim_state`` stays the raw simulator state (resumable physics) and
  ``sim_state_image = [time, qpos_image, qvel]`` is what reproduces the image after ``forward``.
* **Camera names** (adapter table, AG3S core unchanged): ``agentview -> head``,
  ``robot0_eye_in_hand -> left_wrist`` (pi05_libero feeds the wrist image to the ``left_wrist_0_rgb``
  slot, so SL-3 attention token order matches).

robosuite's ``camera_utils`` module cannot be imported in ``.venv-libero`` (it imports ``h5py``,
which the official requirements pin set does not install), so the three formulas are re-stated here;
``tests/benchmark/libero_py38_checks.py`` compares them value-for-value with the robosuite source.
"""
from __future__ import annotations

import collections

import numpy as np

#: (robosuite camera name, V11AB wire camera name). Order = wire ``ag3s/cameras`` order.
CAMERA_TABLE = (("agentview", "head"), ("robot0_eye_in_hand", "left_wrist"))
BASE_BODY = "robot0_base"
MUJOCO_TO_OPENCV_AXES = np.diag([1.0, -1.0, -1.0])
STAMP_MODE = "sim_frozen"
#: mujoco.mjtObj.mjOBJ_GEOM -- segmentation pixels with this object type carry a geom id.
MJOBJ_GEOM = 5
ROBOT_STATE_LAYOUT = ("robot0_joint1", "robot0_joint2", "robot0_joint3", "robot0_joint4", "robot0_joint5",
                      "robot0_joint6", "robot0_joint7", "gripper0_finger_joint1", "gripper0_finger_joint2")
#: Principal-point convention of the K this adapter writes (SL2): pixel centres at integer indices,
#: ``cx = (W - 1) / 2``, ``cy = (H - 1) / 2``.
K_CONVENTION = "pixel_centre"
#: Which simulator state ``robot_state`` is read from (SL2): the kinematic state of the image.
ROBOT_STATE_INSTANT = "image"
#: ``qpos_image`` source codes written by the capture observable.
QPOS_IMAGE_INTEGRATED_BACK = 1   # mj_integratePos(qpos, qvel, -dt) (sample right after mj_step)
QPOS_IMAGE_CURRENT = 0           # data.qpos (sample right after mj_forward; kinematics already match)
#: Max |xpos| mismatch (m) accepted for the chosen ``qpos_image``; larger -> capture refuses.
QPOS_IMAGE_TOLERANCE_M = 1e-9
#: Panda single arm -> V11AB manipulator side (SL2 item 5). The wrist camera is the ``left_wrist`` slot
#: (``CAMERA_TABLE``), so the arm that carries it is ``left`` -- the L slot of the 16-D layout
#: ``[L7, Lg, R7, Rg]`` (used from SL-5).
MANIPULATOR_SIDE = "left"


def wire_camera_name(source_name):
    return dict(CAMERA_TABLE)[source_name]


def conventions():
    """JSON-able description of the frame conventions (written to the run manifest)."""
    return {
        "camera_table": [{"robosuite": source, "wire": wire} for source, wire in CAMERA_TABLE],
        "camera_table_reason": "pi05_libero transform puts the wrist image in the left_wrist_0_rgb slot",
        "base_frame": "MuJoCo body robot0_base (robosuite Panda <body name='base'>; robot0_link0 at the same "
                      "pose); pose from data.xpos/xmat at capture time",
        "T_base_cam": "inv(T_world_base) @ T_world_cam, T_world_cam rotation = cam_xmat @ diag(1,-1,-1) "
                      "(OpenCV axes: +x right, +y down, +z forward)",
        "depth": "z-depth in metres = near / (1 - d (1 - near/far)), near = znear*extent, far = zfar*extent "
                 "(robosuite get_real_depth_map); OpenCV row order = raw[::-1]; wire: uint16 mm, rint, clip 0..65535",
        "K": "f = 0.5 H / tan(fovy/2), cx = (W-1)/2, cy = (H-1)/2 (pixel-centre principal point: pixel centres "
             "at integer indices, optical axis of the symmetric MuJoCo frustum at (W-1)/2); robosuite "
             "get_camera_intrinsic_matrix uses W/2, H/2 (+0.5 px)",
        "K_convention": K_CONVENTION,
        "image_rows": "robosuite IMAGE_CONVENTION 'opengl': raw row 0 = bottom; OpenCV = raw[::-1]",
        "policy_image": "client: raw[::-1, ::-1] then resize_with_pad 256->224 (array identity: "
                        "policy256[i, j] = opencv[i, W-1-j])",
        "robot_state": list(ROBOT_STATE_LAYOUT),
        "robot_state_instant": "image: qpos whose kinematics the render shows = mj_integratePos(qpos, qvel, -dt) after "
                               "mj_step (data.qpos is one substep later), data.qpos after a forced sample following "
                               "mj_forward; chosen by matching data.xpos (tolerance {} m)".format(QPOS_IMAGE_TOLERANCE_M),
        "manipulator_side": {"panda": MANIPULATOR_SIDE,
                             "reason": "the wrist camera is wired as left_wrist (pi05_libero token slot), so the arm "
                                       "carrying it is the left manipulator; L slot of the 16-D layout [L7, Lg, R7, Rg]"},
        "segmentation": "optional (--capture-segmentation): geom id per pixel (-1 = not a geom), OpenCV rows; rendered "
                        "at env.step return, i.e. at sim_state_at_return, not at the image instant",
        "stamp": "MuJoCo data.time (s) at the camera sampling substep, same for all cameras; stamp_mode " + STAMP_MODE,
        "capture_instant": "robosuite samples observables (renders included) inside the physics substep loop of "
                           "env.step (timer-selected substep), not at step return; T/robot_state/stamp/sim_state "
                           "come from the read-only observable "
                           + CAPTURE_OBSERVABLE + " sampled at the same substep, not from env.sim at step return; "
                           "sim_state = raw simulator state there, sim_state_image = same with the image-instant qpos",
    }


# --- Formulas (robosuite camera_utils, restated) ---------------------------------------------------
def intrinsic_matrix(fovy_deg, height, width):
    """robosuite ``get_camera_intrinsic_matrix`` for a camera with vertical fov ``fovy_deg``."""
    f = 0.5 * height / np.tan(float(fovy_deg) * np.pi / 360)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def pixel_centre_intrinsic_matrix(fovy_deg, height, width):
    """The K this adapter writes (``K_CONVENTION``): robosuite's focal length, principal point at the
    pixel-centre optical axis ``((W - 1) / 2, (H - 1) / 2)``."""
    K = intrinsic_matrix(fovy_deg, height, width)
    K[0, 2] = (width - 1) / 2.0
    K[1, 2] = (height - 1) / 2.0
    return K


def real_depth(depth_normalized, extent, znear, zfar):
    """robosuite ``get_real_depth_map`` (computed in float64)."""
    depth = np.asarray(depth_normalized, dtype=np.float64)
    if not (np.all(depth >= 0.0) and np.all(depth <= 1.0)):
        raise ValueError("depth map is not normalized to [0, 1]")
    far = float(zfar) * float(extent)
    near = float(znear) * float(extent)
    return near / (1.0 - depth * (1.0 - near / far))


def pose(position, rotation):
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(position, np.float64).reshape(3)
    return T


def rigid_inverse(T):
    T = np.asarray(T, np.float64)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def world_camera_pose_opencv(cam_xpos, cam_xmat):
    """robosuite ``get_camera_extrinsic_matrix``: camera (OpenCV axes) -> world."""
    return pose(cam_xpos, np.asarray(cam_xmat, np.float64).reshape(3, 3) @ MUJOCO_TO_OPENCV_AXES)


def opengl_rows_to_opencv(image):
    """Raw robosuite render (``IMAGE_CONVENTION == 'opengl'``) -> OpenCV row order."""
    return np.ascontiguousarray(np.asarray(image)[::-1])


# --- Policy image <-> OpenCV pixel ---------------------------------------------------------------
def raw_to_policy_array(raw):
    """Exactly the client's array operation before resizing (``obs[...][::-1, ::-1]``)."""
    return np.ascontiguousarray(np.asarray(raw)[::-1, ::-1])


def resize_with_pad_geometry(cur_height, cur_width, height, width):
    """Size/pad arithmetic of ``openpi_client.image_tools.resize_with_pad`` (identity if same size).

    Returns ``(resized_h, resized_w, pad_h, pad_w)``; the resized image occupies
    ``[pad_h, pad_h + resized_h) x [pad_w, pad_w + resized_w)`` of the output.
    """
    if (int(cur_height), int(cur_width)) == (int(height), int(width)):
        return int(height), int(width), 0, 0
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    return resized_height, resized_width, pad_height, pad_width


def opencv_to_raw_pixel(u, v, height):
    """OpenCV pixel (u = column, v = row) -> raw OpenGL-order array pixel. Continuous coordinates."""
    return np.asarray(u, np.float64), (height - 1) - np.asarray(v, np.float64)


def opencv_to_policy_pixel(u, v, src_hw=(256, 256), out_hw=(224, 224)):
    """OpenCV pixel of the 256 render -> pixel of the policy image the client sends.

    Continuous coordinates, pixel centres at integer indices. Step 1 is the array identity
    ``policy256[i, j] = opencv[i, W-1-j]`` (from ``raw[::-1, ::-1]`` and ``opencv = raw[::-1]``);
    step 2 is PIL's resampling map ``x_out = (x_in + 0.5) * scale - 0.5`` plus the paste offset.
    """
    H, W = int(src_hw[0]), int(src_hw[1])
    rh, rw, ph, pw = resize_with_pad_geometry(H, W, int(out_hw[0]), int(out_hw[1]))
    u1 = (W - 1) - np.asarray(u, np.float64)
    v1 = np.asarray(v, np.float64)
    return (u1 + 0.5) * (rw / W) - 0.5 + pw, (v1 + 0.5) * (rh / H) - 0.5 + ph


def policy_to_opencv_pixel(x, y, src_hw=(256, 256), out_hw=(224, 224)):
    """Inverse of :func:`opencv_to_policy_pixel`."""
    H, W = int(src_hw[0]), int(src_hw[1])
    rh, rw, ph, pw = resize_with_pad_geometry(H, W, int(out_hw[0]), int(out_hw[1]))
    u1 = (np.asarray(x, np.float64) - pw + 0.5) * (W / rw) - 0.5
    v1 = (np.asarray(y, np.float64) - ph + 0.5) * (H / rh) - 0.5
    return (W - 1) - u1, v1


def project_base_points(points_base, K, T_base_cam):
    """Base-frame points (N, 3) -> (u, v, z_cam) with the same pinhole model as AG3S ``backproject``."""
    pts = np.asarray(points_base, np.float64).reshape(-1, 3)
    T_cam_base = rigid_inverse(T_base_cam)
    cam = pts @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    z = cam[:, 2]
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return u, v, z


# --- Capture from a live env ---------------------------------------------------------------------
#: Name of the read-only observable that snapshots the simulator at the camera sampling instant.
CAPTURE_OBSERVABLE = "ag3s_capture_state"
CAPTURE_MODALITY = "ag3s_capture"


def _inner(env):
    """robosuite environment behind a LIBERO ``ControlEnv`` wrapper (or the env itself)."""
    return getattr(env, "env", env)


def robot_qpos_indexes(env):
    robot = _inner(env).robots[0]
    return [int(i) for i in robot._ref_joint_pos_indexes] + [int(i) for i in robot._ref_gripper_joint_pos_indexes]


def install_capture_observable(env, cameras=CAMERA_TABLE):
    """Add an observable that snapshots the simulator at the camera sampling substep.

    Layout: ``[time, qpos, qvel, T_world_base, T_world_cam per camera, qpos_image,
    qpos_image_source, residual_current_m, residual_integrated_back_m]`` (:func:`unpack_capture_state`).

    Why: robosuite samples every observable (camera renders included) inside the physics substep
    loop of ``env.step``, at the substep where the observables' sampling timers fire, not when
    ``step`` returns. ``env.sim`` at return time can therefore be later than the image (moving wrist
    camera, joints, objects). This observable has the cameras' sampling
    rate, so after ``env.reset()`` (which resets every observable's timer) it samples at exactly the
    same substep as the images. It only reads the simulator (a private scratch ``MjData`` answers the
    image-instant question, see :func:`image_instant_qpos`); it adds the obs keys
    ``ag3s_capture_state`` and ``ag3s_capture-state`` and nothing else. Install it before
    ``env.reset()``; it survives hard resets (robosuite keeps added observables, the sensor reads
    ``env.sim`` at call time). Idempotent.
    """
    from robosuite.utils.observables import Observable, sensor

    inner = _inner(env)
    if CAPTURE_OBSERVABLE in inner._observables:
        return
    sources = [source for source, _ in cameras]
    scratch = {}

    @sensor(modality=CAPTURE_MODALITY)
    def ag3s_capture_state(obs_cache):
        sim = inner.sim
        model, data = sim.model, sim.data
        base_id = model.body_name2id(BASE_BODY)
        parts = [np.array([data.time], np.float64), np.array(data.qpos, np.float64), np.array(data.qvel, np.float64),
                 pose(data.xpos[base_id], data.xmat[base_id]).reshape(-1)]
        for source in sources:
            cam_id = model.camera_name2id(source)
            parts.append(world_camera_pose_opencv(data.cam_xpos[cam_id], data.cam_xmat[cam_id]).reshape(-1))
        qpos_image, code, residuals = image_instant_qpos(model._model, data._data, scratch)
        parts += [qpos_image, np.array([code, residuals[0], residuals[1]], np.float64)]
        return np.concatenate(parts)

    inner.add_observable(Observable(name=CAPTURE_OBSERVABLE, sensor=ag3s_capture_state,
                                    sampling_rate=inner.control_freq))


def image_instant_qpos(m, d, scratch=None):
    """``(qpos_image, source_code, (residual_current_m, residual_integrated_back_m))`` for raw MuJoCo
    ``MjModel`` ``m`` / ``MjData`` ``d``.

    The render shows the kinematics stored in ``d`` (``xpos``, ``cam_xpos``, ``geom_xpos``), which
    ``mj_step`` computed *before* integrating ``qpos``. Two candidates are tested by recomputing
    ``mj_kinematics`` on a scratch ``MjData`` and comparing every body position with ``d.xpos``:
    the current ``d.qpos`` and ``mj_integratePos(d.qpos, d.qvel, -dt)``. The one that reproduces
    ``d.xpos`` is the image instant (ties -> current). ``d`` is not modified.
    """
    import mujoco

    if scratch is None:
        scratch = {}
    work = scratch.get("data")
    if work is None:
        work = scratch["data"] = mujoco.MjData(m)
    current = np.array(d.qpos, np.float64)
    back = current.copy()
    mujoco.mj_integratePos(m, back, np.array(d.qvel, np.float64), -float(m.opt.timestep))
    residuals = []
    for candidate in (current, back):
        work.qpos[:] = candidate
        if m.nmocap:
            work.mocap_pos[:] = d.mocap_pos
            work.mocap_quat[:] = d.mocap_quat
        mujoco.mj_kinematics(m, work)
        residuals.append(float(np.max(np.abs(np.asarray(work.xpos) - np.asarray(d.xpos)))))
    if residuals[0] <= residuals[1]:
        return current, QPOS_IMAGE_CURRENT, tuple(residuals)
    return back, QPOS_IMAGE_INTEGRATED_BACK, tuple(residuals)


def unpack_capture_state(vector, nq, nv, cameras=CAMERA_TABLE):
    """Inverse of the ``ag3s_capture_state`` layout -> dict (sim_state is robosuite's flattened state)."""
    v = np.asarray(vector, np.float64).reshape(-1)
    n_state = 1 + nq + nv
    n_poses = 16 * (1 + len(cameras))
    expected = n_state + n_poses + nq + 3
    if v.size != expected:
        raise ValueError("capture state has {} values, expected {}".format(v.size, expected))
    out = {"sim_state": v[:n_state].copy(), "time": float(v[0]), "qpos": v[1:1 + nq].copy(),
           "T_world_base": v[n_state:n_state + 16].reshape(4, 4).copy(), "T_world_cam": collections.OrderedDict()}
    for k, (source, _) in enumerate(cameras):
        offset = n_state + 16 * (1 + k)
        out["T_world_cam"][source] = v[offset:offset + 16].reshape(4, 4).copy()
    offset = n_state + n_poses
    out["qpos_image"] = v[offset:offset + nq].copy()
    out["qpos_image_source"] = int(v[offset + nq])
    out["qpos_image_residual_m"] = v[offset + nq + 1:offset + nq + 3].copy()
    out["sim_state_image"] = out["sim_state"].copy()
    out["sim_state_image"][1:1 + nq] = out["qpos_image"]
    return out


def capture(env, obs, cameras=CAMERA_TABLE, segmentation=False):
    """Per-camera frames for the observation ``obs`` that ``env`` just returned.

    Poses, ``robot_state`` and ``stamp`` come from the ``ag3s_capture_state`` snapshot in ``obs``
    (see :func:`install_capture_observable`), i.e. the instant the images were rendered -- not from
    ``env.sim`` at return time. ``robot_state`` is the image-instant qpos (module docstring); a
    snapshot where neither candidate reproduces the rendered kinematics raises. Reads only: ``obs`` arrays are not modified, the sim is not stepped.
    ``obs`` must come from an env created with ``camera_depths=True``.

    ``segmentation`` renders one extra segmentation pass per camera *now* (``sim.render(...,
    segmentation=True)`` as robosuite ``get_camera_segmentation``); for an ``env.step`` observation
    that is the return-time state, not the image instant -- use it for static scene parts or
    regenerate from ``sim_state``. Per-pixel geom ids, -1 = not a geom.
    Returns an ``OrderedDict`` wire name -> frame dict, in ``cameras`` order.
    """
    import robosuite.macros as macros

    if macros.IMAGE_CONVENTION != "opengl":
        raise RuntimeError("expected robosuite IMAGE_CONVENTION 'opengl', got {!r}".format(macros.IMAGE_CONVENTION))
    if CAPTURE_OBSERVABLE not in obs:
        raise KeyError("{} missing: call install_capture_observable(env) before env.reset()".format(CAPTURE_OBSERVABLE))
    sim = env.sim
    model = sim.model
    extent, znear, zfar = float(model.stat.extent), float(model.vis.map.znear), float(model.vis.map.zfar)
    snapshot = unpack_capture_state(obs[CAPTURE_OBSERVABLE], int(model.nq), int(model.nv), cameras)
    chosen = float(np.min(snapshot["qpos_image_residual_m"]))
    if not chosen <= QPOS_IMAGE_TOLERANCE_M:
        raise RuntimeError("image-instant qpos not found: neither data.qpos nor integratePos(-dt) reproduces "
                           "data.xpos (residuals {} m)".format(snapshot["qpos_image_residual_m"].tolist()))
    T_world_base = snapshot["T_world_base"]
    T_base_world = rigid_inverse(T_world_base)
    robot_state = snapshot["qpos_image"][robot_qpos_indexes(env)].astype(np.float64)

    frames = collections.OrderedDict()
    for source, wire in cameras:
        key = source + "_depth"
        if key not in obs:
            raise KeyError("{} missing: create the env with camera_depths=True".format(key))
        depth_raw = np.asarray(obs[key])[..., 0]
        height, width = depth_raw.shape
        cam_id = model.camera_name2id(source)
        T_world_cam = snapshot["T_world_cam"][source]
        frame = {
            "source_camera": source,
            "depth_m": opengl_rows_to_opencv(real_depth(depth_raw, extent, znear, zfar)).astype(np.float32),
            "rgb": opengl_rows_to_opencv(obs[source + "_image"]),
            "K": pixel_centre_intrinsic_matrix(model.cam_fovy[cam_id], height, width),
            "T_world_cam": T_world_cam,
            "T_world_base": T_world_base.copy(),
            "T_base_cam": T_base_world @ T_world_cam,
            "robot_state": robot_state.copy(),
            "stamp": snapshot["time"],
            "sim_state": snapshot["sim_state"],
            "sim_state_image": snapshot["sim_state_image"],
            "qpos_image_source": snapshot["qpos_image_source"],
            "qpos_image_residual_m": snapshot["qpos_image_residual_m"],
            "fovy_deg": float(model.cam_fovy[cam_id]),
            "near_m": znear * extent,
            "far_m": zfar * extent,
        }
        if segmentation:
            seg = sim.render(camera_name=source, height=height, width=width, segmentation=True)
            seg = opengl_rows_to_opencv(seg)
            frame["segmentation_geom"] = np.where(seg[..., 0] == MJOBJ_GEOM, seg[..., 1], -1).astype(np.int16)
        frames[wire] = frame
    return frames


def model_tables(env):
    """Static name tables that make segmentation ids and ``sim_state`` readable offline."""
    model = env.sim.model
    return {
        "model_geom_names": np.array([model.geom_id2name(i) or "" for i in range(model.ngeom)], dtype=str),
        "model_geom_bodyid": np.asarray(model.geom_bodyid, dtype=np.int32).copy(),
        "model_body_names": np.array([model.body_id2name(i) or "" for i in range(model.nbody)], dtype=str),
        "model_joint_names": np.array([model.joint_id2name(i) or "" for i in range(model.njnt)], dtype=str),
        "robot_qpos_indexes": np.asarray(robot_qpos_indexes(env), dtype=np.int64),
    }
