"""Camera frames -> V11AB wire keys (``AG3S.trajopt.wire``), Python 3.8 replica. numpy only.

The LIBERO client runs in ``.venv-libero`` (Python 3.8) and cannot import AG3S (3.11). This module
re-states the camera part of ``AG3S.trajopt.wire.pack_request`` key for key and dtype for dtype:

==============================  =====================================================
``ag3s/stamp_mode``             ``"sim_frozen"`` (one instant per observation)
``ag3s/cameras``                list of wire camera names; order of the arrays below
``ag3s/depth/<cam>``            (H, W) uint16 millimetres, ``clip(rint(d * 1000), 0, 65535)``
``ag3s/K/<cam>``                (3, 3) float64
``ag3s/T_base_cam/<cam>``       (4, 4) float64
``ag3s/robot_state/<cam>``      (nq,) float64
``ag3s/stamp/<cam>``            float
==============================  =====================================================

Policy-request keys that V11AB also carries (``phase``, ``active_manipulators``, ``reset``, ``seq``,
``exec_feedback``) are not produced: the pi0.5-only condition has no AG3S consumer for them.
``tests/benchmark/test_libero_sl1.py`` checks equality with ``pack_request`` and the round trip
through ``unpack_camera_observations``.

CLI (either interpreter)::

    python benchmark/adapters/libero_panda/wire_pack.py --geom <episode>_geom.npz --out-dir DIR [--frames 0,5]

writes one ``<stem>_wire_<index>.npz`` per selected frame (exclusive; never overwrites).
"""
from __future__ import annotations

import argparse
import collections
import os
import pathlib

import numpy as np

PREFIX = "ag3s/"
DEPTH_SCALE_MM = 1000.0
STAMP_MODE = "stamp_mode"
STAMP_MODES = ("sim_frozen", "render_end")
CAMERA_ARRAY_FIELDS = ("depth", "K", "T_base_cam", "robot_state", "stamp")


def depth_to_mm(depth_m):
    """Metric depth -> wire uint16 millimetres (``AG3S.trajopt.wire.pack_request``)."""
    d = np.asarray(depth_m, np.float64)
    return np.clip(np.rint(d * DEPTH_SCALE_MM), 0, 65535).astype(np.uint16)


def pack_cameras(frames, stamp_mode="sim_frozen"):
    """``frames``: ordered mapping wire camera name -> frame with ``depth_m`` (metres) or ``depth_mm``
    (uint16, already quantized), ``K``, ``T_base_cam``, ``robot_state``, ``stamp``.
    """
    if stamp_mode not in STAMP_MODES:
        raise ValueError("stamp_mode must be one of {}, got {!r}".format(STAMP_MODES, stamp_mode))
    cameras = list(frames)
    out = collections.OrderedDict()
    out[PREFIX + STAMP_MODE] = str(stamp_mode)
    out[PREFIX + "cameras"] = list(cameras)
    for cam in cameras:
        frame = frames[cam]
        if "depth_mm" in frame:
            depth = np.asarray(frame["depth_mm"])
            if depth.dtype != np.uint16:
                raise TypeError("depth_mm must be uint16, got {}".format(depth.dtype))
        else:
            depth = depth_to_mm(frame["depth_m"])
        if depth.ndim != 2:
            raise ValueError("depth must be (H, W), got {}".format(depth.shape))
        out["{}depth/{}".format(PREFIX, cam)] = depth
        out["{}K/{}".format(PREFIX, cam)] = np.asarray(frame["K"], np.float64)
        out["{}T_base_cam/{}".format(PREFIX, cam)] = np.asarray(frame["T_base_cam"], np.float64)
        out["{}robot_state/{}".format(PREFIX, cam)] = np.asarray(frame["robot_state"], np.float64)
        out["{}stamp/{}".format(PREFIX, cam)] = float(frame["stamp"])
    return out


# --- npz form ------------------------------------------------------------------------------------
def to_npz_arrays(wire):
    """Wire dict -> arrays storable by ``np.savez`` (str list -> unicode array, float -> 0-d float64)."""
    arrays = collections.OrderedDict()
    for key, value in wire.items():
        if key == PREFIX + "cameras":
            arrays[key] = np.array(list(value), dtype=str)
        elif key == PREFIX + STAMP_MODE:
            arrays[key] = np.array(str(value))
        elif key.startswith(PREFIX + "stamp/"):
            arrays[key] = np.array(float(value), dtype=np.float64)
        else:
            arrays[key] = np.asarray(value)
    return arrays


def from_npz_arrays(arrays):
    """Inverse of :func:`to_npz_arrays` for a loaded npz (or any mapping of arrays)."""
    wire = collections.OrderedDict()
    for key in arrays:
        value = arrays[key]
        if key == PREFIX + "cameras":
            wire[key] = [str(x) for x in np.asarray(value).reshape(-1)]
        elif key == PREFIX + STAMP_MODE:
            wire[key] = str(np.asarray(value).item())
        elif key.startswith(PREFIX + "stamp/"):
            wire[key] = float(np.asarray(value).item())
        else:
            wire[key] = np.asarray(value)
    return wire


def save_wire_npz(path, wire):
    """Exclusive write (``FileExistsError`` if ``path`` exists)."""
    with open(str(path), "xb") as stream:
        np.savez_compressed(stream, **to_npz_arrays(wire))


def load_wire_npz(path):
    with np.load(str(path), allow_pickle=False) as data:
        return from_npz_arrays({key: data[key] for key in data.files})


# --- From the client's per-episode geometry npz --------------------------------------------------
def geom_cameras(geom):
    return [str(x) for x in np.asarray(geom["cameras"]).reshape(-1)]


def frames_from_geom(geom, index):
    """Frame ``index`` of an ``episodes/<stem>_geom.npz`` (mapping of arrays) as wire-ready frames."""
    frames = collections.OrderedDict()
    for cam in geom_cameras(geom):
        frames[cam] = {
            "depth_mm": np.asarray(geom["depth_mm/" + cam][index]),
            "K": geom["K/" + cam][index],
            "T_base_cam": geom["T_base_cam/" + cam][index],
            "robot_state": geom["robot_state/" + cam][index],
            "stamp": float(geom["stamp/" + cam][index]),
        }
    return frames


def wire_from_geom(geom, index):
    stamp_mode = str(np.asarray(geom["stamp_mode"]).item()) if "stamp_mode" in geom else "sim_frozen"
    return pack_cameras(frames_from_geom(geom, index), stamp_mode=stamp_mode)


def _frame_list(text, count):
    if text in (None, "", "all"):
        return list(range(count))
    return [int(item) for item in str(text).replace(",", " ").split()]


def export(geom_path, out_dir, frames="all"):
    geom_path, out_dir = pathlib.Path(geom_path), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with np.load(str(geom_path), allow_pickle=False) as data:
        geom = {key: data[key] for key in data.files}
    count = int(np.asarray(geom["obs_index"]).shape[0])
    stem = geom_path.name[:-len(".npz")] if geom_path.name.endswith(".npz") else geom_path.name
    for index in _frame_list(frames, count):
        if not 0 <= index < count:
            raise ValueError("frame {} outside 0..{}".format(index, count - 1))
        target = out_dir / "{}_wire_{:04d}.npz".format(stem, index)
        save_wire_npz(target, wire_from_geom(geom, index))
        written.append(str(target))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="geometry npz -> per-frame V11AB wire npz")
    parser.add_argument("--geom", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", default="all", help="'all' or comma separated indices")
    args = parser.parse_args(argv)
    written = export(args.geom, args.out_dir, args.frames)
    print("{} wire frames -> {}".format(len(written), os.path.dirname(written[0]) if written else args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
