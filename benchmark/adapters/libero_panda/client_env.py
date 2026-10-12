"""Process environment for the ``.venv-libero`` (Python 3.8) side. Python 3.8 compatible, stdlib only.

The pod exports ``LD_LIBRARY_PATH`` entries inside a foreign Python's ``dist-packages`` (e.g.
``/usr/local/lib/python3.10/dist-packages/torch/lib``), and ``scripts/setup/env.sh`` puts the checkout
(Python 3.11 sources) on ``PYTHONPATH``. Under that environment the venv's torch 1.11 loads a foreign
``libtorch_python.so`` (undefined symbol ``PyObject_GET_WEAKREFS_LISTPTR``) as soon as LIBERO is imported.

``launch_libero_pi05_only.client_environment`` (SL0) fixed this for the client it spawns. This module is
that function, moved here so a **py3.8 CLI entry point** can apply it to itself (SL3 0.2): the dynamic
loader reads ``LD_LIBRARY_PATH`` once at process start, so a running interpreter cannot repair it in place
-- :func:`reexec_with_client_environment` re-executes the same interpreter and argv with the sanitized
environment (once; guarded by :data:`REEXEC_SENTINEL`). The launcher delegates to the same functions, so
the two cannot drift.
"""
from __future__ import annotations

import os
import pathlib
import sys

#: Worker threads per BLAS / torch / numba / llvmpipe (OSMesa) pool on the shared CPU pod.
CLIENT_THREADS = 4
THREAD_LIMIT_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
                     "LP_NUM_THREADS")
#: Set in the re-executed process; a second re-exec is never attempted.
REEXEC_SENTINEL = "AG3S_LIBERO_CLIENT_ENV"
#: This checkout (``benchmark/adapters/libero_panda/client_env.py`` -> parents[3]).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


def sanitize_library_path(value, project_root=PROJECT_ROOT):
    """Drop ``LD_LIBRARY_PATH`` entries inside a foreign Python's site/dist-packages.

    Driver / CUDA entries are kept; entries below ``project_root`` (this checkout's own venvs) are kept.
    """
    root = str(project_root) + "/"
    kept = []
    for entry in (value or "").split(":"):
        if not entry:
            continue
        foreign = ("/site-packages/" in entry + "/" or "/dist-packages/" in entry + "/") and not (
            entry + "/").startswith(root)
        if not foreign:
            kept.append(entry)
    return ":".join(kept)


def client_environment(base=None, project_root=PROJECT_ROOT):
    """The environment the py3.8 LIBERO side runs in (SL0 client rules, unchanged)."""
    env = dict(os.environ if base is None else base)
    # The py3.8 side relies on its own editable installs; the checkout PYTHONPATH (py3.11 sources) is dropped.
    env.pop("PYTHONPATH", None)
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = sanitize_library_path(env["LD_LIBRARY_PATH"], project_root)
    env["MUJOCO_GL"] = env.get("MUJOCO_GL") or "osmesa"
    # Shared CPU pod: cap BLAS/torch/numba/llvmpipe(OSMesa) worker threads unless already set.
    for key in THREAD_LIMIT_KEYS:
        env.setdefault(key, str(CLIENT_THREADS))
    env.setdefault("LIBERO_CONFIG_PATH", str(pathlib.Path(project_root) / ".runtime/libero"))
    return env


def needs_reexec(environ=None, project_root=PROJECT_ROOT):
    """True if the current environment differs from :func:`client_environment` (and no re-exec yet)."""
    environ = dict(os.environ if environ is None else environ)
    if environ.get(REEXEC_SENTINEL) == "1":
        return False
    return client_environment(environ, project_root) != environ


def reexec_with_client_environment(argv=None, project_root=PROJECT_ROOT):
    """Replace this process with the same interpreter + argv under :func:`client_environment`.

    Call it from a CLI entry point **before** importing numpy / mujoco / LIBERO. No-op when the
    environment is already clean or when this process is the re-executed one.
    """
    if not needs_reexec(project_root=project_root):
        return
    env = client_environment(None, project_root)
    env[REEXEC_SENTINEL] = "1"
    argv = list(sys.argv if argv is None else argv)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable] + argv, env)
