"""Import the **official** VLSA/AEGIS helpers from the SafeLIBERO submodule without modifying it.

``benchmark/SafeLIBERO/main/utils.py`` (THU-RCSCT/vlsa-aegis @ 2457feed) is a flat script module: it imports
``robosuite.utils.camera_utils.get_real_depth_map`` at top level (used only by ``get_point_cloud``, which needs a live
robosuite env and is not called here) and ``cvxpy``. When robosuite is not importable (``.venv-baselines``), a stub
module providing that one name is placed in ``sys.modules`` for the duration of the import and removed afterwards.

Used as is (official code, bit-identical): ``compute_h_ij`` (utils.py:78-102), ``compute_h_coeffs_3d`` (104-152),
``vector_hat`` (40-44), ``project_matrix`` (46-49), ``mvee_cvxpy`` (317-351), ``fit_ellipse`` (431-445).
Not used: ``get_point_cloud`` (155-255; needs the env + the original groundingdino package -> ported in
``perception.py``), ``filtering_points`` (257-316; needs open3d -> ported), ``obstacle_detection`` (447-499; GLM-4.5V
via the zhipuai API -> **not allowed**, replaced, see README).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
SAFELIBERO_ROOT = PROJECT_ROOT / "benchmark" / "SafeLIBERO"
OFFICIAL_UTILS = SAFELIBERO_ROOT / "main" / "utils.py"
OFFICIAL_MAIN = SAFELIBERO_ROOT / "main" / "main_aegis.py"
OFFICIAL_REPO = "https://github.com/THU-RCSCT/vlsa-aegis"
OFFICIAL_SHA = "2457feed5968ae803926e178c8ce8243b9ecdcf9"
_MODULE_NAME = "_aegis_official_utils"


def _stub_robosuite() -> list[str]:
    """Insert minimal ``robosuite.utils.camera_utils`` modules if robosuite is missing; return the inserted names."""
    try:
        import robosuite.utils.camera_utils  # noqa: F401

        return []
    except Exception:
        pass
    inserted = []
    for name in ("robosuite", "robosuite.utils", "robosuite.utils.camera_utils"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            inserted.append(name)

    def get_real_depth_map(*_a, **_k):  # pragma: no cover - only get_point_cloud calls it
        raise RuntimeError("robosuite stub: get_real_depth_map needs a live robosuite env")

    sys.modules["robosuite.utils.camera_utils"].get_real_depth_map = get_real_depth_map
    return inserted


def load_official_utils():
    """The official ``main/utils.py`` as a module (cached)."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    if not OFFICIAL_UTILS.is_file():
        raise FileNotFoundError(f"SafeLIBERO submodule not checked out: {OFFICIAL_UTILS}")
    inserted = _stub_robosuite()
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # no __pycache__ inside the (read-only) submodule
    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, OFFICIAL_UTILS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = dont_write
        for name in inserted:
            sys.modules.pop(name, None)
    sys.modules[_MODULE_NAME] = mod
    return mod


def provenance() -> dict:
    from Baseline.paper_baselines.interface import sha256_file

    return {"repo": OFFICIAL_REPO, "sha": OFFICIAL_SHA, "license": "MIT (repo LICENSE, inherited LIBERO copyright)",
            "utils_py": str(OFFICIAL_UTILS.relative_to(PROJECT_ROOT)), "utils_sha256": sha256_file(OFFICIAL_UTILS),
            "main_aegis_py": str(OFFICIAL_MAIN.relative_to(PROJECT_ROOT)), "main_aegis_sha256": sha256_file(OFFICIAL_MAIN),
            "imported_functions": ["compute_h_ij", "compute_h_coeffs_3d", "vector_hat", "project_matrix", "mvee_cvxpy",
                                   "fit_ellipse"]}


__all__ = ["OFFICIAL_MAIN", "OFFICIAL_REPO", "OFFICIAL_SHA", "OFFICIAL_UTILS", "load_official_utils", "provenance"]
