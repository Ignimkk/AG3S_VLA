"""Expose canonical package names for an archived benchmark used as a numerical reference.

The algorithm files remain exactly the archived revision. Only package entry
points are added, so migrated tests actually execute the old reference code.
"""
from pathlib import Path
import os

def expose_legacy_benchmark(root):
    root = Path(root)
    legacy = root / "benchmark"
    for package in ["AG3S", "Baseline"]:
        folder = root / package
        folder.mkdir(exist_ok=True)
        (folder / "__init__.py").write_text("")
    for method, package in [("ag3s", "AG3S"), ("trajopt", "AG3S"),
                            ("knows_vla", "Baseline"), ("seam_vla", "Baseline")]:
        source = legacy / method
        dest = root / package / method
        if source.exists() and not dest.exists():
            dest.symlink_to(os.path.relpath(source, dest.parent), target_is_directory=True)
    init = legacy / "__init__.py"
    if init.is_symlink():
        init.unlink()
    project = Path(__file__).resolve().parents[2]
    init.write_text((project / "benchmark/__init__.py").read_text())
