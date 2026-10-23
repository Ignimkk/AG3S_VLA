"""Compatibility alias: KNOWS moved to ``Baseline.paper_baselines.methods.knows`` (PAPER-METHODS, 2026-10-23).

Kept because code that must not change still names the old package: ``AG3S/ag3s/stages/geometry.py``
(lazy ``from Baseline.knows_vla.perception.ellipsoid_fit import mvee`` -- without this alias it would silently fall
back to the covariance ellipsoid), ``benchmark/__init__.py`` (``benchmark.knows_vla`` legacy alias) and
``scripts/setup/check_workspace.py``. ``Baseline.knows_vla.<sub>`` resolves to the **same module object** as
``Baseline.paper_baselines.methods.knows.<sub>`` (no second copy of the code). New code imports the new name.
"""
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys

_OLD = __name__  # "Baseline.knows_vla"
_NEW = "Baseline.paper_baselines.methods.knows"


class _AliasLoader(Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        module = import_module(self.target)
        self.original_spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self.original_spec


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(_OLD + "."):
            module = import_module(_NEW + fullname[len(_OLD):])
            return spec_from_loader(fullname, _AliasLoader(module.__name__), is_package=hasattr(module, "__path__"))
        return None


if not any(getattr(f, "_knows_alias", False) for f in sys.meta_path):
    _finder = _AliasFinder()
    _finder._knows_alias = True
    sys.meta_path.insert(0, _finder)

# the package itself: expose the new package's public names
_new = import_module(_NEW)
__path__ = []  # submodules come only through the finder above
globals().update({k: v for k, v in vars(_new).items() if not k.startswith("__")})
