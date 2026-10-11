"""Benchmark tools, with a compatibility import bridge for recorded legacy launchers.

Method files live only in AG3S/ and Baseline/. Legacy Python names resolve to the
same canonical module objects; this does not duplicate code or import another checkout.
"""
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys

_ALIASES = {"benchmark.ag3s": "AG3S.ag3s", "benchmark.trajopt": "AG3S.trajopt",
            "benchmark.knows_vla": "Baseline.knows_vla", "benchmark.seam_vla": "Baseline.seam_vla"}

class _LegacyLoader(Loader):
    def __init__(self, target):
        self.target = target
    def create_module(self, spec):
        module = import_module(self.target)
        self.original_spec = module.__spec__
        return module
    def exec_module(self, module):
        module.__spec__ = self.original_spec

class _LegacyFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for legacy, canonical in _ALIASES.items():
            if fullname == legacy or fullname.startswith(legacy + "."):
                module = import_module(canonical + fullname[len(legacy):])
                return spec_from_loader(fullname, _LegacyLoader(module.__name__), is_package=hasattr(module, "__path__"))
        return None

sys.meta_path.insert(0, _LegacyFinder())

def __getattr__(name):
    target = _ALIASES.get(__name__ + "." + name)
    if target is None:
        raise AttributeError(name)
    return import_module(target)
