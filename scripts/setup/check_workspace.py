"""Read-only checkout, branch, dependency and import-origin check."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "integration/rby1/manipulation/src"),
                str(ROOT / "third_party/openpi/src"),
                str(ROOT / "third_party/openpi/packages/openpi-client/src")]

def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

assert Path(git("rev-parse", "--show-toplevel")) == ROOT
manifest = json.loads((ROOT / "doc/migration/source-manifest.json").read_text())
modules = {}
for module in ["AG3S.ag3s", "AG3S.trajopt", "Baseline.seam_vla", "Baseline.knows_vla", "rby1_manipulation", "openpi"]:
    spec = importlib.util.find_spec(module)
    assert spec is not None, module
    locations = ([spec.origin] if spec.origin else list(spec.submodule_search_locations or []))
    for location in locations:
        assert Path(location).resolve().is_relative_to(ROOT), (module, location)
    modules[module] = locations
for relative, dependency in manifest["dependencies"].items():
    actual = subprocess.check_output(["git", "-C", str(ROOT / relative), "rev-parse", "HEAD"], text=True).strip()
    assert actual == dependency["commit"], (relative, actual)
assert (ROOT / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml").is_file()
assert (ROOT / "third_party/openpi/third_party/libero").resolve() == (ROOT / "benchmark/LIBERO").resolve()
assert (ROOT / "third_party/openpi/third_party/aloha").resolve() == (ROOT / "third_party/aloha").resolve()
for file in (ROOT / ".claude").rglob("*.json"):
    settings = json.loads(file.read_text())
    for rules in settings.get("permissions", {}).values():
        for rule in rules:
            assert "${" not in rule and "/${AG3S_PROJECT_ROOT}/" not in rule, (file, rule)
print(json.dumps({"root":str(ROOT), "branch":git("branch", "--show-current"),
                  "head":git("rev-parse", "HEAD"), "origin":git("remote", "get-url", "origin"),
                  "profile":manifest["profile"], "modules":modules,
                  "dependency_count":len(manifest["dependencies"])}, ensure_ascii=False, indent=2))
