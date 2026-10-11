"""Run a Python module with old-checkout file reads denied, including Python children."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
args = sys.argv[1:]
if len(args) < 2 or args[0] != "-m":
    raise SystemExit("Usage: python scripts/setup/run_without_legacy.py -m pytest ...")
audit_dir = ROOT / ".runtime/no-legacy-audit"
audit_dir.mkdir(parents=True, exist_ok=True)
(audit_dir / "sitecustomize.py").write_text("from scripts.setup.no_legacy_audit import install\ninstall()\n")
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join([str(audit_dir), str(ROOT), env.get("PYTHONPATH", "")])
# Prove that the denial is active before accepting any test result.
probe = """from pathlib import Path
try:
    Path('/mnt/dev/work/CLAUDE.md').read_bytes()
except PermissionError:
    print('old-checkout read denial active')
else:
    raise SystemExit('old-checkout audit hook was not active')
"""
subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, check=True)
raise SystemExit(subprocess.run([sys.executable, *args], cwd=ROOT, env=env).returncode)
