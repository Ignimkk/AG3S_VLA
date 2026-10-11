"""Read-only audit of legacy processes, paths and missing shared dataset aliases."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

from common.run_artifacts import ensure_run
from common.workspace import PROJECT_ROOT, shared_storage_root


def main() -> None:
    result = {'at': datetime.now(timezone(timedelta(hours=9))).isoformat(),
              'old_processes': [], 'inherited_old_environment': {}, 'old_source_imports': []}
    old = '/mnt/dev/work'
    ancestors = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            status = (Path('/proc') / str(pid) / 'status').read_text()
            pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
        except (OSError, StopIteration):
            break
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestors:
            continue
        try:
            cwd = os.readlink(proc / 'cwd')
            exe = os.readlink(proc / 'exe')
            cmd = (proc / 'cmdline').read_bytes()
            if cwd.startswith(old) or exe.startswith(old) or old.encode() in cmd:
                result['old_processes'].append({
                    'pid': int(proc.name), 'cwd': cwd, 'exe': exe,
                    'argv0': cmd.split(b'\0')[0].decode(errors='replace'),
                    'old_in_command': old.encode() in cmd,
                    'python_discovery': b'normalize' in cmd and b'sys.path' in cmd})
        except (OSError, ProcessLookupError):
            pass
    for key in ['VIRTUAL_ENV', 'PYTHONPATH', 'PATH', 'OPENPI_DATA_HOME', 'HF_HOME',
                'LD_LIBRARY_PATH', 'AG3S_OPENPI_PYTHON']:
        value = os.environ.get(key, '')
        if old in value:
            result['inherited_old_environment'][key] = value
    aliases = ['datasets/lerobot-atomic/local/rby1_atomic_basket_14d_v2',
               'datasets/lerobot-transport/local/rby1_dataset_v1']
    result['missing_dataset_aliases'] = [str(shared_storage_root() / name) for name in aliases
                                         if not (shared_storage_root() / name).exists()]
    result['hurdle_configs'] = [str(p) for p in sorted(
        (shared_storage_root() / 'results/workspace/verify/T43/P2/configs').glob('*.json'))]
    for name, module in sys.modules.copy().items():
        path = getattr(module, '__file__', None)
        if path and Path(path).resolve().is_relative_to(Path(old)):
            result['old_source_imports'].append([name, path])
    result['remaining_postprocessing_helpers'] = [
        'T43/CT/p/replay_ct.py', 'T43/W/p/analyze_t39.py', 'T43/W/p/hurdle_metrics.py',
        'T43/W/p/hold_w.py', 'T43/DX/p/dx_common.py']
    run = ensure_run(settings={'audit': 'runtime-dependencies'})
    (run / 'dependency-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
