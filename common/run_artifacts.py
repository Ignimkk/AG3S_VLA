"""Allocate benchmark outputs and preserve run provenance without touching inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from common.workspace import PROJECT_ROOT, shared_storage_root


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(PROJECT_ROOT), *args], text=True).strip()


def results_root() -> Path:
    return shared_storage_root() / 'results' / 'benchmark'


def output_path(path: str | Path) -> Path:
    """Require a destination strictly below benchmark results, resolving symlinks."""
    resolved = Path(path).expanduser().resolve()
    base = results_root().resolve()
    if not resolved.is_relative_to(base) or resolved == base:
        raise ValueError(f'Output must be below {base}: {resolved}')
    return resolved


def create_run(*, run_id: str | None = None, settings: dict | None = None,
               checkpoint: str | None = None) -> Path:
    run_id = run_id or uuid.uuid4().hex[:12]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', run_id):
        raise ValueError('run_id must be a single alphanumeric path component')
    now = datetime.now(timezone(timedelta(hours=9), 'KST'))
    commit = git('rev-parse', 'HEAD')
    path = output_path(results_root() / f'{now:%Y%m%dT%H%M%S}_{commit[:12]}_{run_id}')
    path.mkdir(parents=True, exist_ok=False)
    for name in ['videos', 'logs']:
        (path / name).mkdir()
    source = json.loads((PROJECT_ROOT / 'doc/migration/source-manifest.json').read_text())
    baseline = {}
    for name in ['benchmark', 'pi05_TO_hybrid']:
        ref = f'legacy/{name}/v2'
        baseline[name] = {'ref': ref, 'tag_object': git('rev-parse', ref),
                          'commit': git('rev-parse', ref + '^{commit}')}
    diff = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'diff', '--binary', 'HEAD'])
    (path / 'source.patch').write_bytes(diff)
    untracked = git('ls-files', '--others', '--exclude-standard').splitlines()
    manifest = {
        'schema_version': 1, 'created_at': now.isoformat(), 'run_id': run_id,
        'checkout': str(PROJECT_ROOT), 'branch': git('branch', '--show-current'),
        'commit': commit, 'dirty': bool(git('status', '--porcelain')),
        'purpose': 'evaluation/layout', 'profile': source['profile'],
        'source_diff_sha256': hashlib.sha256(diff).hexdigest(),
        'untracked_sha256': {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                             for name in untracked if (PROJECT_ROOT / name).is_file()},
        'baseline_v2': baseline, 'source_basis': source['sources'],
        'dependencies': source['dependencies'],
        'algorithm_trees': {name: git('rev-parse', 'HEAD:' + name)
                            for name in ['AG3S/ag3s', 'AG3S/trajopt', 'Baseline']},
        'settings': settings or {},
        'checkpoint': checkpoint,
        'storage': {'root': str(shared_storage_root()), 'run': str(path),
                    'videos': str(path / 'videos'), 'logs': str(path / 'logs'), 'rollouts': str(path / 'runs'),
                    'policy_records': str(path / 'policy_records')},
        'executions': [],
    }
    with (path / 'run_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return path


def record_execution(run: str | Path, *, settings: dict, checkpoint: str | None = None) -> Path:
    path = output_path(run)
    with (path / '.manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = path / 'run_manifest.json'
        manifest = json.loads(target.read_text())
        if manifest['checkout'] != str(PROJECT_ROOT) or manifest['commit'] != git('rev-parse', 'HEAD'):
            raise ValueError('Run belongs to another checkout or commit; allocate a new run')
        manifest['executions'].append({'at': datetime.now(timezone(timedelta(hours=9), 'KST')).isoformat(),
                                       'settings': settings, 'checkpoint': checkpoint,
                                       'python': sys.executable})
        temporary = path / ('.manifest-' + uuid.uuid4().hex + '.json')
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(target)
    return path


def ensure_run(*, settings: dict, checkpoint: str | None = None) -> Path:
    selected = os.environ.get('AG3S_RUN_DIR')
    return (record_execution(selected, settings=settings, checkpoint=checkpoint) if selected
            else create_run(settings=settings, checkpoint=checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    parser.add_argument('--checkpoint')
    parser.add_argument('--settings-json', default='{}')
    parser.add_argument('--validate-output')
    parser.add_argument('--command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.validate_output:
        path = output_path(args.validate_output)
        if os.environ.get("AG3S_RUN_DIR") and not path.is_relative_to(output_path(os.environ["AG3S_RUN_DIR"])):
            raise ValueError("Output must belong to the selected run")
        print(path)
    else:
        settings = json.loads(args.settings_json)
        if args.command:
            settings["argv"] = args.command
            settings["environment"] = {key: os.environ.get(key) for key in
                ["XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "OPENPI_DATA_HOME", "MUJOCO_GL", "PYTHONPATH"]}
        path = (create_run(run_id=args.run_id, settings=settings, checkpoint=args.checkpoint)
                if args.run_id else ensure_run(settings=settings, checkpoint=args.checkpoint))
        print(path)


if __name__ == '__main__':
    main()
