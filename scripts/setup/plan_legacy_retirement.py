#!/usr/bin/env python3
"""Read-only, fail-closed plan for historical links; never follows subtree links.

Only Git metadata, bytecode and named caches/environments are excluded. Git
ignore rules do not exclude files: ignored source and runtime launchers matter.
The caller owns mutation, Git-state preservation, and independent verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

EXCLUDED_NAMES = frozenset({
    '.git', '__pycache__', '.cache', '.pytest_cache', '.mypy_cache',
    '.ruff_cache', '.uv-cache', '.tox', '.nox',
})
EXCLUDED_PREFIXES = ('.venv',)
EXCLUDED_SUFFIXES = ('.pyc', '.pyo')
DEFAULT_INPUT_ROOTS = tuple(Path('/mnt/dev/AG3S_VLA-storage') / name for name in (
    'assets', 'datasets', 'checkpoints', 'recordings', 'legacy-inputs',
))


def excluded(name: str) -> bool:
    return (name in EXCLUDED_NAMES or name.startswith(EXCLUDED_PREFIXES)
            or name.endswith(EXCLUDED_SUFFIXES))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def beneath(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def normalized(path: Path) -> Path:
    """Lexical normalization, deliberately without dereferencing symlinks."""
    return Path(os.path.abspath(path))


def inventory_tree(root: Path, *, legacy_root: Path, archive_root: Path,
                   allowed_inputs: tuple[Path, ...]) -> dict[str, Any]:
    """Hash regular files, record links, and reject unsafe/unsupported nodes."""
    canonical_subtree = root
    if beneath(root, legacy_root):
        canonical_subtree = archive_root / root.relative_to(legacy_root)
    entries: dict[str, Any] = {}
    omissions: list[str] = []
    issues: list[dict[str, str]] = []

    def visit(path: Path, rel: str) -> None:
        try:
            info = path.lstat()
        except OSError as error:
            issues.append({'path': rel, 'reason': str(error)})
            return
        if stat.S_ISLNK(info.st_mode):
            raw = os.readlink(path)
            target = normalized(path.parent / raw)
            canonical = target
            if beneath(target, legacy_root):
                canonical = archive_root / target.relative_to(legacy_root)
            safe = beneath(canonical, canonical_subtree) or any(
                beneath(canonical, item) for item in allowed_inputs)
            resolved = path.resolve(strict=False)
            resolved_canonical = resolved
            if beneath(resolved, legacy_root):
                resolved_canonical = archive_root / resolved.relative_to(legacy_root)
            safe_resolved = beneath(resolved_canonical, canonical_subtree) or any(
                beneath(resolved_canonical, item) for item in allowed_inputs)
            entries[rel] = {'kind': 'symlink', 'target': str(canonical)}
            if not safe or not safe_resolved:
                issues.append({'path': rel, 'reason': 'symlink escapes immutable archive/input roots'})
            if not path.exists():
                issues.append({'path': rel, 'reason': 'broken symlink'})
            if beneath(root, archive_root) and beneath(target, legacy_root):
                issues.append({'path': rel, 'reason': 'archive symlink still depends on legacy checkout'})
        elif stat.S_ISDIR(info.st_mode):
            entries[rel] = {'kind': 'directory'}
            with os.scandir(path) as children:
                names = sorted(child.name for child in children)
            for name in names:
                child_rel = name if rel == '.' else f'{rel}/{name}'
                if excluded(name):
                    omissions.append(child_rel)
                else:
                    visit(path / name, child_rel)
        elif stat.S_ISREG(info.st_mode):
            entries[rel] = {'kind': 'file', 'sha256': sha256(path),
                            'size': info.st_size, 'executable': bool(info.st_mode & 0o111)}
        else:
            issues.append({'path': rel, 'reason': 'unsupported special file'})

    # Fail closed if a parent of the requested subtree redirects elsewhere.
    expected = normalized(root)
    if root.parent.resolve() != root.parent:
        issues.append({'path': '.', 'reason': 'subtree parent resolves through a symlink'})
    visit(expected, '.')
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    return {'entries': entries, 'digest': digest, 'excluded_paths': omissions, 'issues': issues}


def compare_trees(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old, new = before['entries'], after['entries']
    return {
        'missing_from_archive': sorted(old.keys() - new.keys()),
        'extra_in_archive': sorted(new.keys() - old.keys()),
        'different': sorted(key for key in old.keys() & new.keys() if old[key] != new[key]),
        'source_issues': before['issues'],
        'archive_issues': after['issues'],
    }


def build_plan(data: dict[str, Any], archive: Path, *, legacy: Path = Path('/mnt/dev/work'),
               allowed_inputs: tuple[Path, ...] = DEFAULT_INPUT_ROOTS) -> dict[str, Any]:
    archive, legacy = normalized(archive), normalized(legacy)
    allowed_inputs = tuple(normalized(item) for item in allowed_inputs)
    if any(item.resolve() != item for item in allowed_inputs):
        raise ValueError('allowed input roots must be physical paths')
    if archive.resolve() != archive or legacy.resolve() != legacy:
        raise ValueError('archive and legacy roots must be physical absolute paths')
    if beneath(archive, legacy) or beneath(legacy, archive):
        raise ValueError('archive and legacy roots must be independent')
    archive_lower = str(archive).lower()
    if 'development' in archive_lower or 'benchmark' in archive.name.lower():
        raise ValueError('target must be a frozen archive, not an active checkout')
    links = data.get('incoming_legacy_links')
    if not isinstance(links, list):
        raise ValueError('inventory must contain incoming_legacy_links array')
    seen: set[str] = set()
    subtrees: dict[str, Any] = {}
    replacements: list[dict[str, Any]] = []
    for item in links:
        path = Path(item['path'])
        if not path.is_absolute() or normalized(path) != path or str(path) in seen:
            raise ValueError(f'invalid/duplicate link path: {path}')
        seen.add(str(path))
        if beneath(path, legacy) or beneath(path, archive):
            raise ValueError(f'incoming link must be outside source/archive: {path}')
        recorded_raw = item['target']
        source = normalized(path.parent / recorded_raw)
        if not beneath(source, legacy) or source == legacy:
            raise ValueError(f'link target is not a legacy subtree: {source}')
        relative = str(source.relative_to(legacy))
        target = archive / relative
        if relative not in subtrees:
            old = inventory_tree(source, legacy_root=legacy, archive_root=archive,
                                 allowed_inputs=allowed_inputs)
            new = inventory_tree(target, legacy_root=legacy, archive_root=archive,
                                 allowed_inputs=allowed_inputs)
            differences = compare_trees(old, new)
            subtrees[relative] = {
                'source': str(source), 'archive': str(target),
                'source_inventory': old, 'archive_inventory': new,
                'differences': differences, 'ready': not any(differences.values()),
            }
        errors: list[str] = []
        if not path.is_symlink():
            errors.append('incoming path is not a symlink')
        elif os.readlink(path) != recorded_raw:
            errors.append('incoming symlink changed since inventory')
        if path.parent.resolve() != path.parent:
            errors.append('incoming symlink parent redirects through symlink')
        ready = subtrees[relative]['ready'] and not errors
        replacements.append({'path': str(path), 'old_target': recorded_raw,
                             'new_target': str(target), 'subtree': relative,
                             'errors': errors, 'ready': ready})
    return {
        'schema_version': 1, 'mode': 'read-only-plan',
        'legacy_root': str(legacy), 'archive_root': str(archive),
        'allowed_input_roots': [str(p) for p in allowed_inputs],
        'exclusion_policy': {'exact_names': sorted(EXCLUDED_NAMES),
                             'prefixes': list(EXCLUDED_PREFIXES),
                             'suffixes': list(EXCLUDED_SUFFIXES),
                             'gitignore_used': False, 'follow_subtree_symlinks': False},
        'subtrees': subtrees, 'replacements': replacements,
        'ready_count': sum(item['ready'] for item in replacements),
        'blocked_count': sum(not item['ready'] for item in replacements),
    }


def validate_plan_current(plan: dict[str, Any]) -> None:
    """Revalidate a generated plan immediately before caller-owned mutations.

    This does not write anything, guarantee concurrency exclusion, or replace
    the caller's responsibility for atomic link replacement and Git snapshots.
    """
    if plan['blocked_count'] or any(not row['ready'] for row in plan['replacements']):
        raise ValueError('plan has blocked replacements')
    inventory = {'incoming_legacy_links': [
        {'path': row['path'], 'target': row['old_target']} for row in plan['replacements']]}
    fresh = build_plan(inventory, Path(plan['archive_root']), legacy=Path(plan['legacy_root']),
                       allowed_inputs=tuple(Path(p) for p in plan['allowed_input_roots']))
    if fresh['blocked_count']:
        raise ValueError('plan is no longer safe')
    for key, subtree in plan['subtrees'].items():
        for side in ('source_inventory', 'archive_inventory'):
            if subtree[side]['digest'] != fresh['subtrees'][key][side]['digest']:
                raise ValueError(f'subtree changed after planning: {key}')



BACKLINK_NAMESPACES = frozenset({
    'checkpoints', 'lerobot_transport', 'lerobot_atomic_20260825',
    'cache', 'backups', 'data', 'outputs', 'logs',
})


def observed_backlink_targets(storage_root: Path) -> dict[str, Path]:
    """Explicit opt-in list for the eight observed legacy backlinks only."""
    suffixes = {
        'checkpoints': 'checkpoints/rby1',
        'lerobot_transport': 'datasets/lerobot-transport',
        'lerobot_atomic_20260825': 'datasets/lerobot-atomic',
        'cache': 'cache/rby1', 'backups': 'backups/rby1',
        'data': 'datasets/rby1', 'outputs': 'results/rby1',
        'logs': 'results/rby1-logs',
    }
    return {name: storage_root / suffix for name, suffix in suffixes.items()}


def node_identity(path: Path, *, include_mtime: bool = True) -> dict[str, int]:
    info = path.lstat()
    result = {'device': info.st_dev, 'inode': info.st_ino, 'mode': info.st_mode}
    if include_mtime:
        result['mtime_ns'] = info.st_mtime_ns
    return result


def build_backlink_plan(rows: list[dict[str, Any]], *,
                        allowed_targets: dict[str, Path],
                        legacy: Path = Path('/mnt/dev/work')) -> dict[str, Any]:
    """Plan bypass of exactly one old backlink, preserving target node identity.

    Unlike frozen source comparison, no input/output directory is traversed or
    hashed. Caller must explicitly opt into each historical namespace target;
    this allowlist is separate from source archival's immutable input policy.
    """
    legacy = normalized(legacy)
    if legacy.resolve() != legacy:
        raise ValueError('legacy root must be physical')
    if not isinstance(rows, list):
        raise ValueError('backlink inventory must be an array')
    allowed: dict[str, Path] = {}
    for name, target in allowed_targets.items():
        if name not in BACKLINK_NAMESPACES:
            raise ValueError(f'unsupported historical namespace: {name}')
        target = normalized(target)
        if beneath(target, legacy) or target.resolve() != target or not target.exists():
            raise ValueError(f'allowed target must be existing physical external storage: {target}')
        allowed[name] = target
    seen: set[str] = set()
    replacements: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row['path'])
        if not path.is_absolute() or normalized(path) != path or str(path) in seen:
            raise ValueError(f'invalid/duplicate incoming backlink: {path}')
        seen.add(str(path))
        if beneath(path, legacy):
            raise ValueError('incoming backlink must be outside legacy checkout')
        old_raw = row['target']
        old = normalized(path.parent / old_raw)
        name = old.name
        if old != legacy / 'pi05_TO_hybrid' / name or name not in allowed:
            raise ValueError(f'legacy backlink is not an opted-in namespace: {old}')
        target = allowed[name]
        errors: list[str] = []
        result: dict[str, Any] = {
            'path': str(path), 'old_target': old_raw, 'legacy_backlink': str(old),
            'new_target': str(target), 'namespace': name,
        }
        try:
            incoming_before = node_identity(path)
            old_before = node_identity(old)
            target_before = node_identity(target, include_mtime=False)
            if not path.is_symlink() or os.readlink(path) != old_raw:
                errors.append('incoming backlink is missing, nonsymlink, or changed')
            if path.parent.resolve() != path.parent or old.parent.resolve() != old.parent:
                errors.append('incoming/legacy backlink parent redirects through symlink')
            if not old.is_symlink():
                errors.append('legacy target must be a single symlink')
            else:
                legacy_raw = os.readlink(old)
                direct = normalized(old.parent / legacy_raw)
                result['legacy_raw_target'] = legacy_raw
                if direct != target or old.resolve(strict=True) != target:
                    errors.append('legacy symlink does not point directly to allowed physical target')
            if path.resolve(strict=True) != target:
                errors.append('incoming backlink resolves outside allowed target')
            if row.get('resolved') is not None and row['resolved'] != str(target):
                errors.append('recorded resolved target differs from allowed target')
            if row.get('mtime_ns') is not None and row['mtime_ns'] != incoming_before['mtime_ns']:
                errors.append('incoming backlink changed since raw inventory')
            result.update({'incoming_identity': incoming_before,
                           'legacy_identity': old_before, 'target_identity': target_before})
            if (incoming_before != node_identity(path) or old_before != node_identity(old)
                    or target_before != node_identity(target, include_mtime=False)):
                errors.append('node identity changed while planning')
        except (OSError, RuntimeError) as error:
            errors.append(str(error))
        result.update({'errors': errors, 'ready': not errors})
        replacements.append(result)
    return {
        'schema_version': 1, 'mode': 'read-only-shared-backlink-plan',
        'legacy_root': str(legacy),
        'allowed_targets': {name: str(target) for name, target in allowed.items()},
        'traverse_storage': False, 'hash_storage': False,
        'replacements': replacements,
        'ready_count': sum(row['ready'] for row in replacements),
        'blocked_count': sum(not row['ready'] for row in replacements),
    }


def validate_backlink_plan_current(plan: dict[str, Any]) -> None:
    """Fail closed if raw targets or any incoming/old/shared node changed."""
    if plan['blocked_count'] or any(not row['ready'] for row in plan['replacements']):
        raise ValueError('backlink plan has blocked replacements')
    rows = [{'path': row['path'], 'target': row['old_target'],
             'resolved': row['new_target']} for row in plan['replacements']]
    fresh = build_backlink_plan(rows, legacy=Path(plan['legacy_root']),
        allowed_targets={key: Path(value) for key, value in plan['allowed_targets'].items()})
    if fresh['blocked_count']:
        raise ValueError('backlink plan is no longer safe')
    for original, current in zip(plan['replacements'], fresh['replacements']):
        for field in ('incoming_identity', 'legacy_identity', 'target_identity', 'legacy_raw_target'):
            if original[field] != current[field]:
                raise ValueError(f'backlink node changed after planning: {original["path"]}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-root', type=Path, default=Path('/mnt/dev/work'))
    args = parser.parse_args()
    plan = build_plan(json.loads(args.inventory.read_text()), args.archive, legacy=args.legacy_root)
    args.output.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'ready': plan['ready_count'], 'blocked': plan['blocked_count'],
                      'subtrees': len(plan['subtrees']), 'output': str(args.output)}))
    return 0 if not plan['blocked_count'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
