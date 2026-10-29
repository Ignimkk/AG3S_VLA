#!/usr/bin/env python3
"""Validate paper manifests and their recorded source identity; never run research code."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import zipfile

from jsonschema import Draft202012Validator

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
ARRAY_KEYS = ('executed_actions.npy', 'inference_states.npy', 'predicted_chunks.npy')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    require(not candidate.is_absolute(), f'Expected workspace-relative path: {relative}')
    resolved = (root / candidate).resolve()
    require(resolved.is_relative_to(root.resolve()), f'Path escapes workspace: {relative}')
    require(resolved.is_file(), f'Missing source file: {relative}')
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def section_sha256(path: Path, heading: str) -> str:
    text = path.read_text()
    matches = [i for i, line in enumerate(text.splitlines(keepends=True)) if line.rstrip() == heading]
    require(len(matches) == 1, f'Expected one research-log section: {heading}')
    lines = text.splitlines(keepends=True)
    first = matches[0]
    last = next((i for i in range(first + 1, len(lines)) if lines[i].startswith('## ')), len(lines))
    # Match the source audit's boundary: exclude the newline immediately before the next heading.
    section = ''.join(lines[first:last])
    if last < len(lines):
        section = section[:-1]
    return hashlib.sha256(section.encode()).hexdigest()


def npz_shapes(path: Path) -> dict[str, list[int]]:
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in ARRAY_KEYS:
            with archive.open(name) as stream:
                require(stream.read(6) == bytes([147]) + b'NUMPY', f'Bad NPY header: {path}/{name}')
                version = stream.read(2)
                require(len(version) == 2 and version[0] in (1, 2, 3), f'Unsupported NPY version: {path}')
                width = 2 if version[0] == 1 else 4
                size = int.from_bytes(stream.read(width), 'little')
                require(size <= 1024 * 1024, f'Unexpected NPY header size: {path}')
                encoding = 'utf-8' if version[0] == 3 else 'latin1'
                header = ast.literal_eval(stream.read(size).decode(encoding))
                require('O' not in str(header['descr']), f'Object dtype is not dimension evidence: {path}')
                result[name] = list(header['shape'])
    return result


def json_pointer(value, pointer: str):
    require(pointer.startswith('/numbers/'), f'Measurement must point inside verify.numbers: {pointer}')
    for segment in pointer.split('/')[1:]:
        key = segment.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def exact_json_equal(left, right) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def validate_manifest(path: Path, root: Path, schema: dict) -> dict:
    manifest = json.loads(path.read_text())
    Draft202012Validator(schema).validate(manifest)
    status = manifest['status']
    if status != 'REGISTERED':
        return {'experiment_id': manifest['experiment_id'], 'status': status}

    cache = {}
    verified = {}
    for item in manifest.get('sources', []):
        file = source_path(root, item['path'])
        actual = section_sha256(file, item['section_heading']) if item.get('hash_scope') == 'section' else sha256(file)
        require(actual == item['sha256'], f'Source changed: {item["path"]}')
        if item['role'] == 'verify':
            require('hash_scope' not in item, 'Verify file needs a full-file hash')
            verified[item['path']] = item['sha256']
            cache[item['path']] = json.loads(file.read_text())
    conditions_list = manifest.get('conditions', [])
    conditions = {condition['condition_id']: condition['baseline_id'] for condition in conditions_list}
    require(len(conditions) == len(conditions_list), 'Duplicate condition ID')
    names = set()
    groups = defaultdict(set)
    counts = Counter()
    for run in manifest.get('runs', []):
        require(run['run_id'] not in names, f'Duplicate run: {run["run_id"]}')
        names.add(run['run_id'])
        condition = run['condition_id']
        require(condition in conditions, f'Unknown condition: {condition}')
        require(run['baseline_id'] == conditions[condition], 'Baseline/condition mismatch')
        pair = (run['episode_id'], run['policy_seed'])
        require(condition not in groups[pair], f'Duplicate pair member: {pair}/{condition}')
        groups[pair].add(condition)
        counts[condition] += 1
        assets = {item['role']: item for item in run['assets']}
        require(len(assets) == len(run['assets']), 'Duplicate run asset role')
        require({'run_manifest', 'trajectory', 'completeness', 'run_check'} <= set(assets), 'Run assets incomplete')
        for item in assets.values():
            require(sha256(source_path(root, item['path'])) == item['sha256'], f'Run source changed: {item["path"]}')
        expected_directory = (root / run['raw_directory']).resolve()
        require(expected_directory.is_dir(), f'Missing raw directory: {run["raw_directory"]}')
        for role in ('run_manifest', 'trajectory', 'completeness'):
            require(source_path(root, assets[role]['path']).parent == expected_directory, 'Run assets come from different directories')
        metadata = json.loads(source_path(root, assets['run_manifest']['path']).read_text())
        scene = metadata['scene']
        require(scene['episode_index'] == run['episode_id'] and metadata['policy_seed'] == run['policy_seed'], 'Run metadata mismatch')
        require(scene['obs_format'] == 'rby1_16d' and scene['action_format'] == 'rby1_16d', 'Run metadata is not 16-D')
        require(scene['policy_config'] == manifest['policy']['config'], 'Policy config mismatch')
        completeness = json.loads(source_path(root, assets['completeness']['path']).read_text())
        check = json.loads(source_path(root, assets['run_check']['path']).read_text())
        require(completeness['pass'] is True and check['complete'] is True, 'Run is incomplete')
        require(check['cond'] == condition and check['ep'] == run['episode_id'] and check['seed'] == run['policy_seed'], 'Completion check belongs to another run')
        shapes = npz_shapes(source_path(root, assets['trajectory']['path']))
        require(shapes == run['dimension_evidence'], 'Recorded array shapes changed')
        require(all(shape[-1] == 16 for shape in shapes.values()), 'Actual array dimension is not 16')
        require(shapes['executed_actions.npy'][0] == completeness['control_records'], 'Control row count mismatch')
        require(shapes['inference_states.npy'][0] == shapes['predicted_chunks.npy'][0] == completeness['planning_records'], 'Planning row count mismatch')
    if 'design' in manifest:
        require(len(names) == manifest['design']['run_count'], 'Design run count mismatch')
    if manifest.get('design', {}).get('paired'):
        require(all(members == set(conditions) for members in groups.values()), 'Incomplete matched pair')
        require(len(groups) == manifest['design']['pair_count'], 'Design pair count mismatch')

    result_ids = set()
    source_linked_results = set()
    for measurement in manifest.get('measurements', []):
        result_id = measurement['result_id']
        require(result_id not in result_ids, f'Duplicate result ID: {result_id}')
        result_ids.add(result_id)
        source_fields = ('source_path', 'source_sha256', 'json_pointer')
        present = [field in measurement for field in source_fields]
        require(all(present) or not any(present), f'Partial source link: {result_id}')
        if all(present):
            key = measurement['source_path']
            require(key in verified and measurement['source_sha256'] == verified[key], 'Measurement source link does not match a verify file')
            actual = json_pointer(cache[key], measurement['json_pointer'])
            require(exact_json_equal(actual, measurement['value']), f'Measurement differs from source: {result_id}')
            source_linked_results.add(result_id)
        condition = measurement.get('condition_id')
        require(not conditions or condition is None or condition in conditions, 'Measurement condition unknown')
        denominator = measurement.get('denominator')
        if denominator is not None and all(present):
            require(exact_json_equal(json_pointer(cache[key], denominator['json_pointer']), denominator['value']), f'Denominator differs from source: {result_id}')
            if condition is not None:
                require(denominator['value'] == counts[condition], 'Denominator does not match registered runs')
    if 'audit' in manifest:
        audit = json.loads(source_path(root, manifest['audit']['path']).read_text())
        require(audit['experiment_id'] == manifest['experiment_id'] and audit['status'] == 'PASS_WITH_LIMITED_SCOPE', 'Source audit mismatch')
        require(audit['verified_run_count'] == len(names), 'Audit run count mismatch')
    return {'experiment_id': manifest['experiment_id'], 'status': 'PASS', 'evidence_scope': manifest['evidence_scope'], 'runs': len(names), 'pairs': len(groups), 'results': sorted(result_ids), 'source_linked_results': sorted(source_linked_results)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', nargs='*', type=Path)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    schema_path = args.root / 'paper/experiments/configs/manifest.schema.json'
    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    paths = args.manifests or sorted((args.root / 'paper/experiments/manifests').glob('*.json'))
    if not paths:
        print(json.dumps({'status': 'NO_REGISTERED_MANIFESTS'}))
        return 1
    seen = set()
    for path in paths:
        try:
            result = validate_manifest(path, args.root, schema)
            require(result['experiment_id'] not in seen, 'Duplicate experiment ID')
            seen.add(result['experiment_id'])
            print(json.dumps(result, ensure_ascii=False))
        except Exception as error:
            print(f'FAIL {path}: {error}', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
