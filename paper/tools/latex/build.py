#!/usr/bin/env python3
"""Validate source provenance and build the separate CVPR draft PDFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(data: object, key: str) -> object:
    for token in key.lstrip('/').split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        data = data[int(token)] if isinstance(data, list) else data[token]
    return data


def check_sources() -> None:
    provenance = json.loads((ROOT / 'paper/template/PROVENANCE.json').read_text())
    original = ROOT / provenance['original_directory']
    for name, expected in provenance['files_sha256'].items():
        if sha256(original / name) != expected:
            raise ValueError(f'Official kit source drift: {name}')
    for folder in ('latex', 'supplementary'):
        for name in ('cvpr.sty', 'ieeenat_fullname.bst'):
            if sha256(ROOT / f'paper/{folder}/{name}') != provenance['files_sha256'][name]:
                raise ValueError(f'Working style source drift: {folder}/{name}')
    subprocess.run(
        [sys.executable, str(ROOT / 'paper/tools/analysis/validate_manifests.py')],
        cwd=ROOT, check=True,
    )
    sidecars = list((ROOT / 'paper/figures').glob('*/figure.json'))
    sidecars += list((ROOT / 'paper/tables').glob('*/table.json'))
    sidecars += [ROOT / 'paper/algorithms/algorithm01_safe_vla.json']
    for path in sidecars:
        data = json.loads(path.read_text())
        if sha256(ROOT / data['render_source']) != data['render_source_sha256']:
            raise ValueError(f'Artifact render source drift: {path.relative_to(ROOT)}')
        for source, expected in data.get('input_sha256', {}).items():
            if sha256(ROOT / source) != expected:
                raise ValueError(f'Artifact input source drift: {source}')
    descriptive = json.loads((ROOT / 'paper/tables/table00_descriptive/table.json').read_text())
    source_path = ROOT / descriptive['source_path']
    if sha256(source_path) != descriptive['source_sha256']:
        raise ValueError('T00 verifier source drift')
    source = json.loads(source_path.read_text())
    rendered = (ROOT / descriptive['render_source']).read_text()
    for cell in descriptive['cells']:
        for kind in ('numerator', 'denominator'):
            if pointer(source, cell[kind + '_pointer']) != cell[kind]:
                raise ValueError(f'T00 {cell["result_id"]}: {kind} differs from source')
        rendered_value = f'{cell["numerator"]}/{cell["denominator"]}'
        if rendered_value not in rendered:
            raise ValueError(f'T00 rendered count missing: {rendered_value}')
    print('PASS kit, working styles, manifests, artifact hashes and T00 source counts', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--target', choices=['main', 'supplementary', 'all'], default='all')
    parser.add_argument('--engine', choices=['auto', 'latexmk', 'tectonic'], default='auto')
    parser.add_argument('--compiler', type=Path, help='Explicit compiler path; also specify --engine')
    parser.add_argument('--offline', action='store_true', help='Use only cached Tectonic resources')
    args = parser.parse_args()
    try:
        check_sources()
        if args.check_only:
            return 0
        engine = args.engine
        if args.compiler and engine == 'auto':
            raise ValueError('Use --engine latexmk or --engine tectonic with --compiler')
        if engine == 'auto':
            engine = next((e for e in ('latexmk', 'tectonic') if shutil.which(e)), '')
        if not engine:
            raise ValueError('No LaTeX compiler found; install/provide latexmk or tectonic')
        compiler = str(args.compiler.resolve()) if args.compiler else shutil.which(engine)
        if not compiler or not Path(compiler).is_file():
            raise ValueError(f'Compiler unavailable: {args.compiler or engine}')
        targets = ['main', 'supplementary'] if args.target == 'all' else [args.target]
        for target in targets:
            folder = ROOT / ('paper/latex' if target == 'main' else 'paper/supplementary')
            filename = 'main.tex' if target == 'main' else 'supplementary.tex'
            output = folder / 'build'
            output.mkdir(exist_ok=True)
            if engine == 'tectonic':
                command = [compiler, '--untrusted', '--keep-logs', '--keep-intermediates',
                           '--outdir', str(output), filename]
                if args.offline:
                    command.insert(1, '--only-cached')
            else:
                command = [compiler, '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                           '-file-line-error', '-outdir=build', filename]
            log = output / 'compiler.log'
            print(f'Building {target} with {engine}; log: {log.relative_to(ROOT)}', flush=True)
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                print('\n'.join(log.read_text(errors='replace').splitlines()[-25:]), file=sys.stderr)
                return result.returncode
            pdf = output / Path(filename).with_suffix('.pdf')
            if not pdf.is_file() or not pdf.read_bytes().startswith(b'%PDF-'):
                raise ValueError(f'Compiler returned success without a PDF: {pdf}')
            print(json.dumps({'target': target, 'status': 'BUILD_PASS',
                              'pdf': str(pdf.relative_to(ROOT)), 'sha256': sha256(pdf)}), flush=True)
        return 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
