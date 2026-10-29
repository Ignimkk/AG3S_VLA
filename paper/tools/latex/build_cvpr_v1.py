#!/usr/bin/env python3
"""Build the full CVPR v1 working draft (all sections) and report layout facts.

Unlike build_method.py / build_ifg.py this script does not gate on figure sidecars or the evidence registry;
it only checks that the official style files are unchanged, compiles with the pinned tectonic, and reports
pages, reference start page, overfull boxes, undefined references, and the counts of \\tbd / \\pend markers.

    python3 paper/tools/latex/build_cvpr_v1.py
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = ROOT / 'paper/latex'
    kit = json.loads((ROOT / 'paper/template/PROVENANCE.json').read_text())
    for name in ('cvpr.sty', 'ieeenat_fullname.bst'):
        assert sha(base / name) == kit['files_sha256'][name], f'Style changed: {name}'
    out = base / 'build'
    out.mkdir(exist_ok=True)
    command = [str(ROOT / 'paper/tools/bin/tectonic'), '--only-cached', '--untrusted',
               '--keep-logs', '--keep-intermediates', '--outdir', str(out), 'main.tex']
    with (out / 'cvpr_v1_compiler.log').open('w') as stream:
        proc = subprocess.run(command, cwd=base, stdout=stream, stderr=subprocess.STDOUT)
    log = (out / 'main.log').read_text(errors='replace') if (out / 'main.log').exists() else ''
    overfull = re.findall(r'Overfull \\[hv]box \([^)]*\) (?:in paragraph at|detected at|has occurred while) lines? ?[\d-]*', log)
    undefined = re.findall(r'(?:Citation|Reference) `([^\']+)\' .*undefined', log)
    pages = None
    reference_page = None
    if proc.returncode == 0 and (out / 'main.pdf').exists():
        text = subprocess.check_output(['pdftotext', '-layout', str(out / 'main.pdf'), '-'], text=True)
        split = [p for p in text.split('\f') if p.strip()]
        pages = len(split)
        for i, p in enumerate(split):
            if re.search(r'(?:^|\s{2,})References(?:\s{2,}|\s*$)', p, re.M):
                reference_page = i + 1
                break
    sections = [base / 'sections' / f for f in sorted(p.name for p in (base / 'sections').glob('*.tex'))]
    figs = list((ROOT / 'paper/figures').glob('fig0*/figure*.tex'))
    body = '\n'.join(p.read_text() for p in sections + figs)
    result = dict(
        status='BUILD_PASS' if proc.returncode == 0 else 'BUILD_FAIL',
        returncode=proc.returncode,
        pdf='paper/latex/build/main.pdf' if proc.returncode == 0 else None,
        pages=pages, references_start_page=reference_page,
        overfull_boxes=len(overfull), overfull_examples=overfull[:8],
        undefined_references=sorted(set(undefined)),
        tbd_markers=len(re.findall(r'\\tbd\{', body)),
        pend_markers=len(re.findall(r'\\pend\{', body)),
        compiler_log='paper/latex/build/cvpr_v1_compiler.log',
    )
    (ROOT / 'paper/notes/CVPR_V1_BUILD.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
