#!/usr/bin/env python3
"""Validate/build the active chapters 1–3 draft, not experiment evidence or submission readiness."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    base = ROOT / 'paper/latex'
    kit = json.loads((ROOT / 'paper/template/PROVENANCE.json').read_text())
    for name in ('cvpr.sty', 'ieeenat_fullname.bst'):
        assert sha(base / name) == kit['files_sha256'][name], f'Style changed: {name}'
    active = '\n'.join(line.split('%', 1)[0] for line in (base/'main.tex').read_text().splitlines())
    for section in ('01_introduction', '02_related_work', '03_method'):
        assert f'\\input{{sections/{section}}}' in active
    for section in ('04_experiments', '05_conclusion', '06_limitations'):
        assert f'\\input{{sections/{section}}}' not in active
    assert 'fig03_attention_to_3d' not in active
    for name in ('fig01_teaser', 'fig02_overview'):
        data = json.loads((ROOT/'paper/figures'/name/'figure.json').read_text())
        assert not data['is_measured_result']
        assert sha(ROOT/data['render_source']) == data['render_source_sha256']
        for path, digest in data['input_sha256'].items():
            assert sha(ROOT/path) == digest, path
    out = base/'build'
    out.mkdir(exist_ok=True)
    command = [str(ROOT/'paper/tools/bin/tectonic'), '--only-cached', '--untrusted',
               '--keep-logs', '--keep-intermediates', '--outdir', str(out), 'main.tex']
    with (out/'method_compiler.log').open('w') as stream:
        subprocess.run(command, cwd=base, stdout=stream, stderr=subprocess.STDOUT, check=True)
    log = (out/'main.log').read_text(errors='replace')
    assert not re.search(r'Overfull \\[hv]box|(?:Citation|Reference) .+ undefined|There were undefined', log), 'Layout/citation warning'
    text = subprocess.check_output(['pdftotext', '-layout', str(out/'main.pdf'), '-'], text=True)
    pages = [p for p in text.split('\f') if p.strip()]
    reference_page = next(i+1 for i,p in enumerate(pages) if re.search(r'\bReferences\s*(?:\d+)?\s*$',p,re.M))
    assert reference_page <= 8, 'Body exceeds provisional page budget'
    for token in ('Introduction', 'Related Work', 'Method', 'ATS:', 'HSR:', 'LTE:', 'CCR:'):
        assert token in text, token
    result = dict(status='BUILD_PASS', scope='CHAPTERS_1_TO_3_METHOD_DRAFT',
        pdf='paper/latex/build/main.pdf', pdf_sha256=sha(out/'main.pdf'),
        pages=len(pages), references_start_page=reference_page,
        overfull_boxes=False, unresolved_references=False, official_style_hashes_match=True,
        experimental_sections_included=False, experiment_evidence_revalidated=False,
        submission_ready=False,
        known_warnings='Existing cached lineno.sty and algorithm.sty UTF-8 warnings.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [base/'main.tex',base/'metadata.tex',*[base/'sections'/f'{n}.tex' for n in
            ('00_abstract','01_introduction','02_related_work','03_method')]]})
    (ROOT/'paper/notes/AG3S_METHOD_REVISION_20261016_BUILD.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__ == '__main__':
    main()
