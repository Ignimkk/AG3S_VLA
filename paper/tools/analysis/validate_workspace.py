#!/usr/bin/env python3
"""Check the CVPR workspace and optionally inspect specified draft PDFs."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[3]
SKILLS = ['cvpr-write', 'cvpr-evidence', 'cvpr-experiment', 'cvpr-figure',
          'cvpr-reference-analysis', 'cvpr-review', 'cvpr-compliance']
AGENTS = ['literature-analyst', 'evidence-auditor', 'cvpr-reviewer']
CHECKLISTS = ['README', 'SUBMISSION_CHECKLIST', 'ANONYMITY_CHECKLIST',
              'PAGE_LIMIT_CHECKLIST', 'REFERENCES_CHECKLIST',
              'SUPPLEMENTARY_CHECKLIST', 'FINAL_AUDIT']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frontmatter(path):
    match = re.match(r'^---\n(.*?)\n---\n', path.read_text(), re.S)
    require(match is not None, f'Missing YAML frontmatter: {path}')
    data = yaml.safe_load(match[1])
    require(isinstance(data, dict), f'Invalid YAML frontmatter: {path}')
    return data


def structure():
    for name in CHECKLISTS:
        require((ROOT / f'paper/submission/{name}.md').is_file(), f'Missing submission document: {name}')
    require(not (ROOT / 'paper/.claude').exists(), 'Duplicate paper/.claude')
    counts = {}
    for rel in ('paper/CLAUDE.md',):
        counts[rel] = len((ROOT / rel).read_text())
        require(counts[rel] <= 200, f'CLAUDE.md exceeds 200 characters: {rel}')
    for name in SKILLS:
        path = ROOT / f'.claude/skills/{name}/SKILL.md'
        data = frontmatter(path)
        require(set(data) == {'name', 'description'}, f'Unexpected skill fields: {name}')
        require(data['name'] == name and isinstance(data['description'], str) and data['description'], f'Invalid skill: {name}')
        require('$ARGUMENTS' in path.read_text(), f'Missing skill arguments: {name}')
    names = [frontmatter(p)['name'] for p in (ROOT / '.claude/agents').glob('*.md')]
    require(len(names) == len(set(names)), 'Duplicate agent names')
    for name in AGENTS:
        data = frontmatter(ROOT / f'.claude/agents/{name}.md')
        require(set(data) == {'name', 'description', 'tools', 'model'}, f'Unexpected agent fields: {name}')
        require(data['name'] == name and data['model'] == 'inherit', f'Invalid agent name/model: {name}')
        tools = {t.strip() for t in data['tools'].split(',')}
        allowed = {'Read', 'Grep', 'Glob', 'WebSearch', 'WebFetch'} if name == 'literature-analyst' else {'Read', 'Grep', 'Glob'}
        require(tools == allowed, f'Unexpected tools: {name}')
    require(frontmatter(ROOT / '.claude/rules/cvpr-workspace.md')['paths'] == ['paper/**'], 'Unexpected rule scope')
    return {'claude_characters': counts,
            'root_claude_policy': 'PRESERVE_USER_RESTORED_FILE; NO_LENGTH_LIMIT',
            'root_claude_sha256': sha(ROOT / 'CLAUDE.md'),
            'skills': SKILLS, 'read_only_agents': AGENTS,
            'submission_documents': CHECKLISTS, 'runtime_invocation': 'NOT_CHECKED'}


def documents():
    paths = [p for p in (ROOT / 'paper').rglob('*.md') if 'cvpr2026_official' not in p.parts and 'build' not in p.parts]
    paths += [ROOT / f'.claude/skills/{name}/SKILL.md' for name in SKILLS]
    paths += [ROOT / f'.claude/agents/{name}.md' for name in AGENTS]
    links = 0
    for path in paths:
        fence = None
        for line in path.read_text().splitlines():
            require(line.rstrip() == line, f'Trailing whitespace: {path}')
            match = re.match(r'^\s*(`{3,}|~{3,})', line)
            if match:
                fence = None if fence == match[1][0] else match[1][0]
                continue
            if fence:
                continue
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', line):
                if '://' in target or target.startswith('#'):
                    continue
                filename, _, anchor = target.partition('#')
                resolved = (path.parent / unquote(filename.strip('<>'))).resolve()
                require(resolved.exists(), f'Broken link: {path}: {target}')
                if anchor and resolved.suffix == '.md':
                    content = resolved.read_text()
                    anchors = set(re.findall(r'<a\s+id="([^"]+)"', content))
                    anchors.update(re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
                                   for heading in re.findall(r'^#+\s+(.+)$', content, re.M))
                    require(unquote(anchor) in anchors, f'Broken anchor: {path}: {target}')
                links += 1
    json_files = [p for p in (ROOT / 'paper').rglob('*.json') if 'build' not in p.parts]
    for path in json_files:
        json.loads(path.read_text())
    scripts = list((ROOT / 'paper/tools').rglob('*.py'))
    for path in scripts:
        ast.parse(path.read_text())
    return {'markdown_files': len(paths), 'local_links': links,
            'json_files': len(json_files), 'python_files': len(scripts)}


def evidence_and_sources():
    result = subprocess.run([sys.executable, str(ROOT / 'paper/tools/latex/build.py'), '--check-only'],
                            cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, result.stdout + result.stderr)
    manifest = json.loads((ROOT / 'paper/experiments/manifests/EXP001_T39_task_outcomes.json').read_text())
    results = {m['result_id'] for m in manifest['measurements']}
    for rid in results:
        require(rid in (ROOT / 'paper/evidence/RESULTS.md').read_text(), f'Missing registered result: {rid}')
    claim_ids = set(re.findall(r'\| (C\d{3}) \|', (ROOT / 'paper/evidence/CLAIMS.md').read_text()))
    artifacts = list((ROOT / 'paper/figures').glob('*/figure.json')) + list((ROOT / 'paper/tables').glob('*/table.json'))
    artifacts += [ROOT / 'paper/algorithms/algorithm01_safe_vla.json']
    ids = []
    for path in artifacts:
        data = json.loads(path.read_text())
        ids.append(data['artifact_id'])
        require(set(data['claim_ids']) <= claim_ids, f'Unknown artifact claim: {path}')
        require(set(data['result_ids']) <= results, f'Unknown artifact result: {path}')
        require(data['artifact_id'] in (ROOT / 'paper/evidence/FIGURES_TABLES.md').read_text(), f'Artifact missing in registry: {path}')
    require(len(ids) == len(set(ids)), 'Duplicate artifact ID')
    return {'source_validator_output': result.stdout.splitlines(), 'registered_results': sorted(results),
            'evidence_scope': manifest['evidence_scope'], 'artifact_ids': sorted(ids),
            'scientific_validity': 'NOT_CERTIFIED'}


def preserved(path):
    baseline = json.loads(path.read_text())
    for rel, expected in baseline.items():
        require(sha(ROOT / rel) == expected, f'Preserved file changed: {rel}')
    return {'baseline_file': str(path), 'file_count': len(baseline)}


def inspect_pdf(path):
    import pymupdf
    document = pymupdf.open(path)
    text = '\n'.join(page.get_text() for page in document)
    for token in ('/mnt/', 'firstauthor@', 'secondauthor@', 'Institution1', 'Institution2', '\ufffd'):
        require(token not in text, f'Unexpected draft PDF text: {token}')
    require(not document.metadata.get('author'), 'Nonempty PDF author metadata')
    fonts = {font[0]: font[3] for page in document for font in page.get_fonts()}
    require(any('NimbusRom' in font for font in fonts.values()), 'Expected Times-family font absent')
    for xref in fonts:
        require(document.extract_font(xref)[3], f'Unembedded font: {fonts[xref]}')
    for page in document:
        require(abs(page.rect.width - 612) < .1 and abs(page.rect.height - 792) < .1, 'Unexpected page size')
        for word in page.get_text('words'):
            require(word[0] >= 0 and word[1] >= 0 and word[2] <= page.rect.width + .1 and word[3] <= page.rect.height + .1, 'Text outside page')
    require('EVIDENCE PENDING' in text, 'Current draft must retain visible evidence-pending status')
    log_path = path.with_suffix('.log')
    require(log_path.is_file(), f'Missing build log: {log_path}')
    log = log_path.read_text(errors='replace')
    require(not re.search(r'LaTeX Font Warning|Overfull \\[hv]box|LaTeX Warning:.*undefined|Undefined control sequence', log),
            f'Unresolved font/layout/reference issue: {log_path}')
    compiler_log = path.parent / 'compiler.log'
    require(compiler_log.is_file(), f'Missing compiler output: {compiler_log}')
    source_paths = list((ROOT / 'paper/latex').glob('*.tex')) + list((ROOT / 'paper/latex/sections').glob('*.tex'))
    source_paths += list((ROOT / 'paper/figures').glob('*/figure.tex')) + list((ROOT / 'paper/tables').glob('*/table.tex'))
    source_paths += list((ROOT / 'paper/algorithms').glob('*.tex'))
    source_paths += [p for p in (ROOT / 'paper/supplementary').rglob('*.tex') if 'build' not in p.parts]
    source_paths += list((ROOT / 'paper/latex').glob('*.bib'))
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'pages': len(document),
            'tex_log_sha256': sha(log_path), 'compiler_log_sha256': sha(compiler_log),
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
            'letter_page': True, 'fonts': sorted(fonts.values()), 'embedded_fonts': True,
            'limited_anonymity_scan': 'PASS', 'full_anonymity': 'NOT_CHECKED',
            'external_uris': sorted({link['uri'] for page in document
                                     for link in page.get_links() if link.get('uri')}),
            'external_uri_policy_review': 'NOT_CHECKED',
            'supplementary_title': 'Supplementary Material' in text,
            'registered_task_counts_present': all(value in text for value in ('30/48', '29/48')),
            'unmeasured_marker_present': 'n/m' in text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--main-pdf', type=Path)
    parser.add_argument('--supplementary-pdf', type=Path)
    args = parser.parse_args()
    report = {'audited_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'submission_ready': False, 'scope': 'Workspace and draft validation; not submission authorization', 'checks': []}
    # Publish an explicit in-progress report before checking links to this report.
    if args.report:
        destination = args.report.resolve()
        require(destination.is_relative_to(ROOT / 'paper'), 'Report must be inside paper/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({**report, 'status': 'IN_PROGRESS'}, indent=2) + '\n')
    def check(name, function):
        try:
            report['checks'].append({'name': name, 'status': 'PASS', 'details': function()})
        except Exception as exc:
            report['checks'].append({'name': name, 'status': 'FAIL', 'error': str(exc)})
    check('structure_and_claude_config', structure)
    check('documents_links_and_syntax', documents)
    check('evidence_template_and_artifacts', evidence_and_sources)
    if args.baseline:
        check('preserved_files', lambda: preserved(args.baseline))
    else:
        report['checks'].append({'name': 'preserved_files', 'status': 'NOT_CHECKED'})
    if args.main_pdf and args.supplementary_pdf:
        def pdfs():
            main_pdf, supplementary = inspect_pdf(args.main_pdf.resolve()), inspect_pdf(args.supplementary_pdf.resolve())
            require(main_pdf['registered_task_counts_present'] and main_pdf['unmeasured_marker_present'], 'Main table content absent')
            require(not main_pdf['supplementary_title'] and supplementary['supplementary_title'], 'Main/supplementary separation failed')
            return {'main': main_pdf, 'supplementary': supplementary}
        check('draft_pdf_inspection', pdfs)
    elif args.main_pdf or args.supplementary_pdf:
        report['checks'].append({'name': 'draft_pdf_inspection', 'status': 'FAIL', 'error': 'Provide both PDF paths'})
    else:
        report['checks'].append({'name': 'draft_pdf_inspection', 'status': 'NOT_CHECKED'})
    failed = any(c['status'] == 'FAIL' for c in report['checks'])
    report['status'] = 'FAIL' if failed else ('PASS_DRAFT_CHECKS' if args.main_pdf and args.supplementary_pdf else 'PASS_STATIC_CHECKS')
    if args.report:
        destination = args.report.resolve()
        require(destination.is_relative_to(ROOT / 'paper'), 'Report must be inside paper/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
