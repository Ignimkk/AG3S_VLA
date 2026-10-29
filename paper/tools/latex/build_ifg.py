#!/usr/bin/env python3
"""Build the IFG development manuscript after validating its active evidence.
Does not assert validation of inactive legacy T39 assets or submission readiness.
"""
import hashlib,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    subprocess.run([sys.executable,str(ROOT/'paper/tools/analysis/audit_ifg_evidence.py')],cwd=ROOT,check=True)
    kit=json.loads((ROOT/'paper/template/PROVENANCE.json').read_text())
    for n in ('cvpr.sty','ieeenat_fullname.bst'):
        assert sha(ROOT/'paper/latex'/n)==kit['files_sha256'][n], f'Style drift: {n}'
    for name in ('fig01_teaser','fig02_overview','fig03_attention_to_3d'):
        j=json.loads((ROOT/'paper/figures'/name/'figure.json').read_text())
        assert sha(ROOT/j['render_source'])==j['render_source_sha256']
        for f,h in j['input_sha256'].items():assert sha(ROOT/f)==h, f'Figure input drift: {f}'
    q=json.loads((ROOT/'paper/figures/ifg_results/exp_manifest.json').read_text())
    for f,h in q['input_sha256'].items():assert sha(ROOT/f)==h, f'Qualitative source drift: {f}'
    out=ROOT/'paper/latex/build';out.mkdir(exist_ok=True)
    command=[str(ROOT/'paper/tools/bin/tectonic'),'--only-cached','--untrusted','--keep-logs','--keep-intermediates','--outdir',str(out),'main.tex']
    with (out/'ifg_compiler.log').open('w') as stream:
        subprocess.run(command,cwd=ROOT/'paper/latex',stdout=stream,stderr=subprocess.STDOUT,check=True)
    log=(out/'main.log').read_text(errors='replace')
    assert not re.search(r'Overfull \\[hv]box|(?:Citation|Reference) .+ undefined|There were undefined',log), 'Unresolved layout/reference warning'
    text=subprocess.check_output(['pdftotext','-layout',str(out/'main.pdf'),'-'],text=True)
    pages=[p for p in text.split('\f') if p.strip()]
    refpage=next(i+1 for i,p in enumerate(pages) if re.search(r'^\s*(?:\d+\s+)?References\b',p,re.M))
    assert refpage<=8, f'Main text extends beyond provisional eight-page limit: {refpage}'
    for token in ('0.837','0.893','0.912','0.998','0.900','0.939','0.933'):
        assert token in text, f'Missing expected manuscript value: {token}'
    result=dict(status='BUILD_PASS',scope='ACTIVE_IFG_DRAFT_ONLY',pdf=str((out/'main.pdf').relative_to(ROOT)),
        pdf_sha256=sha(out/'main.pdf'),pages=len(pages),references_start_page=refpage,
        unresolved_references=False,overfull_boxes=False,
        known_warnings='Cached lineno.sty and algorithm.sty UTF-8 warnings; no body font/style modifications.',
        legacy_EXP001='Not used in this manuscript; legacy full-workspace helper is separate.',
        submission_ready=False)
    supp=ROOT/'paper/supplementary/build';supp.mkdir(exist_ok=True)
    for n in ('cvpr.sty','ieeenat_fullname.bst'):
        assert sha(ROOT/'paper/supplementary'/n)==kit['files_sha256'][n]
    command=[str(ROOT/'paper/tools/bin/tectonic'),'--only-cached','--untrusted','--keep-logs','--keep-intermediates','--outdir',str(supp),'supplementary.tex']
    with (supp/'ifg_compiler.log').open('w') as stream:
        subprocess.run(command,cwd=ROOT/'paper/supplementary',stdout=stream,stderr=subprocess.STDOUT,check=True)
    slog=(supp/'supplementary.log').read_text(errors='replace')
    assert not re.search(r'Overfull \\[hv]box|(?:Citation|Reference) .+ undefined|There were undefined',slog)
    stxt=subprocess.check_output(['pdftotext','-layout',str(supp/'supplementary.pdf'),'-'],text=True)
    result['supplementary']=dict(pdf=str((supp/'supplementary.pdf').relative_to(ROOT)),pages=len([p for p in stxt.split('\f') if p.strip()]),pdf_sha256=sha(supp/'supplementary.pdf'),experiments=9)
    (ROOT/'paper/notes/IFG_REVISION_20261016_B_BUILD.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
