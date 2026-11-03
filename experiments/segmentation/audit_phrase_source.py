#!/usr/bin/env python3
"""Audit oracle-name prompts against independently parsed language prompts.

The parser reads only the recorded instruction. The oracle target is read later
only to check text equality; it is never used to choose the parsed phrase.
"""
import argparse
import hashlib
import json
from pathlib import Path

from AG3S.trajopt.attention_source import _LIBERO_PICK_PLACE, clean_prompt
from Baseline.paper_baselines.eval.offline_grounding import Episode,load_episodes,query_for


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=[]
    for e in sorted(load_episodes(a.source),key=lambda e:(e['task_id'],e['episode_idx'])):
        instruction=e.get('task_description') or ''
        match=_LIBERO_PICK_PLACE.fullmatch(clean_prompt(instruction)) if instruction else None
        # Parse first, using only the instruction; BDDL/scene metadata are not inputs to this step.
        parsed=match.group('target').strip() if match else None
        ep=Episode(e,scene_dir=a.scene);query=query_for(ep,'phrase',{})
        old=query.target_phrase
        rows.append({'episode_id':ep.stem,'task_id':int(e['task_id']),'instruction':instruction,
                     'prompt_only_phrase':parsed,'existing_bddl_phrase':old,
                     'exact_equal':parsed==old,'case_insensitive_equal':parsed.casefold()==old.casefold() if parsed else False})
    out={'source':str(a.source.resolve()),'scene':str(a.scene.resolve()),'episodes':len(rows),
         'exact_matches':sum(r['exact_equal'] for r in rows),'parsed':sum(r['prompt_only_phrase'] is not None for r in rows),
         'rule':'existing LIBERO pick-and-place regex target group; prompt only; no fallback',
         'limitation':'post-hoc input-text audit on seen tasks, not a general parser validation',
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
    print(f"{out['exact_matches']}/{out['episodes']} exact text matches; {out['parsed']} parsed")


if __name__=='__main__':main()
