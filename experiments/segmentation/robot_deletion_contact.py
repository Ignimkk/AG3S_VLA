#!/usr/bin/env python3
"""Describe target-pixel deletion conditioned on image-time gripper contact.

Contact is MuJoCo geom contact, not an oracle grasp or a causal intervention.
Full-image pixel counts are taken from the existing factorial results.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    for k in ('phase','frames','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--shards',type=Path,nargs='+',required=True);a=p.parse_args()
    expected={(x['episode_id'],x['obs_index']) for x in map(json.loads,a.frames.open())}
    rows={}
    for s in a.shards:
        for r in map(json.loads,(s/'frames.jsonl').open()):
            k=r['episode_id'],r['obs_index']
            if k in rows:raise ValueError('duplicate')
            rows[k]=r
    if set(rows)!=expected:raise ValueError('frame mismatch')
    byep=defaultdict(list)
    for stem,t in rows:byep[stem].append(t)
    counts=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(int))))
    for stem,times in byep.items():
        with np.load(a.phase/f'phase_{stem}.npz') as z:
            frames=z['frames'];contact=z['contact_grip_target']
        for t in times:
            idx=np.flatnonzero(frames==t)
            if len(idx)!=1:raise ValueError('contact join mismatch')
            label='gripper_contact' if contact[idx[0]] else 'no_gripper_contact'
            r=rows[(stem,t)]
            for c in ('head','left_wrist'):
                d=counts[c][label][str(r['task_id'])];d['frames']+=1
                for key in ('target_px','target_deleted_px'):d[key]+=r['robot_stats'][c][key]
    out={}
    for c,groups in counts.items():
        out[c]={}
        for group,ts in groups.items():
            vals={t:{**dict(v),'target_deleted_fraction':v['target_deleted_px']/v['target_px'] if v['target_px'] else None} for t,v in ts.items()}
            target=sum(x['target_px'] for x in vals.values());deleted=sum(x['target_deleted_px'] for x in vals.values())
            means=[x['target_deleted_fraction'] for x in vals.values() if x['target_deleted_fraction'] is not None]
            out[c][group]={'frames':sum(x['frames'] for x in vals.values()),'pixel_micro_deleted_fraction':deleted/target if target else None,
                           'equal_task_mean_pooled_fraction':float(np.mean(means)) if means else None,
                           'n_tasks':len(means),'per_task':vals}
    with a.out.open('x') as f:json.dump({'frames':len(rows),'scope':'full image; target pixels',
      'warning':'image-time gripper0 geom contact is not necessarily grasp; observational frame grouping, no causal claim',
      'cameras':out},f,indent=2);f.write('\n')
    print('contact-stratified',len(rows),'frames')


if __name__=='__main__':main()
