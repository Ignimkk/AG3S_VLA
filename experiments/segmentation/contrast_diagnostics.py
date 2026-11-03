#!/usr/bin/env python3
"""Compute pre-named diagnostic contrasts from matched task means."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('--seed',type=int,default=20261101);a=p.parse_args()
    doc=json.loads(a.summary.read_text()); rows={r['arm']:r for r in doc['results']}
    pairs=[('surface_gtseed_gtrobot_privileged','surface_gtrobot_privileged'),
           ('surface_gtrobot_privileged','surface_no_robot_mask')]
    n=len(next(iter(rows.values()))['task_iou']); picks=np.random.default_rng(a.seed).integers(0,n,(10000,n));out=[]
    for aa,bb in pairs:
        d=np.array(rows[aa]['task_iou'])-np.array(rows[bb]['task_iou'])
        out.append({'a':aa,'b':bb,'task_differences':d.tolist(),'delta':float(d.mean()),
                    'ci95':np.quantile(d[picks].mean(1),[.025,.975]).tolist()})
    target=a.summary.with_name('diagnostic_contrasts.json')
    target.write_text(json.dumps({'input':str(a.summary),'task_bootstrap_seed':a.seed,'resamples':10000,'pairs':out},indent=2)+'\n')
    print(target)

if __name__=='__main__': main()
