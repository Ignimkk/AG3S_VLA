#!/usr/bin/env python3
"""Run existing offline harness with explicitly logged YOLOE confidence diagnostic."""
import argparse
import sys
from Baseline.paper_baselines.eval import offline_grounding as harness


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--yoloe-conf',type=float,required=True)
    args,rest=parser.parse_known_args()
    if not 0 < args.yoloe_conf <= 1: raise ValueError('confidence must be in (0,1]')
    original=harness.provider_kwargs
    def configured(spec,size,device,lh_selection=None):
        name,kwargs=original(spec,size,device,lh_selection)
        if name=='yoloe': kwargs['conf']=args.yoloe_conf
        return name,kwargs
    harness.provider_kwargs=configured
    sys.argv=[sys.argv[0]]+rest
    return harness.main(rest)

if __name__=='__main__': raise SystemExit(main())
