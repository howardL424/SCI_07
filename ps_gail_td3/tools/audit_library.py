"""Summarize solver yield, executed expert facts and terminal success."""
import argparse
import json
import numpy as np
from ..common import META, PACKAGE, output_path, fingerprint
from ..expert_data import load_library, validate_transition

def audit(expert,cache,out):
    lib,entries=load_library(expert)
    terminal=np.load(cache,allow_pickle=False)
    if str(terminal['expert_fingerprint'])!=fingerprint(expert):
        raise ValueError('Terminal cache fingerprint mismatch')
    accepted=[e for e in entries if e['ok_use']]
    if len(accepted)!=len(terminal['r_minT']):
        raise ValueError('Terminal cache trajectory count mismatch')
    steps=clipped=both=intercepted=0
    for i,e in enumerate(accepted):
        tr=e['transitions']; _,_,ns=validate_transition(tr)
        steps+=len(ns); clipped+=int(tr['n_clipped'])
        hit=bool(np.asarray(tr['intercepted']).any()); intercepted+=int(hit)
        both+=int(not hit and min(ns[:,0].min(),ns[:,4].min())>=6 and terminal['r_minT'][i]<=10)
    seconds=np.array([e['stageB'].get('solve_seconds',np.nan) for e in entries])
    report=dict(meta=META,expert_fingerprint=fingerprint(expert),attempted=len(entries),
        solver_accepted=len(accepted),solver_usable_rate=len(accepted)/len(entries),
        replay_intercepted=intercepted,both_ok_trajectories=both,
        both_ok_rate_of_attempted=both/len(entries),steps=steps,n_clipped=clipped,
        clipping_rate=clipped/max(steps,1),solver_seconds_mean=float(np.nanmean(seconds)),
        solver_seconds_median=float(np.nanmedian(seconds)),solver_seconds_total=float(np.nansum(seconds)),
        score_coefficients=lib['meta']['scores'])
    assert report['clipping_rate']<=.05
    output_path(out).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expert',default=str(PACKAGE/'data/radau_expert_lib.mat'))
    p.add_argument('--terminal-cache',default=str(PACKAGE/'data/radau_terminal_cache.npz'))
    p.add_argument('--out',default=str(PACKAGE/'logs/expert_audit.json'))
    a=p.parse_args(); audit(a.expert,a.terminal_cache,a.out)

if __name__=='__main__': main()
