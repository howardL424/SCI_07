import argparse
import numpy as np
from ..common import META, PACKAGE, output_path, fingerprint, project_terminal
from ..expert_data import load_library, validate_transition
from ..direct_env import DirectEnv

def precompute(expert, out):
    _, entries=load_library(expert)
    accepted=[e for e in entries if e['ok_use']]
    fields={k:[] for k in ('n_steps','r_minT','reached','diverged','n_steps_stage3',
                           'r_min1','r_min2','intercepted','R1','bonus','both_ok','row_idx')}
    env=DirectEnv()
    try:
        env._start()
        for e in accepted:
            tr=e['transitions']; _,_,ns=validate_transition(tr)
            intercepted=bool(np.asarray(tr['intercepted']).any())
            r1,r2=float(ns[:,0].min()),float(ns[:,4].min())
            rT,reached,diverged,n=np.inf,False,False,0
            if not intercepted:
                rT,reached,diverged,n=env.eng.stage3_rollout(
                    env._ml(tr['terminal_abs']),float(tr['t_end']),dict(dt=.001),nargout=4)
            R1,bonus,both=project_terminal(float(rT),r1,r2,intercepted=intercepted)
            values=(len(ns),rT,reached,diverged,n,r1,r2,intercepted,R1,bonus,both,e['row_idx'])
            for key,value in zip(fields,values): fields[key].append(value)
            print(f"row={e['row_idx']} min=({r1:.3f},{r2:.3f},{float(rT):.3f}) both_ok={both}",flush=True)
        np.savez_compressed(output_path(out),**{k:np.asarray(v) for k,v in fields.items()},
                            **META,expert_path=str(expert),expert_fingerprint=fingerprint(expert),
                            source_mode='ps_direct_accel',stage3_dt=.001,
                            stage3_version='stage3_pn_dt1e-3_v1')
    finally:
        env.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--expert',default=str(PACKAGE/'data/radau_expert_lib.mat'))
    p.add_argument('--out',default=str(PACKAGE/'data/radau_terminal_cache.npz'))
    a=p.parse_args(); precompute(a.expert,a.out)

if __name__=='__main__': main()
