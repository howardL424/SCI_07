"""Persist real physical/reward checks and MATLAB static-analysis findings."""
import json
import numpy as np
from ..common import PACKAGE, META, output_path, project_terminal
from reward_projection import project_process_rewards
from ..expert_data import load_library

def main():
    import matlab.engine
    import matlab
    engine=matlab.engine.start_matlab('-nodesktop -nosplash')
    try:
        engine.addpath(str(PACKAGE/'matlab'),str(PACKAGE/'expert'),str(PACKAGE/'tests'),nargout=0)
        core_report=engine.test_ps_core()
        core=engine.ps_core_init()
        cases=[]
        for r1,r2,intercepted,rT in [(8.,15.,False,5.),(30.,40.,False,30.),(5.,15.,True,np.inf)]:
            fact=dict(r_MD1=r1,r_MD2=r2,intercepted=intercepted)
            actual=float(engine.stage2_process_reward(core,fact))
            obs=np.zeros((1,13)); obs[0,0]=r1; obs[0,4]=r2
            expected=float(project_process_rewards(obs,np.array([intercepted]))[0])
            # Shared main-route projection uses float32 replay rewards.
            assert np.float32(actual)==np.float32(expected)
            R1,bonus,both=project_terminal(rT,r1,r2,intercepted=intercepted)
            cases.append(dict(r1=r1,r2=r2,process=actual,R1=R1,bonus=bonus,both_ok=both))
        comparisons=[]
        library=PACKAGE/'data/radau_expert_lib.mat'
        if library.exists():
            _,entries=load_library(library)
            for e in [e for e in entries if e['ok_use']][:5]:
                ref=np.asarray(engine.ps_reference_minima(
                    matlab.double(np.asarray(e['x0']).reshape(-1,1).tolist()),float(e['Tev']))).reshape(2)
                tr=e['transitions']
                comparisons.append(dict(row_idx=int(e['row_idx']),
                    direct_expert_r_min1=float(tr['r_min1']),direct_expert_r_min2=float(tr['r_min2']),
                    main_fixed_weight_r_min1=float(ref[0]),main_fixed_weight_r_min2=float(ref[1])))
        issues={}
        for file in sorted(PACKAGE.rglob('*.m')):
            findings=engine.ps_checkcode_json(str(file))
            issues[str(file.relative_to(PACKAGE))]=json.loads(findings)
        report=dict(meta=META,ballistic_error=float(core_report['ballistic_error']),
                    equivalent_error=float(core_report['equivalent_error']),
                    reward_cases=cases,reference_comparison=comparisons,static_analysis=issues)
        output_path(PACKAGE/'logs/matlab_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    finally: engine.quit()

if __name__=='__main__': main()
