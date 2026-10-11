import os
import unittest
import numpy as np
from ps_gail_td3.common import PACKAGE, normalize_obs
from ps_gail_td3.expert_data import load_library, validate_transition
from ps_gail_td3.direct_env import DirectEnv

@unittest.skipUnless(os.environ.get('PS_RUN_MATLAB')=='1','Set PS_RUN_MATLAB=1 for real expert replay')
class ExpertConsistencyTests(unittest.TestCase):
    def test_expert_online_replay(self):
        _,entries=load_library(PACKAGE/'data/radau_expert_lib.mat')
        accepted=[e for e in entries if e['ok_use']]
        self.assertTrue(accepted,'No accepted expert trajectories')
        env=DirectEnv()
        try:
            for e in accepted[:5]:
                tr=e['transitions']; s,a,ns=validate_transition(tr)
                obs,_=env.reset_from_abs(e['x0'],float(e['Tev']))
                np.testing.assert_allclose(obs,normalize_obs(s[0]),rtol=0,atol=1e-9)
                for i,action in enumerate(a):
                    obs,_,done,_,info=env.step(action)
                    # Compare raw physical observations, before float32 normalization.
                    np.testing.assert_allclose(env.raw_obs,ns[i],rtol=0,atol=1e-9)
                    self.assertEqual(done,bool(np.asarray(tr['done']).reshape(-1)[i]))
                np.testing.assert_array_equal(env.sabs,np.asarray(tr['terminal_abs']).reshape(18))
        finally: env.close()

if __name__=='__main__': unittest.main()
