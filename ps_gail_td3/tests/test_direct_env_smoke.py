import os
import unittest
import numpy as np
from ps_gail_td3.direct_env import DirectEnv
from ps_gail_td3.common import ROOT

@unittest.skipUnless(os.environ.get('PS_RUN_MATLAB')=='1','Set PS_RUN_MATLAB=1 for real MATLAB integration')
class DirectEnvSmokeTests(unittest.TestCase):
    def test_five_episodes(self):
        env=DirectEnv(ROOT/'samples_clean_val.mat')
        try:
            self.assertGreater(env.n_samples,0)
            for i in range(5):
                obs,_=env.reset(seed=i,options=dict(local_idx=i))
                self.assertEqual(obs.shape,(13,))
                for step in range(1000):
                    obs,reward,done,truncated,info=env.step(np.zeros(2))
                    self.assertTrue(np.isfinite(obs).all()); self.assertTrue(np.isfinite(reward))
                    if done: break
                else: self.fail('Episode did not terminate')
                self.assertIn('both_ok',info)
        finally: env.close()

if __name__=='__main__': unittest.main()
