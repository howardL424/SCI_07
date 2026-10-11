import unittest
import numpy as np
from ps_gail_td3.accel_mapping import AMAX, action_norm_to_phys, phys_to_action_norm, project_action

class AccelMappingTests(unittest.TestCase):
    def test_roundtrip(self):
        a=np.random.default_rng(42).uniform(-1,1,(1000,2))
        self.assertLess(np.max(abs(a-phys_to_action_norm(action_norm_to_phys(a)))),1e-9)

    def test_ball_projection_and_audit(self):
        executed,count=project_action([[0,0],[1,1],[-2,0]])
        self.assertEqual(count,2)
        np.testing.assert_allclose(executed[1],[2**-.5,2**-.5],atol=1e-15)
        self.assertLessEqual(np.linalg.norm(action_norm_to_phys(executed),axis=1).max(),AMAX+1e-12)

    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError): project_action([np.nan,0])

if __name__=='__main__': unittest.main()
