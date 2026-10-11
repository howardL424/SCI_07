import unittest
import numpy as np
import torch
from ps_gail_td3.gail_discriminator import GAILDiscriminator

class GAILProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5); torch.set_num_threads(1)
        self.d=GAILDiscriminator()
        self.s=np.zeros((32,13),np.float32); self.a=np.zeros((32,2),np.float32)

    def test_stale_or_future_negatives_rejected(self):
        for stamp in (1,5):
            with self.assertRaises(AssertionError):
                self.d.update(self.s,self.a,self.s,self.a,np.full(32,stamp),4)

    def test_loss_learns_and_reward_bounds(self):
        expert=self.s+1
        losses=[self.d.update(expert,self.a,self.s,self.a,np.full(32,3),3) for _ in range(25)]
        self.assertLess(losses[-1],losses[0])
        reward=self.d.reward(self.s,self.a)
        self.assertTrue(np.all((-10<=reward)&(reward<=0)))
        self.assertGreater(self.d.diagnostics(expert,self.a),self.d.diagnostics(self.s,self.a))

if __name__=='__main__': unittest.main()
