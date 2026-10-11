import tempfile
from pathlib import Path
import unittest
import numpy as np
import scipy.io
from ps_gail_td3.common import PACKAGE, META, fingerprint
from ps_gail_td3.expert_data import load_expert_data, validate_transition

class ExpertDataTests(unittest.TestCase):
    def test_success_filter_ignores_cached_flags_and_splits_trajectories(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE) as td:
            path=Path(td)/'library.mat'; cache=Path(td)/'cache.npz'
            entries=[]
            for i in range(4):
                ns=np.zeros((13,2)); ns[0]=[10,20] if i!=3 else [4,20]; ns[4]=[10,20]
                tr=dict(state=ns,a_norm=np.zeros((2,2)),next_state=ns,
                        next_state_abs=np.zeros((18,2)),terminal_abs=np.zeros(18),
                        done=[False,True],intercepted=[False,False])
                entries.append(dict(row_idx=i,ok_use=True,transitions=tr))
            scipy.io.savemat(path,dict(results=dict(meta=META,trajectories=entries)))
            # Deliberately wrong cached flags and minima cannot admit failures.
            np.savez(cache,**META,expert_fingerprint=fingerprint(path),stage3_dt=.001,
                     r_minT=[1,2,50,1],both_ok=[0,0,1,1],r_min1=[100]*4,r_min2=[100]*4)
            data=load_expert_data(path,cache)
            self.assertEqual(set(data.train_ids+data.val_ids),{0,1})
            self.assertFalse(set(data.train_ids)&set(data.val_ids))

    def test_terminal_abs_and_done_must_describe_executed_last_step(self):
        tr=dict(state=np.zeros((13,2)),a_norm=np.zeros((2,2)),next_state=np.zeros((13,2)),
                next_state_abs=np.zeros((18,2)),terminal_abs=np.ones(18),done=[False,True])
        with self.assertRaises(ValueError): validate_transition(tr)

if __name__=='__main__': unittest.main()
