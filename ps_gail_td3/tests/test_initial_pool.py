import tempfile
from pathlib import Path
import unittest
import numpy as np
import scipy.io
from ps_gail_td3.common import PACKAGE
from ps_gail_td3.direct_env import DirectEnv

class InitialPoolTests(unittest.TestCase):
    def test_zero_rows_filtered_without_changing_original_ids(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE) as td:
            path=Path(td)/'pool.mat'
            rows=np.zeros((5,14)); rows[[1,4],0]=1
            scipy.io.savemat(path,dict(samples_clean=rows))
            env=DirectEnv(path)
            self.assertEqual(env.n_samples,2); self.assertEqual(env.n_zero_rows,3)
            np.testing.assert_array_equal(env.sample_row_ids,[1,4])

if __name__=='__main__': unittest.main()
