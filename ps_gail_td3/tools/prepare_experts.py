"""Run calibration, Radau generation and terminal preparation in order."""
import argparse
from ..common import ROOT, PACKAGE, output_path
from .precompute_radau_terminals import precompute

def main():
    import matlab
    import matlab.engine
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',default=str(ROOT/'samples_clean.mat'))
    p.add_argument('--count',type=int,default=5)
    p.add_argument('--calibrate',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--out',default=str(PACKAGE/'data/radau_expert_lib.mat'))
    p.add_argument('--terminal-cache',default=str(PACKAGE/'data/radau_terminal_cache.npz'))
    a=p.parse_args()
    calibration=output_path(PACKAGE/'data/radau_calibration.mat')
    out=output_path(a.out)
    e=matlab.engine.start_matlab('-nodesktop -nosplash')
    try:
        e.addpath(str(PACKAGE/'expert'),nargout=0)
        e.ps_expert_setup(nargout=0)
        if a.calibrate:
            e.calibrate_radau_weights(a.samples,matlab.double([list(range(1,6))]),str(calibration),nargout=0)
        e.generate_radau_library(float(a.count),dict(samples_file=a.samples,save_file=str(out),
            calibration_file=str(calibration),resume=a.resume),nargout=0)
    finally: e.quit()
    precompute(out,output_path(a.terminal_cache))

if __name__=='__main__': main()
