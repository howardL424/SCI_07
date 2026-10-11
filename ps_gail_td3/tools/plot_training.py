"""Export fixed-pool reward and success curves from completed training logs."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..common import PACKAGE, output_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',default=str(PACKAGE/'logs/training_curves.png'))
    a=p.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for run in a.runs:
        run=Path(run); evaluation=np.load(run/'eval_logs/evaluations.npz')
        label='GAIL-TD3' if (run/'gail_metrics.jsonl').exists() else 'Pure TD3'
        steps=evaluation['timesteps']
        axes[0].plot(steps,evaluation['results'].mean(axis=1),marker='o',label=label)
        axes[1].plot(steps,evaluation['successes'].mean(axis=1),marker='o',label=label)
    for ax in axes:
        ax.set_xlabel('Environment steps'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel('Fixed-validation mean reward')
    axes[1].set_ylabel('Joint success rate'); axes[1].set_ylim(0,1.05)
    fig.suptitle('Short-run verification; five fixed validation initial conditions')
    fig.savefig(output_path(a.out),dpi=180)
    plt.close(fig)

if __name__=='__main__': main()
