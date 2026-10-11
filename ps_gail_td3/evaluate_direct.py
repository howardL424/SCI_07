"""Deterministic full-test MC evaluation and optional factual comparison rows."""
import argparse
import csv
import json
import re
from datetime import datetime
import numpy as np
from stable_baselines3 import TD3, DDPG
from .common import ROOT, PACKAGE, META, output_path, fingerprint
from .direct_env import DirectEnv

def evaluate(model_path,algo,samples,out,limit=0):
    env=DirectEnv(samples)
    rows=[]
    try:
        model=(TD3 if algo=='td3' else DDPG).load(model_path)
        if model.action_space.shape!=(2,) or model.observation_space.shape!=(13,):
            raise ValueError('Expected a direct acceleration 13D / 2D model')
        n=env.n_samples if not limit else min(limit,env.n_samples)
        for idx in range(n):
            try: obs,info=env.reset(options=dict(local_idx=idx))
            except RuntimeError as exc:
                if 'warm-start' not in str(exc): raise
                rows.append(dict(local_idx=idx,row_idx=int(env.sample_row_ids[idx]),valid=False))
                continue
            total=0.
            while True:
                action=model.predict(obs,deterministic=True)[0]
                obs,reward,done,truncated,info=env.step(action)
                total+=reward
                if done or truncated: break
            keys=('r_min1','r_min2','r_minT','intercepted','evade_ok','strike_ok','both_ok',
                  'R1','bonus','reached','stage2_energy_int','stage3_energy_int','n_clipped')
            row={k:info[k] for k in keys}
            row.update(r_MD1_min=info['r_min1'],r_MD2_min=info['r_min2'],
                       energy_kJ=(info['stage2_energy_int']+info['stage3_energy_int'])/1000)
            row.update(local_idx=idx,row_idx=int(env.sample_row_ids[idx]),valid=True,episode_reward=total)
            rows.append(row); print(row,flush=True)
        valid=[r for r in rows if r['valid']]
        summary=dict(attempted=n,valid=len(valid),warm_failed=n-len(valid),
                     **{k:float(np.mean([r[k] for r in valid])) if valid else None
                        for k in ('evade_ok','strike_ok','both_ok','episode_reward')})
        record=dict(meta=META,model_path=str(model_path),algo=algo,
                    model_sha=fingerprint(model_path),samples_sha=fingerprint(samples),
                    deterministic=True,rows=rows,summary=summary)
        path=output_path(out)
        with open(path.with_suffix('.json'),'w',encoding='utf-8') as f: json.dump(record,f,indent=2)
        with open(path.with_suffix('.log'),'w',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
            f.write(json.dumps(summary)+'\n')
        keys=sorted(set().union(*(r.keys() for r in rows))) if rows else []
        with open(path.with_suffix('.csv'),'w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(path.with_suffix('.npz'),**META,
                            **{k:np.asarray([r.get(k,np.nan) for r in rows]) for k in keys})
        return record
    finally: env.close()

def comparison_table(records,out):
    """Rows reference completed reports; missing evidence stays blank."""
    names=['PN','正弦','纯TD3直驱','GAIL-TD3直驱','主路线BC-TD3']
    references=paper_reference_rows()
    lines=['| 算法 | 规避率 | 打击率 | 联合率 | 来源 |',
           '|---|---:|---:|---:|---|']
    for name in names:
        report=records.get(name)
        if report is None:
            if name in references:
                s=references[name]
                lines.append(f"| {name} | {s['evade_ok']:.2%} | — | {s['both_ok']:.2%} | 论文既有300次结果，未重跑 |")
            else: lines.append(f'| {name} | — | — | — | 未提供结果 |')
            continue
        with open(report,encoding='utf-8') as f: result=json.load(f)
        s=result['summary']
        lines.append(f"| {name} | {s['evade_ok']:.2%} | {s['strike_ok']:.2%} | {s['both_ok']:.2%} | {report} |")
    lines.extend(['', '历史参考来自本地论文 `sections/06_simulation.tex` 的 `tab:perf-nonoise`。',
                  '该表的“目标到达率”包含规避成功条件，故放在联合率列；独立打击率未给出。',
                  '短训/少量初态的运行结果仅作流程验证，不与历史300次结果作性能结论。'])
    output_path(out).write_text('\n'.join(lines)+'\n',encoding='utf-8')

def paper_reference_rows():
    source=ROOT/'论文文稿/overleaf_SCI_07_0914/sections/06_simulation.tex'
    if not source.is_file(): return {}
    text=source.read_text(encoding='utf-8')
    marker=r'\label{tab:perf-nonoise}'
    if marker not in text: return {}
    table=text.split(marker,1)[1].split(r'\end{table}',1)[0]
    references={}
    for label,prefix in [('PN','仅对目标比例导引'),('正弦','程序机动（正弦机动）'),('主路线BC-TD3','BC+TD3')]:
        match=re.search(re.escape(prefix)+r'\s*&\s*([\d.]+)\\%\s*&\s*([\d.]+)\\%',table)
        if match:
            references[label]=dict(evade_ok=float(match[1])/100,both_ok=float(match[2])/100,
                                   source=str(source),source_sha=fingerprint(source))
    return references

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--algo',choices=['td3','ddpg'],default='td3')
    p.add_argument('--samples',default=str(ROOT/'samples_clean_test.mat'))
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--out',default=str(PACKAGE/'logs'/datetime.now().strftime('monte_carlo_eval_%Y%m%d_%H%M%S')))
    p.add_argument('--comparison-inputs',help='JSON mapping algorithm names to existing report JSON files')
    a=p.parse_args(); evaluate(a.model,a.algo,a.samples,a.out,a.limit)
    if a.comparison_inputs:
        with open(a.comparison_inputs,encoding='utf-8') as f: reports=json.load(f)
        comparison_table(reports,PACKAGE/'logs/comparison.md')

if __name__=='__main__': main()
