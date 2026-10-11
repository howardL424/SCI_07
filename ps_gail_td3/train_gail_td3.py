"""Fresh-discriminator GAIL-guided TD3/DDPG, then native SB3 fine tuning.

Run from the repository root: python -m ps_gail_td3.train_gail_td3.
Phase I actor is already the Phase II actor: no cross-network transfer.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from datetime import datetime
import sys
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from .common import ROOT, PACKAGE, META, output_path, fingerprint
from fixed_ic_eval import FixedICEvalEnv, FixedICEvalCallback, save_ic_pool
from .direct_env import DirectEnv
from .expert_data import load_expert_data
from .gail_discriminator import GAILDiscriminator
from .gail_reward import GAILRewardWrapper

DEFAULTS=dict(steps=300000,gail_steps=200000,lr=1e-4,batch=256,buf=300000,
              noise=.1,gamma=.99,tau=.005,learning_starts=1000,policy_delay=3,
              net_arch=[256,256],eval_freq=25000,n_eval=150,grad_clip=10.)

class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,s):
        for stream in self.streams: stream.write(s); stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams: stream.flush()

class ClipGradients(BaseCallback):
    def _on_training_start(self):
        self.handles=[]
        # Optimizer pre-hooks clip the whole parameter norm after backward.
        def clip(opt,args,kwargs):
            torch.nn.utils.clip_grad_norm_(
                [p for group in opt.param_groups for p in group['params']],10.)
        for module in (self.model.actor,self.model.critic):
            self.handles.append(module.optimizer.register_step_pre_hook(clip))
    def _on_step(self): return True
    def _on_training_end(self):
        for h in self.handles: h.remove()

def new_replay(model,capacity):
    return ReplayBuffer(capacity,model.observation_space,model.action_space,
                        device=model.device,n_envs=1,handle_timeout_termination=True)

def train_phase_one(model,env,discriminator,expert,args,callback,out):
    # Dedicated FIFO rollout buffer. Never populated with expert transitions.
    model.replay_buffer=new_replay(model,args.rollout_buffer_size)
    _, callback=model._setup_learn(args.gail_steps,callback,reset_num_timesteps=True)
    callback.on_training_start(locals(),globals())
    rng=np.random.default_rng(args.seed)
    rollout_env=GAILRewardWrapper(env,discriminator)
    iteration=0
    with open(output_path(out/'gail_metrics.jsonl'),'w',encoding='utf-8') as metrics:
        while model.num_timesteps<args.gail_steps:
            iteration+=1
            batch=[]
            # Actor remains fixed until all fresh episodes have been collected.
            for _ in range(args.fresh_episodes):
                obs=rollout_env.reset()[0]
                while True:
                    action=model.predict(obs,deterministic=True)[0]
                    action=np.clip(action+rng.normal(0,args.noise,2),-1,1).astype(np.float32)
                    next_obs,_,done,truncated,info=rollout_env.step(action)
                    # Discriminator sees executed action. TD3 learns the action
                    # actually requested in the Box (projection is environment).
                    batch.append((obs.copy(),action,next_obs.copy(),done or truncated,
                                  info,np.asarray(info['a_norm_executed']),iteration))
                    model.num_timesteps+=1
                    callback.update_locals(locals())
                    if not callback.on_step():
                        raise RuntimeError('Training stopped by callback')
                    obs=next_obs
                    if done or truncated or model.num_timesteps>=args.gail_steps: break
                if model.num_timesteps>=args.gail_steps: break
            s=np.stack([b[0] for b in batch]); a=np.stack([b[5] for b in batch])
            stamps=np.array([b[6] for b in batch])
            # Hui Step2->5: score before updating D, then update D and actor.
            rewards=discriminator.reward(s,a)
            losses=[]
            for _ in range(args.d_steps):
                ei=rng.integers(len(expert.train_s),size=args.d_batch)
                fi=rng.integers(len(batch),size=args.d_batch)
                losses.append(discriminator.update(expert.train_s[ei],expert.train_a[ei],
                              s[fi],a[fi],stamps[fi],iteration))
            for b,reward in zip(batch,rewards):
                model.replay_buffer.add(b[0][None],b[2][None],b[1][None],
                                        np.array([reward]),np.array([b[3]]),[b[4]])
            model._update_current_progress_remaining(model.num_timesteps,args.steps)
            model.train(gradient_steps=args.td3_steps_per_iter,batch_size=args.batch)
            row=dict(iteration=iteration,steps=model.num_timesteps,fresh_steps=len(batch),
                     d_loss=float(np.mean(losses)),reward_mean=float(rewards.mean()),
                     d_expert_train=discriminator.diagnostics(expert.train_s,expert.train_a),
                     d_expert_val=discriminator.diagnostics(expert.val_s,expert.val_a),
                     n_clipped=env.n_clipped_total)
            metrics.write(json.dumps(row)+'\n'); metrics.flush(); print(row,flush=True)
            model.logger.dump(model.num_timesteps)
    callback.on_training_end()
    torch.save(dict(model=discriminator.state_dict(),optimizer=discriminator.optimizer.state_dict(),
                    reward_mode=discriminator.reward_mode),output_path(out/'discriminator.pt'))
    model.save(str(output_path(out/'phase1_model.zip')))
    # Physically discard Phase I buffer before constructing Phase II replay.
    model.replay_buffer=None
    model.replay_buffer=new_replay(model,args.buf)
    model._last_obs=None
    return model.num_timesteps

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('steps','gail_steps','batch','buf','eval_freq','n_eval'):
        p.add_argument('--'+name.replace('_','-'),type=int,default=DEFAULTS[name])
    for name in ('lr','noise'):
        p.add_argument('--'+name,type=float,default=DEFAULTS[name])
    p.add_argument('--algo',choices=['td3','ddpg'],default='td3')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--eval-ic-seed',type=int,default=0)
    p.add_argument('--train-samples',default=str(ROOT/'samples_clean.mat'))
    p.add_argument('--eval-samples',default=str(ROOT/'samples_clean_val.mat'))
    p.add_argument('--expert',default=str(PACKAGE/'data/radau_expert_lib.mat'))
    p.add_argument('--terminal-cache',default=str(PACKAGE/'data/radau_terminal_cache.npz'))
    p.add_argument('--save-dir',default=str(PACKAGE/'checkpoints'))
    p.add_argument('--fresh-episodes',type=int,default=10)
    p.add_argument('--rollout-buffer-size',type=int,default=50000)
    p.add_argument('--td3-steps-per-iter',type=int,default=1000)
    p.add_argument('--d-steps',type=int,default=1)
    p.add_argument('--d-batch',type=int,default=256)
    p.add_argument('--d-lr',type=float,default=3e-4)
    p.add_argument('--reward-mode',choices=['log_d','neg_log_one_minus_d'],default='log_d')
    p.add_argument('--device',default='auto')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv)
    if args.smoke:
        if args.steps==300000: args.steps=5000
        if args.gail_steps==200000: args.gail_steps=1000
        if args.eval_freq==25000: args.eval_freq=1000
        if args.n_eval==150: args.n_eval=5
        if args.td3_steps_per_iter==1000: args.td3_steps_per_iter=10
        if args.save_dir==str(PACKAGE/'checkpoints'): args.save_dir=str(PACKAGE/'checkpoints_smoke')
    if not 0<=args.gail_steps<=args.steps or args.steps<=0:
        p.error('Require 0 <= gail-steps <= steps, steps > 0')
    if any(getattr(args,k)<=0 for k in ('fresh_episodes','rollout_buffer_size','td3_steps_per_iter','d_steps','d_batch','batch','buf','eval_freq')):
        p.error('Buffer, batch, update and evaluation sizes must be positive')
    return args

def run(args):
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    out=output_path(Path(args.save_dir)/datetime.now().strftime('run_%Y%m%d_%H%M%S')/'run_config.json').parent
    cfg=vars(args).copy(); cfg.update(META)
    cfg.update(defaults=DEFAULTS,phase1_target=args.gail_steps,phase2_target=args.steps,
               discriminator=dict(hidden_sizes=[256,256],expert_label=1,policy_label=0,
                                  lr=args.d_lr,batch=args.d_batch,steps=args.d_steps,
                                  reward_mode=args.reward_mode),
               fresh_protocol='same_actor_rollout_then_score_then_D_then_TD3',
               mixed_replay_fallback=False,rollout_noise=args.noise,
               samples_fingerprint=fingerprint(args.train_samples),
               eval_samples_fingerprint=fingerprint(args.eval_samples))
    expert=None
    if args.gail_steps:
        expert=load_expert_data(args.expert,args.terminal_cache,args.seed)
        cfg.update(expert_fingerprint=expert.expert_fingerprint,expert_train_ids=expert.train_ids,
                   expert_val_ids=expert.val_ids,terminal_cache_fingerprint=fingerprint(args.terminal_cache))
    else: cfg['expert_fingerprint']=None
    with open(out/'run_config.json','w',encoding='utf-8') as f: json.dump(cfg,f,indent=2)
    env=DirectEnv(args.train_samples)
    eval_env=None
    try:
        eval_env=FixedICEvalEnv(Monitor(DirectEnv(args.eval_samples)),n_ic=args.n_eval,order_seed=args.eval_ic_seed)
        save_ic_pool(eval_env,output_path(out/'eval_logs/eval_ic_pool.json'))
        cls=TD3 if args.algo=='td3' else DDPG
        kw=dict(learning_rate=args.lr,buffer_size=args.buf,batch_size=args.batch,
                gamma=.99,tau=.005,train_freq=(1,'step'),gradient_steps=1,
                learning_starts=1000,policy_kwargs=dict(net_arch=[256,256]),
                action_noise=NormalActionNoise(np.zeros(2),args.noise*np.ones(2)),
                seed=args.seed,device=args.device,verbose=1)
        if args.algo=='td3': kw['policy_delay']=3
        model=cls('MlpPolicy',Monitor(env),**kw)
        model.set_logger(configure(str(out/'training_logs'),['stdout','csv']))
        callback=CallbackList([CheckpointCallback(save_freq=50000,save_path=str(out),name_prefix=args.algo),
            FixedICEvalCallback(eval_env,n_eval_episodes=len(eval_env.ic_local_idx),
                eval_freq=args.eval_freq,best_model_save_path=str(out/'best'),log_path=str(out/'eval_logs'),
                deterministic=True),ClipGradients()])
        if args.gail_steps:
            d=GAILDiscriminator(args.d_lr,args.reward_mode,device=str(model.device))
            phase_end=train_phase_one(model,env,d,expert,args,callback,out)
        else: phase_end=0
        cfg.update(phase1_actual_end=phase_end,phase2_replay_initial_size=model.replay_buffer.size())
        with open(out/'run_config.json','w',encoding='utf-8') as f: json.dump(cfg,f,indent=2)
        if args.steps>phase_end:
            # Fresh physical-reward replay receives its own baseline warmup.
            model.learning_starts=phase_end+DEFAULTS['learning_starts']
            model.learn(total_timesteps=args.steps-phase_end,callback=callback,reset_num_timesteps=not bool(phase_end))
        model.save(str(output_path(out/'final_model.zip')))
        print(f'Saved {out}')
    finally:
        env.close()
        if eval_env is not None: eval_env.close()

def main():
    args=parse_args()
    log=output_path(PACKAGE/'logs'/datetime.now().strftime('train_%Y%m%d_%H%M%S.log'))
    with open(log,'w',encoding='utf-8') as f, redirect_stdout(Tee(sys.stdout,f)), redirect_stderr(Tee(sys.stderr,f)):
        run(args)

if __name__=='__main__': main()
