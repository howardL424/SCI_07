"""Use a tiny deterministic fixture to verify native TD3 and DDPG handoff."""
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from ps_gail_td3.common import PACKAGE
from ps_gail_td3.gail_discriminator import GAILDiscriminator
from ps_gail_td3.train_gail_td3 import train_phase_one

class FixtureEnv(gym.Env):
    action_space=gym.spaces.Box(-1,1,(2,),dtype=np.float32)
    observation_space=gym.spaces.Box(-5,5,(13,),dtype=np.float32)
    n_clipped_total=0
    def reset(self,seed=None,options=None):
        super().reset(seed=seed); self.t=0
        return np.zeros(13,dtype=np.float32),{}
    def step(self,action):
        self.t+=1
        return np.full(13,self.t/10,np.float32),1.,self.t==4,False,dict(a_norm_executed=action)

class NoopCallback(BaseCallback):
    def _on_step(self): return True

class PhaseSwitchTests(unittest.TestCase):
    def test_td3_and_ddpg_discard_replay_keep_actor_and_resume(self):
        torch.set_num_threads(1)
        for cls in (TD3,DDPG):
            with self.subTest(algo=cls.__name__), tempfile.TemporaryDirectory(dir=PACKAGE) as td:
                env=FixtureEnv()
                model=cls('MlpPolicy',env,buffer_size=64,batch_size=4,
                          policy_kwargs=dict(net_arch=[256,256]),seed=4,device='cpu')
                model.set_logger(configure(td,[]))
                expert=SimpleNamespace(train_s=np.ones((8,13),np.float32),train_a=np.zeros((8,2),np.float32),
                                       val_s=np.ones((4,13),np.float32),val_a=np.zeros((4,2),np.float32))
                args=SimpleNamespace(rollout_buffer_size=16,gail_steps=8,steps=16,
                    seed=4,fresh_episodes=1,noise=.1,d_steps=1,d_batch=4,batch=4,
                    td3_steps_per_iter=2,buf=64)
                actor_id=id(model.actor)
                end=train_phase_one(model,env,GAILDiscriminator(),expert,args,NoopCallback(),Path(td))
                self.assertEqual(end,8); self.assertEqual(model.replay_buffer.size(),0)
                self.assertEqual(id(model.actor),actor_id)
                model.learning_starts=8
                model.learn(8,reset_num_timesteps=False)
                self.assertEqual(model.num_timesteps,16)
                self.assertTrue(np.all(model.replay_buffer.rewards[:8]==1))

if __name__=='__main__': unittest.main()
