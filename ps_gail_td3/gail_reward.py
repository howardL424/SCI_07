"""Phase I reward replacement, leaving the physical environment unchanged."""
import gymnasium as gym
import numpy as np

class GAILRewardWrapper(gym.Wrapper):
    def __init__(self,env,discriminator):
        super().__init__(env)
        self.discriminator=discriminator
        self.obs=None

    def reset(self,**kwargs):
        self.obs,info=self.env.reset(**kwargs)
        return self.obs,info

    def step(self,action):
        next_obs,physical_reward,done,truncated,info=self.env.step(action)
        executed=np.asarray(info['a_norm_executed'])
        reward=float(self.discriminator.reward(self.obs[None],executed[None])[0])
        info=dict(info,physical_reward=physical_reward)
        self.obs=next_obs
        return next_obs,reward,done,truncated,info
