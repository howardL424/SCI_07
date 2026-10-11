"""GAIL discriminator: expert=1, fresh policy=0, Hui log(D) reward."""
import numpy as np
import torch
from torch import nn

class GAILDiscriminator(nn.Module):
    def __init__(self, lr=3e-4, reward_mode='log_d', device='cpu'):
        super().__init__()
        if reward_mode not in ('log_d','neg_log_one_minus_d'):
            raise ValueError(reward_mode)
        self.reward_mode=reward_mode
        self.device=torch.device(device)
        self.net=nn.Sequential(nn.Linear(15,256),nn.ReLU(),nn.Linear(256,256),
                               nn.ReLU(),nn.Linear(256,1))
        self.to(self.device)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)

    def _input(self,s,a):
        return torch.as_tensor(np.concatenate((s,a),axis=-1),dtype=torch.float32,device=self.device)

    def forward(self,s,a):
        return torch.sigmoid(self.net(self._input(s,a)))

    @torch.no_grad()
    def reward(self,s,a):
        d=self(s,a).flatten()
        if self.reward_mode=='log_d':
            return torch.log(d+1e-8).clamp(-10,0).cpu().numpy()
        return (-torch.log(1-d+1e-8)).clamp(0,10).cpu().numpy()

    def update(self,expert_s,expert_a,fresh_s,fresh_a,round_ids,current_round):
        stamps=np.asarray(round_ids)
        if len(stamps)!=len(fresh_s) or np.any(stamps>current_round) or np.any(current_round-stamps>1):
            raise AssertionError('Discriminator negatives must be fresh same-policy rollouts')
        # Standard BCE orientation corresponds to Eq.53 with expert=1.
        real=self.net(self._input(expert_s,expert_a))
        fake=self.net(self._input(fresh_s,fresh_a))
        loss=nn.functional.binary_cross_entropy_with_logits(real,torch.ones_like(real))
        loss+=nn.functional.binary_cross_entropy_with_logits(fake,torch.zeros_like(fake))
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def diagnostics(self,s,a):
        return float(self(s,a).mean())
