function [obs, snext, tnew, reward, fact] = ps_env_step(core, sabs, t, Tev, action)
%ps_env_step - Advance direct dynamics and apply the shared process reward
%   [OBS,SNEXT,TNEW,REWARD,FACT] = ps_env_step(CORE,SABS,T,TEV,ACTION)
%   returns physical transition facts and the Eq.74 process reward.
%   See also ps_step, stage2_process_reward

arguments
    core (1,1) struct
    sabs (18,1) double {mustBeFinite}
    t (1,1) double {mustBeFinite,mustBeNonnegative}
    Tev (1,1) double {mustBeFinite,mustBePositive}
    action (2,1) double {mustBeFinite}
end
[obs,snext,tnew,fact]=ps_step(core,sabs,t,Tev,action);
reward=stage2_process_reward(core,fact);
end
