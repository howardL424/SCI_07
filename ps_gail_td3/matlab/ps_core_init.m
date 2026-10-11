function core = ps_core_init(cfg)
%ps_core_init - Initialize shared helpers for direct acceleration
%   CORE = ps_core_init(CFG) merges physical and reward defaults into CFG.
%   See also ps_step, ps_env_reset

arguments
    cfg (1,1) struct = struct()
end
branch = fileparts(fileparts(mfilename('fullpath')));
root = fileparts(branch);
addpath(root, fullfile(root,'expert_trajectory'), ...
    fullfile(root,'rl_td3','matlab'), fullfile(branch,'expert'));
d = struct('dt',0.1,'dstar',20,'KD1',4,'KD2',4,'g0',9.81, ...
    'kill_radius',6,'proc_coef',0.05,'intercept_pen',50, ...
    'amax_M',8*9.81,'amax_D',10*9.81);
names = fieldnames(d);
for k=1:numel(names)
    if ~isfield(cfg,names{k}); cfg.(names{k})=d.(names{k}); end
end
assert(cfg.dt>0 && cfg.amax_M>0 && cfg.amax_D>0);
core = struct('cfg',cfg,'H',evasion_helpers(), ...
    'meta',ps_transition_meta(),'T_pos',[0 0 0], ...
    'T_state',zeros(1,6),'max_ov_M',cfg.amax_M, ...
    'max_ov_D',cfg.amax_D);
end
