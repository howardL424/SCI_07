function [obs, sabs, Tev, zT0, ok, warm] = ps_env_reset(row, cfg)
%ps_env_reset - Reconstruct a sample and extract the shared warm window
%   [OBS,SABS,TEV,ZT0,OK,WARM] = ps_env_reset(ROW,CFG) initializes stage2.
%   See also ps_env_step, reconstruct_geometry

arguments
    row (1,14) double {mustBeFinite}
    cfg (1,1) struct = struct()
end
core=ps_core_init(cfg);
[M,D1,D2]=reconstruct_geometry(row(:)');
% Use the same single warm trajectory as env_reset, not weight search.
warm=simulate_warmstart_trajectory(M,D1,D2,cfg);
idx=find(warm.stage==2);
ok=~isempty(idx);
if ~ok
    obs=zeros(13,1); sabs=zeros(18,1); Tev=10; zT0=1e4;
    return
end
i0=idx(1); i1=idx(end);
sabs=[warm.M(:,i0);warm.D1(:,i0);warm.D2(:,i0)];
Tev=max(warm.t(i1)-warm.t(i0),1);
zT0=target_zem(sabs(1:6)');
obs=stage2_obs_from_abs(core,sabs,0,Tev);
end
