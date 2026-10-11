function [obs, snext, tnew, fact] = ps_step(core, sabs, t, Tev, action)
%ps_step - Advance three vehicles with direct acceleration and frozen PN
%   [OBS,SNEXT,TNEW,FACT] = ps_step(CORE,SABS,T,TEV,ACTION) performs RK4.
%   See also ps_core_init, ps_env_step

arguments
    core (1,1) struct
    sabs (18,1) double {mustBeFinite}
    t (1,1) double {mustBeFinite,mustBeNonnegative}
    Tev (1,1) double {mustBeFinite,mustBePositive}
    action (2,1) double {mustBeFinite}
end
a = double(action(:));
assert(numel(a)==2 && all(isfinite(a)));
raw = a;
a = min(max(a,-1),1);
a = a/max(1,norm(a));
aM = a*core.max_ov_M;
M = sabs(1:6)'; D1 = sabs(7:12)'; D2 = sabs(13:18)';
H = core.H;
[qy1,qz1,r1] = H.computeLOS(D1(1:3),M(1:3));
[qy2,qz2,r2] = H.computeLOS(D2(1:3),M(1:3));
[~,dy1,dz1] = H.computeMDDot(M,D1,r1,qy1,qz1);
[~,dy2,dz2] = H.computeMDDot(M,D2,r2,qy2,qz2);
aD1 = H.saturate(H.computePN(core.cfg.KD1,D1(6),dy1,dz1,D1(4)),core.max_ov_D);
aD2 = H.saturate(H.computePN(core.cfg.KD2,D2(6),dy2,dz2,D2(4)),core.max_ov_D);
% Match stage2_step: PN commands remain fixed within each RK4 step.
M = H.updateState(M,aM,core.cfg.dt);
D1 = H.updateState(D1,aD1,core.cfg.dt);
D2 = H.updateState(D2,aD2,core.cfg.dt);
snext = [M(:);D1(:);D2(:)]; tnew = t+core.cfg.dt;
obs = stage2_obs_from_abs(core,snext,tnew,Tev);
intercepted = min(obs([1 5])) <= core.cfg.kill_radius;
stage2_done = tnew >= Tev && ~intercepted;
fact = struct('r_MD1',obs(1),'r_MD2',obs(5),'r_MT',obs(9), ...
    'intercepted',intercepted,'stage2_done',stage2_done, ...
    'done',intercepted||stage2_done,'terminated',intercepted||stage2_done, ...
    'a_M_sat',aM,'a_norm_executed',a, ...
    'n_clipped',double(any(abs(a-raw)>1e-12)));
end
