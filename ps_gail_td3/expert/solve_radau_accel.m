function res = solve_radau_accel(x0, Tev, warm, cfg)
%solve_radau_accel - Initialize Radau states and solve or report raw terms
%   RES = solve_radau_accel(X0,TEV,WARM,CFG) records solver diagnostics.
%   See also build_radau_nlp_accel, calibrate_radau_weights

arguments
    x0 (18,1) double {mustBeFinite}
    Tev (1,1) double {mustBePositive,mustBeFinite}
    warm (1,1) struct
    cfg (1,1) struct = struct()
end
P=build_radau_nlp_accel(x0,Tev,cfg); cfg=P.cfg;
idx=find(warm.stage==2); tw=warm.t(idx)-warm.t(idx(1));
Xw=[warm.M(:,idx);warm.D1(:,idx);warm.D2(:,idx)];
if numel(tw)<2
    Xw=[x0,x0]; tw=[0,Tev];
end
sample=@(t) interp1(tw,Xw',min(max(t,0),tw(end)),'linear')';
for i=1:cfg.K+1
    P.opti.set_initial(P.Xstart{i},sample((i-1)*P.h));
    if i<=cfg.K
        P.opti.set_initial(P.Xcoll{i},sample((i-1+P.tau)*P.h));
    end
end
P.opti.set_initial(P.U,zeros(2,cfg.K));
res=struct('ok',false,'debug',false,'U_K',[],'x0',x0,'Tev',Tev, ...
    'cfg',cfg,'miss',[NaN;NaN;NaN],'J',NaN,'terms',[], ...
    'status','not_started','solve_seconds',0);
if cfg.report_terms_only
    res.terms=full(P.opti.debug.value(P.terms,P.opti.initial()));
    res.miss=full(P.opti.debug.value(P.miss,P.opti.initial()));
    res.status='report_terms_only'; return
end
clock0=tic;
try
    sol=P.opti.solve(); value=@(v) sol.value(v);
    res.ok=true; res.status=sol.stats().return_status;
catch err
    res.debug=true; res.status=err.message;
    value=@(v) P.opti.debug.value(v);
end
res.solve_seconds=toc(clock0);
try
    res.U_K=full(value(P.U)); res.miss=full(value(P.miss));
    res.J=full(value(P.J)); res.terms=full(value(P.terms));
catch
    res.U_K=[];
end
res.ok_use=~isempty(res.U_K) && all(isfinite(res.U_K(:))) && ...
    (res.ok || (res.debug && res.miss(1)>=6 && ...
    res.miss(2)>=6 && res.miss(3)<=30));
end
