function tr = extract_accel_transitions(res, cfg)
%extract_accel_transitions - Replay projected interval controls at 0.1s
%   TR = extract_accel_transitions(RES,CFG) stores executed transitions.
%   See also ps_step, generate_radau_library

arguments
    res (1,1) struct
    cfg (1,1) struct = struct()
end
core=ps_core_init(cfg); s=res.x0; t=0; K=size(res.U_K,2);
nmax=ceil(res.Tev/core.cfg.dt)+1;
state=zeros(13,nmax); next=state; action=zeros(2,nmax);
next_abs=zeros(18,nmax); done=false(1,nmax); intercepted=done;
stage2_done=done; times=zeros(1,nmax); clips=0; r1=inf; r2=inf;
n=0;
while t<res.Tev
    n=n+1; interval=min(K,floor(t/(res.Tev/K))+1);
    raw=res.U_K(:,interval)/core.max_ov_M;
    a=min(max(raw,-1),1); a=a/max(1,norm(a));
    clips=clips+double(any(abs(raw-a)>1e-12));
    state(:,n)=stage2_obs_from_abs(core,s,t,res.Tev);
    times(n)=t; action(:,n)=a;
    [next(:,n),s,t,f]=ps_step(core,s,t,res.Tev,a);
    next_abs(:,n)=s; done(n)=f.done; intercepted(n)=f.intercepted;
    stage2_done(n)=f.stage2_done;
    r1=min(r1,f.r_MD1); r2=min(r2,f.r_MD2);
    if f.done; break; end
end
assert(n>0 && sum(done(1:n))==1);
assert(clips/n<=0.05,'Expert clipping >5%%: tighten NLP control bounds');
tr=struct('state',state(:,1:n),'a_norm',action(:,1:n), ...
    'next_state',next(:,1:n),'next_state_abs',next_abs(:,1:n), ...
    'done',done(1:n),'intercepted',intercepted(1:n), ...
    'stage2_done',stage2_done(1:n),'t',times(1:n), ...
    'r_min1',r1,'r_min2',r2,'terminal_abs',s,'t_end',t, ...
    'n_clipped',clips,'meta',ps_transition_meta());
end
