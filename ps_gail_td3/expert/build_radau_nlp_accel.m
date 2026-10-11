function P = build_radau_nlp_accel(x0, Tev, cfg)
%build_radau_nlp_accel - Construct direct acceleration Radau collocation
%   P = build_radau_nlp_accel(X0,TEV,CFG) builds the Eq.23 objective.
%   See also solve_radau_accel, radau_dynamics_accel

arguments
    x0 (18,1) double {mustBeFinite}
    Tev (1,1) double {mustBePositive,mustBeFinite}
    cfg (1,1) struct = struct()
end
ps_expert_setup(); import casadi.*
d=struct('K',20,'deg',3,'KD1',4,'KD2',4,'amax_M',8*9.81, ...
    'amax_D',10*9.81,'dstar',20,'beta_min',3,'Jscale',1e5, ...
    'max_iter',300,'tol',1e-3,'print_level',0,'report_terms_only',false);
names=fieldnames(d);
for k=1:numel(names)
    if ~isfield(cfg,names{k}); cfg.(names{k})=d.(names{k}); end
end
if ~cfg.report_terms_only
    % First five samples_clean rows, 2026-10-10 calibration:
    % s_w=469.110038, s_wE=121.007493, s_wT=3.544474, Jscale=3e5.
    % The loaded calibration file is authoritative for each dataset.
    assert(all(isfield(cfg,{'s_w','s_wE','s_wT','calibration_sha'})), ...
        'Run calibrate_radau_weights first; uncalibrated solves are refused');
else
    cfg.s_w=1; cfg.s_wE=1; cfg.s_wT=1;
end
K=cfg.K; deg=cfg.deg; h=Tev/K;
tau=collocation_points(deg,'radau');
[C,D,B]=collocation_coeff(tau);
opti=Opti(); xs=cell(1,K+1); xc=cell(1,K);
% One constant U per interval, not one control per collocation point.
U=opti.variable(2,K);
opti.subject_to(U >= -cfg.amax_M);
opti.subject_to(U <= cfg.amax_M);
xs{1}=opti.variable(18,1); opti.subject_to(xs{1}==x0);
r1=cell(1,K*deg); r2=r1; energy=0; tc=zeros(1,K*deg);
for i=1:K
    xc{i}=opti.variable(18,deg); Z=[xs{i},xc{i}];
    slope=Z*C;
    opti.subject_to(sumsqr(U(:,i))<=cfg.amax_M^2);
    for j=1:deg
        f=radau_dynamics_accel(xc{i}(:,j),U(:,i),cfg);
        opti.subject_to(slope(:,j)==h*f);
        energy=energy+h*B(j)*sumsqr(U(:,i));
        c=(i-1)*deg+j;
        r1{c}=sqrt(sumsqr(xc{i}(1:3,j)-xc{i}(7:9,j))+1e-6);
        r2{c}=sqrt(sumsqr(xc{i}(1:3,j)-xc{i}(13:15,j))+1e-6);
        tc(c)=(i-1+tau(j))*h;
    end
    xs{i+1}=opti.variable(18,1);
    opti.subject_to(xs{i+1}==Z*D);
end
m1=softmin(vertcat(r1{:}),cfg.beta_min);
m2=softmin(vertcat(r2{:}),cfg.beta_min);
mT=casadi_target_zem(xs{end}(1:6));
opti.subject_to(m1>=6); opti.subject_to(m2>=6);
A=(m1-cfg.dstar)^2+(m2-cfg.dstar)^2;
terms=[0.25*A;0.5*energy;0.5*mT^2];
J=[cfg.s_w,cfg.s_wE,cfg.s_wT]*terms;
opti.minimize(J/cfg.Jscale);
opti.solver('ipopt',struct('expand',true),struct('max_iter',cfg.max_iter, ...
    'tol',cfg.tol,'print_level',cfg.print_level, ...
    'hessian_approximation','limited-memory'));
P=struct('opti',opti,'Xstart',{xs},'Xcoll',{xc},'U',U, ...
    'tcoll',tc,'tau',tau,'h',h,'Tev',Tev,'cfg',cfg, ...
    'miss',[m1;m2;mT],'J',J,'terms',terms);
end

function m=softmin(v,beta)
import casadi.*
vmin=mmin(v); m=vmin-beta*log(sum1(exp(-(v-vmin)/beta)));
end
