function res = solve_ss_trajectory(tau_warm, cfg)
%SOLVE_SS_TRAJECTORY 规避段单打靶优化 (时变 theta + 符号动力学RK4滚动 + IPOPT)。
%   决策变量仅为分段时变权重 theta=[gamma;wT;wE] (K 段, w 固定); 状态由 casadi
%   闭环动力学(式47+式5) RK4 积分得到, 无缺陷约束 -> 鲁棒。目标=固定系数式22。
%   res 字段同 solve_ps_trajectory (ok/theta/tcoll/Xcoll/miss/J)。

    import casadi.*
    if nargin < 2; cfg = struct(); end
    d.K = 10; d.nsub = 12; d.w = 1e4; d.dstar = 20; d.KD1 = 4; d.KD2 = 4;
    d.amax = 8*9.81; d.dfloor = 6; d.beta_min = 3; d.Jscale = 1e5; d.tol = 1e-3;
    % 固定评分系数 (经 calibrate_J_weights 重标定: 令规避/能量/目标三项数量级平衡,
    % 消除"miss 压到下限 + 权重贴边"退化; 旧值 1e4/1.44e3/1e3 致目标项~5e8 主导)
    d.s_w = 2e3; d.s_wE = 55; d.s_wT = 22;
    d.gamma_lo = -3; d.gamma_hi = 3; d.wT_lo = 1; d.wT_hi = 1e5;
    d.wE_lo = 10; d.wE_hi = 1e4;
    d.report_terms_only = false;   % true: 仅在 warm θ 处测三项原始量级 (标定用, 不求解)
    d.max_iter = 300; d.print_level = 0;  % 批处理时 300 iter 足够; 失败条快速退出
    fn = fieldnames(d); for k=1:numel(fn); if ~isfield(cfg,fn{k}); cfg.(fn{k})=d.(fn{k}); end; end

    % ---------------- 抽取规避段 ----------------
    idx2 = find(tau_warm.stage == 2);
    if isempty(idx2) || numel(idx2) < 5
        res = struct('ok', false, 'reason', 'no_evasion_segment'); return;
    end
    i0 = idx2(1); i1 = idx2(end);
    x0 = [tau_warm.M(:,i0); tau_warm.D1(:,i0); tau_warm.D2(:,i0)];
    Tev = tau_warm.t(i1) - tau_warm.t(i0);
    wE0 = tau_warm.meta.cfg.wE; wT0 = tau_warm.meta.cfg.wT;

    params = struct('w', cfg.w, 'KD1', cfg.KD1, 'KD2', cfg.KD2, 'r_star', cfg.dstar);
    K = cfg.K; nsub = cfg.nsub; dt = Tev/(K*nsub);

    % 动力学 Function (返回 Xdot 与 攻弹加速度)
    Xs = SX.sym('Xs',18); ths = SX.sym('ths',3); tts = SX.sym('tts');
    [xd, aM] = casadi_dynamics(Xs, ths, tts, params);
    fdyn = Function('fdyn', {Xs,ths,tts}, {xd, aM});

    opti = Opti();
    Theta = opti.variable(3, K);

    % ---------------- 符号 RK4 滚动 ----------------
    X = x0; t = 0; energy = 0;
    r1 = {}; r2 = {}; tcoll = zeros(1, K*nsub); c = 0;
    for i = 1:K
        th_i = Theta(:,i);
        for s = 1:nsub
            [k1,a1] = fdyn(X,            th_i, t);
            [k2,~ ] = fdyn(X+0.5*dt*k1,  th_i, t+0.5*dt);
            [k3,~ ] = fdyn(X+0.5*dt*k2,  th_i, t+0.5*dt);
            [k4,~ ] = fdyn(X+dt*k3,      th_i, t+dt);
            energy = energy + dt * (a1.'*a1);   % a1 已在动力学内饱和(<=8g)
            c = c+1; tcoll(c) = t;
            r1{end+1} = sqrt(sumsqr(X(1:3)-X(7:9))+1e-6);    %#ok<AGROW>
            r2{end+1} = sqrt(sumsqr(X(1:3)-X(13:15))+1e-6);  %#ok<AGROW>
            X = X + dt/6*(k1+2*k2+2*k3+k4); t = t + dt;
        end
    end
    Xfinal = X;

    miss1 = softmin(vertcat(r1{:}), cfg.beta_min);
    miss2 = softmin(vertcat(r2{:}), cfg.beta_min);
    missT = casadi_target_zem(Xfinal(1:6));

    opti.subject_to(miss1 >= cfg.dfloor);
    opti.subject_to(miss2 >= cfg.dfloor);
    opti.subject_to(cfg.gamma_lo <= Theta(1,:) <= cfg.gamma_hi);
    opti.subject_to(cfg.wT_lo    <= Theta(2,:) <= cfg.wT_hi);
    opti.subject_to(cfg.wE_lo    <= Theta(3,:) <= cfg.wE_hi);

    % 三项 (规避/能量/目标) 分解, 便于标定与诊断
    avoidA = (miss1-cfg.dstar)^2 + (miss2-cfg.dstar)^2;   % 规避项原始量 A
    term_avoid  = 0.25*cfg.s_w*avoidA;                    % = (s_w/4)*A
    term_energy = 0.5*cfg.s_wE*energy;
    term_target = 0.5*cfg.s_wT*missT^2;
    J = term_avoid + term_energy + term_target;
    opti.minimize(J / cfg.Jscale);

    th_warm = repmat([0;wT0;wE0],1,K);
    opti.set_initial(Theta, th_warm);
    opti.solver('ipopt', struct('expand',true), struct('max_iter',cfg.max_iter,'print_level',cfg.print_level, ...
        'tol',cfg.tol,'acceptable_tol',cfg.tol*10,'acceptable_iter',5, ...
        'acceptable_obj_change_tol',1e-5, ...
        'hessian_approximation','limited-memory'));

    % ---------------- 标定模式: 仅在 warm θ 处实测三项原始量级 (不求解) ----------------
    if cfg.report_terms_only
        fterms = opti.to_function('Jterms', {Theta}, ...
            {avoidA, energy, missT^2, miss1, miss2, missT});
        [vA, vE, vTt, vm1, vm2, vmT] = fterms(th_warm);
        res = struct('ok', true);
        res.terms_raw = struct('A', full(vA), 'E', full(vE), 'Tt', full(vTt), ...
                               'miss1', full(vm1), 'miss2', full(vm2), 'missT', full(vmT));
        return;
    end

    res = struct('w_fixed',cfg.w,'x0',x0,'Tev',Tev,'tcoll',tcoll,'K',K,'nsub',nsub);
    try
        sol = opti.solve(); res.ok = true;
        th = full(sol.value(Theta));
        res.theta_K = th;                      % 原始 3×K 分段权重 (供 extract 用)
        res.theta   = repelem(th, 1, nsub);    % 3×(K*nsub) 展开版
        res.miss = [full(sol.value(miss1)); full(sol.value(miss2)); full(sol.value(missT))];
        res.J = full(sol.value(J));
        res.terms = struct('avoid', full(sol.value(term_avoid)), ...
                           'energy', full(sol.value(term_energy)), ...
                           'target', full(sol.value(term_target)), ...
                           'A', full(sol.value(avoidA)), ...
                           'E', full(sol.value(energy)), ...
                           'Tt', full(sol.value(missT^2)));
    catch ME
        res.ok = false; res.reason = ME.message;
        try
            th = full(opti.debug.value(Theta));
            res.theta_K = th;
            res.theta   = repelem(th, 1, nsub);
            res.miss = [full(opti.debug.value(miss1)); full(opti.debug.value(miss2)); full(opti.debug.value(missT))];
            res.J = full(opti.debug.value(J));
        catch; res.theta_K = []; res.theta = []; end
    end
end

function m = softmin(v, beta)
    import casadi.*
    vmin = mmin(v);
    m = vmin - beta * log(sum1(exp(-(v - vmin)/beta)));
end
