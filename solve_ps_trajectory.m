function res = solve_ps_trajectory(tau_warm, cfg)
%SOLVE_PS_TRAJECTORY 用 warm-start 轨迹热启动, 求解规避段伪谱 NLP (式47+式22)。
%   res = solve_ps_trajectory(tau_warm, cfg)
%   tau_warm : simulate_warmstart_trajectory 输出 (需含 stage==2 规避段)
%   cfg      : 透传 build_ps_nlp 的配置 (可选)
%   res 字段:
%     .ok        逻辑   IPOPT 是否求解成功
%     .theta     3xNc   各配点最优时变权重 [gamma; wT; wE]
%     .tcoll     1xNc   配点时间(相对规避段起点)
%     .Xcoll     18xNc  最优状态
%     .miss      3x1    [对D1, 对D2, 对目标] 最近距离
%     .J         标量   最优目标值
%     .w_fixed   标量   固定 w
%     .x0 .Tev   规避段初态与时长

    import casadi.*
    if nargin < 2; cfg = struct(); end

    % ---------------- 抽取规避段 (stage==2) ----------------
    idx2 = find(tau_warm.stage == 2);
    if isempty(idx2) || numel(idx2) < 5
        res = struct('ok', false, 'reason', 'no_evasion_segment');
        return;
    end
    i0 = idx2(1); i1 = idx2(end);
    x0  = [tau_warm.M(:, i0); tau_warm.D1(:, i0); tau_warm.D2(:, i0)];
    t_rel = tau_warm.t(i0:i1) - tau_warm.t(i0);
    Tev = t_rel(end);
    Xwarm = [tau_warm.M(:, i0:i1); tau_warm.D1(:, i0:i1); tau_warm.D2(:, i0:i1)]; % 18 x m

    % warm-start 权重 (用于 theta 初值)
    wE0 = tau_warm.meta.cfg.wE; wT0 = tau_warm.meta.cfg.wT;

    % ---------------- 构造 NLP ----------------
    P = build_ps_nlp(x0, Tev, cfg);
    opti = P.opti; K = P.cfg.K; deg = P.cfg.deg; h = P.h; tau_c = P.tau_c;

    % ---------------- 热启动: 插值 warm-start 状态 ----------------
    interpX = @(tq) interp1(t_rel, Xwarm.', min(max(tq,0),Tev), 'linear', 'extrap').'; % 18 x numel(tq)
    for i = 1:K+1
        opti.set_initial(P.Xstart{i}, interpX((i-1)*h));
    end
    for i = 1:K
        tj = (i-1)*h + tau_c * h;          % 1 x deg
        opti.set_initial(P.Xcoll{i}, interpX(tj));
        opti.set_initial(P.Th{i}, [0; wT0; wE0]);
    end

    % ---------------- 求解 ----------------
    p_opts = struct('expand', true);
    s_opts = struct('max_iter', 500, 'print_level', 0, 'tol', 1e-4, ...
                    'acceptable_tol', 1e-3, 'mu_strategy', 'adaptive');
    opti.solver('ipopt', p_opts, s_opts);

    res = struct('w_fixed', P.cfg.w, 'x0', x0, 'Tev', Tev, 'tcoll', P.tcoll);
    try
        sol = opti.solve();
        res.ok = true;
        res.theta = extract_theta(sol, P);
        res.Xcoll = extract_state(sol, P);
        res.miss  = full(sol.value(P.miss));
        res.J     = full(sol.value(P.J));
    catch ME
        % 降级: 取最后一次迭代值 (若可用)
        res.ok = false; res.reason = ME.message;
        try
            res.theta = extract_theta(opti.debug, P);
            res.Xcoll = extract_state(opti.debug, P);
            res.miss  = full(opti.debug.value(P.miss));
            res.J     = full(opti.debug.value(P.J));
        catch
            res.theta = []; res.Xcoll = [];
        end
    end
end

% 提取分段时变 theta 展开到各配点
function th = extract_theta(sol, P)
    K = P.cfg.K; deg = P.cfg.deg;
    th = zeros(3, K*deg);
    c = 0;
    for i = 1:K
        thi = full(sol.value(P.Th{i}));
        for j = 1:deg
            c = c + 1; th(:, c) = thi;
        end
    end
end

function Xc = extract_state(sol, P)
    K = P.cfg.K; deg = P.cfg.deg; nx = 18;
    Xc = zeros(nx, K*deg);
    c = 0;
    for i = 1:K
        Xi = full(sol.value(P.Xcoll{i}));
        for j = 1:deg
            c = c + 1; Xc(:, c) = Xi(:, j);
        end
    end
end
