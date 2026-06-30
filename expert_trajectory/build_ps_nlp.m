function P = build_ps_nlp(x0, Tev, cfg)
%BUILD_PS_NLP 规避段伪谱 NLP (Radau 配点) 构造器。
%   P = build_ps_nlp(x0, Tev, cfg)
%   决策变量: 各配点状态 X(18) + 分段时变权重 theta=[gamma,wT,wE] (w 固定);
%   目标: 固定系数式22 (规避脱靶->d* + 控制能量 + 目标打击);
%   约束: Radau 缺陷、初值、攻弹过载<=8g、对双防最近距离>=dfloor。
%
%   输入:
%     x0  : 18x1 规避段初态 [M;D1;D2]
%     Tev : 规避段时长(s)
%     cfg : 结构体, 见下方默认值
%   输出 P: 含 opti / 决策变量句柄 / 配点时间, 供 solve_ps_trajectory 设初值并求解。

    import casadi.*

    % ---------------- 默认配置 ----------------
    d.K = 20; d.deg = 3;            % 区间数 / Radau 阶
    d.w = 1e4; d.dstar = 20;        % 式47 内嵌固定参数
    d.KD1 = 4; d.KD2 = 4;
    d.amax = 8 * 9.81;              % 攻弹过载上限
    d.dfloor = 6;                   % 非拦截下限(m)
    d.beta_min = 3;                 % softmin 平滑尺度(m)
    d.Jscale = 1e6;                 % 目标函数缩放(改善数值条件)
    % 固定评分系数 (式22, 阶段A 常数)
    d.s_w = 1e4; d.s_wE = 1.44e3; d.s_wT = 1e3;
    % theta 边界
    d.gamma_lo = -3;   d.gamma_hi = 3;
    d.wT_lo = 1;       d.wT_hi = 1e5;
    d.wE_lo = 10;      d.wE_hi = 1e4;
    if nargin < 3 || isempty(cfg); cfg = struct(); end
    fn = fieldnames(d);
    for k = 1:numel(fn)
        if ~isfield(cfg, fn{k}); cfg.(fn{k}) = d.(fn{k}); end
    end

    K = cfg.K; deg = cfg.deg; h = Tev / K;
    nx = 18;
    params = struct('w', cfg.w, 'KD1', cfg.KD1, 'KD2', cfg.KD2, 'r_star', cfg.dstar);

    % ---------------- Radau 配点系数 ----------------
    tau_c = collocation_points(deg, 'radau');     % 1 x deg, 落在 (0,1]
    [C, Dm, Bq] = collocation_coeff(tau_c);       % C:(deg+1)xdeg, Dm:(deg+1)x1, Bq:1xdeg

    opti = Opti();

    Xstart = cell(1, K+1);    % 各区间起点状态
    Xcoll  = cell(1, K);      % 各区间配点状态 (nx x deg)
    Th     = cell(1, K);      % 各区间时变权重 [gamma; wT; wE]
    tcoll  = zeros(1, K*deg); % 各配点全局时间
    rMD1 = {}; rMD2 = {};     % 各配点弹间距离 (求最近脱靶)

    Xstart{1} = opti.variable(nx, 1);
    opti.subject_to(Xstart{1} == x0);       % 初值约束

    J_energy = 0;
    cpt = 0;
    for i = 1:K
        Xcoll{i} = opti.variable(nx, deg);
        Th{i}    = opti.variable(3, 1);
        Z = [Xstart{i}, Xcoll{i}];          % nx x (deg+1)
        Pidot = Z * C;                       % nx x deg, 配点导数

        ti = (i-1) * h;
        for j = 1:deg
            tij = ti + tau_c(j) * h;
            [fj, aMj, ~, ~] = casadi_dynamics(Xcoll{i}(:,j), Th{i}, tij, params);
            opti.subject_to(Pidot(:,j) == h * fj);   % 缺陷约束

            % 能量项 (式22 第二项)
            J_energy = J_energy + h * Bq(j) * (aMj.' * aMj);
            % 过载约束
            opti.subject_to(aMj.' * aMj <= cfg.amax^2);

            % 弹间距离 (供脱靶项与非拦截约束)
            Mc = Xcoll{i}(1:3, j); D1c = Xcoll{i}(7:9, j); D2c = Xcoll{i}(13:15, j);
            cpt = cpt + 1;
            tcoll(cpt) = tij;
            rMD1{end+1} = sqrt(sumsqr(Mc - D1c) + 1e-6); %#ok<AGROW>
            rMD2{end+1} = sqrt(sumsqr(Mc - D2c) + 1e-6); %#ok<AGROW>
        end

        Xstart{i+1} = opti.variable(nx, 1);
        opti.subject_to(Xstart{i+1} == Z * Dm);          % 区间衔接
    end

    % ---------------- 脱靶量 ----------------
    % 双防: 规避窗口内最近距离 (softmin 近似 z_i(t_fi))
    v1 = vertcat(rMD1{:});  v2 = vertcat(rMD2{:});
    miss1 = softmin(v1, cfg.beta_min);
    miss2 = softmin(v2, cfg.beta_min);
    % 目标: 规避末端对目标的预测脱靶量 z_T(t_fT) (规避后才命中, 故用末态 ZEM 而非窗口距离)
    missT = casadi_target_zem(Xstart{K+1}(1:6));

    % 非拦截硬约束 (式: 终端脱靶 >= dfloor)
    opti.subject_to(miss1 >= cfg.dfloor);
    opti.subject_to(miss2 >= cfg.dfloor);

    % ---------------- 目标函数 (固定系数式22) ----------------
    s_w1 = cfg.s_w / 2; s_w2 = cfg.s_w / 2;   % gamma=0 参考分配
    J = 0.5 * s_w1 * (miss1 - cfg.dstar)^2 ...
      + 0.5 * s_w2 * (miss2 - cfg.dstar)^2 ...
      + 0.5 * cfg.s_wE * J_energy ...
      + 0.5 * cfg.s_wT * missT^2;
    opti.minimize(J / cfg.Jscale);

    % ---------------- theta 边界 ----------------
    for i = 1:K
        opti.subject_to(cfg.gamma_lo <= Th{i}(1) <= cfg.gamma_hi);
        opti.subject_to(cfg.wT_lo    <= Th{i}(2) <= cfg.wT_hi);
        opti.subject_to(cfg.wE_lo    <= Th{i}(3) <= cfg.wE_hi);
    end

    % ---------------- 打包 ----------------
    P = struct();
    P.opti = opti;
    P.Xstart = Xstart; P.Xcoll = Xcoll; P.Th = Th;
    P.tcoll = tcoll;
    P.J = J; P.miss = [miss1; miss2; missT];
    P.cfg = cfg; P.Tev = Tev; P.h = h; P.tau_c = tau_c;
    P.params = params;
end

% softmin 下估计 (<= 真实 min): 用 log-sum-exp 稳定化避免大距离下溢
function m = softmin(v, beta)
    import casadi.*
    vmin = mmin(v);
    m = vmin - beta * log(sum1(exp(-(v - vmin) / beta)));
end
