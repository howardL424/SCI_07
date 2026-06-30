function tr = extract_expert_transitions(tau_warm, res_ss, cfg)
%EXTRACT_EXPERT_TRANSITIONS 从单打靶优化结果提取 (s,a) 对供 GAIL 训练。
%   tr = extract_expert_transitions(tau_warm, res_ss)
%   以 dt_out≈0.1s 数值 RK4 重跑规避段 (stage2), 每步提取13维相对态势 state + 4维 action。
%   action 取当前时刻所在 K 段的分段常值 theta*(t)。
%
%   输入:
%     tau_warm : solve_warmstart 返回的阶段A轨迹 (含 .stage / .M / .D1 / .D2 列)
%     res_ss   : solve_ss_trajectory 返回结果, 必须含:
%                  .theta_K  [3×K]  原始分段权重 [gamma; wT; wE]
%                  .Tev      规避段总时长 (s)
%                  .x0       规避段初态 [xM; xD1; xD2] 18×1
%     cfg      : 可选配置, 缺省见 defaults
%
%   输出 tr:
%     .state     [N_steps × 13]  相对态势量 (见注释)
%     .action    [N_steps × 4]   [w=1e4, gamma(t), wT(t), wE(t)]
%     .t         [N_steps × 1]   时间 (s, 从 0 到 Tev)
%     .state_abs [N_steps × 18]  绝对物理状态备查 [xM(6), xD1(6), xD2(6)]
%
%   state 定义 (13维):
%     1  r_MD1    : 攻-D1 距离 (m)
%     2  rdot_MD1 : 攻-D1 接近率 (m/s; 正值=远离=安全)
%     3  qy_MD1   : LOS 高低角 D1→M (rad)
%     4  qz_MD1   : LOS 方位角 D1→M (rad)
%     5  r_MD2    : 攻-D2 距离 (m)
%     6  rdot_MD2 : 攻-D2 接近率 (m/s)
%     7  qy_MD2   : LOS 高低角 D2→M (rad)
%     8  qz_MD2   : LOS 方位角 D2→M (rad)
%     9  r_MT     : 攻-目标距离 (m)
%     10 qy_MT    : 攻看目标高低角 M→T (rad)
%     11 qz_MT    : 攻看目标方位角 M→T (rad)
%     12 V_M      : 攻弹速度 (m/s)
%     13 tgo_norm : 归一化剩余规避时间 (Tev-t)/Tev ∈ [0,1]

    %% -------- 默认配置 --------
    setup_expert_path();
    if nargin < 3; cfg = struct(); end
    d.dt_out = 0.1;    % 输出时间步长 (s); 每条约 Tev/0.1 ≈ 100-150 步
    d.w      = 1e4;    % 固定总权重 w
    d.dstar  = 20;     % 固定期望脱靶量 d* (m)
    d.KD1    = 4;      % 防御弹1 PN 增益
    d.KD2    = 4;      % 防御弹2 PN 增益
    d.g0     = 9.81;   % 重力加速度
    fn = fieldnames(d);
    for k = 1:numel(fn)
        if ~isfield(cfg, fn{k}); cfg.(fn{k}) = d.(fn{k}); end
    end

    %% -------- 校验输入 --------
    if ~isfield(res_ss,'theta_K') || isempty(res_ss.theta_K)
        tr = []; return;
    end

    %% -------- 初始化辅助函数 --------
    H = evasion_helpers();
    computeLOS                  = H.computeLOS;
    computeMDDot                = H.computeMDDot;
    computePN                   = H.computePN;
    computeOptimalCmd_duo_target= H.computeOptimalCmd_duo_target;
    updateState                 = H.updateState;
    saturate                    = H.saturate;

    T_pos   = [0, 0, 0];                   % 目标位置 (原点)
    T_state = [0, 0, 0, 0, 0, 0];          % 目标状态 (静止, V=0)
    max_ov_M = 8  * cfg.g0;
    max_ov_D = 10 * cfg.g0;

    %% -------- 抽取规避段初态 --------
    idx2 = find(tau_warm.stage == 2);
    if isempty(idx2)
        tr = []; return;
    end
    i0 = idx2(1);
    % 转为行向量 (updateState 返回 1×6)
    M  = tau_warm.M(:,  i0)';
    D1 = tau_warm.D1(:, i0)';
    D2 = tau_warm.D2(:, i0)';

    Tev     = res_ss.Tev;
    dt      = cfg.dt_out;
    theta_K = res_ss.theta_K;   % 3×K 分段权重
    K       = size(theta_K, 2);
    seg_dur = Tev / K;           % 每段时长 (s)

    w = cfg.w;

    %% -------- 预分配 --------
    nmax       = ceil(Tev / dt) + 5;
    state_buf  = zeros(nmax, 13);
    action_buf = zeros(nmax, 4);
    sabs_buf   = zeros(nmax, 18);
    t_buf      = zeros(nmax, 1);
    n = 0;

    %% -------- 主仿真循环 --------
    t = 0;
    while t < Tev - 1e-9
        % --- 当前所在 K 段 → 取对应 theta ---
        seg_idx = min(floor(t / seg_dur) + 1, K);
        gamma_k = theta_K(1, seg_idx);
        wT_k    = theta_K(2, seg_idx);
        wE_k    = theta_K(3, seg_idx);
        a1 = w / (1 + exp(-gamma_k));
        a2 = w * exp(-gamma_k) / (1 + exp(-gamma_k));

        % --- 视线几何 ---
        [qy_D1M, qz_D1M, r_MD1] = computeLOS(D1(1:3), M(1:3));   % D1→M (与 simulate 一致)
        [qy_D2M, qz_D2M, r_MD2] = computeLOS(D2(1:3), M(1:3));   % D2→M
        [qy_MT,  qz_MT,  r_MT]  = computeLOS(M(1:3),  T_pos);    % M→T (state用)
        [qy_TM,  qz_TM,  r_TM]  = computeLOS(T_pos,   M(1:3));   % T→M (式47通道T用)

        % --- 视线角速率 ---
        [rdot_MD1, qd_y1, qd_z1]     = computeMDDot(M, D1, r_MD1, qy_D1M, qz_D1M);
        [rdot_MD2, qd_y2, qd_z2]     = computeMDDot(M, D2, r_MD2, qy_D2M, qz_D2M);
        [rdot_TM,  qd_yTM, qd_zTM]   = computeMDDot(M, T_state, r_TM, qy_TM, qz_TM);

        % --- 防御弹 PN 指令 ---
        a_D1_cmd = saturate(computePN(cfg.KD1, D1(6), qd_y1, qd_z1, D1(4)), max_ov_D);
        a_D2_cmd = saturate(computePN(cfg.KD2, D2(6), qd_y2, qd_z2, D2(4)), max_ov_D);

        % --- 预估终端时刻 (式48/49) ---
        rd1 = sign(rdot_MD1) * max(abs(rdot_MD1), 1e-3);
        rd2 = sign(rdot_MD2) * max(abs(rdot_MD2), 1e-3);
        rdT = sign(rdot_TM)  * max(abs(rdot_TM),  1e-3);
        tf1 = t - r_MD1 / rd1;
        tf2 = t - r_MD2 / rd2;
        tfT = t - r_TM  / rdT;

        % --- 最优制导指令 (式47) ---
        a_M_cmd = computeOptimalCmd_duo_target( ...
            qy_D1M, qz_D1M, qd_y1, qd_z1, r_MD1, rdot_MD1, t, tf1, ...
            qy_D2M, qz_D2M, qd_y2, qd_z2, r_MD2, rdot_MD2, tf2, ...
            qy_TM,  qz_TM,  qd_yTM, qd_zTM, r_TM, rdot_TM, tfT, ...
            M(4), M(5), D1(4), D1(5), D2(4), D2(5), ...
            a_D1_cmd, a_D2_cmd, a1, a2, wT_k, wE_k, cfg.dstar);
        a_M_cmd = saturate(a_M_cmd, max_ov_M);

        % --- 记录当前步 (t 时刻的状态与对应动作) ---
        n = n + 1;
        tgo_norm = max(0, (Tev - t) / Tev);
        state_buf(n, :) = [r_MD1, rdot_MD1, qy_D1M, qz_D1M, ...
                           r_MD2, rdot_MD2, qy_D2M, qz_D2M, ...
                           r_MT, qy_MT, qz_MT, M(6), tgo_norm];
        action_buf(n, :) = [w, gamma_k, wT_k, wE_k];
        sabs_buf(n, :)   = [M(:)', D1(:)', D2(:)'];
        t_buf(n)         = t;

        % --- RK4 状态更新 ---
        M  = updateState(M,  a_M_cmd,  dt);
        D1 = updateState(D1, a_D1_cmd, dt);
        D2 = updateState(D2, a_D2_cmd, dt);
        t  = t + dt;
    end

    %% -------- 打包输出 --------
    tr.state     = state_buf(1:n, :);
    tr.action    = action_buf(1:n, :);
    tr.state_abs = sabs_buf(1:n, :);
    tr.t         = t_buf(1:n);
end
