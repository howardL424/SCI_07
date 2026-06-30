function [obs_next, state_abs_next, reward, terminated, info_vec] = ...
        env_step(state_abs, t, Tev, r_MT0, action_norm, cfg_in)
%ENV_STEP  执行一步环境交互（dt=0.1s）：式47 制导 + RK4 更新 → 新 obs + reward。
%
%   [obs_next, state_abs_next, reward, terminated, info_vec] = ...
%       env_step(state_abs, t, Tev, r_MT0, action_norm)
%   [obs_next, state_abs_next, reward, terminated, info_vec] = ...
%       env_step(state_abs, t, Tev, r_MT0, action_norm, cfg_in)
%
%   输入:
%     state_abs   [18×1] 当前绝对状态 [M(6); D1(6); D2(6)]
%     t           当前仿真时刻 (s，从规避段起点算起)
%     Tev         规避段总时长估计 (s)
%     r_MT0       规避段开始时攻-目标距离 (m)，reward 归一化基准
%     action_norm [3×1] 归一化动作 [gamma_n, wT_n, wE_n] ∈ [-1,1]
%     cfg_in      可选配置，字段：dt(默认0.1), dstar(20), KD1(4), KD2(4),
%                                  KM(3), g0(9.81), k1(1.0), kT(0.5), kE(1e-6)
%
%   输出:
%     obs_next       [13×1] 新状态向量 (定义同 env_reset)
%     state_abs_next [18×1] 新绝对状态
%     reward         标量奖励
%     terminated     逻辑，本 episode 是否结束
%     info_vec       [5×1] 调试信息 [r_MD1, r_MD2, r_MT, intercepted, reached]
%
%   动作反归一化:
%     gamma = action_norm(1) * 3          → [-3, 3]
%     wT    = 10^(action_norm(2) + 3)     → [1e2, 1e4]
%     wE    = 10^(action_norm(3) + 3)     → [1e2, 1e4]
%     w     = 1e4 (固定)
%     d*    = 20 m (固定)

    % ---- 默认配置 ----
    d.dt    = 0.1;    d.dstar = 20.0;
    d.KD1   = 4;      d.KD2   = 4;    d.KM = 3;
    d.g0    = 9.81;
    d.k1    = 1.0;    d.kT    = 0.5;  d.kE = 1e-6;
    d.kill_radius  = 6.0;    % 被拦截判定距离 (m)
    d.reach_radius = 50.0;   % 到达目标判定距离 (m)
    if nargin < 6 || isempty(cfg_in); cfg_in = struct(); end
    fn = fieldnames(d);
    for k = 1:numel(fn)
        if ~isfield(cfg_in, fn{k}); cfg_in.(fn{k}) = d.(fn{k}); end
    end

    dt     = cfg_in.dt;
    dstar  = cfg_in.dstar;
    KD1    = cfg_in.KD1;   KD2 = cfg_in.KD2;
    k1     = cfg_in.k1;    kT  = cfg_in.kT;  kE = cfg_in.kE;
    max_ov_M = 8  * cfg_in.g0;
    max_ov_D = 10 * cfg_in.g0;

    T_pos   = [0, 0, 0];
    T_state = [0, 0, 0, 0, 0, 0];

    % ---- 拆解绝对状态 ----
    M  = state_abs(1:6)';    % 1×6
    D1 = state_abs(7:12)';
    D2 = state_abs(13:18)';

    % ---- 反归一化动作 ----
    W     = 1e4;
    gamma = double(action_norm(1)) * 3.0;
    wT    = 10^(double(action_norm(2)) + 3);
    wE    = 10^(double(action_norm(3)) + 3);
    w1    = W / (1 + exp(-gamma));
    w2    = W * exp(-gamma) / (1 + exp(-gamma));

    % ---- 获取辅助函数句柄 ----
    H = evasion_helpers();
    computeLOS                   = H.computeLOS;
    computeMDDot                 = H.computeMDDot;
    computeMTDot                 = H.computeMTDot;
    computePN                    = H.computePN;
    computeOptimalCmd_duo_target = H.computeOptimalCmd_duo_target;
    updateState                  = H.updateState;
    saturate                     = H.saturate;

    % ---- 计算当前视线几何 ----
    [qy_D1M, qz_D1M, r_MD1] = computeLOS(D1(1:3), M(1:3));
    [qy_D2M, qz_D2M, r_MD2] = computeLOS(D2(1:3), M(1:3));
    [qy_MT,  qz_MT,  r_MT]  = computeLOS(M(1:3),  T_pos);
    [qy_TM,  qz_TM,  r_TM]  = computeLOS(T_pos,   M(1:3));   % 式47 通道T

    [rdot_MD1, qd_y1, qd_z1]     = computeMDDot(M, D1, r_MD1, qy_D1M, qz_D1M);
    [rdot_MD2, qd_y2, qd_z2]     = computeMDDot(M, D2, r_MD2, qy_D2M, qz_D2M);
    [rdot_TM,  qd_yTM, qd_zTM]   = computeMDDot(M, T_state, r_TM, qy_TM, qz_TM);

    % ---- 防御弹 PN 指令 ----
    a_D1 = saturate(computePN(KD1, D1(6), qd_y1, qd_z1, D1(4)), max_ov_D);
    a_D2 = saturate(computePN(KD2, D2(6), qd_y2, qd_z2, D2(4)), max_ov_D);

    % ---- 预估终端时刻 (式48/49) ----
    rd1 = sign(rdot_MD1) * max(abs(rdot_MD1), 1e-3);
    rd2 = sign(rdot_MD2) * max(abs(rdot_MD2), 1e-3);
    rdT = sign(rdot_TM)  * max(abs(rdot_TM),  1e-3);
    tf1 = t - r_MD1 / rd1;
    tf2 = t - r_MD2 / rd2;
    tfT = t - r_TM  / rdT;

    % ---- 攻击弹最优制导指令（式47）----
    a_M = computeOptimalCmd_duo_target( ...
        qy_D1M, qz_D1M, qd_y1, qd_z1, r_MD1, rdot_MD1, t, tf1, ...
        qy_D2M, qz_D2M, qd_y2, qd_z2, r_MD2, rdot_MD2,    tf2, ...
        qy_TM,  qz_TM,  qd_yTM, qd_zTM, r_TM, rdot_TM,   tfT, ...
        M(4), M(5), D1(4), D1(5), D2(4), D2(5), ...
        a_D1, a_D2, w1, w2, wT, wE, dstar);
    a_M = saturate(a_M, max_ov_M);

    % ---- RK4 状态更新 ----
    M_new  = updateState(M,  a_M,  dt);
    D1_new = updateState(D1, a_D1, dt);
    D2_new = updateState(D2, a_D2, dt);
    t_new  = t + dt;

    % ---- 计算新 obs ----
    [qy_D1M_n, qz_D1M_n, r_MD1_n] = computeLOS(D1_new(1:3), M_new(1:3));
    [qy_D2M_n, qz_D2M_n, r_MD2_n] = computeLOS(D2_new(1:3), M_new(1:3));
    [qy_MT_n,  qz_MT_n,  r_MT_n]  = computeLOS(M_new(1:3),  T_pos);
    [rdot_MD1_n, ~, ~] = computeMDDot(M_new, D1_new, r_MD1_n, qy_D1M_n, qz_D1M_n);
    [rdot_MD2_n, ~, ~] = computeMDDot(M_new, D2_new, r_MD2_n, qy_D2M_n, qz_D2M_n);

    tgo_norm_n = max(0.0, (Tev - t_new) / Tev);

    obs_next = [r_MD1_n; rdot_MD1_n; qy_D1M_n; qz_D1M_n;
                r_MD2_n; rdot_MD2_n; qy_D2M_n; qz_D2M_n;
                r_MT_n;  qy_MT_n;   qz_MT_n;  M_new(6);  tgo_norm_n];   % 13×1

    state_abs_next = [M_new(:); D1_new(:); D2_new(:)];   % 18×1

    % ---- 奖励计算 ----
    % 规避距离项：鼓励保持与两防距离
    r_evade  = k1 * (log10(max(r_MD1_n, 1.0)) + log10(max(r_MD2_n, 1.0)));
    % 目标距离项：鼓励靠近目标（归一化）
    r_target = -kT * (r_MT_n / max(r_MT0, 1.0));
    % 能量惩罚
    r_energy = -kE * (a_M(1)^2 + a_M(2)^2) * dt;

    reward = r_evade + r_target + r_energy;

    % ---- 终止条件判断 ----
    intercepted = (min(r_MD1_n, r_MD2_n) <= cfg_in.kill_radius);
    reached     = (r_MT_n <= cfg_in.reach_radius) && (t_new > Tev * 0.5);
    timeout     = (t_new >= Tev * 1.5);   % 超时（1.5倍估计时长）

    if intercepted
        reward     = reward - 100.0;
        terminated = true;
    elseif reached
        reward     = reward + 100.0;
        terminated = true;
    elseif timeout
        terminated = true;
    else
        terminated = false;
    end

    info_vec = [r_MD1_n; r_MD2_n; r_MT_n; double(intercepted); double(reached)];
end
