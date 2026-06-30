function tau = simulate_warmstart_trajectory(M0, D10, D20, cfg)
%SIMULATE_WARMSTART_TRAJECTORY 阶段A: 固定权重 RK4 全段可行轨迹 (warm-start)。
%   tau = simulate_warmstart_trajectory(M0, D10, D20, cfg)
%   全段流程: PN 打目标 -> 双防 OBSM 规避 -> PN 恢复打目标, 用于为阶段B 伪谱
%   NLP 提供物理可行、三阶段可跑通的初值。权重全程固定 (不调用 BP/LM)。
%
%   输入:
%     M0, D10, D20 : 各飞行器初态 [x,y,z,theta,psi_v,V] (1x6), 见 reconstruct_geometry
%     cfg          : 可选配置结构体, 缺省见下方 defaults
%   输出 tau (结构体, 列为时间序列):
%     .t           1xn   时间
%     .M/.D1/.D2   6xn   各飞行器状态轨迹
%     .a_M         2xn   攻击弹法向加速度指令 [a_y; a_z]
%     .a_D1/.a_D2  2xn   防御弹加速度指令
%     .stage       1xn   阶段标号 (1 PN打目标 / 2 规避 / 3 PN恢复)
%     .r_MT/.r_MD1/.r_MD2  1xn  相对距离
%     .t_start_pen .t_pen_end  规避起止时刻
%     .r_min1 .r_min2 .r_minT  最小弹间/弹目距离
%     .intercepted 逻辑   是否被拦截 (min r_MD < 6m)
%     .success     逻辑   双防是否均规避成功
%     .meta        配置与初态备份

    % ---------------- 默认配置 ----------------
    d.KM = 3; d.KD1 = 4; d.KD2 = 4;
    d.w = 1e4; d.gamma = 0; d.wE = 1.44e3; d.wT = 1e3; d.dstar = 20;
    d.dt = 1e-3; d.t_max = 60; d.r_safe = 8000; d.g0 = 9.81;
    d.T = [0, 0, 0];
    d.reach_radius = 50;         % 弹目距离小于此值视为"到达目标"
    d.tail_time = 0.3;           % 到达目标后再多跑的时长(s)
    d.diverge_time = 2.0;        % stage3 持续远离目标超过此时长则判定发散早停(s)
    if nargin < 4 || isempty(cfg); cfg = struct(); end
    fn = fieldnames(d);
    for k = 1:numel(fn)
        if ~isfield(cfg, fn{k}); cfg.(fn{k}) = d.(fn{k}); end
    end

    H = evasion_helpers();
    computeLOS = H.computeLOS; computeMDDot = H.computeMDDot; computeMTDot = H.computeMTDot;
    computePN = H.computePN; updateState = H.updateState; saturate = H.saturate;
    computeOptimalCmd = H.computeOptimalCmd;
    computeOptimalCmd_duo_target = H.computeOptimalCmd_duo_target;

    KM = cfg.KM; KD1 = cfg.KD1; KD2 = cfg.KD2;
    w = cfg.w; gamma = cfg.gamma; wE = cfg.wE; wT = cfg.wT; r_star = cfg.dstar;
    dt = cfg.dt; t_max = cfg.t_max; r_safe = cfg.r_safe; T = cfg.T;
    T_state = [T(1), T(2), T(3), 0, 0, 0];   % 目标作为原点处静止物体 (式47 通道T)
    max_ov_M = 8 * cfg.g0; max_ov_D = 10 * cfg.g0;

    a1 = w / (1 + exp(-gamma));            % D1 规避项权重 (式23)
    a2 = w * exp(-gamma) / (1 + exp(-gamma));% D2 规避项权重

    % ---------------- 预分配 ----------------
    nmax = ceil(t_max / dt) + 2;
    tau.t   = zeros(1, nmax);
    tau.M   = zeros(6, nmax); tau.D1 = zeros(6, nmax); tau.D2 = zeros(6, nmax);
    tau.a_M = zeros(2, nmax); tau.a_D1 = zeros(2, nmax); tau.a_D2 = zeros(2, nmax);
    tau.stage = zeros(1, nmax);
    tau.r_MT  = zeros(1, nmax); tau.r_MD1 = zeros(1, nmax); tau.r_MD2 = zeros(1, nmax);
    tau.tf1 = zeros(1, nmax); tau.tf2 = zeros(1, nmax);

    % ---------------- 状态与标志 ----------------
    M = M0; D1 = D10; D2 = D20;
    t = 0; n = 0;
    penetration_started = false; t_start_pen = NaN; t_pen_end = NaN;
    f1 = 0; f2 = 0;                 % 各防御弹规避成功标志 (r_dot>0)
    intercepted = false;
    r_min1 = inf; r_min2 = inf; r_minT = inf; t_minT = NaN;
    reached = false; t_reach = NaN; % 是否到达目标附近
    diverged = false; t_away = NaN;  % stage3 远离目标计时

    while t < t_max
        n = n + 1;

        % --- 视线几何 ---
        [q_theta, q_psi, r_MT]   = computeLOS(M(1:3), T);    % M->T (供PN阶段)
        [q_y_MD1, q_z_MD1, r_MD1] = computeLOS(D1(1:3), M(1:3));
        [q_y_MD2, q_z_MD2, r_MD2] = computeLOS(D2(1:3), M(1:3));
        [q_y_TM, q_z_TM, r_TM]   = computeLOS(T_state(1:3), M(1:3)); % T->M (式47 通道T)

        % --- 视线角速率 ---
        [r_dot_MD1, qd_y1, qd_z1] = computeMDDot(M, D1, r_MD1, q_y_MD1, q_z_MD1);
        [r_dot_MD2, qd_y2, qd_z2] = computeMDDot(M, D2, r_MD2, q_y_MD2, q_z_MD2);
        [qd_theta, qd_psi]        = computeMTDot(M, r_MT, q_theta, q_psi);   % PN 用
        [r_dot_TM, qd_y_TM, qd_z_TM] = computeMDDot(M, T_state, r_TM, q_y_TM, q_z_TM); % 式47 通道T

        % --- 规避触发 ---
        if ~penetration_started && min(r_MD1, r_MD2) <= r_safe
            penetration_started = true;
            t_start_pen = t;
        end

        % --- 防御弹始终比例导引 ---
        a_D1_cmd = computePN(KD1, D1(6), qd_y1, qd_z1, D1(4));
        a_D2_cmd = computePN(KD2, D2(6), qd_y2, qd_z2, D2(4));

        % --- 预估终端时刻 (式48) ---
        rdot1 = sign(r_dot_MD1) * max(abs(r_dot_MD1), 1e-3);
        rdot2 = sign(r_dot_MD2) * max(abs(r_dot_MD2), 1e-3);
        rdotT = sign(r_dot_TM) * max(abs(r_dot_TM), 1e-3);
        tf1 = t - r_MD1 / rdot1;
        tf2 = t - r_MD2 / rdot2;
        tfT = t - r_TM / rdotT;                 % 目标打击终端时刻 (式49)

        % --- 攻击弹指令 (三阶段) ---
        if ~penetration_started
            a_M_cmd = computePN(KM, M(6), qd_theta, qd_psi, M(4));
            stage = 1;
        elseif t <= max(tf1, tf2)
            t_go1 = tf1 - t; t_go2 = tf2 - t;
            if abs(r_MD1 - r_MD2) > 3000
                % 距离差过大: 仅对更近的防御弹单弹规避 (论文 §4.2 step b)
                if r_MD1 <= r_MD2
                    a_M_cmd = computeOptimalCmd(q_y_MD1, q_z_MD1, qd_y1, qd_z1, ...
                        r_MD1, r_dot_MD1, t, tf1, M(4), M(5), D1(4), D1(5), ...
                        a_D1_cmd, w, wE, r_star);
                else
                    a_M_cmd = computeOptimalCmd(q_y_MD2, q_z_MD2, qd_y2, qd_z2, ...
                        r_MD2, r_dot_MD2, t, tf2, M(4), M(5), D2(4), D2(5), ...
                        a_D2_cmd, w, wE, r_star);
                end
            elseif min(t_go1, t_go2) > 0.002
                % 双防规避 + 目标打击一体化最优制导 (式47)
                a_M_cmd = computeOptimalCmd_duo_target( ...
                    q_y_MD1, q_z_MD1, qd_y1, qd_z1, r_MD1, r_dot_MD1, t, tf1, ...
                    q_y_MD2, q_z_MD2, qd_y2, qd_z2, r_MD2, r_dot_MD2, tf2, ...
                    q_y_TM, q_z_TM, qd_y_TM, qd_z_TM, r_TM, r_dot_TM, tfT, ...
                    M(4), M(5), D1(4), D1(5), D2(4), D2(5), ...
                    a_D1_cmd, a_D2_cmd, a1, a2, wT, wE, r_star);
            elseif t_go1 > 0.002
                a_M_cmd = computeOptimalCmd(q_y_MD1, q_z_MD1, qd_y1, qd_z1, ...
                    r_MD1, r_dot_MD1, t, tf1, M(4), M(5), D1(4), D1(5), ...
                    a_D1_cmd, w, wE, r_star);
            elseif t_go2 > 0.002
                a_M_cmd = computeOptimalCmd(q_y_MD2, q_z_MD2, qd_y2, qd_z2, ...
                    r_MD2, r_dot_MD2, t, tf2, M(4), M(5), D2(4), D2(5), ...
                    a_D2_cmd, w, wE, r_star);
            else
                a_M_cmd = computePN(KM, M(6), qd_theta, qd_psi, M(4));
            end
            stage = 2;
            t_pen_end = t;
        else
            a_M_cmd = computePN(KM, M(6), qd_theta, qd_psi, M(4));
            stage = 3;
        end

        % --- 过载饱和 ---
        a_M_cmd  = saturate(a_M_cmd,  max_ov_M);
        a_D1_cmd = saturate(a_D1_cmd, max_ov_D);
        a_D2_cmd = saturate(a_D2_cmd, max_ov_D);

        % --- 记录 (状态在更新前, 与 t 对齐) ---
        tau.t(n) = t;
        tau.M(:, n) = M(:); tau.D1(:, n) = D1(:); tau.D2(:, n) = D2(:);
        tau.a_M(:, n) = a_M_cmd; tau.a_D1(:, n) = a_D1_cmd; tau.a_D2(:, n) = a_D2_cmd;
        tau.stage(n) = stage;
        tau.r_MT(n) = r_MT; tau.r_MD1(n) = r_MD1; tau.r_MD2(n) = r_MD2;
        tau.tf1(n) = tf1; tau.tf2(n) = tf2;

        % --- 状态更新 ---
        M  = updateState(M,  a_M_cmd,  dt);
        D1 = updateState(D1, a_D1_cmd, dt);
        D2 = updateState(D2, a_D2_cmd, dt);

        % --- 标志更新 ---
        if min(r_MD1, r_MD2) <= 6 && ~intercepted
            intercepted = true;
        end
        if r_dot_MD1 > 0 && f1 == 0 && penetration_started; f1 = 1; end
        if r_dot_MD2 > 0 && f2 == 0 && penetration_started; f2 = 1; end
        r_min1 = min(r_min1, r_MD1);
        r_min2 = min(r_min2, r_MD2);
        if r_MT < r_minT; r_minT = r_MT; t_minT = t; end

        % --- 终止: 规避结束后(stage3)到达目标附近, 再跑 tail_time 收尾 ---
        if stage == 3 && r_MT < cfg.reach_radius && ~reached
            reached = true; t_reach = t;
        end
        if reached && t > t_reach + cfg.tail_time
            break;
        end
        % --- 早停: stage3 中持续远离目标 (PN 无法回收) 判定发散 ---
        if stage == 3 && ~reached && n >= 2 && tau.r_MT(n) > tau.r_MT(n-1)
            if isnan(t_away); t_away = t; end
            if t - t_away > cfg.diverge_time
                diverged = true; break;
            end
        elseif stage == 3
            t_away = NaN;   % 重新接近目标则复位计时
        end

        t = t + dt;
    end

    % ---------------- 裁剪 ----------------
    idx = 1:n;
    tau.t = tau.t(idx);
    tau.M = tau.M(:, idx); tau.D1 = tau.D1(:, idx); tau.D2 = tau.D2(:, idx);
    tau.a_M = tau.a_M(:, idx); tau.a_D1 = tau.a_D1(:, idx); tau.a_D2 = tau.a_D2(:, idx);
    tau.stage = tau.stage(idx);
    tau.r_MT = tau.r_MT(idx); tau.r_MD1 = tau.r_MD1(idx); tau.r_MD2 = tau.r_MD2(idx);
    tau.tf1 = tau.tf1(idx); tau.tf2 = tau.tf2(idx);

    tau.t_start_pen = t_start_pen;
    tau.t_pen_end   = t_pen_end;
    tau.r_min1 = r_min1; tau.r_min2 = r_min2;
    tau.r_minT = r_minT; tau.t_minT = t_minT;
    tau.intercepted = intercepted;
    tau.reached = reached; tau.t_reach = t_reach;
    tau.diverged = diverged;
    tau.success = (f1 == 1 && f2 == 1) && ~intercepted;
    tau.meta = struct('M0', M0, 'D10', D10, 'D20', D20, 'cfg', cfg, ...
                      'a1', a1, 'a2', a2);
end
