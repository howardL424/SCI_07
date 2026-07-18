function [obs_next, state_abs_next, t_new, fact] = ...
        stage2_step(core, state_abs, t, Tev, action_norm)
%STAGE2_STEP  预初始化核心上的一步 stage2 动力学（物理事实，不含奖励常数）。
%
%   [obs_next, state_abs_next, t_new, fact] = ...
%       stage2_step(core, state_abs, t, Tev, action_norm)
%
%   输入:
%     core        stage2_core_init 返回值
%     state_abs   [18×1] 当前绝对态 [M(6); D1(6); D2(6)]
%     t           当前时刻 (s)
%     Tev         规避段时长 (s)
%     action_norm [3×1] 已映射/裁剪的 TD3 动作 ∈ [-1,1]
%
%   输出 fact 字段:
%     .r_MD1 .r_MD2 .r_MT
%     .intercepted  (min(r_MD)≤kill_radius)
%     .stage2_done  (t_new≥Tev 且未拦截)
%     .terminated   (intercepted || stage2_done)
%     .done         同 terminated（schema 别名）
%
%   硬性约定: 调用方必须先 phys_to_action_norm 再传入；本函数不再二次裁剪。

    cfg = core.cfg;
    H   = core.H;
    dt  = cfg.dt;

    M  = state_abs(1:6)';
    D1 = state_abs(7:12)';
    D2 = state_abs(13:18)';

    W     = cfg.w;
    gamma = double(action_norm(1)) * 3.0;
    wT    = 10^(double(action_norm(2)) * 1.5 + 3.5);
    wE    = 10^(double(action_norm(3)) + 3);
    w1    = W / (1 + exp(-gamma));
    w2    = W * exp(-gamma) / (1 + exp(-gamma));

    computeLOS                   = H.computeLOS;
    computeMDDot                 = H.computeMDDot;
    computePN                    = H.computePN;
    computeOptimalCmd_duo_target = H.computeOptimalCmd_duo_target;
    updateState                  = H.updateState;
    saturate                     = H.saturate;

    T_pos   = core.T_pos;
    T_state = core.T_state;

    [qy_D1M, qz_D1M, r_MD1] = computeLOS(D1(1:3), M(1:3));
    [qy_D2M, qz_D2M, r_MD2] = computeLOS(D2(1:3), M(1:3));
    [qy_MT,  qz_MT,  r_MT]  = computeLOS(M(1:3),  T_pos); %#ok<ASGLU>
    [qy_TM,  qz_TM,  r_TM]  = computeLOS(T_pos,   M(1:3));

    [rdot_MD1, qd_y1, qd_z1]   = computeMDDot(M, D1, r_MD1, qy_D1M, qz_D1M);
    [rdot_MD2, qd_y2, qd_z2]   = computeMDDot(M, D2, r_MD2, qy_D2M, qz_D2M);
    [rdot_TM,  qd_yTM, qd_zTM] = computeMDDot(M, T_state, r_TM, qy_TM, qz_TM);

    a_D1 = saturate(computePN(cfg.KD1, D1(6), qd_y1, qd_z1, D1(4)), core.max_ov_D);
    a_D2 = saturate(computePN(cfg.KD2, D2(6), qd_y2, qd_z2, D2(4)), core.max_ov_D);

    rd1 = sign(rdot_MD1) * max(abs(rdot_MD1), 1e-3);
    rd2 = sign(rdot_MD2) * max(abs(rdot_MD2), 1e-3);
    rdT = sign(rdot_TM)  * max(abs(rdot_TM),  1e-3);
    tf1 = t - r_MD1 / rd1;
    tf2 = t - r_MD2 / rd2;
    tfT = t - r_TM  / rdT;

    a_M = computeOptimalCmd_duo_target( ...
        qy_D1M, qz_D1M, qd_y1, qd_z1, r_MD1, rdot_MD1, t, tf1, ...
        qy_D2M, qz_D2M, qd_y2, qd_z2, r_MD2, rdot_MD2,    tf2, ...
        qy_TM,  qz_TM,  qd_yTM, qd_zTM, r_TM, rdot_TM,   tfT, ...
        M(4), M(5), D1(4), D1(5), D2(4), D2(5), ...
        a_D1, a_D2, w1, w2, wT, wE, cfg.dstar);
    a_M = saturate(a_M, core.max_ov_M);

    M_new  = updateState(M,  a_M,  dt);
    D1_new = updateState(D1, a_D1, dt);
    D2_new = updateState(D2, a_D2, dt);
    t_new  = t + dt;

    [qy_D1M_n, qz_D1M_n, r_MD1_n] = computeLOS(D1_new(1:3), M_new(1:3));
    [qy_D2M_n, qz_D2M_n, r_MD2_n] = computeLOS(D2_new(1:3), M_new(1:3));
    [qy_MT_n,  qz_MT_n,  r_MT_n]  = computeLOS(M_new(1:3),  T_pos);
    [rdot_MD1_n, ~, ~] = computeMDDot(M_new, D1_new, r_MD1_n, qy_D1M_n, qz_D1M_n);
    [rdot_MD2_n, ~, ~] = computeMDDot(M_new, D2_new, r_MD2_n, qy_D2M_n, qz_D2M_n);

    tgo_norm_n = max(0.0, (Tev - t_new) / max(Tev, 1e-6));
    obs_next = [r_MD1_n; rdot_MD1_n; qy_D1M_n; qz_D1M_n;
                r_MD2_n; rdot_MD2_n; qy_D2M_n; qz_D2M_n;
                r_MT_n;  qy_MT_n;   qz_MT_n;  M_new(6);  tgo_norm_n];
    state_abs_next = [M_new(:); D1_new(:); D2_new(:)];

    intercepted = (min(r_MD1_n, r_MD2_n) <= cfg.kill_radius);
    stage2_done = (t_new >= Tev) && ~intercepted;
    terminated  = intercepted || stage2_done;

    fact = struct();
    fact.r_MD1       = r_MD1_n;
    fact.r_MD2       = r_MD2_n;
    fact.r_MT        = r_MT_n;
    fact.intercepted = logical(intercepted);
    fact.stage2_done = logical(stage2_done);
    fact.terminated  = logical(terminated);
    fact.done        = logical(terminated);
end
