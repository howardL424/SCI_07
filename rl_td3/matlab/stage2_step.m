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
%     .a_M          [2×1] 饱和前指令（诊断）
%     .a_M_sat      [2×1] 饱和后指令
%     .saturated_8g 逻辑，|a_M| 触达 8g 盒
%     .old_box_feasible 比值反解 (wT,wE) 是否落在旧权重盒（诊断）
%
%   硬性约定: 调用方必须先 phys_to_action_norm 再传入；本函数不再二次裁剪。
%   比值版本 (ratio_log_box_v1) 直接把 ρE、ρT、γ 送入式(47)，等价于
%   以 wE=1 传入相对权重，避免任意组合再裁旧盒。

    cfg = core.cfg;
    H   = core.H;
    dt  = cfg.dt;
    if isfield(cfg, 'action_mapping_version') && ~isempty(cfg.action_mapping_version)
        amap = char(cfg.action_mapping_version);
    else
        amap = char(core.meta.action_mapping_version);
    end

    M  = state_abs(1:6)';
    D1 = state_abs(7:12)';
    D2 = state_abs(13:18)';

    [r1, r2, rT, gamma, old_ok] = local_decode_ratios(action_norm, amap, cfg.w);

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

    % 式(47) 只依赖相对权重：传入 (w1/wE,w2/wE,wT/wE) 且 wE=1
    a_M_raw = computeOptimalCmd_duo_target( ...
        qy_D1M, qz_D1M, qd_y1, qd_z1, r_MD1, rdot_MD1, t, tf1, ...
        qy_D2M, qz_D2M, qd_y2, qd_z2, r_MD2, rdot_MD2,    tf2, ...
        qy_TM,  qz_TM,  qd_yTM, qd_zTM, r_TM, rdot_TM,   tfT, ...
        M(4), M(5), D1(4), D1(5), D2(4), D2(5), ...
        a_D1, a_D2, r1, r2, rT, 1.0, cfg.dstar);
    a_M = saturate(a_M_raw, core.max_ov_M);
    saturated_8g = any(abs(a_M_raw(:)) > core.max_ov_M + 1e-12) || ...
                   any(abs(a_M(:) - a_M_raw(:)) > 1e-12);

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
    fact.r_MD1            = r_MD1_n;
    fact.r_MD2            = r_MD2_n;
    fact.r_MT             = r_MT_n;
    fact.intercepted      = logical(intercepted);
    fact.stage2_done      = logical(stage2_done);
    fact.terminated       = logical(terminated);
    fact.done             = logical(terminated);
    fact.a_M              = a_M_raw(:);
    fact.a_M_sat          = a_M(:);
    fact.saturated_8g     = logical(saturated_8g);
    fact.old_box_feasible = logical(old_ok);
    fact.gamma            = gamma;
    fact.action_mapping_version = amap;
end

function [r1, r2, rT, gamma, old_ok] = local_decode_ratios(action_norm, amap, w)
    a = double(action_norm(:));
    switch amap
        case 'ratio_log_box_v1'
            rhoE  = 10^(a(1) + 1.0);
            rhoT  = 10^(a(2) * 2.5 + 0.5);
            gamma = a(3) * 3.0;
            wE = double(w) / max(rhoE, 1e-12);
            wT = rhoT * wE;
        case 'td3_box_log_v1'
            gamma = a(1) * 3.0;
            wT    = 10^(a(2) * 1.5 + 3.5);
            wE    = 10^(a(3) + 3.0);
            rhoE  = double(w) / max(wE, 1e-12);
            rhoT  = wT / max(wE, 1e-12);
        otherwise
            error('stage2_step:UnknownActionMap', '未知 action_mapping_version=%s', amap);
    end
    eg = exp(-gamma);
    r1 = rhoE / (1.0 + eg);
    r2 = rhoE * eg / (1.0 + eg);
    rT = rhoT;
    old_ok = (wT >= 1e2 - 1e-9) && (wT <= 1e5 + 1e-6) && ...
             (wE >= 1e2 - 1e-9) && (wE <= 1e4 + 1e-6) && ...
             (gamma >= -3.0 - 1e-12) && (gamma <= 3.0 + 1e-12);
end
