function [obs_next, state_abs_next, t_new, fact, ident_state] = ...
        stage2_step(core, state_abs, t, Tev, action_norm, ident_state)
%STAGE2_STEP  预初始化核心上的一步 stage2 动力学（物理事实，不含奖励常数）。
%
%   [obs_next, state_abs_next, t_new, fact] = ...
%       stage2_step(core, state_abs, t, Tev, action_norm)
%   [obs_next, state_abs_next, t_new, fact, ident_state] = ...
%       stage2_step(core, state_abs, t, Tev, action_norm, ident_state)
%
%   输入:
%     core         stage2_core_init 返回值
%     state_abs    [18×1] 当前绝对态 [M(6); D1(6); D2(6)]
%     t            当前时刻 (s)
%     Tev          规避段时长 (s)
%     action_norm  [3×1] 已映射/裁剪的 TD3 动作 ∈ [-1,1]
%     ident_state  可选；辨识器显式状态（跨步持久化）
%
%   coeff_mode (core.cfg):
%     analytic_truth  — 现有解析真值路径（默认）
%     analytic_noisy  — 带噪测量代入解析 computeBC
%     sparse_ffrls    — Kalman + L1 FF-RLS 辨识系数
%   ident_apply_cmd=false 时仍跑辨识诊断，但指令用 analytic_truth（影子模式）。
%
%   硬性约定: 调用方必须先 phys_to_action_norm 再传入；本函数不再二次裁剪。

    if nargin < 6
        ident_state = [];
    end

    cfg = core.cfg;
    H   = core.H;
    dt  = cfg.dt;
    if isfield(cfg, 'action_mapping_version') && ~isempty(cfg.action_mapping_version)
        amap = char(cfg.action_mapping_version);
    else
        amap = char(core.meta.action_mapping_version);
    end
    if ~isfield(cfg, 'coeff_mode') || isempty(cfg.coeff_mode)
        coeff_mode = 'analytic_truth';
    else
        coeff_mode = char(cfg.coeff_mode);
    end
    if ~isfield(cfg, 'ident_apply_cmd')
        apply_cmd = true;
    else
        apply_cmd = logical(cfg.ident_apply_cmd);
    end

    M  = state_abs(1:6)';
    D1 = state_abs(7:12)';
    D2 = state_abs(13:18)';

    [r1, r2, rT, gamma, old_ok] = local_decode_ratios(action_norm, amap, cfg.w);

    computeLOS                   = H.computeLOS;
    computeMDDot                 = H.computeMDDot;
    computePN                    = H.computePN;
    computeOptimalCmd_duo_target = H.computeOptimalCmd_duo_target;
    computeOptimalCmd_duo_target_coeffs = H.computeOptimalCmd_duo_target_coeffs;
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

    % 防御弹 PN 始终用真值几何（测量噪声只作用于攻击方制导系数路径）
    a_D1 = saturate(computePN(cfg.KD1, D1(6), qd_y1, qd_z1, D1(4)), core.max_ov_D);
    a_D2 = saturate(computePN(cfg.KD2, D2(6), qd_y2, qd_z2, D2(4)), core.max_ov_D);

    [tf1, tgo1] = local_intercept_time(t, r_MD1, rdot_MD1);
    [tf2, tgo2] = local_intercept_time(t, r_MD2, rdot_MD2);
    [tfT, tgoT] = local_intercept_time(t, r_TM, rdot_TM);
    tf1_truth = tf1;
    tf2_truth = tf2;
    tfT_truth = tfT;

    ident_diag = struct();
    use_analytic = true;
    c = [];

    if ~strcmp(coeff_mode, 'analytic_truth')
        if isempty(ident_state)
            ident_state = stage2_ident_init(cfg);
        end
        truth_geo = struct();
        truth_geo.D1 = struct('r', r_MD1, 'rdot', rdot_MD1, ...
            'qy', qy_D1M, 'qz', qz_D1M, 'qdy', qd_y1, 'qdz', qd_z1, ...
            'V', D1(6));
        truth_geo.D2 = struct('r', r_MD2, 'rdot', rdot_MD2, ...
            'qy', qy_D2M, 'qz', qz_D2M, 'qdy', qd_y2, 'qdz', qd_z2, ...
            'V', D2(6));
        truth_geo.T  = struct('r', r_TM,  'rdot', rdot_TM,  ...
            'qy', qy_TM,  'qz', qz_TM,  'qdy', qd_yTM, 'qdz', qd_zTM, ...
            'V', 0.0);
        truth_geo.theta_M = M(4);  truth_geo.psi_M = M(5);
        truth_geo.theta_D1 = D1(4); truth_geo.psi_D1 = D1(5);
        truth_geo.theta_D2 = D2(4); truth_geo.psi_D2 = D2(5);

        [ident_state, coeffs, ident_diag] = stage2_ident_step( ...
            ident_state, truth_geo, a_D1, a_D2, H);

        if strcmp(coeff_mode, 'analytic_noisy')
            c = coeffs.analytic_noisy;
            use_analytic = false;
        elseif strcmp(coeff_mode, 'sparse_ffrls') && apply_cmd
            c = coeffs;
            use_analytic = false;
        end
        % sparse_ffrls + ident_apply_cmd=false：影子模式，仍用解析真值指令
    end

    % 非真值系数路径的剩余时间同样由该路径可用的 r/rdot 计算，
    % 避免式(34)–(37)的积分增益继续读取真值几何。
    if ~use_analytic
        [tf1, tgo1] = local_intercept_time(t, c.r1, c.rdot1);
        [tf2, tgo2] = local_intercept_time(t, c.r2, c.rdot2);
        [tfT, tgoT] = local_intercept_time(t, c.rT, c.rdotT);
    end

    if use_analytic
        a_M_raw = computeOptimalCmd_duo_target( ...
            qy_D1M, qz_D1M, qd_y1, qd_z1, r_MD1, rdot_MD1, t, tf1, ...
            qy_D2M, qz_D2M, qd_y2, qd_z2, r_MD2, rdot_MD2,    tf2, ...
            qy_TM,  qz_TM,  qd_yTM, qd_zTM, r_TM, rdot_TM,   tfT, ...
            M(4), M(5), D1(4), D1(5), D2(4), D2(5), ...
            a_D1, a_D2, r1, r2, rT, 1.0, cfg.dstar);
    else
        % 与解析路径一致：Q*tgo 上限钳制；tgo/tf 使用本系数路径的时间轴
        Q1c = local_clamp_Q(c.Q1, tgo1);
        Q2c = local_clamp_Q(c.Q2, tgo2);
        QTc = local_clamp_Q(c.QT, tgoT);
        a_M_raw = computeOptimalCmd_duo_target_coeffs( ...
            Q1c, c.b1, c.vr1, c.r1, c.rdot1, tgo1, tf1, ...
            Q2c, c.b2, c.vr2, c.r2, c.rdot2, tgo2, tf2, ...
            QTc, c.bT, c.vrT, c.rT, c.rdotT, tgoT, tfT, ...
            t, r1, r2, rT, 1.0, cfg.dstar);
    end

    % 诊断基准始终为解析真值指令，避免 sparse_ffrls 闭环应用时
    % “预测指令减自身”退化为恒零。真值仅用于诊断，不进入控制。
    if ~strcmp(coeff_mode, 'analytic_truth') && isstruct(ident_diag) && ...
            isfield(ident_diag, 'theta_hat')
        a_truth_ref = computeOptimalCmd_duo_target( ...
            qy_D1M, qz_D1M, qd_y1, qd_z1, r_MD1, rdot_MD1, t, tf1_truth, ...
            qy_D2M, qz_D2M, qd_y2, qd_z2, r_MD2, rdot_MD2, tf2_truth, ...
            qy_TM, qz_TM, qd_yTM, qd_zTM, r_TM, rdot_TM, tfT_truth, ...
            M(4), M(5), D1(4), D1(5), D2(4), D2(5), ...
            a_D1, a_D2, r1, r2, rT, 1.0, cfg.dstar);
        if strcmp(coeff_mode, 'sparse_ffrls')
            [tf1_id, tgo1_id] = local_intercept_time(t, coeffs.r1, coeffs.rdot1);
            [tf2_id, tgo2_id] = local_intercept_time(t, coeffs.r2, coeffs.rdot2);
            [tfT_id, tgoT_id] = local_intercept_time(t, coeffs.rT, coeffs.rdotT);
            Q1c = local_clamp_Q(coeffs.Q1, tgo1_id);
            Q2c = local_clamp_Q(coeffs.Q2, tgo2_id);
            QTc = local_clamp_Q(coeffs.QT, tgoT_id);
            a_pred = computeOptimalCmd_duo_target_coeffs( ...
                Q1c, coeffs.b1, coeffs.vr1, coeffs.r1, coeffs.rdot1, tgo1_id, tf1_id, ...
                Q2c, coeffs.b2, coeffs.vr2, coeffs.r2, coeffs.rdot2, tgo2_id, tf2_id, ...
                QTc, coeffs.bT, coeffs.vrT, coeffs.rT, coeffs.rdotT, tgoT_id, tfT_id, ...
                t, r1, r2, rT, 1.0, cfg.dstar);
        else
            a_pred = a_M_raw;
        end
        ident_diag.a_cmd_err = a_pred(:) - a_truth_ref(:);
        ident_diag.a_cmd_err_sat = ...
            saturate(a_pred, core.max_ov_M) - ...
            saturate(a_truth_ref, core.max_ov_M);
        ident_diag.a_cmd_pred = a_pred(:);
        ident_diag.a_cmd_truth_ref = a_truth_ref(:);
    end

    a_M = saturate(a_M_raw, core.max_ov_M);
    saturated_8g = any(abs(a_M_raw(:)) > core.max_ov_M + 1e-12) || ...
                   any(abs(a_M(:) - a_M_raw(:)) > 1e-12);

    if ~isempty(ident_state)
        ident_state = stage2_ident_commit(ident_state, a_M);
    end

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
    fact.coeff_mode       = coeff_mode;
    fact.ident_diag       = ident_diag;
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

function Q = local_clamp_Q(Q, tgo)
    Q = max(double(Q), 0);
    if abs(tgo) > 1e-12 && Q * tgo > 100
        Q = 100 / tgo;
    end
end

function [tf, tgo] = local_intercept_time(t, r, rdot)
    rd = sign(rdot) * max(abs(rdot), 1e-3);
    if rd == 0
        rd = 1e-3;
    end
    tf = t - r / rd;
    tgo = tf - t;
end
