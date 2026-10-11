function [ident, coeffs, id_diag] = stage2_ident_step(ident, truth_geo, a_D1, a_D2, H)
%STAGE2_IDENT_STEP  单拍测量噪声 + Kalman + 因果 sparse FF-RLS。
%
%   [ident, coeffs, id_diag] = stage2_ident_step(ident, truth_geo, a_D1, a_D2, H)
%
%   truth_geo 字段:
%     .D1/.D2/.T : .r .rdot .qy .qz .qdy .qdz
%     .theta_M .psi_M .theta_D1 .psi_D1 .theta_D2 .psi_D2
%
%   时序（因果）:
%     1) 对真值几何加噪
%     2) Kalman 更新 (r,rdot) → (r_hat,rdot_hat,rddot_hat)
%     3) 若有上一拍：y=rddot_hat(k)，phi=[r_hat(k-1); a_M_sat(k-1); 1]
%     4) 输出本拍制导系数（首拍 = noisy 解析 warm-start，不做 RLS）
%
%   参数时序:
%     vartheta[k-1] 在 [t_{k-1},t_k) 内零阶保持；第 k 拍的因果
%     回归与辨识误差均对应缓存的 vartheta[k-1]。当前几何计算得到的
%     vartheta[k] 在本拍末提交，供下一周期使用。
%
%   下一拍提交实际饱和指令请调用 stage2_ident_commit(ident, a_M_sat)。

    if nargin < 5 || isempty(H)
        H = evasion_helpers();
    end
    computeBC = H.computeBC;
    computePN = H.computePN;
    saturate = H.saturate;

    noise = stage2_measurement_noise(ident, ident.step_count);
    nD1 = noise.D1;
    nD2 = noise.D2;
    nT = noise.T;
    nAtt = noise.att;
    ident.noise_draw = noise.draw_end;

    mD1 = local_apply_noise(truth_geo.D1, nD1);
    mD2 = local_apply_noise(truth_geo.D2, nD2);
    mT  = local_apply_noise(truth_geo.T,  nT);

    theta_M_m = truth_geo.theta_M + nAtt.theta_M;
    psi_M_m   = truth_geo.psi_M   + nAtt.psi_M;
    theta_D1_m = truth_geo.theta_D1 + nAtt.theta_D1;
    psi_D1_m   = truth_geo.psi_D1   + nAtt.psi_D1;
    theta_D2_m = truth_geo.theta_D2 + nAtt.theta_D2;
    psi_D2_m   = truth_geo.psi_D2   + nAtt.psi_D2;

    % 式(9)：用可用量测和已知 PN 模型重建 NCV 加速度。
    % 该估计仅用于 noisy 解析基线及首拍 C1 warm-start，不进入植物动力学。
    VD1 = local_field_or(truth_geo.D1, 'V', 0.0);
    VD2 = local_field_or(truth_geo.D2, 'V', 0.0);
    a_D1_pn = saturate(computePN(ident.KD1, VD1, mD1.qdy, mD1.qdz, theta_D1_m), ...
        ident.max_ov_D);
    a_D2_pn = saturate(computePN(ident.KD2, VD2, mD2.qdy, mD2.qdz, theta_D2_m), ...
        ident.max_ov_D);

    [B11t, B21t, C11t] = computeBC(truth_geo.D1.qy, truth_geo.D1.qz, ...
        truth_geo.theta_M, truth_geo.psi_M, truth_geo.theta_D1, truth_geo.psi_D1, a_D1(1), a_D1(2));
    [B12t, B22t, C12t] = computeBC(truth_geo.D2.qy, truth_geo.D2.qz, ...
        truth_geo.theta_M, truth_geo.psi_M, truth_geo.theta_D2, truth_geo.psi_D2, a_D2(1), a_D2(2));
    [B1Tt, B2Tt, ~] = computeBC(truth_geo.T.qy, truth_geo.T.qz, ...
        truth_geo.theta_M, truth_geo.psi_M, 0, 0, 0, 0);

    A21_1t = truth_geo.D1.qdy^2 + truth_geo.D1.qdz^2 * cos(truth_geo.D1.qy)^2;
    A21_2t = truth_geo.D2.qdy^2 + truth_geo.D2.qdz^2 * cos(truth_geo.D2.qy)^2;
    A21_Tt = truth_geo.T.qdy^2  + truth_geo.T.qdz^2  * cos(truth_geo.T.qy)^2;

    [B11n, B21n, C11n] = computeBC(mD1.qy, mD1.qz, theta_M_m, psi_M_m, ...
        theta_D1_m, psi_D1_m, a_D1_pn(1), a_D1_pn(2));
    [B12n, B22n, C12n] = computeBC(mD2.qy, mD2.qz, theta_M_m, psi_M_m, ...
        theta_D2_m, psi_D2_m, a_D2_pn(1), a_D2_pn(2));
    [B1Tn, B2Tn, ~] = computeBC(mT.qy, mT.qz, theta_M_m, psi_M_m, 0, 0, 0, 0);
    A21_1n = mD1.qdy^2 + mD1.qdz^2 * cos(mD1.qy)^2;
    A21_2n = mD2.qdy^2 + mD2.qdz^2 * cos(mD2.qy)^2;
    A21_Tn = mT.qdy^2  + mT.qdz^2  * cos(mT.qy)^2;

    theta_warm = { ...
        [A21_1n; B11n; B21n; C11n], ...
        [A21_2n; B12n; B22n; C12n], ...
        [A21_Tn; B1Tn; B2Tn] };
    theta_true_current = { ...
        [A21_1t; B11t; B21t; C11t], ...
        [A21_2t; B12t; B22t; C12t], ...
        [A21_Tt; B1Tt; B2Tt] };

    meas_pack = {mD1, mD2, mT};
    do_rls = ident.has_prev;
    eta = ident.rls_eta;
    rls_gated = false(3, 1);
    theta_true_causal = cell(1, 3);
    theta_noisy_causal = cell(1, 3);
    for i = 1:3
        if ident.ch(i).has_theta_true_hold
            theta_true_causal{i} = ident.ch(i).theta_true_hold;
        else
            theta_true_causal{i} = nan(ident.ch(i).n_theta, 1);
        end
        if isfield(ident.ch(i), 'has_theta_noisy_hold') && ...
                ident.ch(i).has_theta_noisy_hold
            theta_noisy_causal{i} = ident.ch(i).theta_noisy_hold;
        else
            theta_noisy_causal{i} = nan(ident.ch(i).n_theta, 1);
        end
    end

    for i = 1:3
        z = [meas_pack{i}.r; meas_pack{i}.rdot];
        S = local_scale(ident, ident.ch(i).n_theta);

        if ~ident.ch(i).initialized
            th_phys = theta_warm{i};
            th_phys(1) = max(th_phys(1), 0);
            phi0 = zeros(ident.ch(i).n_theta, 1);
            phi0(1) = z(1);
            if ident.ch(i).n_theta == 4
                phi0(4) = 1;
            end
            rddot0 = phi0' * th_phys;
            ident.ch(i).xi = [z(1); z(2); rddot0];
            ident.ch(i).P_xi = diag([ident.sigma_r^2, ...
                ident.sigma_rdot^2, ident.kf_sigma_a0^2]);
            ident.ch(i).theta_sc = S * th_phys;   % 内部存缩放坐标
            ident.ch(i).P = local_cap_covariance_trace( ...
                ident.rls_P0 * eye(ident.ch(i).n_theta), ident.rls_P_trace_max);
            ident.ch(i).initialized = true;
        else
            [ident.ch(i).xi, ident.ch(i).P_xi] = local_kf_predict_update( ...
                ident.ch(i).xi, ident.ch(i).P_xi, z, ...
                ident.F_xi, ident.Q_xi, ident.H_xi, ident.R_xi);
        end

        rls_gated(i) = local_endgame_gate(ident, ident.ch(i).xi);
        if do_rls && ~rls_gated(i)
            y = ident.ch(i).xi(3);
            if ident.ch(i).n_theta == 4
                phi = [ident.ch(i).r_hat_prev; ident.a_M_sat_prev(1); ident.a_M_sat_prev(2); 1];
            else
                phi = [ident.ch(i).r_hat_prev; ident.a_M_sat_prev(1); ident.a_M_sat_prev(2)];
            end
            phi_bar = S \ phi;
            [th_sc, P] = stage2_ffrls_update( ...
                ident.ch(i).theta_sc, ident.ch(i).P, phi_bar, y, ...
                ident.rls_lambda, eta, ident.rls_P_trace_max);
            ident.ch(i).theta_sc = th_sc;
            ident.ch(i).P = P;
        end

        ident.ch(i).r_hat_prev = ident.ch(i).xi(1);
    end

    % 真值几何（仅供诊断）；r̈_true 用上一拍距离率有限差分。
    % 第 k 拍回归对应 [t_{k-1},t_k)，故以缓存的 vartheta[k-1] 为因果真值。
    r_true = [truth_geo.D1.r; truth_geo.D2.r; truth_geo.T.r];
    rdot_true = [truth_geo.D1.rdot; truth_geo.D2.rdot; truth_geo.T.rdot];
    rddot_true = nan(3, 1);
    frozen_model_residual = nan(3, 1);
    if do_rls && isfield(ident, 'rdot_true_prev') && ...
            isfield(ident, 'r_true_prev') && ...
            numel(ident.rdot_true_prev) == 3 && numel(ident.r_true_prev) == 3
        rddot_true = (rdot_true - ident.rdot_true_prev(:)) / ident.dt;
        for i = 1:3
            if ident.ch(i).n_theta == 4
                phi_true_prev = [ident.r_true_prev(i); ...
                    ident.a_M_sat_prev(1); ident.a_M_sat_prev(2); 1];
            else
                phi_true_prev = [ident.r_true_prev(i); ...
                    ident.a_M_sat_prev(1); ident.a_M_sat_prev(2)];
            end
            frozen_model_residual(i) = rddot_true(i) - ...
                phi_true_prev' * theta_true_causal{i};
        end
    end

    coeffs = struct();
    for i = 1:3
        S = local_scale(ident, ident.ch(i).n_theta);
        th = S \ ident.ch(i).theta_sc;
        th(1) = max(th(1), 0);
        Q = sqrt(th(1));
        b = [th(2); th(3)];
        if ident.ch(i).n_theta == 4
            vr = -th(4);
        else
            vr = 0;
        end
        r_hat = ident.ch(i).xi(1);
        rdot_hat = ident.ch(i).xi(2);
        switch i
            case 1
                coeffs.Q1 = Q; coeffs.b1 = b; coeffs.vr1 = vr;
                coeffs.r1 = r_hat; coeffs.rdot1 = rdot_hat;
                coeffs.theta_hat_D1 = th;
            case 2
                coeffs.Q2 = Q; coeffs.b2 = b; coeffs.vr2 = vr;
                coeffs.r2 = r_hat; coeffs.rdot2 = rdot_hat;
                coeffs.theta_hat_D2 = th;
            case 3
                coeffs.QT = Q; coeffs.bT = b; coeffs.vrT = vr;
                coeffs.rT = r_hat; coeffs.rdotT = rdot_hat;
                coeffs.theta_hat_T = th;
        end
    end

    coeffs.analytic_noisy = struct( ...
        'Q1', sqrt(max(A21_1n,0)), 'b1', [B11n; B21n], 'vr1', -C11n, ...
        'r1', mD1.r, 'rdot1', mD1.rdot, ...
        'Q2', sqrt(max(A21_2n,0)), 'b2', [B12n; B22n], 'vr2', -C12n, ...
        'r2', mD2.r, 'rdot2', mD2.rdot, ...
        'QT', sqrt(max(A21_Tn,0)), 'bT', [B1Tn; B2Tn], 'vrT', 0, ...
        'rT', mT.r, 'rdotT', mT.rdot);
    coeffs.analytic_truth = struct( ...
        'Q1', sqrt(max(A21_1t,0)), 'b1', [B11t; B21t], 'vr1', -C11t, ...
        'r1', truth_geo.D1.r, 'rdot1', truth_geo.D1.rdot, ...
        'Q2', sqrt(max(A21_2t,0)), 'b2', [B12t; B22t], 'vr2', -C12t, ...
        'r2', truth_geo.D2.r, 'rdot2', truth_geo.D2.rdot, ...
        'QT', sqrt(max(A21_Tt,0)), 'bT', [B1Tt; B2Tt], 'vrT', 0, ...
        'rT', truth_geo.T.r, 'rdotT', truth_geo.T.rdot);

    ident.step_count = ident.step_count + 1;

    S1 = local_scale(ident, 4); S2 = local_scale(ident, 4); ST = local_scale(ident, 3);
    th_hat = {S1 \ ident.ch(1).theta_sc, S2 \ ident.ch(2).theta_sc, ST \ ident.ch(3).theta_sc};
    id_diag = struct();
    % 兼容字段 theta_true 明确表示因果真值 vartheta[k-1]。
    id_diag.theta_true = theta_true_causal;
    id_diag.theta_true_causal = theta_true_causal;
    id_diag.theta_true_current = theta_true_current;
    id_diag.theta_noisy = theta_warm;
    id_diag.theta_noisy_causal = theta_noisy_causal;
    id_diag.theta_hat = th_hat;
    q_hat = [sqrt(max(th_hat{1}(1), 0)); sqrt(max(th_hat{2}(1), 0)); ...
        sqrt(max(th_hat{3}(1), 0))];
    q_true_causal = [sqrt(max(theta_true_causal{1}(1), 0)); ...
        sqrt(max(theta_true_causal{2}(1), 0)); ...
        sqrt(max(theta_true_causal{3}(1), 0))];
    q_true_current = [sqrt(max(theta_true_current{1}(1), 0)); ...
        sqrt(max(theta_true_current{2}(1), 0)); ...
        sqrt(max(theta_true_current{3}(1), 0))];
    id_diag.qdot_err = q_hat - q_true_causal;
    id_diag.qdot_err_causal = id_diag.qdot_err;
    id_diag.qdot_err_command = q_hat - q_true_current;
    id_diag.P_trace = zeros(3, 1);
    id_diag.P_min_eig = zeros(3, 1);
    id_diag.P_max_eig = zeros(3, 1);
    for i = 1:3
        ev = eig(0.5 * (ident.ch(i).P + ident.ch(i).P'));
        id_diag.P_trace(i) = trace(ident.ch(i).P);
        id_diag.P_min_eig(i) = min(real(ev));
        id_diag.P_max_eig(i) = max(real(ev));
    end
    id_diag.rddot_hat = [ident.ch(1).xi(3); ident.ch(2).xi(3); ident.ch(3).xi(3)];
    id_diag.r_hat = [ident.ch(1).xi(1); ident.ch(2).xi(1); ident.ch(3).xi(1)];
    id_diag.rdot_hat = [ident.ch(1).xi(2); ident.ch(2).xi(2); ident.ch(3).xi(2)];
    id_diag.r_true = r_true;
    id_diag.rdot_true = rdot_true;
    id_diag.rddot_true = rddot_true;
    id_diag.frozen_model_residual = frozen_model_residual;
    % did_rls 表示"本拍三通道是否都真正执行了回归"；端局门控生效时为 false。
    id_diag.did_rls = do_rls && ~any(rls_gated);
    id_diag.rls_gated = rls_gated;
    id_diag.meas_r = [mD1.r; mD2.r; mT.r];
    id_diag.meas_rdot = [mD1.rdot; mD2.rdot; mT.rdot];
    id_diag.meas_qy = [mD1.qy; mD2.qy; mT.qy];
    id_diag.meas_qz = [mD1.qz; mD2.qz; mT.qz];
    id_diag.a_cmd_err = [NaN; NaN];  % 由 stage2_step 在算出双路径指令后回填
    id_diag.a_cmd_err_sat = [NaN; NaN];
    id_diag.a_D_pn = [a_D1_pn(:); a_D2_pn(:)];

    % 提交当前边界真值，作为下一周期 [t_k,t_{k+1}) 的冻结参数。
    for i = 1:3
        ident.ch(i).theta_true_hold = theta_true_current{i};
        ident.ch(i).has_theta_true_hold = true;
        ident.ch(i).theta_noisy_hold = theta_warm{i};
        ident.ch(i).has_theta_noisy_hold = true;
    end
    ident.r_true_prev = r_true;
    ident.rdot_true_prev = rdot_true;
end

function gated = local_endgame_gate(ident, xi)
%LOCAL_ENDGAME_GATE  仅凭 KF 后验判断本通道是否进入回归失效的交汇末段。
    gated = false;
    r_hat = xi(1);
    rdot_hat = xi(2);
    if ident.rls_gate_r_min > 0 && r_hat < ident.rls_gate_r_min
        gated = true;
        return;
    end
    if ident.rls_gate_tgo_min > 0 && rdot_hat < 0
        tgo_hat = -r_hat / rdot_hat;
        if tgo_hat < ident.rls_gate_tgo_min
            gated = true;
        end
    end
end

function S = local_scale(ident, n)
    if n == 4
        S = diag([ident.rls_r0, ident.rls_a0, ident.rls_a0, 1]);
    else
        S = diag([ident.rls_r0, ident.rls_a0, ident.rls_a0]);
    end
end

function [xi, P] = local_kf_predict_update(xi, P, z, F, Q, H, R)
    xi_p = F * xi;
    P_p = F * P * F' + Q;
    S = H * P_p * H' + R;
    K = P_p * H' / S;
    xi = xi_p + K * (z - H * xi_p);
    P = (eye(3) - K * H) * P_p;
    P = 0.5 * (P + P');
end

function P = local_cap_covariance_trace(P, p_trace_max)
    trP = real(trace(P));
    if isfinite(p_trace_max) && trP > p_trace_max
        P = P * (p_trace_max / trP);
        P = 0.5 * (P + P');
    end
end

function v = local_field_or(s, name, default)
    if isfield(s, name) && ~isempty(s.(name))
        v = double(s.(name));
    else
        v = default;
    end
end

function m = local_apply_noise(g, n)
    m = struct();
    m.r    = g.r + n.r;
    m.rdot = g.rdot + n.rdot;
    m.qy   = g.qy + n.qy;
    m.qz   = g.qz + n.qz;
    m.qdy  = g.qdy + n.qdy;
    m.qdz  = g.qdz + n.qdz;
end
