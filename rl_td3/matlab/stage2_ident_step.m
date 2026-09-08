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
%     3) 若有上一拍：y=rddot_hat(k)，phi=[r_meas(k-1); a_M_sat(k-1); 1]
%     4) 输出本拍制导系数（首拍 = noisy 解析 warm-start，不做 RLS）
%
%   下一拍提交实际饱和指令请调用 stage2_ident_commit(ident, a_M_sat)。

    if nargin < 5 || isempty(H)
        H = evasion_helpers();
    end
    computeBC = H.computeBC;
    computePN = H.computePN;
    saturate = H.saturate;

    [ident, nD1] = local_noise_channel(ident);
    [ident, nD2] = local_noise_channel(ident);
    [ident, nT]  = local_noise_channel(ident);
    [ident, nAtt] = local_noise_att(ident);

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

    meas_pack = {mD1, mD2, mT};
    do_rls = ident.has_prev;
    eta = ident.rls_eta;

    for i = 1:3
        z = [meas_pack{i}.r; meas_pack{i}.rdot];
        S = local_scale(ident, ident.ch(i).n_theta);

        if ~ident.ch(i).initialized
            ident.ch(i).xi = [z(1); z(2); 0];
            ident.ch(i).P_xi = diag([ident.kf_sigma_r^2, ...
                ident.kf_sigma_rdot^2, ident.kf_sigma_a0^2]);
            th_phys = theta_warm{i};
            th_phys(1) = max(th_phys(1), 0);
            ident.ch(i).theta_sc = S * th_phys;   % 内部存缩放坐标
            ident.ch(i).P = ident.rls_P0 * eye(ident.ch(i).n_theta);
            ident.ch(i).initialized = true;
        else
            [ident.ch(i).xi, ident.ch(i).P_xi] = local_kf_predict_update( ...
                ident.ch(i).xi, ident.ch(i).P_xi, z, ...
                ident.F_xi, ident.Q_xi, ident.H_xi, ident.R_xi);
        end

        if do_rls
            y = ident.ch(i).xi(3);
            if ident.ch(i).n_theta == 4
                phi = [ident.ch(i).r_meas_prev; ident.a_M_sat_prev(1); ident.a_M_sat_prev(2); 1];
            else
                phi = [ident.ch(i).r_meas_prev; ident.a_M_sat_prev(1); ident.a_M_sat_prev(2)];
            end
            phi_bar = S \ phi;
            [th_sc, P] = local_ffrls(ident.ch(i).theta_sc, ident.ch(i).P, ...
                phi_bar, y, ident.rls_lambda);
            th_sc = local_soft_threshold(th_sc, eta);
            % A21_sc = r0 * A21 => A21 = A21_sc / r0
            th_phys = S \ th_sc;
            th_phys(1) = max(th_phys(1), 0);
            ident.ch(i).theta_sc = S * th_phys;
            ident.ch(i).P = P;
        end

        ident.ch(i).r_hat_prev = ident.ch(i).xi(1);
        ident.ch(i).r_meas_prev = z(1);
    end

    % 真值几何（供影子 RMSE）；r̈_true 用上一拍距离率有限差分
    r_true = [truth_geo.D1.r; truth_geo.D2.r; truth_geo.T.r];
    rdot_true = [truth_geo.D1.rdot; truth_geo.D2.rdot; truth_geo.T.rdot];
    rddot_true = nan(3, 1);
    if do_rls && isfield(ident, 'rdot_true_prev') && numel(ident.rdot_true_prev) == 3
        rddot_true = (rdot_true - ident.rdot_true_prev(:)) / ident.dt;
    end
    ident.rdot_true_prev = rdot_true;

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
    th_true = {[A21_1t; B11t; B21t; C11t], [A21_2t; B12t; B22t; C12t], [A21_Tt; B1Tt; B2Tt]};
    id_diag = struct();
    id_diag.theta_true = th_true;
    id_diag.theta_noisy = theta_warm;
    id_diag.theta_hat = th_hat;
    q_hat = [sqrt(max(th_hat{1}(1), 0)); sqrt(max(th_hat{2}(1), 0)); ...
        sqrt(max(th_hat{3}(1), 0))];
    q_true = [sqrt(max(th_true{1}(1), 0)); sqrt(max(th_true{2}(1), 0)); ...
        sqrt(max(th_true{3}(1), 0))];
    id_diag.qdot_err = q_hat - q_true;
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
    id_diag.did_rls = do_rls;
    id_diag.meas_r = [mD1.r; mD2.r; mT.r];
    id_diag.meas_rdot = [mD1.rdot; mD2.rdot; mT.rdot];
    id_diag.a_cmd_err = [NaN; NaN];  % 由 stage2_step 在算出双路径指令后回填
    id_diag.a_cmd_err_sat = [NaN; NaN];
    id_diag.a_D_pn = [a_D1_pn(:); a_D2_pn(:)];
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

function [theta, P] = local_ffrls(theta, P, phi, y, lambda)
    phi = phi(:);
    denom = lambda + phi' * P * phi;
    K = (P * phi) / denom;
    err = y - phi' * theta;
    theta = theta + K * err;
    P = (1 / lambda) * (P - K * phi' * P);
    P = 0.5 * (P + P');
end

function v = local_soft_threshold(v, eta)
    if eta <= 0
        return;
    end
    v = sign(v) .* max(abs(v) - eta, 0);
end

function [ident, n] = local_noise_channel(ident)
    n = struct();
    [ident, n.r]    = local_randn(ident, ident.sigma_r);
    [ident, n.rdot] = local_randn(ident, ident.sigma_rdot);
    [ident, n.qy]   = local_randn(ident, ident.sigma_q);
    [ident, n.qz]   = local_randn(ident, ident.sigma_q);
    [ident, n.qdy]  = local_randn(ident, ident.sigma_qdot);
    [ident, n.qdz]  = local_randn(ident, ident.sigma_qdot);
end

function [ident, n] = local_noise_att(ident)
    n = struct();
    [ident, n.theta_M]  = local_randn(ident, ident.sigma_theta);
    [ident, n.psi_M]    = local_randn(ident, ident.sigma_psi);
    [ident, n.theta_D1] = local_randn(ident, ident.sigma_theta);
    [ident, n.psi_D1]   = local_randn(ident, ident.sigma_psi);
    [ident, n.theta_D2] = local_randn(ident, ident.sigma_theta);
    [ident, n.psi_D2]   = local_randn(ident, ident.sigma_psi);
end

function [ident, x] = local_randn(ident, sigma)
    ident.noise_draw = ident.noise_draw + 1;
    u1 = local_hash01(ident.noise_seed, ident.noise_draw, 1);
    u2 = local_hash01(ident.noise_seed, ident.noise_draw, 2);
    u1 = max(u1, 1e-12);
    z = sqrt(-2 * log(u1)) * cos(2 * pi * u2);
    x = sigma * z;
end

function u = local_hash01(seed, draw, lane)
    v = double(seed) * 1.0e-3 + double(draw) * 17.0 + double(lane) * 0.37;
    u = abs(sin(v * 12.9898) * 43758.5453);
    u = u - floor(u);
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
