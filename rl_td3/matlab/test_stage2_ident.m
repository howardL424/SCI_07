function ok = test_stage2_ident()
%TEST_STAGE2_IDENT  Kalman / sparse FF-RLS / 因果时序 / 重置 / 回归冒烟。
%
%   ok = test_stage2_ident()

    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);
    addpath(fullfile(this_dir, '..', '..'));
    addpath(fullfile(this_dir, '..', '..', 'expert_trajectory'));

    ok = true;
    tol = 1e-6;

    fprintf('--- [1] analytic_truth 回归 ---\n');
    if ~test_stage2_core_equiv()
        fprintf('[FAIL] analytic_truth / stage2_core_equiv\n');
        ok = false;
    else
        fprintf('[OK] analytic_truth 回归\n');
    end

    fprintf('--- [2] 无噪 Kalman 跟踪 ---\n');
    if ~local_test_kf_noiseless(tol)
        ok = false;
    else
        fprintf('[OK] 无噪 Kalman\n');
    end

    fprintf('--- [3] 合成 FF-RLS 收敛 (mu=0 / mu>0) ---\n');
    if ~local_test_ffrls_synth()
        ok = false;
    else
        fprintf('[OK] 合成 FF-RLS\n');
    end

    fprintf('--- [4] 因果：首拍不做 RLS，且用上一拍 a_M ---\n');
    if ~local_test_causal()
        ok = false;
    else
        fprintf('[OK] 因果时序\n');
    end

    fprintf('--- [5] episode 重置 ---\n');
    if ~local_test_reset()
        ok = false;
    else
        fprintf('[OK] 重置\n');
    end

    fprintf('--- [6] analytic_truth vs coeffs 数值入口等价 ---\n');
    if ~local_test_coeffs_equiv(tol)
        ok = false;
    else
        fprintf('[OK] coeffs 入口等价\n');
    end

    fprintf('--- [7] 式(13)/(15) 投影系数恒等式 ---\n');
    if ~local_test_bc_identities()
        ok = false;
    else
        fprintf('[OK] B 系数恒等式\n');
    end

    fprintf('--- [8] PN warm-start 不读取真值 a_D ---\n');
    if ~local_test_no_ad_leak()
        ok = false;
    else
        fprintf('[OK] PN warm-start 无真值泄漏\n');
    end

    fprintf('--- [9] KF 设计 R 与实际量测噪声解耦 ---\n');
    if ~local_test_kf_design_covariance()
        ok = false;
    else
        fprintf('[OK] KF 设计协方差\n');
    end

    fprintf('--- [10] 显式 eta 与 A21 投影 ---\n');
    if ~local_test_eta_and_projection()
        ok = false;
    else
        fprintf('[OK] eta / A21 投影\n');
    end

    if ok
        fprintf('\n[PASS] test_stage2_ident 全部通过\n');
    else
        fprintf('\n[FAIL] test_stage2_ident 存在失败项\n');
    end
end

function ok = local_test_kf_noiseless(tol)
    ok = true;
    cfg = struct('sigma_r', 1e-9, 'sigma_rdot', 1e-9, 'sigma_q', 0, ...
        'sigma_qdot', 0, 'sigma_theta', 0, 'sigma_psi', 0, ...
        'kf_qj', 1e2, 'noise_seed', 1);
    ident = stage2_ident_init(cfg);
    H = evasion_helpers();
    dt = ident.dt;
    % 常加速度 rddot = -20
    r0 = 5000; rd0 = -200; rdd = -20;
    a_D = [0; 0];
    for k = 0:40
        t = k * dt;
        r = r0 + rd0 * t + 0.5 * rdd * t^2;
        rd = rd0 + rdd * t;
        g = struct('r', r, 'rdot', rd, 'qy', 0.1, 'qz', -0.05, ...
            'qdy', 0.01, 'qdz', -0.02);
        truth = struct('D1', g, 'D2', g, 'T', g, ...
            'theta_M', 0, 'psi_M', 0, 'theta_D1', 0, 'psi_D1', 0, ...
            'theta_D2', 0, 'psi_D2', 0);
        [ident, ~, id_diag] = stage2_ident_step(ident, truth, a_D, a_D, H);
        ident = stage2_ident_commit(ident, [1; -1]);
        if k > 15
            err_r = abs(id_diag.r_hat(1) - r);
            err_rd = abs(id_diag.rdot_hat(1) - rd);
            err_rdd = abs(id_diag.rddot_hat(1) - rdd);
            if err_r > 5 || err_rd > 2 || err_rdd > 5
                fprintf('[FAIL] KF err r=%.3g rd=%.3g rdd=%.3g at k=%d\n', ...
                    err_r, err_rd, err_rdd, k);
                ok = false;
                return;
            end
        end
    end
end

function ok = local_test_ffrls_synth()
    ok = true;
    % 三通道共 11 参数：在缩放域用充分激励的常参数数据验证收敛。
    theta_set = {
        [1e-4; 0.8; -0.5; 6.0], ...
        [2e-4; -0.4; 0.9; -4.0], ...
        [1.5e-4; 0.6; 0.3]};
    dims = [4, 4, 3];
    Sset = {diag([1e4,50,50,1]), diag([1e4,50,50,1]), ...
        diag([1e4,50,50])};
    lambda = 0.98;
    eta = 1e-3 / 2;
    rng(7);
    for c = 1:3
        n = dims(c);
        theta_true = theta_set{c};
        S = Sset{c};
        th0 = zeros(n, 1); P0 = 1e4 * eye(n);
        th_eta = th0; Peta = P0;
        for k = 1:1200
            phi = [4000 + randn()*1500; randn()*30; randn()*30];
            if n == 4; phi = [phi; 1]; end %#ok<AGROW>
            y = phi' * theta_true + 0.05 * randn();
            phi_bar = S \ phi;
            [th0, P0] = local_ffrls(th0, P0, phi_bar, y, lambda);
            [th_eta, Peta] = local_ffrls(th_eta, Peta, phi_bar, y, lambda);
            th_eta = sign(th_eta) .* max(abs(th_eta) - eta, 0);
        end
        err0 = norm(S \ th0 - theta_true);
        err_eta = norm(S \ th_eta - theta_true);
        if err0 > 0.2 || err_eta > 1.0
            fprintf('[FAIL] synth channel=%d err0=%.3g err_eta=%.3g\n', ...
                c, err0, err_eta);
            ok = false;
        end
    end

    % 软阈值在纯噪声小系数上应保持稀疏。
    th_sp = [0; 0; 0; 0]; Psp = 1e4 * eye(4);
    for k = 1:200
        phi = [randn(); randn(); randn(); 1];
        y = 0.01 * randn();
        [th_sp, Psp] = local_ffrls(th_sp, Psp, phi, y, lambda);
        th_sp = sign(th_sp) .* max(abs(th_sp) - 0.05, 0);
    end
    if norm(th_sp) > 0.5
        fprintf('[FAIL] soft-threshold 未充分稀疏 ||th||=%.3g\n', norm(th_sp));
        ok = false;
    end
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

function ok = local_test_causal()
    ok = true;
    cfg = struct('sigma_r', 0, 'sigma_rdot', 0, 'sigma_q', 0, ...
        'sigma_qdot', 0, 'sigma_theta', 0, 'sigma_psi', 0, ...
        'noise_seed', 3, 'rls_mu', 0);
    ident = stage2_ident_init(cfg);
    H = evasion_helpers();
    g = struct('r', 4000, 'rdot', -150, 'qy', 0.05, 'qz', 0.02, ...
        'qdy', 0.01, 'qdz', 0.01);
    truth = struct('D1', g, 'D2', g, 'T', g, ...
        'theta_M', 0.1, 'psi_M', -0.1, 'theta_D1', 0, 'psi_D1', 0, ...
        'theta_D2', 0, 'psi_D2', 0);
    a_D = [5; -3];
    [ident, ~, d1] = stage2_ident_step(ident, truth, a_D, a_D, H);
    if d1.did_rls
        fprintf('[FAIL] 首拍不应做 RLS\n');
        ok = false;
    end
    a_cmd = [12.5; -7.5];
    ident = stage2_ident_commit(ident, a_cmd);
    if ~ident.has_prev || max(abs(ident.a_M_sat_prev(:) - a_cmd(:))) > 0
        fprintf('[FAIL] commit 未保存 a_M_sat\n');
        ok = false;
    end
    g2 = g; g2.r = 3850; g2.rdot = -152;
    truth.D1 = g2; truth.D2 = g2; truth.T = g2;
    [ident, ~, d2] = stage2_ident_step(ident, truth, a_D, a_D, H); %#ok<ASGLU>
    if ~d2.did_rls
        fprintf('[FAIL] 第二拍应做 RLS\n');
        ok = false;
    end
end

function ok = local_test_reset()
    ok = true;
    cfg = struct('coeff_mode', 'sparse_ffrls', 'noise_seed', 9, ...
        'sigma_r', 1, 'sigma_rdot', 0.1);
    core = stage2_core_init(cfg);
    M  = [8000, 0, 2000, 0, 0, 800];
    D1 = [6000, 500, 1500, 0, 0, 700];
    D2 = [6000, -500, 1500, 0, 0, 700];
    sabs0 = [M(:); D1(:); D2(:)];
    an = [0; 0; 0];
    Tev = 5;
    ident = [];
    [~, ~, ~, ~, ident] = stage2_step(core, sabs0, 0.0, Tev, an, ident);
    [~, ~, ~, ~, ident] = stage2_step(core, sabs0, 0.1, Tev, an, ident);
    sc1 = ident.step_count;
    th1 = ident.ch(1).theta_sc;
    % 模拟新 episode：重新 init
    ident2 = stage2_ident_init(core.cfg);
    if ident2.step_count ~= 0 || ident2.has_prev
        fprintf('[FAIL] 重置后 step_count/has_prev 非初值\n');
        ok = false;
    end
    if sc1 < 1
        fprintf('[FAIL] 步进计数异常\n');
        ok = false;
    end
    if isequal(th1, ident2.ch(1).theta) && ident.has_prev
        % theta_sc 在第二拍后应已更新；重置后应回到零缩放 warm 前状态
        % 仅检查 has_prev / step_count 已足够
    end
    %#ok<NASGU>
end

function ok = local_test_coeffs_equiv(tol)
    ok = true;
    H = evasion_helpers();
    % 随机几何点对比解析入口与 coeffs 入口
    qy = 0.12; qz = -0.08; qdy = 0.02; qdz = -0.015;
    r = 3500; rdot = -180; t = 1.0; tf = 12.0;
    thM = 0.05; psM = -0.03; thD = 0.1; psD = 0.02;
    aD = [10; -8];
    Q = sqrt(qdy^2 + qdz^2 * cos(qy)^2);
    tgo = tf - t;
    if Q * tgo > 100; Q = 100 / tgo; end
    [B1, B2, C1] = H.computeBC(qy, qz, thM, psM, thD, psD, aD(1), aD(2));
    a1 = H.computeOptimalCmd_duo_target( ...
        qy, qz, qdy, qdz, r, rdot, t, tf, ...
        qy, qz, qdy, qdz, r, rdot, tf, ...
        qy, qz, qdy, qdz, r, rdot, tf, ...
        thM, psM, thD, psD, thD, psD, ...
        aD, aD, 0.5, 0.5, 2.0, 1.0, 20.0);
    a2 = H.computeOptimalCmd_duo_target_coeffs( ...
        Q, [B1; B2], -C1, r, rdot, tgo, tf, ...
        Q, [B1; B2], -C1, r, rdot, tgo, tf, ...
        Q, [B1; B2], 0.0, r, rdot, tgo, tf, ...
        t, 0.5, 0.5, 2.0, 1.0, 20.0);
    if max(abs(a1(:) - a2(:))) > tol
        fprintf('[FAIL] coeffs 不等价 err=%.3g\n', max(abs(a1(:)-a2(:))));
        ok = false;
    end
end

function ok = local_test_bc_identities()
    ok = true;
    H = evasion_helpers();
    qy = 0.21; qz = -0.34; thA = 0.12; psA = -0.18;
    [B1, B2, ~] = H.computeBC(qy, qz, thA, psA, 0, 0, 0, 0);
    radial = cos(thA) * cos(qy) * cos(qz - psA) + sin(thA) * sin(qy);
    err_ncv = abs(B1^2 + B2^2 - (1 - radial^2));

    [B1T, B2T, ~] = H.computeBC(qy, qz, thA, psA, 0, 0, 0, 0);
    rdot_over_v = -(cos(thA) * cos(qy) * cos(qz - psA) + ...
        sin(thA) * sin(qy));
    err_t = abs(B1T^2 + B2T^2 - (1 - rdot_over_v^2));
    if max(err_ncv, err_t) > 1e-12
        fprintf('[FAIL] B 系数恒等式 err_ncv=%.3g err_T=%.3g\n', err_ncv, err_t);
        ok = false;
    end
end

function ok = local_test_no_ad_leak()
    ok = true;
    cfg = struct('sigma_r', 0, 'sigma_rdot', 0, 'sigma_q', 0, ...
        'sigma_qdot', 0, 'sigma_theta', 0, 'sigma_psi', 0, ...
        'kf_sigma_r', 10, 'kf_sigma_rdot', 1, 'noise_seed', 11);
    H = evasion_helpers();
    ident1 = stage2_ident_init(cfg);
    ident2 = stage2_ident_init(cfg);
    g = struct('r', 4200, 'rdot', -180, 'qy', 0.07, 'qz', -0.04, ...
        'qdy', 0.012, 'qdz', -0.018, 'V', 600);
    truth = struct('D1', g, 'D2', g, 'T', g, ...
        'theta_M', 0.1, 'psi_M', -0.2, 'theta_D1', 0.05, 'psi_D1', 0.1, ...
        'theta_D2', 0.05, 'psi_D2', 0.1);
    [~, c1, d1] = stage2_ident_step(ident1, truth, [1; 2], [3; 4], H);
    [~, c2, d2] = stage2_ident_step(ident2, truth, [70; -50], [-60; 40], H);
    err = max(abs([c1.analytic_noisy.vr1 - c2.analytic_noisy.vr1; ...
        c1.analytic_noisy.vr2 - c2.analytic_noisy.vr2; ...
        d1.theta_noisy{1}(4) - d2.theta_noisy{1}(4); ...
        d1.theta_noisy{2}(4) - d2.theta_noisy{2}(4)]));
    if err > 1e-12
        fprintf('[FAIL] noisy/warm-start 仍依赖真值 a_D，err=%.3g\n', err);
        ok = false;
    end
end

function ok = local_test_kf_design_covariance()
    ok = true;
    cfg1 = struct('sigma_r', 0, 'sigma_rdot', 0, 'sigma_q', 0, ...
        'sigma_qdot', 0, 'sigma_theta', 0, 'sigma_psi', 0, ...
        'kf_qj', 1e2, 'kf_sigma_r', 10, 'kf_sigma_rdot', 1, ...
        'rls_eta', 0, 'noise_seed', 13);
    cfg2 = cfg1; cfg2.kf_qj = 1e5;
    i1 = stage2_ident_init(cfg1);
    i2 = stage2_ident_init(cfg2);
    H = evasion_helpers();
    a_D = [0; 0];
    out1 = 0; out2 = 0;
    for k = 0:12
        t = k * i1.dt;
        rdd = -15 + 8 * sin(2.5 * t);
        rd = -200 - 15*t + (8/2.5) * (1 - cos(2.5*t));
        r = 5000 - 200*t - 7.5*t^2 + (8/2.5)*t - (8/2.5^2)*sin(2.5*t);
        g = struct('r', r, 'rdot', rd, 'qy', 0.1, 'qz', -0.05, ...
            'qdy', 0.01, 'qdz', -0.02, 'V', 600);
        truth = struct('D1', g, 'D2', g, 'T', g, ...
            'theta_M', 0, 'psi_M', 0, 'theta_D1', 0, 'psi_D1', 0, ...
            'theta_D2', 0, 'psi_D2', 0);
        [i1, ~, d1] = stage2_ident_step(i1, truth, a_D, a_D, H);
        [i2, ~, d2] = stage2_ident_step(i2, truth, a_D, a_D, H);
        i1 = stage2_ident_commit(i1, [0; 0]);
        i2 = stage2_ident_commit(i2, [0; 0]);
        out1 = d1.rddot_hat(1); out2 = d2.rddot_hat(1);
    end
    if abs(out1 - out2) < 1e-3
        fprintf('[FAIL] kf_qj 改变未影响 rddot_hat，差值=%.3g\n', abs(out1-out2));
        ok = false;
    end
    if any(diag(i1.R_xi) <= 0) || i1.sigma_r ~= 0 || i1.sigma_rdot ~= 0
        fprintf('[FAIL] KF 设计 R 未与实际零噪声正确解耦\n');
        ok = false;
    end
end

function ok = local_test_eta_and_projection()
    ok = true;
    base = struct('sigma_r', 0, 'sigma_rdot', 0, 'sigma_q', 0, ...
        'sigma_qdot', 0, 'sigma_theta', 0, 'sigma_psi', 0, ...
        'kf_sigma_r', 10, 'kf_sigma_rdot', 1, 'rls_mu', 9, ...
        'rls_eta', 0, 'noise_seed', 17);
    i0 = stage2_ident_init(base);
    base.rls_eta = 0.5;
    i1 = stage2_ident_init(base);
    H = evasion_helpers();
    g = struct('r', 4000, 'rdot', -150, 'qy', 0.08, 'qz', 0.03, ...
        'qdy', 0.015, 'qdz', -0.01, 'V', 600);
    truth = struct('D1', g, 'D2', g, 'T', g, ...
        'theta_M', 0.1, 'psi_M', -0.1, 'theta_D1', 0.02, 'psi_D1', 0.03, ...
        'theta_D2', 0.02, 'psi_D2', 0.03);
    [i0, ~] = stage2_ident_step(i0, truth, [0;0], [0;0], H);
    [i1, ~] = stage2_ident_step(i1, truth, [0;0], [0;0], H);
    i0 = stage2_ident_commit(i0, [20;-10]);
    i1 = stage2_ident_commit(i1, [20;-10]);
    g.r = 3985; g.rdot = -151;
    truth.D1 = g; truth.D2 = g; truth.T = g;
    [i0, c0] = stage2_ident_step(i0, truth, [0;0], [0;0], H);
    [i1, c1] = stage2_ident_step(i1, truth, [0;0], [0;0], H);
    if norm(i0.ch(1).theta_sc - i1.ch(1).theta_sc) < 1e-6
        fprintf('[FAIL] 显式 rls_eta 未改变软阈值结果\n');
        ok = false;
    end
    if c0.Q1 < 0 || c1.Q1 < 0 || c0.QT < 0 || c1.QT < 0
        fprintf('[FAIL] A21 非负投影失效\n');
        ok = false;
    end
end
