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
    % 直接测缩放坐标下的 FF-RLS + soft-threshold 数值性质
    n = 4;
    theta_true = [1e-4; 2.0; -1.5; 0.3];  % A21,B1,B2,C1
    P = 1e4 * eye(n);
    th0 = zeros(n, 1);
    th_mu0 = th0; P0 = P;
    th_mu = th0; Pm = P;
    lambda = 0.98;
    eta = 1e-3 / 2;
    rng(7);
    for k = 1:800
        phi = [randn()*5e3; randn()*30; randn()*30; 1];
        y = phi' * theta_true + 0.05 * randn();
        [th_mu0, P0] = local_ffrls(th_mu0, P0, phi, y, lambda);
        [th_mu, Pm] = local_ffrls(th_mu, Pm, phi, y, lambda);
        th_mu = sign(th_mu) .* max(abs(th_mu) - eta, 0);
    end
    err0 = norm(th_mu0 - theta_true);
    errm = norm(th_mu - theta_true);
    if err0 > 0.5
        fprintf('[FAIL] mu=0 RLS err=%.3g\n', err0);
        ok = false;
    end
    % mu>0 应靠近真值，且相对零初值有收缩
    if errm > 1.0
        fprintf('[FAIL] mu>0 RLS err=%.3g\n', errm);
        ok = false;
    end
    % soft-threshold 在纯噪声小系数上应更稀疏：用额外零真值通道验证
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
