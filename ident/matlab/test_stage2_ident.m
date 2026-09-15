function ok = test_stage2_ident()
%TEST_STAGE2_IDENT  参数辨识阶段2完整回归测试。
%
%   ok = test_stage2_ident()
%
%   覆盖量测协方差、KF、因果回归、ZOH真值、拟冻结残差、
%   三种时间轴、协方差迹投影、固定参数收敛及85维诊断协议。

    this_dir = fileparts(mfilename('fullpath'));
    gail_root = fullfile(this_dir, '..', '..');
    addpath(this_dir);
    addpath(fullfile(gail_root, 'rl_td3', 'matlab'));
    addpath(gail_root);
    addpath(fullfile(gail_root, 'expert_trajectory'));

    names = { ...
        'analytic_truth 核心回归', ...
        '配置与量测协方差', ...
        'KF 灵敏度与解析初值', ...
        '软阈值与 A21 投影', ...
        '因果回归使用上一拍 KF 距离', ...
        '逐拍 ZOH 真值与拟冻结残差', ...
        '三种系数路径的 tgo 时间轴', ...
        'RLS 协方差迹投影', ...
        '固定参数带噪持续激励收敛', ...
        'episode 重置', ...
        '85维 MATLAB 诊断协议', ...
        '投影系数与 coeffs 入口回归'};
    tests = { ...
        @test_stage2_core_equiv, ...
        @local_test_config_and_Rxi, ...
        @local_test_kf_and_initialization, ...
        @local_test_threshold_and_projection, ...
        @local_test_causal_rhat, ...
        @local_test_zoh_and_frozen_residual, ...
        @local_test_tgo_paths, ...
        @local_test_covariance_cap, ...
        @local_test_fixed_parameter_convergence, ...
        @local_test_reset, ...
        @local_test_info_protocol, ...
        @local_test_coeff_regressions};

    ok = true;
    for k = 1:numel(tests)
        fprintf('--- [%d/%d] %s ---\n', k, numel(tests), names{k});
        try
            passed = logical(tests{k}());
        catch ME
            passed = false;
            fprintf('[FAIL] %s: %s\n', ME.identifier, ME.message);
            for j = 1:min(numel(ME.stack), 3)
                fprintf('       %s:%d\n', ME.stack(j).name, ME.stack(j).line);
            end
        end
        if passed
            fprintf('[OK] %s\n', names{k});
        else
            ok = false;
        end
    end

    if ok
        fprintf('\n[PASS] test_stage2_ident 全部通过\n');
    else
        fprintf('\n[FAIL] test_stage2_ident 存在失败项\n');
    end
end

function ok = local_test_config_and_Rxi()
    ok = true;
    core = stage2_core_init(struct());
    ident = stage2_ident_init(core.cfg);
    ok = local_check(core.cfg.sigma_r == 10.0, ...
        '默认 sigma_r 应为 10') && ok;
    ok = local_check(core.cfg.sigma_rdot == 0.5, ...
        '默认 sigma_rdot 应为 0.5') && ok;
    ok = local_check(max(abs(diag(ident.R_xi) - [100; 0.25])) < 1e-14, ...
        'R_xi 未由实际量测噪声直接构成') && ok;
    ok = local_check(~isfield(core.cfg, 'kf_sigma_r') && ...
        ~isfield(core.cfg, 'kf_sigma_rdot'), ...
        '配置仍生成已删除的 kf_sigma_r/kf_sigma_rdot') && ok;

    bad = {-1, 0};
    for i = 1:numel(bad)
        ok = local_expect_error( ...
            @() stage2_ident_init(struct('sigma_r', bad{i}, 'sigma_rdot', 0.5)), ...
            'stage2_ident_init:BadMeasurementNoise') && ok;
        ok = local_expect_error( ...
            @() stage2_ident_init(struct('sigma_r', 10, 'sigma_rdot', bad{i})), ...
            'stage2_ident_init:BadMeasurementNoise') && ok;
    end
end

function ok = local_test_kf_and_initialization()
    ok = true;
    H = evasion_helpers();
    cfg1 = local_cfg();
    cfg1.kf_qj = 1e2;
    cfg1.noise_seed = 13;
    cfg2 = cfg1;
    cfg2.kf_qj = 1e5;
    i1 = stage2_ident_init(cfg1);
    i2 = stage2_ident_init(cfg2);

    for k = 0:12
        t = k * i1.dt;
        rd = -200 - 15*t + (8/2.5) * (1 - cos(2.5*t));
        r = 5000 - 200*t - 7.5*t^2 + ...
            (8/2.5)*t - (8/2.5^2)*sin(2.5*t);
        truth = local_truth(local_geo(r, rd, 0.1, -0.05, 0.01, -0.02));
        [i1, ~, d1] = stage2_ident_step(i1, truth, [0;0], [0;0], H);
        [i2, ~, d2] = stage2_ident_step(i2, truth, [0;0], [0;0], H);
        i1 = stage2_ident_commit(i1, [0;0]);
        i2 = stage2_ident_commit(i2, [0;0]);
    end
    ok = local_check(abs(d1.rddot_hat(1) - d2.rddot_hat(1)) > 1e-3, ...
        '改变 kf_qj 未改变 rddot_hat，kf_qj 疑似死旋钮') && ok;

    cfg0 = local_cfg();
    cfg0.noise_seed = 21;
    ia = stage2_ident_init(cfg0);
    ib = stage2_ident_init(cfg0);
    g = local_geo(4200, -180, 0.07, -0.04, 0.012, -0.018);
    truth = local_truth(g);
    [~, ~, da] = stage2_ident_step(ia, truth, [1;2], [3;4], H);
    [~, ~, db] = stage2_ident_step(ib, truth, [70;-50], [-60;40], H);

    for c = 1:3
        th0 = da.theta_noisy{c};
        expected = da.meas_r(c) * th0(1);
        if numel(th0) == 4
            expected = expected + th0(4);
        end
        ok = local_check(abs(da.rddot_hat(c) - expected) < 1e-10, ...
            sprintf('通道%d KF 第三状态未按 noisy analytic rddot0 初始化', c)) && ok;
    end
    warm_diff = max(abs([ ...
        da.theta_noisy{1}(4) - db.theta_noisy{1}(4); ...
        da.theta_noisy{2}(4) - db.theta_noisy{2}(4)]));
    ok = local_check(warm_diff < 1e-12, ...
        'NCV C1 warm-start 读取了传入的真值 a_D') && ok;
end

function ok = local_test_threshold_and_projection()
    ok = true;
    theta = [-1; 1; -2; 0.25];
    [out, P] = stage2_ffrls_update( ...
        theta, zeros(4), zeros(4,1), 0, 0.98, 0.5, Inf);
    expected = [0; 0.5; -1.5; 0];
    ok = local_check(max(abs(out - expected)) < 1e-14, ...
        '软阈值应只收缩 B1/B2/C1，负 A21 应投影到0') && ok;
    ok = local_check(norm(P, 'fro') == 0, '零协方差不应凭空变化') && ok;

    [out2, ~] = stage2_ffrls_update( ...
        [0.25; 1; -2], zeros(3), zeros(3,1), 0, 0.98, 0.5, Inf);
    ok = local_check(abs(out2(1) - 0.25) < 1e-14, ...
        '正 A21 被软阈值错误收缩') && ok;
end

function ok = local_test_causal_rhat()
    ok = true;
    H = evasion_helpers();
    cfg = local_cfg();
    cfg.sigma_r = 50;
    cfg.sigma_rdot = 2;
    cfg.noise_seed = 31;
    cfg.rls_eta = 0;
    cfg.rls_P_trace_max = Inf;
    ident = stage2_ident_init(cfg);

    truth1 = local_truth(local_geo(4300, -170, 0.08, 0.02, 0.012, -0.011));
    [ident, ~, d1] = stage2_ident_step(ident, truth1, [0;0], [0;0], H);
    ok = local_check(~d1.did_rls, '首拍不应执行 RLS') && ok;
    ident = stage2_ident_commit(ident, [12;-7]);

    truth2 = local_truth(local_geo(4283, -171, 0.081, 0.019, 0.013, -0.010));
    [ident, ~, d2] = stage2_ident_step(ident, truth2, [0;0], [0;0], H);
    ident = stage2_ident_commit(ident, [-9;15]);
    ok = local_check(max(abs(d2.r_hat - d2.meas_r)) > 1e-6, ...
        '测试构造失败：第二拍 r_hat 与原始量测未拉开') && ok;

    before = ident;
    truth3 = local_truth(local_geo(4266, -172, 0.082, 0.018, 0.011, -0.012));
    [after, ~, d3] = stage2_ident_step(ident, truth3, [0;0], [0;0], H);
    alt_gap = zeros(3,1);
    for c = 1:3
        n = before.ch(c).n_theta;
        S = local_scale(cfg, n);
        if n == 4
            phi = [before.ch(c).r_hat_prev; before.a_M_sat_prev(:); 1];
            phi_alt = [d2.meas_r(c); before.a_M_sat_prev(:); 1];
        else
            phi = [before.ch(c).r_hat_prev; before.a_M_sat_prev(:)];
            phi_alt = [d2.meas_r(c); before.a_M_sat_prev(:)];
        end
        [expected, ~] = stage2_ffrls_update( ...
            before.ch(c).theta_sc, before.ch(c).P, S \ phi, ...
            d3.rddot_hat(c), cfg.rls_lambda, cfg.rls_eta, ...
            cfg.rls_P_trace_max);
        [alternate, ~] = stage2_ffrls_update( ...
            before.ch(c).theta_sc, before.ch(c).P, S \ phi_alt, ...
            d3.rddot_hat(c), cfg.rls_lambda, cfg.rls_eta, ...
            cfg.rls_P_trace_max);
        ok = local_check(max(abs(after.ch(c).theta_sc - expected)) < 1e-9, ...
            sprintf('通道%d RLS 未使用上一拍 r_hat', c)) && ok;
        alt_gap(c) = max(abs(after.ch(c).theta_sc - alternate));
    end
    ok = local_check(max(alt_gap) > 1e-8, ...
        'r_hat/r_meas 反事实更新无法区分，因果距离测试无效') && ok;
end

function ok = local_test_zoh_and_frozen_residual()
    ok = true;
    H = evasion_helpers();
    cfg = local_cfg();
    cfg.noise_seed = 41;
    cfg.rls_eta = 0;
    ident = stage2_ident_init(cfg);

    g1 = local_geo(4000, -150, 0.05, 0.02, 0.010, 0.008);
    truth1 = local_truth(g1);
    [ident, ~, d1] = stage2_ident_step(ident, truth1, [5;-3], [6;2], H);
    for c = 1:3
        ok = local_check(all(isnan(d1.theta_true_causal{c})), ...
            sprintf('通道%d 首拍 causal 真值应为 NaN', c)) && ok;
        ok = local_check(all(isnan(d1.theta_true{c})), ...
            sprintf('通道%d 兼容 theta_true 应映射 causal 真值', c)) && ok;
    end

    a_prev = [12.5; -7.5];
    ident = stage2_ident_commit(ident, a_prev);
    g2 = local_geo(3984.8, -153, 0.055, 0.017, 0.014, 0.006);
    truth2 = local_truth(g2);
    truth2.D2.qy = -0.04;
    truth2.D2.qdz = -0.009;
    truth2.T.qz = -0.03;
    [~, ~, d2] = stage2_ident_step(ident, truth2, [4;-2], [7;1], H);

    for c = 1:3
        ok = local_check(max(abs( ...
            d2.theta_true_causal{c} - d1.theta_true_current{c})) < 1e-12, ...
            sprintf('通道%d vartheta[k-1] 缓存 off-by-one', c)) && ok;
    end
    changed = max(abs(local_flat_theta(d2.theta_true_current) - ...
        local_flat_theta(d2.theta_true_causal)));
    ok = local_check(changed > 1e-8, ...
        '测试构造失败：vartheta[k] 与 vartheta[k-1] 未拉开') && ok;

    for c = 1:3
        th_prev = d1.theta_true_current{c};
        if numel(th_prev) == 4
            phi_prev = [d1.r_true(c); a_prev(:); 1];
        else
            phi_prev = [d1.r_true(c); a_prev(:)];
        end
        rdd_avg = (d2.rdot_true(c) - d1.rdot_true(c)) / cfg.dt;
        expected = rdd_avg - phi_prev' * th_prev;
        ok = local_check(abs(d2.frozen_model_residual(c) - expected) < 1e-10, ...
            sprintf('通道%d frozen residual 未使用 k-1 数据', c)) && ok;
    end
end

function ok = local_test_tgo_paths()
    ok = true;
    H = evasion_helpers();
    state0 = local_state();
    action = [0.2; -0.4; 0.3];
    Tev = 5;
    base = local_cfg();
    base.sigma_r = 300;
    base.sigma_rdot = 20;
    base.noise_seed = 51;
    base.action_mapping_version = 'ratio_log_box_v1';
    [r1, r2, rT] = local_decode_ratios(action);

    cfg_n = base;
    cfg_n.coeff_mode = 'analytic_noisy';
    core_n = stage2_core_init(cfg_n);
    [~, ~, ~, fact_n] = stage2_step(core_n, state0, 0, Tev, action, []);
    d_n = fact_n.ident_diag;
    c_n = local_coeff_pack(d_n.theta_noisy, d_n.meas_r, d_n.meas_rdot);
    expected_n = local_coeff_cmd(H, c_n, d_n.meas_r, d_n.meas_rdot, ...
        0, r1, r2, rT, base.dstar);
    wrong_n = local_coeff_cmd(H, c_n, d_n.r_true, d_n.rdot_true, ...
        0, r1, r2, rT, base.dstar);
    ok = local_check(max(abs(fact_n.a_M - expected_n)) < 1e-8, ...
        'analytic_noisy 未使用 noisy r/rdot 时间轴') && ok;
    ok = local_check(max(abs(fact_n.a_M - wrong_n)) > 1e-5, ...
        'analytic_noisy 时间轴反事实不可区分，测试无效') && ok;

    cfg_t = base;
    cfg_t.coeff_mode = 'analytic_truth';
    core_t = stage2_core_init(cfg_t);
    [~, ~, ~, fact_t] = stage2_step(core_t, state0, 0, Tev, action);
    c_t = local_coeff_pack( ...
        d_n.theta_true_current, d_n.r_true, d_n.rdot_true);
    expected_t = local_coeff_cmd(H, c_t, d_n.r_true, d_n.rdot_true, ...
        0, r1, r2, rT, base.dstar);
    ok = local_check(max(abs(fact_t.a_M - expected_t)) < 1e-8, ...
        'analytic_truth 未使用真值 r/rdot 时间轴') && ok;
    ok = local_check(max(abs(d_n.a_cmd_pred - fact_n.a_M)) < 1e-12 && ...
        max(abs(d_n.a_cmd_truth_ref - fact_t.a_M)) < 1e-8, ...
        'analytic_noisy 指令诊断未对齐 analytic_truth 参考') && ok;
    ok = local_check(max(abs(d_n.a_cmd_err - ...
        (d_n.a_cmd_pred-d_n.a_cmd_truth_ref))) < 1e-12, ...
        'analytic_noisy 指令误差定义不一致') && ok;

    cfg_s = base;
    cfg_s.coeff_mode = 'sparse_ffrls';
    cfg_s.ident_apply_cmd = true;
    core_s = stage2_core_init(cfg_s);
    [~, state_s1, t_s1, ~, ident_s] = ...
        stage2_step(core_s, state0, 0, Tev, action, []);
    [~, ~, ~, fact_s] = ...
        stage2_step(core_s, state_s1, t_s1, Tev, action, ident_s);
    d_s = fact_s.ident_diag;
    c_s = local_coeff_pack(d_s.theta_hat, d_s.r_hat, d_s.rdot_hat);
    expected_s = local_coeff_cmd(H, c_s, d_s.r_hat, d_s.rdot_hat, ...
        t_s1, r1, r2, rT, base.dstar);
    wrong_s = local_coeff_cmd(H, c_s, d_s.r_true, d_s.rdot_true, ...
        t_s1, r1, r2, rT, base.dstar);
    ok = local_check(max(abs(fact_s.a_M - expected_s)) < 1e-8, ...
        'sparse_ffrls 未使用 KF 后验 r_hat/rdot_hat 时间轴') && ok;
    ok = local_check(max(abs(fact_s.a_M - wrong_s)) > 1e-5, ...
        'sparse_ffrls 时间轴反事实不可区分，测试无效') && ok;
    ok = local_check(max(abs(d_s.a_cmd_err - ...
        (d_s.a_cmd_pred-d_s.a_cmd_truth_ref))) < 1e-12, ...
        'sparse_ffrls 指令误差未相对 analytic_truth 参考计算') && ok;

    cfg_h = cfg_s;
    cfg_h.ident_apply_cmd = false;
    core_h = stage2_core_init(cfg_h);
    [~, state_h1, t_h1, ~, ident_h] = ...
        stage2_step(core_h, state0, 0, Tev, action, []);
    [~, ~, ~, fact_h] = ...
        stage2_step(core_h, state_h1, t_h1, Tev, action, ident_h);
    d_h = fact_h.ident_diag;
    c_h = local_coeff_pack(d_h.theta_hat, d_h.r_hat, d_h.rdot_hat);
    expected_h = local_coeff_cmd(H, c_h, d_h.r_hat, d_h.rdot_hat, ...
        t_h1, r1, r2, rT, base.dstar);
    wrong_h = local_coeff_cmd(H, c_h, d_h.r_true, d_h.rdot_true, ...
        t_h1, r1, r2, rT, base.dstar);
    ok = local_check(max(abs(d_h.a_cmd_pred - expected_h)) < 1e-8, ...
        '影子预测指令未使用辨识路径自己的时间轴') && ok;
    ok = local_check(max(abs(d_h.a_cmd_pred - wrong_h)) > 1e-5, ...
        '影子时间轴反事实不可区分，测试无效') && ok;

end

function ok = local_test_covariance_cap()
    ok = true;
    cap = 0.7;
    cfg = local_cfg();
    cfg.rls_P_trace_max = cap;
    ident = stage2_ident_init(cfg);
    for c = 1:3
        ok = local_check(trace(ident.ch(c).P) <= cap + 1e-12, ...
            sprintf('通道%d 初始化 P 超出迹上限', c)) && ok;
    end

    rng(61);
    theta = zeros(4,1);
    P = 1e4 * eye(4);
    for k = 1:500
        phi = randn(4,1);
        y = randn();
        [theta, P] = stage2_ffrls_update( ...
            theta, P, phi, y, 0.98, 0, cap);
        ev = eig(0.5 * (P + P'));
        ok = local_check(trace(P) <= cap + 1e-10, ...
            sprintf('第%d次更新后 P 超出迹上限', k)) && ok;
        ok = local_check(norm(P-P', 'fro') < 1e-12 && min(ev) >= -1e-10, ...
            sprintf('第%d次更新后 P 未保持对称半正定', k)) && ok;
        if ~ok
            return;
        end
    end

    theta0 = [0.2; -0.4; 0.7];
    P0 = [3, 0.2, 0.1; 0.2, 2, 0.3; 0.1, 0.3, 1];
    phi0 = [0.5; -1; 2];
    y0 = 1.3;
    [actual_t, actual_P] = stage2_ffrls_update( ...
        theta0, P0, phi0, y0, 0.97, 0, Inf);
    [standard_t, standard_P] = local_standard_rls( ...
        theta0, P0, phi0, y0, 0.97);
    standard_t(1) = max(standard_t(1), 0);
    ok = local_check(max(abs(actual_t-standard_t)) < 1e-12 && ...
        max(abs(actual_P(:)-standard_P(:))) < 1e-12, ...
        'P_trace_max=Inf 未恢复标准 FF-RLS') && ok;
end

function ok = local_test_fixed_parameter_convergence()
    ok = true;
    theta_set = { ...
        [1e-4; 0.8; -0.5; 6.0], ...
        [1.5e-4; 0.6; 0.3]};
    Sset = {diag([1e4,50,50,1]), diag([1e4,50,50])};
    caps = [Inf, 1.0];
    rng(71);

    for c = 1:2
        theta_true = theta_set{c};
        S = Sset{c};
        n = numel(theta_true);
        phi_all = zeros(n, 2000);
        y_all = zeros(1, 2000);
        for k = 1:2000
            phi = [4000 + randn()*1500; randn()*30; randn()*30; 1];
            phi = phi(1:n);
            phi_all(:,k) = phi;
            y_all(k) = phi' * theta_true + 0.05 * randn();
        end
        for j = 1:numel(caps)
            cap = caps(j);
            theta_sc = zeros(n,1);
            if isinf(cap)
                P = 1e4 * eye(n);
            else
                P = (cap/n) * eye(n);
            end
            for k = 1:size(phi_all,2)
                [theta_sc, P] = stage2_ffrls_update( ...
                    theta_sc, P, S \ phi_all(:,k), y_all(k), ...
                    0.98, 0, cap);
            end
            theta_hat = S \ theta_sc;
            scale = [5e-5; 0.05; 0.05; 0.5];
            err_norm = norm((theta_hat - theta_true) ./ scale(1:n));
            ok = local_check(err_norm < 1.0, sprintf( ...
                '固定参数收敛失败: n=%d, Pcap=%g, normalized error=%.3g', ...
                n, cap, err_norm)) && ok;
        end
    end
    % eta>0 的近端估计存在稳态偏差；精确收敛测试用 eta=0，
    % 稀疏收缩的索引与幅值由 local_test_threshold_and_projection 单测。
end

function ok = local_test_reset()
    ok = true;
    cfg = local_cfg();
    cfg.coeff_mode = 'sparse_ffrls';
    cfg.noise_seed = 81;
    core = stage2_core_init(cfg);
    ident = [];
    [~, state1, t1, ~, ident] = ...
        stage2_step(core, local_state(), 0, 5, [0;0;0], ident);
    [~, ~, ~, ~, ident] = ...
        stage2_step(core, state1, t1, 5, [0;0;0], ident);
    ok = local_check(ident.step_count >= 2 && ident.has_prev, ...
        '测试 episode 未进入已提交状态') && ok;

    ident2 = stage2_ident_init(core.cfg);
    ok = local_check(ident2.step_count == 0 && ~ident2.has_prev, ...
        '新 episode 重置后 step_count/has_prev 非初值') && ok;
    ok = local_check(all(~[ident2.ch.initialized]), ...
        '新 episode 重置后 KF 通道仍为 initialized') && ok;
end

function ok = local_test_info_protocol()
    ok = true;
    cfg = local_cfg();
    cfg.coeff_mode = 'sparse_ffrls';
    cfg.ident_apply_cmd = false;
    cfg.noise_seed = 91;
    state0 = local_state();
    action = [0.1; -0.2; 0.3];
    clear env_step
    [~, state1, ~, ~, info1] = ...
        env_step(state0, 0, 5, 0, action, cfg);
    ok = local_check(numel(info1) == 85, ...
        sprintf('首拍 info_vec 长度应为85，实际%d', numel(info1))) && ok;
    ok = local_check(all(isnan(info1(19:29))) && ...
        all(isnan(info1(58:68))), ...
        '首拍 causal 参数误差/真值应为 NaN') && ok;
    ok = local_check(all(isfinite(info1(47:57))) && ...
        all(isfinite(info1(69:79))), ...
        '首拍 theta_hat/theta_true_current 应为有限值') && ok;

    [~, ~, ~, ~, info2] = ...
        env_step(state1, 0.1, 5, 0, action, cfg);
    ok = local_check(numel(info2) == 85, ...
        '第二拍 info_vec 长度应保持85') && ok;
    ok = local_check(max(abs(info2(58:68) - info1(69:79))) < 1e-12, ...
        'MATLAB 协议 causal 真值未映射上一拍 current 真值') && ok;
    ok = local_check(max(abs(info2(19:29) - ...
        (info2(47:57)-info2(58:68)))) < 1e-12, ...
        'MATLAB 协议 theta_err_causal 切片不一致') && ok;
end

function ok = local_test_coeff_regressions()
    ok = true;
    H = evasion_helpers();
    qy = 0.12;
    qz = -0.08;
    qdy = 0.02;
    qdz = -0.015;
    r = 3500;
    rdot = -180;
    t = 1.0;
    tf = 12.0;
    thM = 0.05;
    psM = -0.03;
    thD = 0.1;
    psD = 0.02;
    aD = [10; -8];
    Q = sqrt(qdy^2 + qdz^2 * cos(qy)^2);
    tgo = tf - t;
    [B1, B2, C1] = H.computeBC( ...
        qy, qz, thM, psM, thD, psD, aD(1), aD(2));
    a1 = H.computeOptimalCmd_duo_target( ...
        qy, qz, qdy, qdz, r, rdot, t, tf, ...
        qy, qz, qdy, qdz, r, rdot, tf, ...
        qy, qz, qdy, qdz, r, rdot, tf, ...
        thM, psM, thD, psD, thD, psD, ...
        aD, aD, 0.5, 0.5, 2.0, 1.0, 20.0);
    a2 = H.computeOptimalCmd_duo_target_coeffs( ...
        Q, [B1;B2], -C1, r, rdot, tgo, tf, ...
        Q, [B1;B2], -C1, r, rdot, tgo, tf, ...
        Q, [B1;B2], 0, r, rdot, tgo, tf, ...
        t, 0.5, 0.5, 2.0, 1.0, 20.0);
    ok = local_check(max(abs(a1-a2)) < 1e-6, ...
        'analytic_truth 与 coeffs 数值入口不等价') && ok;

    radial = cos(thM)*cos(qy)*cos(qz-psM) + sin(thM)*sin(qy);
    ok = local_check(abs(B1^2+B2^2-(1-radial^2)) < 1e-12, ...
        '式(13)/(15) B系数投影恒等式失败') && ok;
end

function cfg = local_cfg()
    cfg = struct( ...
        'dt', 0.1, 'g0', 9.81, ...
        'sigma_r', 10.0, 'sigma_rdot', 0.5, ...
        'sigma_q', 0, 'sigma_qdot', 0, ...
        'sigma_theta', 0, 'sigma_psi', 0, ...
        'kf_qj', 1e4, 'kf_sigma_a0', 50, ...
        'rls_lambda', 0.98, 'rls_mu', 1e-3, 'rls_eta', 0.1, ...
        'rls_P0', 1e4, 'rls_P_trace_max', 1.0, ...
        'rls_r0', 1e4, 'rls_a0', 50, ...
        'KD1', 4, 'KD2', 4, 'noise_seed', 0, ...
        'dstar', 20, 'w', 1e4);
end

function g = local_geo(r, rdot, qy, qz, qdy, qdz)
    g = struct('r', r, 'rdot', rdot, 'qy', qy, 'qz', qz, ...
        'qdy', qdy, 'qdz', qdz, 'V', 600);
end

function truth = local_truth(g)
    truth = struct('D1', g, 'D2', g, 'T', g, ...
        'theta_M', 0.1, 'psi_M', -0.1, ...
        'theta_D1', 0.02, 'psi_D1', 0.03, ...
        'theta_D2', -0.01, 'psi_D2', -0.02);
end

function state = local_state()
    M = [8000, 0, 2000, 0.08, -0.05, 800];
    D1 = [6000, 500, 1500, 0.02, 0.03, 700];
    D2 = [6100, -600, 1400, -0.01, -0.02, 700];
    state = [M(:); D1(:); D2(:)];
end

function S = local_scale(cfg, n)
    if n == 4
        S = diag([cfg.rls_r0, cfg.rls_a0, cfg.rls_a0, 1]);
    else
        S = diag([cfg.rls_r0, cfg.rls_a0, cfg.rls_a0]);
    end
end

function flat = local_flat_theta(theta)
    flat = [];
    for i = 1:numel(theta)
        flat = [flat; theta{i}(:)]; %#ok<AGROW>
    end
end

function c = local_coeff_pack(theta, r, rdot)
    c = struct();
    c.Q1 = sqrt(max(theta{1}(1),0));
    c.b1 = theta{1}(2:3);
    c.vr1 = -theta{1}(4);
    c.r1 = r(1);
    c.rdot1 = rdot(1);
    c.Q2 = sqrt(max(theta{2}(1),0));
    c.b2 = theta{2}(2:3);
    c.vr2 = -theta{2}(4);
    c.r2 = r(2);
    c.rdot2 = rdot(2);
    c.QT = sqrt(max(theta{3}(1),0));
    c.bT = theta{3}(2:3);
    c.vrT = 0;
    c.rT = r(3);
    c.rdotT = rdot(3);
end

function a = local_coeff_cmd(H, c, axis_r, axis_rdot, ...
        t, r1, r2, rT, dstar)
    [tf1, tgo1] = local_intercept_time(t, axis_r(1), axis_rdot(1));
    [tf2, tgo2] = local_intercept_time(t, axis_r(2), axis_rdot(2));
    [tfT, tgoT] = local_intercept_time(t, axis_r(3), axis_rdot(3));
    Q1 = local_clamp_Q(c.Q1, tgo1);
    Q2 = local_clamp_Q(c.Q2, tgo2);
    QT = local_clamp_Q(c.QT, tgoT);
    a = H.computeOptimalCmd_duo_target_coeffs( ...
        Q1, c.b1, c.vr1, c.r1, c.rdot1, tgo1, tf1, ...
        Q2, c.b2, c.vr2, c.r2, c.rdot2, tgo2, tf2, ...
        QT, c.bT, c.vrT, c.rT, c.rdotT, tgoT, tfT, ...
        t, r1, r2, rT, 1.0, dstar);
    a = a(:);
end

function [r1, r2, rT] = local_decode_ratios(action)
    rhoE = 10^(action(1) + 1);
    rhoT = 10^(action(2) * 2.5 + 0.5);
    gamma = action(3) * 3;
    eg = exp(-gamma);
    r1 = rhoE / (1 + eg);
    r2 = rhoE * eg / (1 + eg);
    rT = rhoT;
end

function [tf, tgo] = local_intercept_time(t, r, rdot)
    rd = sign(rdot) * max(abs(rdot), 1e-3);
    if rd == 0
        rd = 1e-3;
    end
    tf = t - r / rd;
    tgo = tf - t;
end

function Q = local_clamp_Q(Q, tgo)
    Q = max(double(Q), 0);
    if abs(tgo) > 1e-12 && Q * tgo > 100
        Q = 100 / tgo;
    end
end

function [theta, P] = local_standard_rls(theta, P, phi, y, lambda)
    K = (P * phi) / (lambda + phi' * P * phi);
    theta = theta + K * (y - phi' * theta);
    P = (1/lambda) * (P - K * phi' * P);
    P = 0.5 * (P + P');
end

function ok = local_check(condition, message)
    ok = logical(condition);
    if ~ok
        fprintf('[FAIL] %s\n', message);
    end
end

function ok = local_expect_error(fun, identifier)
    ok = false;
    try
        fun();
    catch ME
        ok = strcmp(ME.identifier, identifier);
    end
    if ~ok
        fprintf('[FAIL] 未触发预期错误 %s\n', identifier);
    end
end
