%% 纯 PN / 正弦程序机动 — 300 次蒙特卡洛评估
%  初态池：samples_clean_test.mat（与 TD3 evaluate_td3 同一 test-IC，300 条有效行）
%  制导：  对齐 plot_table2_matlab/main_chunPN_0412_duo.m
%          与 plot_table2_matlab/main_zhengxian_0412_duo.m
%  成功判据（与 rl_td3 一致）：
%    双规避成功  evade_ok : r_MD1_min≥6m 且 r_MD2_min≥6m 且未被拦截
%    打击成功    strike_ok: r_minT≤10m 且未被拦截
%    联合成功    both_ok  : evade_ok && strike_ok
%
%  用法：
%    main_mc_pn_sine_300                 % 默认 300 局，两种方法都跑（无噪）
%    main_mc_pn_sine_300('N', 5)         % 烟雾：只跑前 5 局
%    main_mc_pn_sine_300('methods', {'pn'})  % 只跑纯 PN
%    main_mc_pn_sine_300('enable_noise', true, 'sigma_scale', 3)  % 带测量噪声

function main_mc_pn_sine_300(varargin)
    this_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(fileparts(this_dir));  % GAIL/
    addpath(this_dir);
    addpath(fullfile(project_root, 'expert_trajectory'));

    p = inputParser;
    addParameter(p, 'N', 300, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'methods', {'pn', 'sine'}, @(c) iscell(c) || ischar(c) || isstring(c));
    addParameter(p, 'samples', fullfile(project_root, 'samples_clean_test.mat'), @ischar);
    addParameter(p, 'seed', 0, @isnumeric);
    addParameter(p, 'kill_radius', 6.0, @isnumeric);
    addParameter(p, 'strike_ok', 10.0, @isnumeric);
    addParameter(p, 'progress_every', 10, @isnumeric);
    addParameter(p, 'enable_noise', false, @(x) islogical(x) || isnumeric(x));
    % addParameter(p, 'sigma_scale', 1.0, @isnumeric);   % 原默认：1×
    addParameter(p, 'sigma_scale', 30.0, @isnumeric);     % 2026-08-06：改为 30×
    addParameter(p, 'noise_seed', 0, @isnumeric);
    addParameter(p, 'noise_hold_dt', 0.1, @isnumeric);
    addParameter(p, 'out_dir', '', @ischar);
    parse(p, varargin{:});
    opt = p.Results;

    if ischar(opt.methods) || isstring(opt.methods)
        methods = {char(opt.methods)};
    else
        methods = opt.methods;
    end

    rng(opt.seed);

    S = load(opt.samples, 'samples_clean');
    sc = S.samples_clean;
    valid = any(sc ~= 0, 2);
    sc = sc(valid, :);
    n_pool = size(sc, 1);
    N = min(opt.N, n_pool);
    fprintf('初态池: %s\n', opt.samples);
    fprintf('有效行 %d / 取前 %d 条（与 TD3 全量 test-IC 顺序一致）\n', n_pool, N);
    fprintf('判据: evade r_MD≥%.0fm, strike r_minT≤%.0fm\n', ...
        opt.kill_radius, opt.strike_ok);

    sigma_q = 0.1 * pi / 180 * opt.sigma_scale;
    noise_on = logical(opt.enable_noise);
    if noise_on
        fprintf(['测量噪声: ON  scale=×%g  σ_q=%.4g rad (%.3f°)  ' ...
            'σ_qdot=%.4g  σ_r=%.4g  hold=%.2fs\n\n'], ...
            opt.sigma_scale, sigma_q, rad2deg(sigma_q), ...
            0.001 * opt.sigma_scale, 0.0, opt.noise_hold_dt);
    else
        fprintf('测量噪声: OFF\n\n');
    end

    cfg = struct( ...
        'kill_radius', opt.kill_radius, ...
        'strike_ok', opt.strike_ok, ...
        'enable_noise', noise_on, ...
        'sigma_q', sigma_q, ...
        'sigma_qdot', 0.001 * opt.sigma_scale, ...
        'sigma_theta', sigma_q, ...
        'sigma_psi', sigma_q, ...
        'sigma_r', 0.0, ...
        'sigma_rdot', 0.0, ...
        'noise_hold_dt', opt.noise_hold_dt);

    if isempty(opt.out_dir)
        out_dir = fullfile(this_dir, 'results');
    else
        out_dir = opt.out_dir;
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    ts = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>

    summary_rows = {};
    for im = 1:numel(methods)
        mode = lower(char(methods{im}));
        fprintf('========== 开始方法: %s (%d 局) ==========\n', upper(mode), N);
        t0 = tic;

        r_MD1 = nan(N, 1);
        r_MD2 = nan(N, 1);
        r_minT = nan(N, 1);
        evade_ok = false(N, 1);
        strike_ok = false(N, 1);
        both_ok = false(N, 1);
        intercepted = false(N, 1);
        row_idx = zeros(N, 1);
        energy_kJ = nan(N, 1);
        energy_from_pen_kJ = nan(N, 1);

        for i = 1:N
            row = sc(i, :);
            [M0, D10, D20] = reconstruct_geometry(row);
            cfg_i = cfg;
            % 每局独立噪声序列：与 TD3 analytic_noisy MC 对齐
            cfg_i.noise_seed = double(opt.noise_seed) + (i - 1);
            ep = simulate_baseline_episode(M0, D10, D20, mode, cfg_i);

            r_MD1(i) = ep.r_MD1_min;
            r_MD2(i) = ep.r_MD2_min;
            r_minT(i) = ep.r_minT;
            evade_ok(i) = ep.evade_ok;
            strike_ok(i) = ep.strike_ok;
            both_ok(i) = ep.both_ok;
            intercepted(i) = ep.intercepted;
            row_idx(i) = i;
            energy_kJ(i) = ep.energy_kJ;
            energy_from_pen_kJ(i) = ep.energy_from_pen_kJ;

            if mod(i, opt.progress_every) == 0 || i == N
                fprintf('  [%3d/%3d] evade=%.1f%%  both=%.1f%%  | 本局 rMD=[%.1f,%.1f] rT=%.2f  E=%.3fkJ%s\n', ...
                    i, N, ...
                    100 * mean(evade_ok(1:i)), ...
                    100 * mean(both_ok(1:i)), ...
                    ep.r_MD1_min, ep.r_MD2_min, ep.r_minT, ep.energy_kJ, ...
                    tern(ep.intercepted, ' [拦截]', ''));
            end
        end

        elapsed = toc(t0);
        stats = print_and_pack_stats(mode, N, evade_ok, strike_ok, both_ok, ...
            intercepted, r_MD1, r_MD2, r_minT, energy_kJ, energy_from_pen_kJ, ...
            opt.kill_radius, opt.strike_ok, elapsed);

        result = struct( ...
            'mode', mode, ...
            'N', N, ...
            'samples_file', opt.samples, ...
            'kill_radius', opt.kill_radius, ...
            'strike_ok_th', opt.strike_ok, ...
            'enable_noise', noise_on, ...
            'sigma_scale', opt.sigma_scale, ...
            'noise_seed0', opt.noise_seed, ...
            'r_MD1_min', r_MD1, ...
            'r_MD2_min', r_MD2, ...
            'r_minT', r_minT, ...
            'energy_kJ', energy_kJ, ...
            'energy_from_pen_kJ', energy_from_pen_kJ, ...
            'evade_ok', evade_ok, ...
            'strike_ok', strike_ok, ...
            'both_ok', both_ok, ...
            'intercepted', intercepted, ...
            'row_idx', row_idx, ...
            'stats', stats, ...
            'elapsed_s', elapsed);

        tag = '';
        if noise_on
            tag = sprintf('_noisy_x%g', opt.sigma_scale);
        end
        mat_path = fullfile(out_dir, sprintf('mc_%s_N%d%s_%s.mat', mode, N, tag, ts));
        save(mat_path, 'result');
        fprintf('已保存: %s\n\n', mat_path);

        summary_rows{end+1} = stats; %#ok<AGROW>
    end

    % 汇总表
    fprintf('==================== 汇总 ====================\n');
    fprintf('%-8s  %6s  %12s  %12s  %12s  %10s  %14s\n', ...
        '方法', 'N', '双规避成功', '打击成功', '联合成功', '被拦截', '全过程能耗kJ');
    for k = 1:numel(summary_rows)
        s = summary_rows{k};
        fprintf('%-8s  %6d  %5d/%-4d=%4.1f%%  %5d/%-4d=%4.1f%%  %5d/%-4d=%4.1f%%  %5d/%-4d  med=%.4f\n', ...
            upper(s.mode), s.N, ...
            s.n_evade, s.N, 100*s.rate_evade, ...
            s.n_strike, s.N, 100*s.rate_strike, ...
            s.n_both, s.N, 100*s.rate_both, ...
            s.n_intercepted, s.N, s.energy_kJ_median);
    end
    fprintf('==============================================\n');
end

function s = print_and_pack_stats(mode, N, evade_ok, strike_ok, both_ok, ...
        intercepted, r_MD1, r_MD2, r_minT, energy_kJ, energy_from_pen_kJ, ...
        kill_r, strike_th, elapsed)
    n_evade = sum(evade_ok);
    n_strike = sum(strike_ok);
    n_both = sum(both_ok);
    n_int = sum(intercepted);
    finite_T = isfinite(r_minT);
    finite_E = isfinite(energy_kJ);

    fprintf('\n------------------------------------------------------------\n');
    fprintf('  评估结果：%s  （%d 轮，耗时 %.1f min）\n', upper(mode), N, elapsed/60);
    fprintf('------------------------------------------------------------\n');
    fprintf('  双防均规避成功  (d1,d2≥%.0fm)     : %3d/%d = %.1f%%\n', ...
        kill_r, n_evade, N, 100*n_evade/N);
    fprintf('  打击成功        (r_minT≤%.0fm)   : %3d/%d = %.1f%%\n', ...
        strike_th, n_strike, N, 100*n_strike/N);
    fprintf('  规避+打击 全部达标                : %3d/%d = %.1f%%\n', ...
        n_both, N, 100*n_both/N);
    fprintf('  被拦截          (r_MD<%.0fm)      : %3d/%d = %.1f%%\n', ...
        kill_r, n_int, N, 100*n_int/N);
    fprintf('  最小脱靶距离 D1  mean±std (m)     : %.1f ± %.1f\n', mean(r_MD1), std(r_MD1));
    fprintf('  最小脱靶距离 D2  mean±std (m)     : %.1f ± %.1f\n', mean(r_MD2), std(r_MD2));
    if any(finite_T)
        fprintf('  终端脱靶 r_minT  mean±std (m)     : %.2f ± %.2f  (n=%d)\n', ...
            mean(r_minT(finite_T)), std(r_minT(finite_T)), nnz(finite_T));
    else
        fprintf('  终端脱靶 r_minT  : (无有效样本)\n');
    end
    if any(finite_E)
        e = energy_kJ(finite_E);
        q = quantile(e, [0.25, 0.5, 0.75]);
        fprintf('  全过程能耗 Energy mean±std (kJ)  : %.4f ± %.4f  median=%.4f  IQR=[%.4f,%.4f]\n', ...
            mean(e), std(e), q(2), q(1), q(3));
        e2 = energy_from_pen_kJ(isfinite(energy_from_pen_kJ));
        if ~isempty(e2)
            fprintf('    规避触发后（含末端） median     : %.4f kJ\n', median(e2));
        end
        if n_both > 0
            e_ok = energy_kJ(both_ok & isfinite(energy_kJ));
            if ~isempty(e_ok)
                fprintf('    联合成功子集 median            : %.4f kJ  (n=%d)\n', ...
                    median(e_ok), numel(e_ok));
            end
        end
    else
        fprintf('  全过程能耗 Energy : (无有效样本)\n');
    end
    fprintf('------------------------------------------------------------\n');

    if any(finite_E)
        e = energy_kJ(finite_E);
        q = quantile(e, [0.25, 0.5, 0.75]);
        e_mean = mean(e); e_std = std(e); e_med = q(2);
    else
        e_mean = NaN; e_std = NaN; e_med = NaN;
    end

    s = struct( ...
        'mode', mode, ...
        'N', N, ...
        'n_evade', n_evade, ...
        'n_strike', n_strike, ...
        'n_both', n_both, ...
        'n_intercepted', n_int, ...
        'rate_evade', n_evade/N, ...
        'rate_strike', n_strike/N, ...
        'rate_both', n_both/N, ...
        'rate_intercepted', n_int/N, ...
        'energy_kJ_mean', e_mean, ...
        'energy_kJ_std', e_std, ...
        'energy_kJ_median', e_med);
end

function s = tern(cond, a, b)
    if cond, s = a; else, s = b; end
end
