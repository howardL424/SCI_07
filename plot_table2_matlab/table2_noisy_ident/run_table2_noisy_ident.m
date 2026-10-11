function out = run_table2_noisy_ident(opts)
%RUN_TABLE2_NOISY_IDENT  Table-2 初值下带测量噪声的 TD3 / BC-TD3 轨迹复算与出图。
%
%   out = run_table2_noisy_ident()
%   out = run_table2_noisy_ident(opts)
%
%   与 plot_table2_matlab 下原有脚本的关系：
%     - 动力学、初值、三段结构、能耗定义完全沿用
%       table2_td3_pure/main_table2_td3_pure.m 与 table2_bctd3/main_table2_bctd3.m
%     - 唯一改动是 stage2 把 coeff_mode 从 analytic_truth 换成带噪路径，
%       并跨拍持久化辨识状态（stage2_step 的第 6 输入 / 第 5 输出）
%     - 原目录文件一律不修改
%
%   opts 字段（均可省略）:
%     .preset       噪声预设，见 noisy_ident_cfg，默认 'plan1'
%     .sigma_scale  所有 σ 的统一倍率，默认 30（原默认 1 已注释保留）
%     .noise_seed   噪声种子，默认 0
%     .coeff_mode   'analytic_noisy'（默认）| 'sparse_ffrls'
%     .obs_noise    策略 13 维观测是否同样加噪，默认 false
%     .out_tag      结果/图子目录与文件名后缀；默认按 sigma_scale 自动生成
%                   （scale=1 时为空，否则 'sigma_x<scale>'）
%     .py_exe       python 解释器路径
%
%   输出:
%     results/<out_tag>/record_table2_<algo>_noisy.mat
%     figures/<out_tag>/trajectory_<algo>_noisy.{png,pdf,fig}
%     figures/<out_tag>/trajectory_<algo>_truth_vs_noisy.{png,pdf,fig}
%
%   See also NOISY_IDENT_CFG, SIMULATE_TABLE2_NOISY_EPISODE, PLOT_NOISY_VS_CLEAN.

%% 0. 参数与路径
if nargin < 1 || isempty(opts)
    opts = struct();
end
opts = local_default(opts, 'preset', 'plan1');
% opts = local_default(opts, 'sigma_scale', 1);   % 原默认
opts = local_default(opts, 'sigma_scale', 30);    % 2026-08-06 定标
opts = local_default(opts, 'noise_seed', 0);
opts = local_default(opts, 'coeff_mode', 'analytic_noisy');
opts = local_default(opts, 'obs_noise', false);
opts = local_default(opts, 'py_exe', 'E:\Python\venvs\rl_pytorch\Scripts\python.exe');
if ~isfield(opts, 'out_tag')
    if abs(opts.sigma_scale - 1) < 1e-12
        opts.out_tag = '';
    else
        opts.out_tag = sprintf('sigma_x%g', opts.sigma_scale);
    end
end

this_dir  = fileparts(mfilename('fullpath'));
plot_root = fileparts(this_dir);
gail_root = fileparts(plot_root);
addpath(this_dir);
addpath(plot_root);                                  % plot_table2_trajectories
addpath(fullfile(plot_root, 'table2_bctd3'));
addpath(gail_root);
addpath(fullfile(gail_root, 'rl_td3', 'matlab'));
addpath(fullfile(gail_root, 'ident', 'matlab'));
addpath(fullfile(gail_root, 'expert_trajectory'));

results_dir = fullfile(this_dir, 'results', opts.out_tag);
figures_dir = fullfile(this_dir, 'figures', opts.out_tag);
if ~isfolder(results_dir); mkdir(results_dir); end
if ~isfolder(figures_dir); mkdir(figures_dir); end

%% 1. 噪声配置
ncfg = noisy_ident_cfg(opts.preset, opts.noise_seed, opts.sigma_scale);
fprintf(['测量噪声 (%s, ×%g): σ_q=%.4g rad (%.3f°)  σ_qdot=%.4g rad/s  ' ...
    'σ_r=%.4g m  σ_rdot=%.4g m/s  seed=%d\n'], ...
    ncfg.preset, ncfg.sigma_scale, ncfg.sigma_q, rad2deg(ncfg.sigma_q), ...
    ncfg.sigma_qdot, ncfg.sigma_r, ncfg.sigma_rdot, ncfg.noise_seed);

runs = struct( ...
    'tag',   {'td3', 'bctd3'}, ...
    'label', {'Pure TD3', 'BC-TD3'}, ...
    'algo',  {'TD3_pure', 'BC_TD3'}, ...
    'prefer_td3bc', {false, true}, ...
    'model_path', { ...
        fullfile(gail_root, 'td3_baseline', 'checkpoints', ...
            'training_300k_20260730_1917', 'best', 'best_model'), ...
        fullfile(gail_root, 'rl_td3', 'checkpoints', ...
            'training_300k_20260730_1917', 'best', 'best_model')}, ...
    'clean_file', { ...
        fullfile(plot_root, 'table2_td3_pure', 'record_table2_td3_pure.mat'), ...
        fullfile(plot_root, 'table2_bctd3', 'record_table2_bctd3.mat')});

%% 2. 带噪轨迹求解
methods_noisy = repmat(struct('tag', '', 'label', '', 'record', struct()), ...
    1, numel(runs));
for k = 1:numel(runs)
    ep = struct();
    ep.model_path   = runs(k).model_path;
    ep.prefer_td3bc = runs(k).prefer_td3bc;
    ep.algo         = runs(k).algo;
    ep.coeff_mode   = opts.coeff_mode;
    ep.ident        = ncfg;
    ep.obs_noise    = opts.obs_noise;
    ep.py_exe       = opts.py_exe;
    ep.verbose      = true;

    record = simulate_table2_noisy_episode(ep); %#ok<NASGU>
    out_mat = fullfile(results_dir, ...
        sprintf('record_table2_%s_noisy.mat', runs(k).tag));
    save(out_mat, 'record');
    fprintf('已保存: %s\n', out_mat);

    methods_noisy(k).tag = [runs(k).tag, '_noisy'];
    methods_noisy(k).label = sprintf('%s (noisy meas. x%g, %s)', ...
        runs(k).label, ncfg.sigma_scale, opts.coeff_mode);
    methods_noisy(k).record = record;
end

%% 3. 三维轨迹 + 最近接近点放大图（复用原绘图函数，不修改原文件）
pcfg = struct();
pcfg.output_dir = figures_dir;
pcfg.save_figures = true;
pcfg.close_after_save = false;
pcfg.save_fig = true;
pcfg.dpi = 600;
pcfg.view = [132, 21];
pcfg.zoom_half_time = 0.20;
plot_table2_trajectories(methods_noisy, pcfg);

%% 4. 与原真值系数轨迹对照
pairs = struct('tag', {}, 'label', {}, 'clean', {}, 'noisy', {});
for k = 1:numel(runs)
    if ~isfile(runs(k).clean_file)
        fprintf('缺少真值系数轨迹，跳过对照图: %s\n', runs(k).clean_file);
        continue;
    end
    S = load(runs(k).clean_file);
    pairs(end + 1) = struct('tag', runs(k).tag, ...
        'label', sprintf('%s (noise x%g)', runs(k).label, ncfg.sigma_scale), ...
        'clean', S.record, ...
        'noisy', methods_noisy(k).record); %#ok<AGROW>
end
if ~isempty(pairs)
    plot_noisy_vs_clean(pairs, pcfg);
end

%% 5. 汇总
fprintf('\n==== Table-2 带测量噪声结果汇总 (%s ×%g / %s) ====\n', ...
    ncfg.preset, ncfg.sigma_scale, opts.coeff_mode);
fprintf('%-10s %10s %10s %10s %10s %8s %8s\n', ...
    'algo', 'r_min1', 'r_min2', 'r_minT', 'Energy', 'evade', 'strike');
for k = 1:numel(methods_noisy)
    r = methods_noisy(k).record;
    fprintf('%-10s %10.3f %10.3f %10.3f %10.3f %8d %8d\n', ...
        runs(k).tag, r.r_min1, r.r_min2, r.r_minT, r.Energy, ...
        r.evade_ok, r.strike_ok);
    idx = find(r.stage == 2);
    fprintf('%-10s 系数相对误差 (D1/D2/T) 均值: %.3g / %.3g / %.3g\n', '', ...
        mean(r.theta_rel_err(1, idx), 'omitnan'), ...
        mean(r.theta_rel_err(2, idx), 'omitnan'), ...
        mean(r.theta_rel_err(3, idx), 'omitnan'));
end
fprintf('\n图已输出至:\n  %s\n', figures_dir);

out = struct();
out.ident_cfg = ncfg;
out.methods = methods_noisy;
out.results_dir = results_dir;
out.figures_dir = figures_dir;
end


function s = local_default(s, name, value)
if ~isfield(s, name) || isempty(s.(name))
    s.(name) = value;
end
end
