function record = simulate_table2_noisy_episode(opts)
%SIMULATE_TABLE2_NOISY_EPISODE  Table-2 初值下带测量噪声的单局轨迹求解。
%
%   record = simulate_table2_noisy_episode(opts)
%
%   三段结构与 table2_td3_pure/main_table2_td3_pure.m、
%   table2_bctd3/main_table2_bctd3.m 完全一致：
%     stage1  PN warm-start（dt=1e-3）至 r_safe=8000，无噪声
%     stage2  策略每 0.1 s 输出权重 → 式(47) 制导，测量噪声在此段注入
%     stage3  PN 恢复打目标，无噪声
%
%   噪声注入点与参数辨识（Plan 1）一致：真值几何先加噪，再走
%   stage2_ident_step（Kalman + sparse FF-RLS），由 coeff_mode 决定制导系数来源：
%     analytic_noisy  带噪测量直接代入解析 computeBC
%     sparse_ffrls    Kalman + L1 FF-RLS 在线辨识系数
%     analytic_truth  解析真值（无噪对照）
%   防御弹 PN 始终用真值几何（见 stage2_step 注释）。
%
%   opts 字段:
%     .model_path    策略路径 .../best/best_model（可省略 .zip）
%     .prefer_td3bc  true=BC-TD3；false=纯 TD3
%     .algo          记录用算法名，如 'TD3_pure'
%     .coeff_mode    制导系数来源，默认 'analytic_noisy'
%     .ident         noisy_ident_cfg 返回的噪声/KF/RLS 配置
%     .obs_noise     策略 13 维观测是否同样加噪，默认 false
%                    （Plan 1 约定策略看干净观测，噪声只走系数路径）
%     .py_exe        python 解释器路径
%     .verbose       是否打印进度，默认 true
%
%   See also NOISY_IDENT_CFG, MAIN_TABLE2_NOISY_IDENT, STAGE2_STEP.

%% 0. 参数与路径
if nargin < 1 || isempty(opts)
    opts = struct();
end
opts = local_default(opts, 'prefer_td3bc', true);
opts = local_default(opts, 'algo', 'policy');
opts = local_default(opts, 'coeff_mode', 'analytic_noisy');
opts = local_default(opts, 'ident', noisy_ident_cfg());
opts = local_default(opts, 'obs_noise', false);
opts = local_default(opts, 'py_exe', 'E:\Python\venvs\rl_pytorch\Scripts\python.exe');
opts = local_default(opts, 'verbose', true);
if ~isfield(opts, 'model_path') || isempty(opts.model_path)
    error('simulate_table2_noisy_episode:NoModel', '必须提供 opts.model_path');
end

this_dir  = fileparts(mfilename('fullpath'));
plot_root = fileparts(this_dir);            % .../plot_table2_matlab
gail_root = fileparts(plot_root);           % .../GAIL
addpath(this_dir);
addpath(fullfile(plot_root, 'table2_bctd3'));   % td3_load_policy / predict / obs 归一化
addpath(gail_root);
addpath(fullfile(gail_root, 'rl_td3', 'matlab'));
addpath(fullfile(gail_root, 'ident', 'matlab'));
addpath(fullfile(gail_root, 'expert_trajectory'));

%% 1. Table 2 初值（与 main_changjing3_0421_LM.m 一致）
% 状态: [x, y, z, θ, ψ_v, V]；y 轴指天
M0  = [10000, 6000, 3500, deg2rad(-20), deg2rad(155), 500];  % Evader A
D10 = [1000,  0,    0,    deg2rad(40),  deg2rad(-20),  600];
D20 = [-1000, 0,    0,    deg2rad(40),  deg2rad(-20),  600];
T_pos = [0, 0, 0];

%% 2. 配置
cfg_ws = struct();
cfg_ws.KM = 3; cfg_ws.KD1 = 4; cfg_ws.KD2 = 4;
cfg_ws.dt = 1e-3; cfg_ws.t_max = 60; cfg_ws.r_safe = 8000;
cfg_ws.g0 = 9.81; cfg_ws.T = T_pos;
cfg_ws.w = 1e4; cfg_ws.gamma = 0; cfg_ws.wE = 1.44e3; cfg_ws.wT = 1e3;
cfg_ws.dstar = 20;

cfg_s2 = struct();
cfg_s2.dt = 0.1;
cfg_s2.dstar = 20.0;
cfg_s2.KD1 = 4; cfg_s2.KD2 = 4; cfg_s2.KM = 3;
cfg_s2.g0 = 9.81;
cfg_s2.kill_radius = 6.0;
cfg_s2.w = 1e4;
cfg_s2.action_mapping_version = 'ratio_log_box_v1';
cfg_s2.coeff_mode = char(opts.coeff_mode);
cfg_s2.ident_apply_cmd = true;      % 非影子模式：辨识结果真的进指令

ncfg = opts.ident;
fn = setdiff(fieldnames(ncfg), {'preset', 'sigma_scale'});
for k = 1:numel(fn)
    cfg_s2.(fn{k}) = ncfg.(fn{k});
end

%% 3. 加载策略
if opts.verbose
    fprintf('加载策略 (%s): %s\n', opts.algo, opts.model_path);
end
bridge = td3_load_policy(opts.model_path, opts.py_exe, 'cpu', opts.prefer_td3bc);

%% 4. stage1 warm-start（无噪，与原脚本逐点一致）
if opts.verbose
    fprintf('stage1 warm-start …\n');
end
tau = simulate_warmstart_trajectory(M0, D10, D20, cfg_ws);
idx2 = find(tau.stage == 2);
if isempty(idx2)
    error('simulate_table2_noisy_episode:NoEvade', ...
        'warm-start 未进入规避段(stage==2)，请检查初值/r_safe');
end
i0 = idx2(1);
i_e = idx2(end);
t_pen0 = tau.t(i0);
Tev = max(tau.t(i_e) - tau.t(i0), 1.0);
state_abs = [tau.M(:, i0); tau.D1(:, i0); tau.D2(:, i0)];
if opts.verbose
    fprintf('突防起点: t=%.3fs, Tev=%.3fs, r_MD1=%.1f, r_MD2=%.1f\n', ...
        t_pen0, Tev, tau.r_MD1(i0), tau.r_MD2(i0));
end

%% 5. 记录初始化（stage1 降采样到 ~0.1 s）
ds = max(1, round(0.1 / cfg_ws.dt));
idx1 = 1:ds:i0;
n1 = numel(idx1);
record = struct();
record.t       = tau.t(idx1);
record.pos_M   = tau.M(1:3, idx1);
record.pos_D1  = tau.D1(1:3, idx1);
record.pos_D2  = tau.D2(1:3, idx1);
record.a_M     = tau.a_M(:, idx1);
record.a_D1    = tau.a_D1(:, idx1);
record.a_D2    = tau.a_D2(:, idx1);
record.r_MT    = tau.r_MT(idx1);
record.r_MD1   = tau.r_MD1(idx1);
record.r_MD2   = tau.r_MD2(idx1);
record.stage   = tau.stage(idx1);
record.action_norm = nan(3, n1);
record.gamma   = nan(1, n1);
record.wT      = nan(1, n1);
record.wE      = nan(1, n1);
record.theta_rel_err = nan(3, n1);   % D1/D2/T 制导系数相对误差
record.r_est_err     = nan(3, n1);   % D1/D2/T 距离量测/估计误差 (m)

%% 6. stage2：带测量噪声的策略闭环
core = stage2_core_init(cfg_s2);
ident_state = [];
if opts.obs_noise
    obs_rng = RandStream('mt19937ar', 'Seed', 1e4 + ncfg.noise_seed);
else
    obs_rng = [];
end

t = 0.0;
terminated = false;
intercepted = false;
stage2_done = false;
n_s2 = 0;
r_min1 = inf; r_min2 = inf;

if opts.verbose
    fprintf('stage2 闭环（coeff_mode=%s, obs_noise=%d）…\n', ...
        cfg_s2.coeff_mode, opts.obs_noise);
end
tic;
while ~terminated
    obs_phys = stage2_obs_from_abs(core, state_abs, t, Tev);
    if opts.obs_noise
        obs_phys = local_add_obs_noise(obs_phys, ncfg, obs_rng);
    end
    action = td3_predict(bridge, normalize_obs_scale(obs_phys));

    [~, state_abs, t_new, fact, ident_state] = ...
        stage2_step(core, state_abs, t, Tev, action, ident_state);
    t = t_new;
    n_s2 = n_s2 + 1;

    [gamma, wT, wE] = action_norm_to_phys(action, ...
        cfg_s2.action_mapping_version, cfg_s2.w);
    [th_err, r_err] = local_ident_errors(fact.ident_diag, cfg_s2.coeff_mode);

    record.t      = [record.t, t_pen0 + t];
    record.pos_M  = [record.pos_M,  state_abs(1:3)];
    record.pos_D1 = [record.pos_D1, state_abs(7:9)];
    record.pos_D2 = [record.pos_D2, state_abs(13:15)];
    record.a_M    = [record.a_M,  fact.a_M_sat(:)];
    record.a_D1   = [record.a_D1, nan(2, 1)];
    record.a_D2   = [record.a_D2, nan(2, 1)];
    record.r_MT   = [record.r_MT,  fact.r_MT];
    record.r_MD1  = [record.r_MD1, fact.r_MD1];
    record.r_MD2  = [record.r_MD2, fact.r_MD2];
    record.stage  = [record.stage, 2];
    record.action_norm = [record.action_norm, action(:)];
    record.gamma  = [record.gamma, gamma];
    record.wT     = [record.wT, wT];
    record.wE     = [record.wE, wE];
    record.theta_rel_err = [record.theta_rel_err, th_err];
    record.r_est_err     = [record.r_est_err, r_err];

    r_min1 = min(r_min1, fact.r_MD1);
    r_min2 = min(r_min2, fact.r_MD2);
    intercepted = logical(fact.intercepted);
    stage2_done = logical(fact.stage2_done);
    terminated  = logical(fact.terminated);

    if opts.verbose && mod(n_s2, 20) == 0
        fprintf('  step=%3d t=%.2fs r_MD1=%.1f r_MD2=%.1f γ=%.2f\n', ...
            n_s2, t, fact.r_MD1, fact.r_MD2, gamma);
    end
end
if opts.verbose
    fprintf('stage2 结束: steps=%d, wall=%.2fs, intercepted=%d, stage2_done=%d\n', ...
        n_s2, toc, intercepted, stage2_done);
    fprintf('  r_min1=%.4f m, r_min2=%.4f m\n', r_min1, r_min2);
end

%% 7. stage3：PN 打目标（无噪）
r_minT = nan;
if stage2_done && ~intercepted
    cfg_s3 = struct('dt', 1e-3, 'traj_sample_dt', 0.1, 'T', T_pos);
    [r_minT, reached, diverged, n3, traj3] = stage3_rollout(state_abs, t, cfg_s3);
    if opts.verbose
        fprintf('stage3: r_minT=%.4f m, reached=%d, diverged=%d, steps=%d\n', ...
            r_minT, reached, diverged, n3);
    end

    if ~isempty(traj3.t)
        t3_abs = t_pen0 + traj3.t;
        i_start = 1;
        if abs(t3_abs(1) - record.t(end)) < 1e-9
            i_start = 2;      % 跳过与 stage2 末点重合的首采样点
        end
        if i_start <= numel(traj3.t)
            sel = i_start:numel(traj3.t);
            n3s = numel(sel);
            record.t      = [record.t, t3_abs(sel)];
            record.pos_M  = [record.pos_M,  traj3.pos_M(:, sel)];
            record.pos_D1 = [record.pos_D1, traj3.pos_D1(:, sel)];
            record.pos_D2 = [record.pos_D2, traj3.pos_D2(:, sel)];
            record.a_M    = [record.a_M,  traj3.a_M(:, sel)];
            record.a_D1   = [record.a_D1, nan(2, n3s)];
            record.a_D2   = [record.a_D2, nan(2, n3s)];
            record.r_MT   = [record.r_MT,  traj3.r_MT(sel)];
            record.r_MD1  = [record.r_MD1, traj3.r_MD1(sel)];
            record.r_MD2  = [record.r_MD2, traj3.r_MD2(sel)];
            record.stage  = [record.stage, 3 * ones(1, n3s)];
            record.action_norm = [record.action_norm, nan(3, n3s)];
            record.gamma  = [record.gamma, nan(1, n3s)];
            record.wT     = [record.wT, nan(1, n3s)];
            record.wE     = [record.wE, nan(1, n3s)];
            record.theta_rel_err = [record.theta_rel_err, nan(3, n3s)];
            record.r_est_err     = [record.r_est_err, nan(3, n3s)];
        end
    end
elseif opts.verbose
    fprintf('跳过 stage3（拦截或未正常结束规避段）\n');
end

%% 8. 规避段能耗 Energy = 0.5*∫||a_M||²dt/1000 [kJ]
idx_evade = find(record.stage == 2);
if numel(idx_evade) >= 2
    a_cmd = hypot(record.a_M(1, idx_evade), record.a_M(2, idx_evade));
    Energy = 0.5 * trapz(record.t(idx_evade), a_cmd.^2) / 1000;
    t_pen_end = record.t(idx_evade(end));
else
    Energy = 0;
    t_pen_end = t_pen0;
end

%% 9. 汇总
record.M0 = M0; record.D10 = D10; record.D20 = D20; record.T = T_pos;
record.t_pen0 = t_pen0; record.t_pen_end = t_pen_end; record.Tev = Tev;
record.r_min1 = r_min1; record.r_min2 = r_min2; record.r_minT = r_minT;
record.Energy = Energy;
record.intercepted = intercepted; record.stage2_done = stage2_done;
record.n_stage2 = n_s2;
record.algo = char(opts.algo);
record.model_path = char(opts.model_path);
record.coeff_mode = cfg_s2.coeff_mode;
record.obs_noise = logical(opts.obs_noise);
record.ident_cfg = ncfg;
record.evade_ok = (r_min1 >= cfg_s2.kill_radius) && (r_min2 >= cfg_s2.kill_radius);
record.strike_ok = isfinite(r_minT) && (r_minT <= 10);

if opts.verbose
    fprintf('==== %s / %s ====\n', record.algo, record.coeff_mode);
    fprintf('规避成功(双防≥6m): %d  |  打击成功(r_minT≤10m): %d\n', ...
        record.evade_ok, record.strike_ok);
    fprintf('r_min1=%.4f  r_min2=%.4f  r_minT=%.4f  Energy=%.4f kJ\n\n', ...
        r_min1, r_min2, r_minT, Energy);
end
end


%% -------- 局部函数 --------
function s = local_default(s, name, value)
if ~isfield(s, name) || isempty(s.(name))
    s.(name) = value;
end
end


function obs = local_add_obs_noise(obs, ncfg, rs)
%LOCAL_ADD_OBS_NOISE  按同一组 σ 给 13 维观测加噪（速度与 tgo 不加噪）。
sd = [ncfg.sigma_r; ncfg.sigma_rdot; ncfg.sigma_q; ncfg.sigma_q;
      ncfg.sigma_r; ncfg.sigma_rdot; ncfg.sigma_q; ncfg.sigma_q;
      ncfg.sigma_r; ncfg.sigma_q;    ncfg.sigma_q; 0;  0];
obs = obs(:) + sd .* randn(rs, 13, 1);
end


function [th_err, r_err] = local_ident_errors(id_diag, coeff_mode)
%LOCAL_IDENT_ERRORS  本拍实际使用的系数相对真值的误差，以及距离通道误差。
th_err = nan(3, 1);
r_err  = nan(3, 1);
if ~isstruct(id_diag) || ~isfield(id_diag, 'theta_true')
    return;
end

if strcmp(coeff_mode, 'sparse_ffrls')
    th_used = id_diag.theta_hat;
    r_used  = id_diag.r_hat;
else
    th_used = id_diag.theta_noisy;
    r_used  = id_diag.meas_r;
end

for i = 1:3
    if isfield(id_diag, 'theta_true_current')
        % 本拍实际用于制导的估计应与下一周期 vartheta[k] 比较。
        tt = id_diag.theta_true_current{i}(:);
    else
        tt = id_diag.theta_true{i}(:);
    end
    tu = th_used{i}(:);
    nrm = norm(tt);
    if nrm > 0
        th_err(i) = norm(tu - tt) / nrm;
    end
end
r_err = r_used(:) - id_diag.r_true(:);
end
