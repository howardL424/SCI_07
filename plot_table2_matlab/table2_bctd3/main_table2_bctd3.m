clc; clear;
%% BC-TD3 制导律轨迹求解 — 论文 Table 2 初值（场景3）
% 流程对齐 GAIL 训练环境:
%   stage1: PN 打目标（warm-start，dt=1e-3）至 r_safe
%   stage2: BC-TD3 每 0.1s 输出权重 → 式47 制导（stage2_step）
%   stage3: PN 恢复打目标（stage3_rollout）
%
% 模型: rl_td3/checkpoints/training_300k_20260730_1917/best/best_model
% 观测: obs_scale_0714_v1；动作: ratio_log_box_v1
%
% 依赖: MATLAB Python 接口指向 rl_pytorch venv（见 td3_load_policy）。
% 若 py 不可用，设 UsePythonDriver=true，改走同目录 run_table2_bctd3_traj.py。
% 本目录: GAIL/plot_table2_matlab/table2_bctd3/

%% 0. 路径与开关
this_dir = fileparts(mfilename('fullpath'));
plot_root = fileparts(this_dir);           % .../plot_table2_matlab
gail_root = fileparts(plot_root);          % .../GAIL
addpath(this_dir);
addpath(gail_root);
addpath(fullfile(gail_root, 'rl_td3', 'matlab'));
addpath(fullfile(gail_root, 'ident', 'matlab'));
addpath(fullfile(gail_root, 'expert_trajectory'));

UsePythonDriver = false;   % true: 整轨交给 Python+MATLAB Engine，本脚本只读结果绘图
py_exe = 'E:\Python\venvs\rl_pytorch\Scripts\python.exe';
model_path = fullfile(gail_root, 'rl_td3', 'checkpoints', ...
    'training_300k_20260730_1917', 'best', 'best_model');
out_mat = fullfile(this_dir, 'record_table2_bctd3.mat');

if UsePythonDriver
    cmd = sprintf('"%s" "%s" --model "%s" --out "%s"', ...
        py_exe, ...
        fullfile(this_dir, 'run_table2_bctd3_traj.py'), ...
        model_path, out_mat);
    fprintf('调用 Python 驱动:\n%s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('Python 驱动失败，exit=%d', status);
    end
    S = load(out_mat);
    record = S.record;
    plot_table2_bctd3(record);
    return;
end

%% 1. Table 2 初值（与 main_changjing3_0421_LM.m 一致）
% 状态: [x, y, z, θ, ψ_v, V]；y 轴指天
M0  = [10000, 6000, 3500, deg2rad(-20), deg2rad(155), 500];
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
cfg_s2.coeff_mode = 'analytic_truth';

%% 3. 加载 BC-TD3
fprintf('加载策略: %s\n', model_path);
bridge = td3_load_policy(model_path, py_exe, 'cpu');

%% 4. stage1 warm-start（固定权重，仅用于找规避起点与 Tev）
fprintf('stage1 warm-start …\n');
tau = simulate_warmstart_trajectory(M0, D10, D20, cfg_ws);
idx2 = find(tau.stage == 2);
if isempty(idx2)
    error('warm-start 未进入规避段(stage==2)，请检查初值/r_safe');
end
i0 = idx2(1);
i_e = idx2(end);
t_pen0 = tau.t(i0);
Tev = max(tau.t(i_e) - tau.t(i0), 1.0);
M  = tau.M(:, i0)';
D1 = tau.D1(:, i0)';
D2 = tau.D2(:, i0)';
state_abs = [M(:); D1(:); D2(:)];
fprintf('突防起点: t=%.3fs, Tev=%.3fs, r_MD1=%.1f, r_MD2=%.1f\n', ...
    t_pen0, Tev, tau.r_MD1(i0), tau.r_MD2(i0));

%% 5. 预分配记录（stage1 来自 warm-start，stage2/3 后续追加）
% 降采样 stage1 到 ~0.1s，便于与 stage2 拼接画图
ds = max(1, round(0.1 / cfg_ws.dt));
idx1 = 1:ds:i0;
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
record.action_norm = nan(3, numel(idx1));
record.gamma   = nan(1, numel(idx1));
record.wT      = nan(1, numel(idx1));
record.wE      = nan(1, numel(idx1));

%% 6. stage2：BC-TD3 闭环
core = stage2_core_init(cfg_s2);
t = 0.0;
terminated = false;
intercepted = false;
stage2_done = false;
n_s2 = 0;
r_min1 = inf; r_min2 = inf;

fprintf('stage2 BC-TD3 闭环 …\n');
tic;
while ~terminated
    obs_phys = stage2_obs_from_abs(core, state_abs, t, Tev);
    obs_n = normalize_obs_scale(obs_phys);
    action = td3_predict(bridge, obs_n);

    [~, state_abs, t_new, fact] = stage2_step(core, state_abs, t, Tev, action);
    t = t_new;
    n_s2 = n_s2 + 1;

    [gamma, wT, wE] = action_norm_to_phys(action, cfg_s2.action_mapping_version, cfg_s2.w);

    t_abs = t_pen0 + t;
    record.t      = [record.t, t_abs];
    record.pos_M  = [record.pos_M,  state_abs(1:3)];
    record.pos_D1 = [record.pos_D1, state_abs(7:9)];
    record.pos_D2 = [record.pos_D2, state_abs(13:15)];
    record.a_M    = [record.a_M,  fact.a_M_sat(:)];
    record.a_D1   = [record.a_D1, nan(2,1)];   % stage2 内未单独缓存防方加速度
    record.a_D2   = [record.a_D2, nan(2,1)];
    record.r_MT   = [record.r_MT,  fact.r_MT];
    record.r_MD1  = [record.r_MD1, fact.r_MD1];
    record.r_MD2  = [record.r_MD2, fact.r_MD2];
    record.stage  = [record.stage, 2];
    record.action_norm = [record.action_norm, action(:)];
    record.gamma  = [record.gamma, gamma];
    record.wT     = [record.wT, wT];
    record.wE     = [record.wE, wE];

    r_min1 = min(r_min1, fact.r_MD1);
    r_min2 = min(r_min2, fact.r_MD2);
    intercepted = logical(fact.intercepted);
    stage2_done = logical(fact.stage2_done);
    terminated  = logical(fact.terminated);

    if mod(n_s2, 20) == 0
        fprintf('  step=%3d t=%.2fs r_MD1=%.1f r_MD2=%.1f γ=%.2f\n', ...
            n_s2, t, fact.r_MD1, fact.r_MD2, gamma);
    end
end
fprintf('stage2 结束: steps=%d, wall=%.2fs, intercepted=%d, stage2_done=%d\n', ...
    n_s2, toc, intercepted, stage2_done);
fprintf('  r_min1=%.4f m, r_min2=%.4f m\n', r_min1, r_min2);

%% 7. stage3：PN 打目标
r_minT = nan;
if stage2_done && ~intercepted
    fprintf('stage3 PN rollout …\n');
    cfg_s3 = struct('dt', 1e-3, 'traj_sample_dt', 0.1, 'T', T_pos);
    [r_minT, reached, diverged, n3, traj3] = stage3_rollout(state_abs, t, cfg_s3);
    fprintf('stage3: r_minT=%.4f m, reached=%d, diverged=%d, steps=%d\n', ...
        r_minT, reached, diverged, n3);

    % 跳过与 stage2 末点重合的首采样点
    if ~isempty(traj3.t)
        t3_abs = t_pen0 + traj3.t;
        i_start = 1;
        if abs(t3_abs(1) - record.t(end)) < 1e-9
            i_start = 2;
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
        end
    end
else
    fprintf('跳过 stage3（拦截或未正常结束规避段）\n');
end

%% 8. 规避段能耗（与 main_changjing3_0421_LM.m 一致）
% Energy = 0.5 * ∫ ||a_M||² dt / 1000  [kJ]，积分区间为 stage2（突防段）
idx_evade = find(record.stage == 2);
if numel(idx_evade) >= 2
    a_cmd = sqrt(record.a_M(1, idx_evade).^2 + record.a_M(2, idx_evade).^2);
    Energy = 0.5 * trapz(record.t(idx_evade), a_cmd.^2) / 1000;
    t_pen_end = record.t(idx_evade(end));
elseif ~isempty(idx_evade)
    Energy = 0;
    t_pen_end = record.t(idx_evade(end));
else
    Energy = 0;
    t_pen_end = t_pen0;
end
record.Energy = Energy;
record.t_pen_end = t_pen_end;

%% 9. 汇总
record.M0 = M0; record.D10 = D10; record.D20 = D20; record.T = T_pos;
record.t_pen0 = t_pen0; record.Tev = Tev;
record.r_min1 = r_min1; record.r_min2 = r_min2; record.r_minT = r_minT;
record.intercepted = intercepted; record.stage2_done = stage2_done;
record.model_path = model_path;
record.n_stage2 = n_s2;

evade_ok = (r_min1 >= cfg_s2.kill_radius) && (r_min2 >= cfg_s2.kill_radius);
strike_ok = isfinite(r_minT) && (r_minT <= 10);
fprintf('\n==== Table2 + BC-TD3 结果 ====\n');
fprintf('规避成功(双防≥6m): %d  |  打击成功(r_minT≤10m): %d\n', evade_ok, strike_ok);
fprintf('r_min1=%.4f  r_min2=%.4f  r_minT=%.4f\n', r_min1, r_min2, r_minT);
fprintf('规避段机动能量消耗 Energy = %.4f kJ  (t∈[%.3f, %.3f] s)\n', ...
    Energy, t_pen0, record.t_pen_end);

save(out_mat, 'record');
fprintf('已保存: %s\n', out_mat);

plot_table2_bctd3(record);


%% -------- 本地绘图函数 --------
function plot_table2_bctd3(record)
    T = record.T;
    M0 = record.M0; D10 = record.D10; D20 = record.D20;

    figure('Name', '3-D Evasion Trajectory (BC-TD3)');
    plot3(record.pos_M(1,:), record.pos_M(3,:), record.pos_M(2,:), 'k-', 'LineWidth', 1.2); hold on;
    plot3(record.pos_D1(1,:), record.pos_D1(3,:), record.pos_D1(2,:), 'r--', 'LineWidth', 1.2);
    plot3(record.pos_D2(1,:), record.pos_D2(3,:), record.pos_D2(2,:), 'b-.', 'LineWidth', 1.2);
    plot3(T(1), T(3), T(2), 'k^', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    plot3(M0(1), M0(3), M0(2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    plot3(D10(1), D10(3), D10(2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    plot3(D20(1), D20(3), D20(2), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    % 标出突防起点
    if isfield(record, 't_pen0')
        [~, ip] = min(abs(record.t - record.t_pen0));
        plot3(record.pos_M(1,ip), record.pos_M(3,ip), record.pos_M(2,ip), ...
            'ks', 'MarkerSize', 9, 'MarkerFaceColor', 'y');
    end
    xlabel('x/m'); ylabel('z/m'); zlabel('y/m');
    legend('Evader A', 'D1', 'D2', 'T', 'A0', 'D10', 'D20', 'pen start');
    title('3-D Evasion Trajectory (BC-TD3)');
    grid on; view(3);
    set(gca, 'SortMethod', 'childorder');

    figure('Name', 'Accelerations of Evader A (BC-TD3)');
    plot(record.t, record.a_M(1,:), 'b-', 'LineWidth', 1.2); hold on;
    plot(record.t, record.a_M(2,:), 'r--', 'LineWidth', 1.2);
    if isfield(record, 't_pen0')
        xline(record.t_pen0, 'g--', 'LineWidth', 1);
    end
    xlabel('t/s'); ylabel('Acceleration/(m/s^2)');
    legend('a_{yA}', 'a_{zA}', 'pen start');
    title('Accelerations of Evader A (BC-TD3)'); grid on;
    set(gca, 'SortMethod', 'childorder');

    figure('Name', 'Relative Distances (BC-TD3)');
    plot(record.t, record.r_MD1, 'r-', 'LineWidth', 1.2); hold on;
    plot(record.t, record.r_MD2, 'b-', 'LineWidth', 1.2);
    plot(record.t, record.r_MT, 'k-', 'LineWidth', 1.2);
    yline(6, 'm--', 'LineWidth', 1);
    if isfield(record, 't_pen0')
        xline(record.t_pen0, 'g--', 'LineWidth', 1);
    end
    xlabel('t/s'); ylabel('Distance/m');
    legend('r_{MD1}', 'r_{MD2}', 'r_{MT}', 'kill=6m', 'pen start');
    title('Relative Distances (BC-TD3)'); grid on;

    if isfield(record, 'gamma') && any(isfinite(record.gamma))
        figure('Name', 'BC-TD3 Guidance Weights');
        yyaxis left;
        plot(record.t, record.gamma, 'k-', 'LineWidth', 1.2); hold on;
        ylabel('\gamma');
        yyaxis right;
        semilogy(record.t, record.wT, 'b-', 'LineWidth', 1.0); hold on;
        semilogy(record.t, record.wE, 'r--', 'LineWidth', 1.0);
        ylabel('w_T / w_E');
        xlabel('t/s');
        legend('\gamma', 'w_T', 'w_E');
        title('BC-TD3 Output Weights (stage2)'); grid on;
    end
end
