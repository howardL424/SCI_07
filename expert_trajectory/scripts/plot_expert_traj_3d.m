%% 绘制 expert_lib_1000.mat 中三条专家规避轨迹 (各一张 3D 图)
%  数据格式与 generate_ps_expert_library 输出一致:
%    results.trajectories(k).transitions.state_abs  [N×18] = [xM(6), xD1(6), xD2(6)]
%    坐标: state = [x, y, z, theta, psi_v, V], y 轴指天, 目标 T 在原点
%
%  用法: 在 expert_trajectory/scripts/ 下运行 (ziyong/ 下保留原版副本)

clear; clc; close all;
addpath(fileparts(fileparts(mfilename('fullpath'))));  % expert_trajectory/
setup_expert_path();

%% -------- 配置 --------
dataFile = expert_lib_1000_file();
% 从可用轨迹中均匀抽取 3 条; 也可改为固定下标, 如 pickK = [10, 200, 500];
pickK = [];   % 留空则自动均匀抽样（0639注：现在均匀抽则三个轨迹全部被拦截）

%% -------- 加载 --------
if ~isfile(dataFile)
    error('plot_expert_traj_3d:FileNotFound', '未找到文件: %s', dataFile);
end
d = load(dataFile, 'results');
r = d.results;

validIdx = find(arrayfun(@(tr) tr.ok_use && ~isempty(tr.transitions), r.trajectories));
if numel(validIdx) < 3
    error('plot_expert_traj_3d:InsufficientData', '可用轨迹不足 3 条 (当前 %d 条)', numel(validIdx));
end

if isempty(pickK)
    pickK = validIdx(round(linspace(1, numel(validIdx), 3)));
else
    pickK = pickK(:)';
    if any(~ismember(pickK, validIdx))
        bad = pickK(~ismember(pickK, validIdx));
        error('plot_expert_traj_3d:InvalidPick', '指定下标不可用: %s', mat2str(bad));
    end
end

fprintf('从 %d 条可用轨迹中绘制 k = [%s]\n', numel(validIdx), num2str(pickK));

%% -------- 逐条绘制 (每条独立 figure) --------
for ii = 1:numel(pickK)
    k = pickK(ii);
    tr = r.trajectories(k);
    plot_one_expert_traj_3d(tr, k, ii);
end

%% ===== 局部函数 =====
function plot_one_expert_traj_3d(tr, k, figIdx)
    ex = tr.transitions;
    sa = ex.state_abs;          % N×18
    st = ex.state;              % N×13 相对态势
    t  = ex.t;

    % 绝对位置 [x,y,z] -> plot3 用 (x, z, y), 与 generate_ps_expert_library.plot_trajectory 一致
    xM  = sa(:, 1);  yM  = sa(:, 2);  zM  = sa(:, 3);
    xD1 = sa(:, 7);  yD1 = sa(:, 8);  zD1 = sa(:, 9);
    xD2 = sa(:,13);  yD2 = sa(:,14);  zD2 = sa(:,15);

    geo = tr.geo;
    okMode = 'unknown';
    if ~isempty(tr.stageB) && isfield(tr.stageB, 'ok_mode')
        okMode = tr.stageB.ok_mode;
    end

    figure('Name', sprintf('专家轨迹 #%d (batch k=%d)', figIdx, k), ...
           'NumberTitle', 'off', 'Color', 'w');

    % --- 左: 3D 航迹 (规避段) ---
    subplot(1, 2, 1);
    plot3(xM,  zM,  yM,  'k-',  'LineWidth', 1.4); hold on;
    plot3(xD1, zD1, yD1, 'r--', 'LineWidth', 1.2);
    plot3(xD2, zD2, yD2, 'b-.', 'LineWidth', 1.2);
    plot3(0, 0, 0, 'g^', 'MarkerSize', 9, 'MarkerFaceColor', 'g');
    plot3(xM(1),  zM(1),  yM(1),  'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
    plot3(xM(end),zM(end),yM(end),'ks', 'MarkerFaceColor', 'w', 'MarkerSize', 7);
    xlabel('x / m'); ylabel('z / m'); zlabel('y / m');
    grid on; view(3);
    legend('A (攻)', 'D1', 'D2', 'T', 'A_{start}', 'A_{end}', 'Location', 'best');
    title(sprintf(['规避段 3D 航迹 | k=%d row=%d\n', ...
                   'rAD0=%.0f m, \\beta=%.1f°, mode=%s, %d 步'], ...
                  k, tr.idx, geo.rAD0, geo.beta, okMode, numel(t)));
    set(gca, 'SortMethod', 'childorder');

    % --- 右: 相对距离 ---
    subplot(1, 2, 2);
    plot(t, st(:, 1), 'r-', 'LineWidth', 1.2); hold on;
    plot(t, st(:, 5), 'b-', 'LineWidth', 1.2);
    plot(t, st(:, 9), 'k--', 'LineWidth', 1.2);
    yline(20, 'g:', 'd^*=20 m');
    yline(8000, 'm:', 'r_{safe}');
    xlabel('t / s'); ylabel('距离 / m');
    grid on;
    legend('r_{MD1}', 'r_{MD2}', 'r_{MT}', 'Location', 'best');
    title('相对距离 (规避段)');
end
