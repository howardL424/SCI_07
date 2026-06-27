function results = generate_ps_expert_library(N, opts)
%GENERATE_PS_EXPERT_LIBRARY 伪谱专家轨迹批处理主入口 (由 main_jiaBP3 改造)。
%   results = generate_ps_expert_library(N, opts)
%   流程: samples_clean 初态 -> 阶段A RK4 warm-start -> (阶段B 伪谱 NLP)
%         -> 提取 (s,a) -> 汇总。当前已实现阶段A; 阶段B 由 opts.run_stageB 控制。
%
%   输入:
%     N    : 处理的样本条数 (默认 10, P0 验证规模)
%     opts : 可选, 字段:
%            .run_stageB (默认 false) 是否运行阶段B 伪谱 NLP
%            .plot       (默认 true)  是否绘制第一条轨迹
%            .save       (默认 false) 是否保存 expert_lib.mat
%            .cfg        传给 simulate_warmstart_trajectory 的配置
%   输出 results: 含 trajectories(k) 与统计信息的结构体。

    if nargin < 1 || isempty(N); N = 10; end
    if nargin < 2; opts = struct(); end
    if ~isfield(opts, 'run_stageB'); opts.run_stageB = false; end
    if ~isfield(opts, 'plot');       opts.plot = true; end
    if ~isfield(opts, 'save');       opts.save = false; end
    if ~isfield(opts, 'cfg');        opts.cfg = struct(); end

    % ---------------- 载入样本 ----------------
    S = load('samples_clean.mat', 'samples_clean');
    sc = S.samples_clean;
    valid = find(any(sc ~= 0, 2));         % 去掉补零行 (clean 后约 760 行)
    nValid = numel(valid);
    N = min(N, nValid);
    fprintf('samples_clean: %d 有效行, 本次处理前 %d 条\n', nValid, N);

    if opts.run_stageB
        add_casadi_path();
    end

    % ---------------- 主循环 ----------------
    trajectories = struct('idx', {}, 'X', {}, 'tau', {}, 'stageB', {});
    nSucc = 0; nInt = 0;
    t0 = tic;
    for k = 1:N
        row = sc(valid(k), :);
        [M0, D10, D20, geo] = reconstruct_geometry(row);
        [tau, wE_used] = solve_warmstart(M0, D10, D20, opts.cfg);

        trajectories(k).idx = valid(k);
        trajectories(k).X   = row;
        trajectories(k).geo = geo;
        trajectories(k).tau = tau;
        trajectories(k).wE_used = wE_used;
        trajectories(k).stageB = [];

        nSucc = nSucc + double(tau.success);
        nInt  = nInt  + double(tau.intercepted);
        fprintf(['  [%3d/%3d] row=%4d rAD0=%.0f beta=%.0f wE=%4d | ', ...
                 'r_min1=%.2f r_min2=%.2f r_minT=%.1f | succ=%d int=%d reach=%d (%d步)\n'], ...
                 k, N, valid(k), geo.rAD0, geo.beta, wE_used, ...
                 tau.r_min1, tau.r_min2, tau.r_minT, tau.success, tau.intercepted, tau.reached, numel(tau.t));
    end
    fprintf('阶段A 完成: 成功 %d/%d, 被拦截 %d/%d, 用时 %.1fs\n', ...
            nSucc, N, nInt, N, toc(t0));

    results = struct();
    results.trajectories = trajectories;
    results.N = N;
    results.nSucc = nSucc;
    results.nInt = nInt;

    % ---------------- 可视化 (第一条) ----------------
    if opts.plot && N >= 1
        plot_trajectory(trajectories(1).tau, trajectories(1).geo);
    end

    % ---------------- 保存 ----------------
    if opts.save
        save('expert_lib.mat', 'results', '-v7.3');
        fprintf('已保存 expert_lib.mat\n');
    end
end

% -------- 绘制单条轨迹的三维航迹与相对距离 --------
function plot_trajectory(tau, geo)
    figure('Name', '阶段A warm-start 轨迹');
    subplot(1, 2, 1);
    plot3(tau.M(1,:),  tau.M(3,:),  tau.M(2,:),  'k-',  'LineWidth', 1.2); hold on;
    plot3(tau.D1(1,:), tau.D1(3,:), tau.D1(2,:), 'r--', 'LineWidth', 1.2);
    plot3(tau.D2(1,:), tau.D2(3,:), tau.D2(2,:), 'b-.', 'LineWidth', 1.2);
    plot3(0, 0, 0, 'k^', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    plot3(tau.M(1,1),  tau.M(3,1),  tau.M(2,1),  'ko', 'MarkerFaceColor', 'k');
    xlabel('x/m'); ylabel('z/m'); zlabel('y/m'); grid on; view(3);
    legend('A', 'D1', 'D2', 'T', 'A_0', 'Location', 'best');
    title(sprintf('3D 航迹 (rAD0=%.0f, beta=%.0f)', geo.rAD0, geo.beta));
    set(gca, 'SortMethod', 'childorder');

    subplot(1, 2, 2);
    plot(tau.t, tau.r_MD1, 'r-',  'LineWidth', 1.2); hold on;
    plot(tau.t, tau.r_MD2, 'b-',  'LineWidth', 1.2);
    plot(tau.t, tau.r_MT,  'k--', 'LineWidth', 1.2);
    yline(20, 'g:', 'd^*=20'); yline(8000, 'm:', 'r_{safe}');
    xlabel('t/s'); ylabel('距离/m'); grid on;
    legend('r_{MD1}', 'r_{MD2}', 'r_{MT}', 'Location', 'best');
    title('相对距离');
end

% -------- 定位并添加 CasADi 路径 (阶段B 用) --------
function add_casadi_path()
    casadiRoot = 'E:\matlab\casadi-3.7.2-windows64-matlab2018b';
    toolboxRoot = fullfile(matlabroot, 'toolbox', 'casadi');
    if isfolder(toolboxRoot) && isfile(fullfile(toolboxRoot, 'casadiMEX.mexw64'))
        addpath(toolboxRoot);
    elseif isfolder(casadiRoot) && isfile(fullfile(casadiRoot, 'casadiMEX.mexw64'))
        addpath(casadiRoot);
    else
        error('generate_ps_expert_library:MissingCasADi', ...
            '未找到 CasADi, 请检查路径 %s 或 toolbox/casadi', casadiRoot);
    end
end
