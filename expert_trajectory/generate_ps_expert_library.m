function results = generate_ps_expert_library(N, opts)
%GENERATE_PS_EXPERT_LIBRARY 伪谱专家轨迹批处理主入口。
%   results = generate_ps_expert_library(N, opts)
%   流程: samples_clean 初态 -> 阶段A RK4 warm-start (solve_warmstart)
%         -> [阶段B 单打靶 NLP (solve_ss_trajectory)]
%         -> [extract_expert_transitions 抽 (s,a)]
%         -> 汇总保存 expert_lib.mat
%
%   输入:
%     N    : 处理的样本条数 (默认 10; P0 验证)
%     opts : 可选, 字段:
%            .run_stageB       (默认 false)  是否运行阶段B NLP
%            .plot             (默认 true)   是否绘制第一条阶段A轨迹
%            .save             (默认 false)  是否保存 expert_lib.mat
%            .save_file        (默认 'expert_lib.mat')  保存文件名
%            .checkpoint_every (默认 50)     每处理多少条保存一次中间结果; 0=禁用
%            .cfg              传给 simulate_warmstart_trajectory 的配置
%            .ss_cfg           传给 solve_ss_trajectory 的配置
%            .ex_cfg           传给 extract_expert_transitions 的配置
%   输出 results: 含 trajectories(k) 与统计信息的结构体。
%
%   expert_lib.mat 格式:
%     results.expert_sa  [N_total_steps × 17]  第1-13列=state, 14-17列=action
%     results.trajectories(k).{idx,X,geo,tau,stageB,transitions,ok_use}
%     results.stats.*
%
%   路径约定:
%     samples_clean.mat 位于 GAIL 项目根目录;
%     默认输出 expert_lib.mat 保存于 expert_trajectory/ 目录。

    setup_expert_path();
    [expertRoot, projectRoot] = project_paths();

    %% -------- 参数默认 --------
    if nargin < 1 || isempty(N); N = 10; end
    if nargin < 2; opts = struct(); end
    if ~isfield(opts,'run_stageB'); opts.run_stageB = false;         end
    if ~isfield(opts,'plot');       opts.plot        = true;          end
    if ~isfield(opts,'save');       opts.save        = false;         end
    if ~isfield(opts,'save_file');  opts.save_file   = fullfile(expertRoot, 'expert_lib.mat'); end
    if ~isfield(opts,'cfg');              opts.cfg              = struct(); end
    if ~isfield(opts,'ss_cfg');           opts.ss_cfg           = struct(); end
    if ~isfield(opts,'ex_cfg');           opts.ex_cfg           = struct(); end
    if ~isfield(opts,'checkpoint_every'); opts.checkpoint_every = 50;       end
    opts.save_file = resolve_data_path(opts.save_file, expertRoot);

    %% -------- 加载样本 --------
    samples_file = fullfile(projectRoot, 'samples_clean.mat');
    if ~isfile(samples_file)
        error('generate_ps_expert_library:MissingSamples', ...
            '未找到 %s, 请先在项目根目录生成 samples_clean.mat', samples_file);
    end
    S = load(samples_file, 'samples_clean');
    sc = S.samples_clean;
    valid = find(any(sc ~= 0, 2));
    nValid = numel(valid);
    N = min(N, nValid);
    fprintf('samples_clean: %d 有效行, 本次处理前 %d 条\n', nValid, N);

    if opts.run_stageB
        add_casadi_path();
    end

    %% -------- 主循环 --------
    trajectories = struct('idx',{},'X',{},'geo',{},'tau',{},'stageB',{},'transitions',{},'ok_use',{});
    nA_succ = 0; nA_int = 0; nB_ok = 0; nB_dbg = 0; nB_fail = 0;
    expert_sa_list = {};   % cell array, 每条轨迹一个 [N_steps×17] 矩阵

    t0_total = tic;
    for k = 1:N
        row = sc(valid(k), :);
        [M0, D10, D20, geo] = reconstruct_geometry(row);

        % ---- 阶段A ----
        [tau, wE_used] = solve_warmstart(M0, D10, D20, opts.cfg);
        nA_succ = nA_succ + double(tau.success);
        nA_int  = nA_int  + double(tau.intercepted);

        trajectories(k).idx    = valid(k);
        trajectories(k).X      = row;
        trajectories(k).geo    = geo;
        trajectories(k).tau    = tau;
        trajectories(k).wE_used= wE_used;
        trajectories(k).stageB = [];
        trajectories(k).transitions = [];
        trajectories(k).ok_use = false;

        fprintf(['  [%3d/%3d] row=%4d rAD0=%.0f beta=%.1f wE=%4d | ', ...
                 'r_min1=%.1f r_min2=%.1f r_minT=%.0f | succA=%d int=%d reach=%d\n'], ...
                 k, N, valid(k), geo.rAD0, geo.beta, wE_used, ...
                 tau.r_min1, tau.r_min2, tau.r_minT, tau.success, tau.intercepted, tau.reached);

        % ---- 阶段B (可选) ----
        if ~opts.run_stageB || ~tau.success
            continue;
        end

        t_b = tic;
        res = solve_ss_trajectory(tau, opts.ss_cfg);
        dt_b = toc(t_b);

        % 判断可用性: ok=true 直接用; ok=false 但 debug theta 存在且 miss 充分时降级使用
        ok_use  = false;
        ok_mode = 'fail';
        if res.ok
            ok_use  = true;
            ok_mode = 'ok';
            nB_ok   = nB_ok + 1;
        elseif ~isempty(res.theta_K) && isfield(res,'miss') && ~isempty(res.miss)
            dfloor = 6;
            if isfield(opts.ss_cfg,'dfloor'); dfloor = opts.ss_cfg.dfloor; end
            if res.miss(1) >= dfloor && res.miss(2) >= dfloor
                ok_use  = true;
                ok_mode = 'debug';
                nB_dbg  = nB_dbg + 1;
            else
                nB_fail = nB_fail + 1;
            end
        else
            nB_fail = nB_fail + 1;
        end

        res.ok_mode = ok_mode;
        trajectories(k).stageB = res;
        trajectories(k).ok_use = ok_use;

        miss_str = '';
        if isfield(res,'miss') && ~isempty(res.miss)
            miss_str = sprintf('miss=[%.1f,%.1f,%.1f] ', res.miss(1),res.miss(2),res.miss(3));
        end
        reason_str = get_reason_str(res);
        fprintf('         stageB %s(%s) %s用时%.1fs\n', ok_mode, reason_str, miss_str, dt_b);

        % ---- 提取 (s,a) ----
        if ok_use
            tr = extract_expert_transitions(tau, res, opts.ex_cfg);
            if ~isempty(tr) && size(tr.state,1) > 0
                trajectories(k).transitions = tr;
                sa = [tr.state, tr.action];    % N_steps × 17
                expert_sa_list{end+1} = sa;    %#ok<AGROW>
                fprintf('         extracted %d (s,a) pairs\n', size(sa,1));
            end
        end

        % ---- 分段 checkpoint 保存 ----
        cp = opts.checkpoint_every;
        if opts.save && cp > 0 && mod(k, cp) == 0
            results_ckpt = build_results(trajectories, expert_sa_list, N, ...
                nA_succ, nA_int, nB_ok, nB_dbg, nB_fail, toc(t0_total)); %#ok<NASGU>
            [fdir, fname, fext] = fileparts(opts.save_file);
            ckpt_file = fullfile(fdir, sprintf('%s_ckpt%04d%s', fname, k, fext));
            save(ckpt_file, 'results_ckpt', '-v7.3');
            fprintf('  [checkpoint] 已保存 %s\n', ckpt_file);
        end
    end

    elapsed = toc(t0_total);
    fprintf('\n===== 批处理完成 =====\n');
    fprintf('  阶段A: 成功 %d/%d, 被拦截 %d/%d\n', nA_succ, N, nA_int, N);
    if opts.run_stageB
        fprintf('  阶段B: ok=%d, debug降级=%d, fail=%d (共尝试 %d 条)\n', ...
                nB_ok, nB_dbg, nB_fail, nA_succ);
        n_sa = numel(expert_sa_list);
        total_steps = sum(cellfun(@(x)size(x,1), expert_sa_list));
        fprintf('  (s,a): %d 条轨迹提取, 共 %d 步\n', n_sa, total_steps);
    end
    fprintf('  用时 %.1f s (%.1f min)\n', elapsed, elapsed/60);

    %% -------- 汇总 --------
    results = build_results(trajectories, expert_sa_list, N, ...
        nA_succ, nA_int, nB_ok, nB_dbg, nB_fail, elapsed);

    %% -------- 可视化 (第一条阶段A) --------
    if opts.plot && N >= 1
        plot_trajectory(trajectories(1).tau, trajectories(1).geo);
    end

    %% -------- 最终保存 --------
    if opts.save
        save(opts.save_file, 'results', '-v7.3');
        fprintf('已保存最终结果 %s  (expert_sa: %d x %d)\n', ...
                opts.save_file, size(results.expert_sa,1), size(results.expert_sa,2));
    end
end

% =========================================================
% 辅助: 组装 results 结构体 (供主函数和 checkpoint 共用)
% =========================================================
function results = build_results(trajectories, expert_sa_list, N, ...
        nA_succ, nA_int, nB_ok, nB_dbg, nB_fail, elapsed)
    results = struct();
    results.trajectories = trajectories;
    results.N = N;
    results.stats = struct('nA_succ',nA_succ,'nA_int',nA_int, ...
                           'nB_ok',nB_ok,'nB_dbg',nB_dbg,'nB_fail',nB_fail, ...
                           'elapsed_s',elapsed);
    if ~isempty(expert_sa_list)
        results.expert_sa = vertcat(expert_sa_list{:});
    else
        results.expert_sa = zeros(0, 17);
    end
end

% =========================================================
% 辅助: 生成 stageB 失败原因的短描述
% =========================================================
function s = get_reason_str(res)
    if res.ok
        s = '';
    elseif isfield(res,'reason') && ~isempty(res.reason)
        r = res.reason;
        if numel(r) > 50; r = r(1:50); end
        s = r;
    else
        s = 'unknown';
    end
end

% =========================================================
% 辅助: 绘制单条阶段A轨迹
% =========================================================
function plot_trajectory(tau, geo)
    figure('Name','阶段A warm-start 轨迹');
    subplot(1,2,1);
    plot3(tau.M(1,:),  tau.M(3,:),  tau.M(2,:),  'k-',  'LineWidth',1.2); hold on;
    plot3(tau.D1(1,:), tau.D1(3,:), tau.D1(2,:), 'r--', 'LineWidth',1.2);
    plot3(tau.D2(1,:), tau.D2(3,:), tau.D2(2,:), 'b-.', 'LineWidth',1.2);
    plot3(0,0,0,'k^','MarkerSize',8,'MarkerFaceColor','k');
    plot3(tau.M(1,1),tau.M(3,1),tau.M(2,1),'ko','MarkerFaceColor','k');
    xlabel('x/m'); ylabel('z/m'); zlabel('y/m'); grid on; view(3);
    legend('A','D1','D2','T','A_0','Location','best');
    title(sprintf('3D航迹 (rAD0=%.0f, beta=%.1f)',geo.rAD0,geo.beta));
    set(gca,'SortMethod','childorder');

    subplot(1,2,2);
    plot(tau.t,tau.r_MD1,'r-','LineWidth',1.2); hold on;
    plot(tau.t,tau.r_MD2,'b-','LineWidth',1.2);
    plot(tau.t,tau.r_MT, 'k--','LineWidth',1.2);
    yline(20,'g:','d^*=20'); yline(8000,'m:','r_{safe}');
    xlabel('t/s'); ylabel('距离/m'); grid on;
    legend('r_{MD1}','r_{MD2}','r_{MT}','Location','best');
    title('相对距离');
end

% =========================================================
% 辅助: 定位并添加 CasADi 路径
% =========================================================
function p = resolve_data_path(p, defaultDir)
    if isempty(fileparts(p))
        p = fullfile(defaultDir, p);
    end
end

function add_casadi_path()
    casadiRoot  = 'E:\matlab\casadi-3.7.2-windows64-matlab2018b';
    toolboxRoot = fullfile(matlabroot,'toolbox','casadi');
    if isfolder(toolboxRoot) && isfile(fullfile(toolboxRoot,'casadiMEX.mexw64'))
        addpath(toolboxRoot);
    elseif isfolder(casadiRoot) && isfile(fullfile(casadiRoot,'casadiMEX.mexw64'))
        addpath(casadiRoot);
    else
        error('generate_ps_expert_library:MissingCasADi', ...
            '未找到 CasADi, 请检查路径 %s 或 toolbox/casadi', casadiRoot);
    end
end
