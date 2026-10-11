function results = generate_ps_expert_library(N, opts)
%GENERATE_PS_EXPERT_LIBRARY 伪谱专家轨迹批处理主入口。
%   results = generate_ps_expert_library(N, opts)
%   流程: samples_clean 初态 -> 阶段A RK4 warm-start (solve_warmstart)
%         -> [阶段B 单打靶 NLP (solve_ss_trajectory)]
%         -> [extract_expert_transitions 抽物理 transition（出库即对齐）]
%         -> 汇总保存 expert_lib.mat
%
%   输入:
%     N    : 处理的样本条数 (默认 10; P0 验证); Inf=使用 samples_clean 全部有效行
%     opts : 可选, 字段:
%            .run_stageB       (默认 false)  是否运行阶段B NLP
%            .plot             (默认 true)   是否绘制第一条阶段A轨迹
%            .save             (默认 false)  是否保存 expert_lib.mat
%            .save_file        (默认 'expert_lib.mat')  保存文件名
%            .checkpoint_every (默认 200)    每处理多少条保存一次中间结果; 0=禁用
%            .resume_ckpt      (默认 '')      checkpoint 文件路径, 非空则从断点续跑
%            .valid_pos        (默认 [])      指定 valid 行序号子集, 用于分批
%            .cfg              传给 simulate_warmstart_trajectory 的配置
%            .ss_cfg           传给 solve_ss_trajectory 的配置
%            .ex_cfg           传给 extract_expert_transitions 的配置
%   输出 results: 含 trajectories(k) 与统计信息的结构体。
%
%   训练权威源 = trajectories(k).transitions（phys_transition_v1），含
%     action_norm / next_state / done / intercepted / terminal_abs 等。
%   results.expert_sa 仅兼容/QA（state+原始 action），不作为训练切轨迹依据。
%   stage3 不在本函数内运行；主库落盘后另跑 precompute_expert_terminals.py sidecar。
%
%   expert_lib.mat 格式:
%     results.trajectories(k).{idx,X,geo,tau,stageB,transitions,ok_use}
%     results.expert_sa  [N_total_steps × 17]  QA 扁平表（非训练权威）
%     results.schema_version / stage2_dynamics_version / ...
%     results.stats.*
%
%   路径约定:
%     % samples_clean.mat 位于 GAIL 项目根目录;          % 原初态，还原时改回
%     初态: D:\LTY\yan\yanerxia\GAIL\samples_clean_kuoda_train.mat
%     % 默认输出 expert_lib.mat 保存于 expert_trajectory/ 目录。
%     默认输出与 checkpoint 保存于 E:\matlab\result_1010。

    setup_expert_path();
    [expertRoot, projectRoot] = project_paths(); %#ok<ASGLU>
    % expertRoot 保留，便于还原旧输出目录。还原时注释下一行，取消再下一行注释。
    resultDir = 'E:\matlab\result_1010';
    % resultDir = expertRoot;
    if ~isfolder(resultDir)
        mkdir(resultDir);
    end

    %% -------- 参数默认 --------
    if nargin < 1 || isempty(N); N = 10; end
    if nargin < 2; opts = struct(); end
    if ~isfield(opts,'run_stageB'); opts.run_stageB = false;         end
    if ~isfield(opts,'plot');       opts.plot        = true;          end
    if ~isfield(opts,'save');       opts.save        = false;         end
    % if ~isfield(opts,'save_file');  opts.save_file   = fullfile(expertRoot, 'expert_lib.mat'); end
    if ~isfield(opts,'save_file');  opts.save_file   = fullfile(resultDir, 'expert_lib.mat'); end
    if ~isfield(opts,'cfg');              opts.cfg              = struct(); end
    if ~isfield(opts,'ss_cfg');           opts.ss_cfg           = struct(); end
    if ~isfield(opts,'ex_cfg');           opts.ex_cfg           = struct(); end
    % if ~isfield(opts,'checkpoint_every'); opts.checkpoint_every = 50;       end
    if ~isfield(opts,'checkpoint_every'); opts.checkpoint_every = 200;      end
    if ~isfield(opts,'resume_ckpt');      opts.resume_ckpt      = '';       end
    if ~isfield(opts,'valid_pos');        opts.valid_pos        = [];       end
    % opts.save_file = resolve_data_path(opts.save_file, expertRoot);
    opts.save_file = resolve_data_path(opts.save_file, resultDir);
    outParent = fileparts(opts.save_file);
    if ~isempty(outParent) && ~isfolder(outParent)
        mkdir(outParent);
    end
    if ~isempty(opts.resume_ckpt)
        % opts.resume_ckpt = resolve_data_path(opts.resume_ckpt, expertRoot);
        opts.resume_ckpt = resolve_data_path(opts.resume_ckpt, resultDir);
    end

    %% -------- 加载样本 --------
    % samples_file = fullfile(projectRoot, 'samples_clean.mat');
    samples_file = fullfile(projectRoot, 'samples_clean_kuoda_train.mat');
    if ~isfile(samples_file)
        error('generate_ps_expert_library:MissingSamples', ...
            '未找到 %s', samples_file);
    end
    S = load(samples_file, 'samples_clean');
    sc = S.samples_clean;
    valid = find(any(sc ~= 0, 2));
    nValid = numel(valid);
    if isscalar(N) && isinf(N)
        N = nValid;
    else
        N = min(N, nValid);
    end
    select_pos = 1:N;
    if ~isempty(opts.valid_pos)
        pos = unique(opts.valid_pos(:).');
        pos = pos(pos >= 1 & pos <= nValid);
        select_pos = pos;
        N = numel(select_pos);
        fprintf('%s: %d 有效行, 指定子集 %d 条\n', samples_file, nValid, N);
    else
        fprintf('%s: %d 有效行, 本次处理 %d 条\n', samples_file, nValid, N);
    end

    if opts.run_stageB
        add_casadi_path();
    end

    %% -------- 断点续跑 (可选) --------
    resume_k = 0;
    elapsed_prev = 0;
    trajectories = struct('idx',{},'X',{},'geo',{},'tau',{},'stageB',{},'transitions',{},'ok_use',{});
    nA_succ = 0; nA_int = 0; nB_ok = 0; nB_dbg = 0; nB_fail = 0;
    expert_sa_list = {};   % cell array, 每条轨迹一个 [N_steps×17] 矩阵

    if ~isempty(opts.resume_ckpt)
        if ~isfile(opts.resume_ckpt)
            error('generate_ps_expert_library:MissingCheckpoint', ...
                '未找到 checkpoint: %s', opts.resume_ckpt);
        end
        tmp = load(opts.resume_ckpt, 'results_ckpt');
        prev = tmp.results_ckpt;
        resume_k = numel(prev.trajectories);
        if resume_k >= N
            fprintf('[resume] checkpoint 已含 %d/%d 条, 无需续跑\n', resume_k, N);
            results = prev;
            results.valid_pos = select_pos;
            results.valid_idx = valid(select_pos);
            if opts.save
                save(opts.save_file, 'results', '-v7.3');
                fprintf('已保存最终结果 %s\n', opts.save_file);
            end
            return;
        end
        trajectories = prev.trajectories;
        [nA_succ, nA_int, nB_ok, nB_dbg, nB_fail, expert_sa_list] = ...
            restore_batch_state(prev);
        if isfield(prev.stats, 'elapsed_s')
            elapsed_prev = prev.stats.elapsed_s;
        end
        fprintf('[resume] 从 %s 恢复 %d/%d 条, 续跑 %d:%d\n', ...
            opts.resume_ckpt, resume_k, N, resume_k + 1, N);
    end

    %% -------- 主循环 --------
    t0_total = tic;
    for k = (resume_k + 1):N
        vpos = select_pos(k);
        row = sc(valid(vpos), :);
        [M0, D10, D20, geo] = reconstruct_geometry(row);

        % ---- 阶段A ----
        [tau, wE_used] = solve_warmstart(M0, D10, D20, opts.cfg);
        nA_succ = nA_succ + double(tau.success);
        nA_int  = nA_int  + double(tau.intercepted);

        trajectories(k).idx    = valid(vpos);
        trajectories(k).valid_pos = vpos;
        trajectories(k).X      = row;
        trajectories(k).geo    = geo;
        trajectories(k).tau    = tau;
        trajectories(k).wE_used= wE_used;
        trajectories(k).stageB = [];
        trajectories(k).transitions = [];
        trajectories(k).ok_use = false;

        fprintf(['  [%3d/%3d] row=%4d rAD0=%.0f beta=%.1f wE=%4d | ', ...
                 'r_min1=%.1f r_min2=%.1f r_minT=%.0f | succA=%d int=%d reach=%d\n'], ...
                 k, N, valid(vpos), geo.rAD0, geo.beta, wE_used, ...
                 tau.r_min1, tau.r_min2, tau.r_minT, tau.success, tau.intercepted, tau.reached);

        % ---- 阶段B (可选) ----
        if ~opts.run_stageB || ~tau.success
            continue;
        end

        t_b = tic;
        res = solve_ss_trajectory(tau, opts.ss_cfg);
        dt_b = toc(t_b);

        % 判断可用性: ok=true 直接用; ok=false 但 debug theta 存在且 miss 充分时降级使用
        % 硬门槛: miss1/miss2 >= dfloor(6m) 且 missT <= missT_ceil(30m)
        ok_use  = false;
        ok_mode = 'fail';
        dfloor     = 6;    if isfield(opts.ss_cfg,'dfloor');     dfloor     = opts.ss_cfg.dfloor;     end
        missT_ceil = 30;   if isfield(opts.ss_cfg,'missT_ceil'); missT_ceil = opts.ss_cfg.missT_ceil; end
        if res.ok
            ok_use  = true;
            ok_mode = 'ok';
            nB_ok   = nB_ok + 1;
        elseif ~isempty(res.theta_K) && isfield(res,'miss') && ~isempty(res.miss)
            missT_ok = (numel(res.miss) < 3) || (res.miss(3) <= missT_ceil);
            if res.miss(1) >= dfloor && res.miss(2) >= dfloor && missT_ok
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

        % ---- 提取物理 transition（出库即对齐；不含 stage3）----
        if ok_use
            tr = extract_expert_transitions(tau, res, opts.ex_cfg);
            if ~isempty(tr) && size(tr.state,1) > 0
                assert_phys_transition(tr, k);
                trajectories(k).transitions = tr;
                % expert_sa 仅 QA：原始 state + 物理 action（非 action_norm）
                sa = [tr.state, tr.action];    % N_steps × 17
                expert_sa_list{end+1} = sa;    %#ok<AGROW>
                n_int = sum(tr.intercepted(:));
                fprintf(['         phys transition %d 步  schema=%s  ', ...
                         'intercept_steps=%d  t_end=%.2f  clip=%d\n'], ...
                    size(tr.state,1), char(tr.schema_version), n_int, ...
                    tr.t_end, tr.n_actions_clipped);
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

    elapsed = elapsed_prev + toc(t0_total);
    fprintf('\n===== 批处理完成 =====\n');
    fprintf('  阶段A: 成功 %d/%d, 被拦截 %d/%d\n', nA_succ, N, nA_int, N);
    if opts.run_stageB
        fprintf('  阶段B: ok=%d, debug降级=%d, fail=%d (共尝试 %d 条)\n', ...
                nB_ok, nB_dbg, nB_fail, nA_succ);
        n_sa = numel(expert_sa_list);
        total_steps = sum(cellfun(@(x)size(x,1), expert_sa_list));
        fprintf('  物理 transition: %d 条轨迹, 共 %d 步（expert_sa 仅 QA）\n', ...
        n_sa, total_steps);
    end
    fprintf('  用时 %.1f s (%.1f min)\n', elapsed, elapsed/60);
    fprintf('  提示: stage3 sidecar 请事后运行 precompute_expert_terminals.py\n');

    %% -------- 汇总 --------
    results = build_results(trajectories, expert_sa_list, N, ...
        nA_succ, nA_int, nB_ok, nB_dbg, nB_fail, elapsed);
    results.valid_pos = select_pos;
    results.valid_idx = valid(select_pos);

    %% -------- 可视化 (第一条阶段A) --------
    if opts.plot && N >= 1
        plot_trajectory(trajectories(1).tau, trajectories(1).geo);
    end

    %% -------- 最终保存 --------
    if opts.save
        % 主库先安全落盘（含物理 transition）；不绑定 stage3
        save(opts.save_file, 'results', '-v7.3');
        fprintf('已保存最终结果 %s\n', opts.save_file);
        fprintf('  训练权威: trajectories(*).transitions  schema=%s\n', ...
            results.schema_version);
        fprintf('  expert_sa(QA): %d x %d\n', ...
            size(results.expert_sa,1), size(results.expert_sa,2));
        fprintf('  stage3: 未运行（sidecar 另算）\n');
    end
end

function assert_phys_transition(tr, k)
% 新库出库必须带齐权威字段；缺字段立刻失败，避免静默写成旧格式。
    req = {'state','action','state_abs','t', ...
           'action_norm','next_state','next_state_abs', ...
           'done','intercepted','stage2_done', ...
           'r_min1','r_min2','terminal_abs','t_end', ...
           'schema_version','stage2_dynamics_version', ...
           'action_mapping_version','obs_norm_version'};
    for i = 1:numel(req)
        if ~isfield(tr, req{i})
            error('generate_ps_expert_library:BadTransition', ...
                '轨迹 #%d transitions 缺少字段 %s（phys_transition_v1）', k, req{i});
        end
    end
    if size(tr.next_state, 1) ~= size(tr.state, 1)
        error('generate_ps_expert_library:BadTransition', ...
            '轨迹 #%d next_state 行数与 state 不一致', k);
    end
    if size(tr.action_norm, 2) ~= 3
        error('generate_ps_expert_library:BadTransition', ...
            '轨迹 #%d action_norm 应为 N×3', k);
    end
end

% =========================================================
% 辅助: 组装 results 结构体 (供主函数和 checkpoint 共用)
% =========================================================
function [nA_succ, nA_int, nB_ok, nB_dbg, nB_fail, expert_sa_list] = restore_batch_state(prev)
    nA_succ = prev.stats.nA_succ;
    nA_int  = prev.stats.nA_int;
    nB_ok   = prev.stats.nB_ok;
    nB_dbg  = prev.stats.nB_dbg;
    nB_fail = prev.stats.nB_fail;
    expert_sa_list = {};
    for i = 1:numel(prev.trajectories)
        tr = prev.trajectories(i);
        if tr.ok_use && ~isempty(tr.transitions) && isfield(tr.transitions, 'state') ...
                && size(tr.transitions.state, 1) > 0
            expert_sa_list{end+1} = [tr.transitions.state, tr.transitions.action]; %#ok<AGROW>
        end
    end
end

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
    % expert_sa 仅兼容/QA；训练权威 = per-trajectory 物理 transition
    results.expert_sa_role = 'qa_compat_only';
    results.train_authority = 'per_trajectory_phys_transition';
    results.stage3_in_library = false;

    meta = transition_schema_meta();
    results.schema_version          = meta.schema_version;
    results.stage2_dynamics_version = meta.stage2_dynamics_version;
    results.action_mapping_version  = meta.action_mapping_version;
    results.obs_norm_version        = meta.obs_norm_version;

    % 汇总物理层拦截计数（供 sidecar / 验收）
    n_phys = 0; n_phys_int = 0; n_phys_steps = 0;
    for i = 1:numel(trajectories)
        tr = trajectories(i);
        if ~tr.ok_use || isempty(tr.transitions) || ~isfield(tr.transitions, 'state')
            continue;
        end
        n_phys = n_phys + 1;
        n_phys_steps = n_phys_steps + size(tr.transitions.state, 1);
        if isfield(tr.transitions, 'intercepted') && any(tr.transitions.intercepted(:))
            n_phys_int = n_phys_int + 1;
        end
    end
    results.stats.n_phys_traj = n_phys;
    results.stats.n_phys_steps = n_phys_steps;
    results.stats.n_phys_intercepted = n_phys_int;
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
