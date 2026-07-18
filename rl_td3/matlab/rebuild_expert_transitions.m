function aligned = rebuild_expert_transitions(expert_path, out_path, opts)
%REBUILD_EXPERT_TRANSITIONS  按 env_step 语义重放专家轨迹，重建 (s,a,s',r,done)。
%
%   aligned = rebuild_expert_transitions(expert_path, out_path)
%   aligned = rebuild_expert_transitions(expert_path, out_path, opts)
%
%   权威源：results.trajectories(k).transitions
%   从每条轨迹首个 state_abs 出发，按记录动作顺序调用 stage2_step；
%   动作先归一化并裁剪到 [-1,1]^3，与在线 RL 可执行动作一致。
%   helper/config 经 stage2_core_init 预初始化，不逐步重建。
%
%   opts 字段（均可选）:
%     .max_traj   最多处理有效轨迹数（调试/冒烟）
%     .verbose    默认 true
%
%   输出 aligned（并 save 到 out_path）:
%     states/actions/next_states/rewards/dones/traj_ids  （扁平，物理观测 + 归一化动作）
%     n_steps/intercepted/terminal_abs/t_end/source_idx/r_min1/r_min2
%     expert_path, n_actions_clipped, n_steps_recorded

    if nargin < 3 || isempty(opts); opts = struct(); end
    if ~isfield(opts, 'max_traj'); opts.max_traj = inf; end
    if ~isfield(opts, 'verbose');  opts.verbose  = true; end

    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);
    addpath(fullfile(this_dir, '..', '..'));
    addpath(fullfile(this_dir, '..', '..', 'expert_trajectory'));

    if opts.verbose
        fprintf('[rebuild] 加载专家库: %s\n', expert_path);
    end
    S = load(expert_path);
    if ~isfield(S, 'results')
        error('文件中无 results 字段: %s', expert_path);
    end
    results = S.results;
    n_lib = numel(results.trajectories);

    % 预初始化共享动力学核心（整库只一次）
    core = stage2_core_init(struct());
    meta = core.meta;
    dt   = core.cfg.dt;

    % 预分配（按专家库总记录步数上界）
    n_cap = 0;
    for k = 1:n_lib
        n_cap = n_cap + local_n_steps(results.trajectories(k));
    end
    if ~isfinite(opts.max_traj)
        % keep n_cap
    else
        n_cap = min(n_cap, max(1, round(opts.max_traj)) * 200);
    end

    states      = zeros(n_cap, 13);
    actions     = zeros(n_cap, 3);
    next_states = zeros(n_cap, 13);
    rewards     = zeros(n_cap, 1);
    dones       = false(n_cap, 1);
    traj_ids    = zeros(n_cap, 1);

    n_steps_list   = [];
    n_rec_list     = [];
    intercepted_l  = [];
    terminal_abs_l = [];
    t_end_list     = [];
    source_idx_l   = [];
    r_min1_list    = [];
    r_min2_list    = [];
    n_clip_total   = 0;
    n_written      = 0;
    n_done         = 0;

    for k = 1:n_lib
        tr = results.trajectories(k);
        n_rec = local_n_steps(tr);
        if n_rec <= 0
            continue;
        end
        if n_done >= opts.max_traj
            break;
        end

        [state, action, state_abs] = local_get_arrays(tr);
        if isempty(state) || isempty(action) || isempty(state_abs)
            if opts.verbose
                fprintf('[rebuild] 跳过 traj#%d: 缺字段\n', k);
            end
            continue;
        end
        n_rec = min([n_rec, size(state,1), size(action,1), size(state_abs,1)]);
        if n_rec <= 0
            continue;
        end

        sabs = state_abs(1, :)';
        obs  = state(1, :)';
        t    = 0.0;
        Tev  = n_rec * dt;
        r_min1 = inf;
        r_min2 = inf;
        intercepted = false;
        n_local = 0;
        n_clip_local = 0;

        for i = 1:n_rec
            gamma = action(i, 2);
            wT    = action(i, 3);
            wE    = action(i, 4);
            [an, clipped] = phys_to_action_norm(gamma, wT, wE);
            n_clip_local = n_clip_local + double(clipped);

            [obs_next, sabs_next, t_new, fact] = ...
                stage2_step(core, sabs, t, Tev, an);
            reward = stage2_process_reward(core, fact);

            r_min1 = min(r_min1, fact.r_MD1);
            r_min2 = min(r_min2, fact.r_MD2);
            intercepted = logical(fact.intercepted);

            n_written = n_written + 1;
            n_local   = n_local + 1;
            if n_written > size(states, 1)
                % 动态扩容
                grow = size(states, 1);
                states      = [states;      zeros(grow, 13)]; %#ok<AGROW>
                actions     = [actions;     zeros(grow, 3)];  %#ok<AGROW>
                next_states = [next_states; zeros(grow, 13)]; %#ok<AGROW>
                rewards     = [rewards;     zeros(grow, 1)];  %#ok<AGROW>
                dones       = [dones;       false(grow, 1)];  %#ok<AGROW>
                traj_ids    = [traj_ids;    zeros(grow, 1)];  %#ok<AGROW>
            end

            states(n_written, :)      = obs';
            actions(n_written, :)     = an(:)';
            next_states(n_written, :) = obs_next(:)';
            rewards(n_written)        = reward;
            dones(n_written)          = logical(fact.terminated);
            traj_ids(n_written)       = n_done;  % 0-based 有效轨迹序号

            sabs = sabs_next;
            t    = t_new;
            if fact.terminated
                break;
            end
            obs = obs_next;
        end

        % 若循环耗尽动作仍未 terminated（数值边界），强制末步 done
        if n_local > 0 && ~dones(n_written)
            dones(n_written) = true;
        end

        n_steps_list(end+1, 1)   = n_local; %#ok<AGROW>
        n_rec_list(end+1, 1)     = n_rec; %#ok<AGROW>
        intercepted_l(end+1, 1)  = double(intercepted); %#ok<AGROW>
        terminal_abs_l(end+1, :) = sabs(:)'; %#ok<AGROW>
        t_end_list(end+1, 1)     = t; %#ok<AGROW>
        source_idx_l(end+1, 1)   = k; %#ok<AGROW>
        r_min1_list(end+1, 1)    = r_min1; %#ok<AGROW>
        r_min2_list(end+1, 1)    = r_min2; %#ok<AGROW>
        n_clip_total = n_clip_total + n_clip_local;
        n_done = n_done + 1;

        if opts.verbose && (n_done == 1 || mod(n_done, 50) == 0)
            fprintf('  [%d] src=#%d n_rec=%d n_use=%d intercept=%d clip=%d\n', ...
                n_done, k, n_rec, n_local, intercepted, n_clip_local);
        end
    end

    aligned = struct();
    aligned.expert_path       = char(expert_path);
    aligned.n_traj            = int32(n_done);
    aligned.n_steps_total     = int32(n_written);
    aligned.states            = states(1:n_written, :);
    aligned.actions           = actions(1:n_written, :);
    aligned.next_states       = next_states(1:n_written, :);
    aligned.rewards           = rewards(1:n_written);
    aligned.dones             = dones(1:n_written);
    aligned.traj_ids          = int32(traj_ids(1:n_written));
    aligned.n_steps           = int32(n_steps_list);
    aligned.n_steps_recorded  = int32(n_rec_list);
    aligned.intercepted       = int8(intercepted_l);
    aligned.terminal_abs      = terminal_abs_l;
    aligned.t_end             = t_end_list;
    aligned.source_idx        = int32(source_idx_l);
    aligned.r_min1            = r_min1_list;
    aligned.r_min2            = r_min2_list;
    aligned.n_actions_clipped = int32(n_clip_total);
    aligned.schema_version          = meta.schema_version;
    aligned.stage2_dynamics_version = meta.stage2_dynamics_version;
    aligned.action_mapping_version  = meta.action_mapping_version;
    aligned.obs_norm_version        = meta.obs_norm_version

    if nargin >= 2 && ~isempty(out_path)
        out_dir = fileparts(out_path);
        if ~isempty(out_dir) && exist(out_dir, 'dir') ~= 7
            mkdir(out_dir);
        end
        save(out_path, 'aligned', '-v7.3');
        if opts.verbose
            fprintf('[rebuild] 完成 %d 条轨迹 / %d 步 → %s\n', ...
                n_done, n_written, out_path);
            fprintf('  拦截比例=%.1f%%  动作裁剪步数=%d\n', ...
                100 * mean(intercepted_l), n_clip_total);
        end
    end
end

% ---- helpers ----

function n = local_n_steps(tr)
    n = 0;
    if isempty(tr) || ~isfield(tr, 'transitions') || isempty(tr.transitions)
        return;
    end
    trans = tr.transitions;
    if isfield(trans, 't') && ~isempty(trans.t)
        n = numel(trans.t);
    elseif isfield(trans, 'state') && ~isempty(trans.state)
        n = size(trans.state, 1);
    end
end

function [state, action, state_abs] = local_get_arrays(tr)
    state = []; action = []; state_abs = [];
    if ~isfield(tr, 'transitions') || isempty(tr.transitions)
        return;
    end
    trans = tr.transitions;
    if isfield(trans, 'state');     state = as_NxD(trans.state, 13); end
    if isfield(trans, 'action');    action = as_NxD(trans.action, 4); end
    if isfield(trans, 'state_abs'); state_abs = as_NxD(trans.state_abs, 18); end
end

function A = as_NxD(A, D)
    A = double(A);
    if isempty(A)
        return;
    end
    if size(A, 2) == D
        return;
    end
    if size(A, 1) == D
        A = A';
        return;
    end
    error('数组无法解释为 (N,%d): [%s]', D, num2str(size(A)));
end
