function out_path = smoke_phys_schema_lib(out_path, n_traj)
%SMOKE_PHYS_SCHEMA_LIB  构造含 phys_transition_v1 的微型 expert_lib（不跑 IPOPT）。
%   用于验收 generate 落盘字段 + load/precompute 新库路径。

    if nargin < 1 || isempty(out_path)
        this_dir = fileparts(mfilename('fullpath'));
        out_path = fullfile(this_dir, '..', 'data', 'expert_lib_phys_smoke.mat');
    end
    if nargin < 2 || isempty(n_traj); n_traj = 3; end

    this_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(this_dir, '..', '..', 'expert_trajectory'));
    addpath(this_dir);
    setup_expert_path();
    core = stage2_core_init(struct());
    meta = core.meta;

    trajectories = struct( ...
        'idx', {}, 'X', {}, 'geo', {}, 'tau', {}, 'stageB', {}, ...
        'transitions', {}, 'ok_use', {});

    expert_sa_list = {};
    rng(0);
    for k = 1:n_traj
        M  = [8000 + 100*k, 0, 2000, 0, 0, 800];
        D1 = [6000,  500, 1500, 0, 0, 700];
        D2 = [6000, -500, 1500, 0, 0, 700];
        sabs = [M(:); D1(:); D2(:)];
        Tev = 1.0;  % 10 步
        t = 0;
        obs = stage2_obs_from_abs(core, sabs, t, Tev);

        nmax = 20;
        state_buf = zeros(nmax, 13);
        action_buf = zeros(nmax, 4);
        anorm_buf = zeros(nmax, 3);
        next_buf = zeros(nmax, 13);
        next_abs_buf = zeros(nmax, 18);
        sabs_buf = zeros(nmax, 18);
        t_buf = zeros(nmax, 1);
        done_buf = false(nmax, 1);
        int_buf = false(nmax, 1);
        s2_buf = false(nmax, 1);
        n = 0;
        r_min1 = inf; r_min2 = inf;

        while t < Tev - 1e-9
            gamma = 0.2 * sin(t);
            wT = 1e3; wE = 5e2;
            [an, ~] = phys_to_action_norm(gamma, wT, wE);
            [obs_next, sabs_next, t_new, fact] = stage2_step(core, sabs, t, Tev, an);
            r_min1 = min(r_min1, fact.r_MD1);
            r_min2 = min(r_min2, fact.r_MD2);
            n = n + 1;
            state_buf(n,:) = obs(:)';
            action_buf(n,:) = [1e4, gamma, wT, wE];
            anorm_buf(n,:) = an(:)';
            next_buf(n,:) = obs_next(:)';
            next_abs_buf(n,:) = sabs_next(:)';
            sabs_buf(n,:) = sabs(:)';
            t_buf(n) = t;
            done_buf(n) = fact.done;
            int_buf(n) = fact.intercepted;
            s2_buf(n) = fact.stage2_done;
            sabs = sabs_next; obs = obs_next; t = t_new;
            if fact.terminated; break; end
        end
        if ~done_buf(n); done_buf(n) = true; s2_buf(n) = true; end

        tr = struct();
        tr.state = state_buf(1:n,:);
        tr.action = action_buf(1:n,:);
        tr.action_phys = action_buf(1:n,:);
        tr.action_norm = anorm_buf(1:n,:);
        tr.next_state = next_buf(1:n,:);
        tr.next_state_abs = next_abs_buf(1:n,:);
        tr.state_abs = sabs_buf(1:n,:);
        tr.t = t_buf(1:n);
        tr.done = done_buf(1:n);
        tr.intercepted = int_buf(1:n);
        tr.stage2_done = s2_buf(1:n);
        tr.r_min1 = r_min1;
        tr.r_min2 = r_min2;
        tr.terminal_abs = sabs(:)';
        tr.t_end = t;
        tr.n_actions_clipped = int32(0);
        tr.schema_version = meta.schema_version;
        tr.stage2_dynamics_version = meta.stage2_dynamics_version;
        tr.action_mapping_version = meta.action_mapping_version;
        tr.obs_norm_version = meta.obs_norm_version;

        trajectories(k).idx = k;
        trajectories(k).X = zeros(1, 18);
        trajectories(k).geo = struct();
        trajectories(k).tau = struct('success', true);
        trajectories(k).stageB = struct('ok', true, 'ok_mode', 'ok');
        trajectories(k).transitions = tr;
        trajectories(k).ok_use = true;
        expert_sa_list{end+1} = [tr.state, tr.action]; %#ok<AGROW>
    end

    results = struct();
    results.trajectories = trajectories;
    results.N = n_traj;
    results.expert_sa = vertcat(expert_sa_list{:});
    results.expert_sa_role = 'qa_compat_only';
    results.train_authority = 'per_trajectory_phys_transition';
    results.stage3_in_library = false;
    results.schema_version = meta.schema_version;
    results.stage2_dynamics_version = meta.stage2_dynamics_version;
    results.action_mapping_version = meta.action_mapping_version;
    results.obs_norm_version = meta.obs_norm_version;
    results.stats = struct('n_phys_traj', n_traj, 'nA_succ', n_traj, ...
        'nB_ok', n_traj, 'nB_dbg', 0, 'nB_fail', 0, 'nA_int', 0, 'elapsed_s', 0);

    save(out_path, 'results', '-v7.3');
    fprintf('[smoke] 写入 %s  (%d 轨, schema=%s)\n', out_path, n_traj, meta.schema_version);
end
