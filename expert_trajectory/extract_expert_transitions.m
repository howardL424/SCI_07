function tr = extract_expert_transitions(tau_warm, res_ss, cfg)
%EXTRACT_EXPERT_TRANSITIONS 从单打靶优化结果提取物理 transition（新 schema）。
%   tr = extract_expert_transitions(tau_warm, res_ss)
%   tr = extract_expert_transitions(tau_warm, res_ss, cfg)
%
%   以 dt_out≈0.1s 数值 RK4 重跑规避段 (stage2)，与在线 env_step 共用
%   stage2_core_init / stage2_step。动作先映射裁剪到 [-1,1]^3，再用同一动作积分。
%
%   兼容字段（旧工具可读）:
%     .state     [N×13]  相对态势
%     .action    [N×4]   [w=1e4, gamma, wT, wE] 原始分段 theta（审计）
%     .t         [N×1]   时间 (s)
%     .state_abs [N×18]  绝对物理状态
%
%   新增权威字段（每步）:
%     .action_norm   [N×3]  裁剪后 TD3 动作
%     .next_state    [N×13] 执行裁剪动作后的真实 s'
%     .next_state_abs[N×18]
%     .done          [N×1] logical
%     .intercepted   [N×1] logical  （物理事实；不在此写 -50）
%     .stage2_done   [N×1] logical
%     .action_phys   [N×4]  与 action 同义（审计别名）
%
%   轨迹级:
%     .r_min1 .r_min2 .terminal_abs .t_end
%     .schema_version .stage2_dynamics_version
%     .action_mapping_version .obs_norm_version
%     .n_actions_clipped
%
%   state 定义 (13维): 同前（r_MD*/rdot/qy/qz, r_MT, V_M, tgo_norm）

    %% -------- 默认配置 --------
    setup_expert_path();
    if nargin < 3; cfg = struct(); end
    d.dt_out = 0.1;
    d.w      = 1e4;
    d.dstar  = 20;
    d.KD1    = 4;
    d.KD2    = 4;
    d.g0     = 9.81;
    fn = fieldnames(d);
    for k = 1:numel(fn)
        if ~isfield(cfg, fn{k}); cfg.(fn{k}) = d.(fn{k}); end
    end

    %% -------- 校验输入 --------
    if ~isfield(res_ss,'theta_K') || isempty(res_ss.theta_K)
        tr = []; return;
    end

    %% -------- 预初始化共享动力学核心（只一次）--------
    core_cfg = struct();
    core_cfg.dt    = cfg.dt_out;
    core_cfg.dstar = cfg.dstar;
    core_cfg.KD1   = cfg.KD1;
    core_cfg.KD2   = cfg.KD2;
    core_cfg.g0    = cfg.g0;
    core_cfg.w     = cfg.w;
    core = stage2_core_init(core_cfg);
    meta = core.meta;

    %% -------- 抽取规避段初态 --------
    idx2 = find(tau_warm.stage == 2);
    if isempty(idx2)
        tr = []; return;
    end
    i0 = idx2(1);
    M  = tau_warm.M(:,  i0)';
    D1 = tau_warm.D1(:, i0)';
    D2 = tau_warm.D2(:, i0)';
    sabs = [M(:); D1(:); D2(:)];

    Tev     = res_ss.Tev;
    dt      = cfg.dt_out;
    theta_K = res_ss.theta_K;
    K       = size(theta_K, 2);
    seg_dur = Tev / K;
    w       = cfg.w;

    %% -------- 预分配 --------
    nmax        = ceil(Tev / dt) + 5;
    state_buf   = zeros(nmax, 13);
    action_buf  = zeros(nmax, 4);
    anorm_buf   = zeros(nmax, 3);
    next_buf    = zeros(nmax, 13);
    next_abs_buf= zeros(nmax, 18);
    sabs_buf    = zeros(nmax, 18);
    t_buf       = zeros(nmax, 1);
    done_buf    = false(nmax, 1);
    int_buf     = false(nmax, 1);
    s2done_buf  = false(nmax, 1);
    n = 0;
    n_clip = 0;
    r_min1 = inf;
    r_min2 = inf;

    %% -------- 主仿真循环：先裁剪动作，再同动作积分 --------
    t = 0;
    obs = stage2_obs_from_abs(core, sabs, t, Tev);
    while t < Tev - 1e-9
        seg_idx = min(floor(t / seg_dur) + 1, K);
        gamma_k = theta_K(1, seg_idx);
        wT_k    = theta_K(2, seg_idx);
        wE_k    = theta_K(3, seg_idx);

        [an, clipped] = phys_to_action_norm(gamma_k, wT_k, wE_k);
        n_clip = n_clip + double(clipped);

        [obs_next, sabs_next, t_new, fact] = ...
            stage2_step(core, sabs, t, Tev, an);

        r_min1 = min(r_min1, fact.r_MD1);
        r_min2 = min(r_min2, fact.r_MD2);

        n = n + 1;
        state_buf(n, :)    = obs(:)';
        action_buf(n, :)   = [w, gamma_k, wT_k, wE_k];
        anorm_buf(n, :)    = an(:)';
        next_buf(n, :)     = obs_next(:)';
        next_abs_buf(n, :) = sabs_next(:)';
        sabs_buf(n, :)     = sabs(:)';
        t_buf(n)           = t;
        done_buf(n)        = fact.done;
        int_buf(n)         = fact.intercepted;
        s2done_buf(n)      = fact.stage2_done;

        sabs = sabs_next;
        obs  = obs_next;
        t    = t_new;

        % 拦截立刻截断；正常 stage2 结束亦截断
        if fact.terminated
            break;
        end
    end

    if n == 0
        tr = []; return;
    end

    % 数值边界：循环耗尽仍未 terminated 时强制末步 done
    if ~done_buf(n)
        done_buf(n) = true;
        s2done_buf(n) = true;
    end

    %% -------- 打包输出 --------
    tr = struct();
    tr.state          = state_buf(1:n, :);
    tr.action         = action_buf(1:n, :);
    tr.action_phys    = action_buf(1:n, :);   % 与 action 同义（审计）
    tr.action_norm    = anorm_buf(1:n, :);
    tr.next_state     = next_buf(1:n, :);
    tr.next_state_abs = next_abs_buf(1:n, :);
    tr.state_abs      = sabs_buf(1:n, :);
    tr.t              = t_buf(1:n);
    tr.done           = done_buf(1:n);
    tr.intercepted    = int_buf(1:n);
    tr.stage2_done    = s2done_buf(1:n);

    tr.r_min1         = r_min1;
    tr.r_min2         = r_min2;
    tr.terminal_abs   = sabs(:)';          % 执行最后一拍后的真实末态
    tr.t_end          = t;
    tr.n_actions_clipped = int32(n_clip);

    tr.schema_version          = meta.schema_version;
    tr.stage2_dynamics_version = meta.stage2_dynamics_version;
    tr.action_mapping_version  = meta.action_mapping_version;
    tr.obs_norm_version        = meta.obs_norm_version;
end
