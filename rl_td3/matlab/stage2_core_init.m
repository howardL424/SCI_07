function core = stage2_core_init(cfg_in)
%STAGE2_CORE_INIT  预初始化 stage2 一步动力学核心（helper + 配置）。
%
%   core = stage2_core_init()
%   core = stage2_core_init(cfg_in)
%
%   大循环中应只调用一次本函数，再反复 stage2_step(core, ...)。
%   禁止在逐步循环里反复 evasion_helpers() / 重建默认配置。
%
%   core 字段:
%     .cfg   合并后的仿真/奖励/辨识配置
%     .H     evasion_helpers() 句柄包
%     .meta  transition_schema_meta() 版本元数据
%     .T_pos / .T_state  静止目标
%
%   辨识相关 cfg（默认 analytic_truth，不影响旧入口）:
%     .coeff_mode = 'analytic_truth' | 'analytic_noisy' | 'sparse_ffrls'
%     .ident_apply_cmd = true/false（false=影子：估参但不改指令）
%     .sigma_q .sigma_qdot .sigma_theta .sigma_psi .sigma_r .sigma_rdot
%     .kf_qj .kf_sigma_a0 .rls_lambda .rls_mu .rls_P0 .rls_r0 .rls_a0
%     .noise_seed

    if nargin < 1 || isempty(cfg_in); cfg_in = struct(); end

    d = struct();
    d.dt            = 0.1;
    d.dstar         = 20.0;
    d.KD1           = 4;
    d.KD2           = 4;
    d.KM            = 3;
    d.g0            = 9.81;
    d.kill_radius   = 6.0;
    d.proc_coef     = 0.05;
    d.intercept_pen = 50.0;   % 仅在线/投影用；物理 transition 不写死 -50
    d.w             = 1e4;
    meta0 = transition_schema_meta();
    d.action_mapping_version = meta0.action_mapping_version;

    % ---- 参数辨识 / 测量噪声（Plan 1）----
    d.coeff_mode       = 'analytic_truth';
    d.ident_apply_cmd  = true;
    d.sigma_q          = 0.1 * pi / 180;   % 0.1 deg
    d.sigma_qdot       = 0.001;            % rad/s
    d.sigma_theta      = 0.1 * pi / 180;
    d.sigma_psi        = 0.1 * pi / 180;
    d.sigma_r          = 0.0;              % m（2026-08-01：占位 10→0）
    d.sigma_rdot       = 0.0;              % m/s（2026-08-01：占位 1→0）
    d.kf_qj            = 1.0e4;
    d.kf_sigma_a0      = 50.0;
    d.rls_lambda       = 0.98;
    d.rls_mu           = 1.0e-3;           % 2026-08-01：曾误改为 0，已回退
    d.rls_P0           = 1.0e4;
    d.rls_r0           = 1.0e4;
    d.rls_a0           = 50.0;
    d.noise_seed       = 0;

    fn = fieldnames(d);
    for k = 1:numel(fn)
        if ~isfield(cfg_in, fn{k})
            cfg_in.(fn{k}) = d.(fn{k});
        end
    end
    cfg_in.coeff_mode = char(cfg_in.coeff_mode);
    cfg_in.ident_apply_cmd = logical(cfg_in.ident_apply_cmd);

    core = struct();
    core.cfg = cfg_in;
    core.H   = evasion_helpers();
    core.meta = meta0;
    % cfg 可覆盖 meta 中的动作映射版本（A/B 对照）
    core.meta.action_mapping_version = char(cfg_in.action_mapping_version);
    core.T_pos   = [0, 0, 0];
    core.T_state = [0, 0, 0, 0, 0, 0];
    core.max_ov_M = 8  * cfg_in.g0;
    core.max_ov_D = 10 * cfg_in.g0;
end
