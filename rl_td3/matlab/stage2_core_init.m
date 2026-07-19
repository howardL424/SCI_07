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
%     .cfg   合并后的仿真/奖励配置
%     .H     evasion_helpers() 句柄包
%     .meta  transition_schema_meta() 版本元数据
%     .T_pos / .T_state  静止目标

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

    fn = fieldnames(d);
    for k = 1:numel(fn)
        if ~isfield(cfg_in, fn{k})
            cfg_in.(fn{k}) = d.(fn{k});
        end
    end

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
