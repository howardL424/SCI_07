function meta = transition_schema_meta()
%TRANSITION_SCHEMA_META  物理 transition schema / 动力学 / 动作盒 / 观测归一化版本。
%
%   权威物理层字段见 BC_TD3_USAGE.md「物理 transition schema」。
%   变更 stage2 公式 / 动作映射 / OBS_SCALE 时必须 bump 对应 version。

    meta = struct();
    meta.schema_version          = 'phys_transition_v1';
    meta.stage2_dynamics_version = 'stage2_rk4_eq47_v1';
    meta.action_mapping_version  = 'td3_box_log_v1';
    meta.obs_norm_version        = 'obs_scale_0714_v1';
end
