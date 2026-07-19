function [obs_next, state_abs_next, reward, terminated, info_vec] = ...
        env_step(state_abs, t, Tev, zT0, action_norm, cfg_in)
%ENV_STEP  执行一步环境交互（dt=0.1s）：式47 制导 + RK4 更新 → 新 obs + reward。
%
%   [obs_next, state_abs_next, reward, terminated, info_vec] = ...
%       env_step(state_abs, t, Tev, zT0, action_norm)
%   [obs_next, state_abs_next, reward, terminated, info_vec] = ...
%       env_step(state_abs, t, Tev, zT0, action_norm, cfg_in)
%
%   输入:
%     state_abs   [18×1] 当前绝对状态 [M(6); D1(6); D2(6)]
%     t           当前仿真时刻 (s，从规避段起点算起)
%     Tev         规避段总时长估计 (s)
%     zT0         保留参数（兼容旧接口；新奖励不再使用）
%     action_norm [3×1] 归一化动作 [gamma_n, wT_n, wE_n] ∈ [-1,1]
%     cfg_in      可选配置
%
%   输出:
%     obs_next       [13×1] 新状态向量
%     state_abs_next [18×1] 新绝对状态
%     reward         标量过程奖励（终端 R1/+100 由 Python 在 stage3 后追加）
%     terminated     逻辑，本 episode（规避段）是否结束
%     info_vec       [5×1] [r_MD1; r_MD2; r_MT; intercepted; stage2_done]
%
%   动力学实现委托 stage2_core_init + stage2_step（与专家抽取共用）。
%   本函数对 helper/config 做 persistent 缓存，避免逐步重建。
%
%   过程奖励（每步）:
%     R2 = log10(min(r_MD1,d*)-5) if r_MD1>6 else 0
%     R3 = log10(min(r_MD2,d*)-5) if r_MD2>6 else 0
%     reward = 0.05*(R2+R3)
%   被拦截 (min(r_MD)≤6): reward=-50，过程项归零，terminated=true
%   规避段结束 (t≥Tev): terminated=true，由 Python 调 stage3_rollout 结算 R1/+100
%
%   动作反归一化见 action_norm_to_phys / transition_schema_meta.action_mapping_version
%   （默认 ratio_log_box_v1: [ρE_n, ρT_n, γ_n]）。

    %#ok<*NASGU>
    if nargin < 6 || isempty(cfg_in); cfg_in = struct(); end
    if nargin >= 4 && ~isempty(zT0)
        % zT0 保留兼容，不参与新奖励
    end

    persistent core cfg_key
    key = cfg_cache_key(cfg_in);
    if isempty(core) || ~strcmp(cfg_key, key)
        core = stage2_core_init(cfg_in);
        cfg_key = key;
    end

    [obs_next, state_abs_next, ~, fact] = ...
        stage2_step(core, state_abs, t, Tev, action_norm);

    reward     = stage2_process_reward(core, fact);
    terminated = fact.terminated;
    info_vec   = [fact.r_MD1; fact.r_MD2; fact.r_MT; ...
                  double(fact.intercepted); double(fact.stage2_done)];
end

function key = cfg_cache_key(cfg_in)
% 用稳定字段串作 cache key；缺省字段由 stage2_core_init 填默认值。
    names = {'dt','dstar','KD1','KD2','KM','g0','kill_radius','proc_coef','intercept_pen','w'};
    parts = cell(1, numel(names));
    for i = 1:numel(names)
        n = names{i};
        if isfield(cfg_in, n)
            parts{i} = sprintf('%s=%g', n, double(cfg_in.(n)));
        else
            parts{i} = sprintf('%s=default', n);
        end
    end
    if isfield(cfg_in, 'action_mapping_version') && ~isempty(cfg_in.action_mapping_version)
        parts{end+1} = sprintf('amap=%s', char(cfg_in.action_mapping_version)); %#ok<AGROW>
    else
        parts{end+1} = 'amap=default'; %#ok<AGROW>
    end
    key = strjoin(parts, ';');
end
