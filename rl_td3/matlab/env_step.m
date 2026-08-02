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
%     info_vec       [9×1] 基础；sparse_ffrls / analytic_noisy 时扩展为 [32×1]
%       [r_MD1; r_MD2; r_MT; intercepted; stage2_done;
%        a_M_sat_y; a_M_sat_z; a_M_raw_y; a_M_raw_z]
%       （后 4 项供轨迹落盘/画图；旧脚本只读前 5 项仍兼容）
%       扩展（仅辨识路径，旧脚本忽略）:
%         [10:12]  r_err D1/D2/T
%         [13:15]  rdot_err
%         [16:18]  rddot_err（相对真值 rdot 有限差分）
%         [19:29]  theta_err 共 11 维 (D1:4 + D2:4 + T:3)
%         [30]     did_rls
%         [31:32]  a_cmd_err_y / a_cmd_err_z（估参指令 − 本拍采用指令）
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

    persistent core cfg_key ident_state
    key = cfg_cache_key(cfg_in);
    if isempty(core) || ~strcmp(cfg_key, key)
        core = stage2_core_init(cfg_in);
        cfg_key = key;
        ident_state = [];  % 配置变化时清空辨识状态，防跨局泄漏
    end

    % 新 episode（t=0）重置辨识器，保证不同局之间不泄漏
    need_ident = isfield(core.cfg, 'coeff_mode') && ...
        ~strcmp(char(core.cfg.coeff_mode), 'analytic_truth');
    if need_ident
        if isempty(ident_state) || abs(double(t)) < 1e-15
            ident_state = stage2_ident_init(core.cfg);
        end
        [obs_next, state_abs_next, ~, fact, ident_state] = ...
            stage2_step(core, state_abs, t, Tev, action_norm, ident_state);
    else
        [obs_next, state_abs_next, ~, fact] = ...
            stage2_step(core, state_abs, t, Tev, action_norm);
    end

    reward     = stage2_process_reward(core, fact);
    terminated = fact.terminated;
    a_sat = fact.a_M_sat(:);
    a_raw = fact.a_M(:);
    info_vec   = [fact.r_MD1; fact.r_MD2; fact.r_MT; ...
                  double(fact.intercepted); double(fact.stage2_done); ...
                  a_sat(1); a_sat(2); a_raw(1); a_raw(2)];
    if isfield(fact, 'ident_diag') && isstruct(fact.ident_diag) && ...
            isfield(fact.ident_diag, 'r_hat')
        info_vec = [info_vec; local_pack_ident_diag(fact.ident_diag)]; %#ok<AGROW>
    end
end

function v = local_pack_ident_diag(d)
% 固定 23×1：KF 误差 9 + θ 误差 11 + did_rls + a_cmd_err 2
    r_err = d.r_hat(:) - d.r_true(:);
    rd_err = d.rdot_hat(:) - d.rdot_true(:);
    rdd_err = d.rddot_hat(:) - d.rddot_true(:);
    rdd_err(~isfinite(rdd_err)) = NaN;
    th_e = [];
    for i = 1:3
        th_e = [th_e; d.theta_hat{i}(:) - d.theta_true{i}(:)]; %#ok<AGROW>
    end
    if isfield(d, 'a_cmd_err') && numel(d.a_cmd_err) >= 2
        a_err = d.a_cmd_err(1:2);
        a_err = a_err(:);
    else
        a_err = [NaN; NaN];
    end
    v = [r_err; rd_err; rdd_err; th_e; double(d.did_rls); a_err];
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
    % 辨识 / 测量噪声相关字段
    id_names = { ...
        'coeff_mode','ident_apply_cmd','noise_seed', ...
        'sigma_q','sigma_qdot','sigma_theta','sigma_psi','sigma_r','sigma_rdot', ...
        'kf_qj','kf_sigma_a0','rls_lambda','rls_mu','rls_P0','rls_r0','rls_a0'};
    for i = 1:numel(id_names)
        n = id_names{i};
        if ~isfield(cfg_in, n) || isempty(cfg_in.(n))
            parts{end+1} = sprintf('%s=default', n); %#ok<AGROW>
        elseif ischar(cfg_in.(n)) || isstring(cfg_in.(n))
            parts{end+1} = sprintf('%s=%s', n, char(cfg_in.(n))); %#ok<AGROW>
        elseif islogical(cfg_in.(n))
            parts{end+1} = sprintf('%s=%d', n, double(cfg_in.(n))); %#ok<AGROW>
        else
            parts{end+1} = sprintf('%s=%g', n, double(cfg_in.(n))); %#ok<AGROW>
        end
    end
    key = strjoin(parts, ';');
end
