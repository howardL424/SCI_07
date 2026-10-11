function [obs, source] = stage2_actor_obs_from_abs( ...
        core, state_abs, t, Tev, ident_prior)
%STAGE2_ACTOR_OBS_FROM_ABS  按显式量测口径构造冻结Actor的13维观测。
%
%   actor_obs_mode:
%     truth    - 历史兼容：13维均按原真值几何构造；
%     noisy    - 距离/距离率/LOS角使用本拍原始含噪量测；
%     filtered - 三通道距离及距离率使用KF后验，LOS角仍用含噪量测。
%
%   V_M视为机载自身状态可准确获得，tgo_norm为确定性时间量。传入的
%   ident_prior应为上一拍处理并提交控制后的辨识状态；函数只做预览，
%   不修改真实辨识状态。

    if nargin < 5; ident_prior = []; end
    truth_obs = stage2_obs_from_abs(core, state_abs, t, Tev);
    mode = local_mode(core.cfg);
    source = struct('mode', mode, 'measurement_step', NaN, ...
        'range_source', 'truth', 'los_source', 'truth');
    if strcmp(mode, 'truth')
        obs = truth_obs;
        return;
    end

    step_index = max(0, round(double(t) / max(double(core.cfg.dt), 1e-12)));
    noise = stage2_measurement_noise(core.cfg, step_index);
    M = double(state_abs(1:6))';
    [qy_TM, qz_TM, r_TM] = core.H.computeLOS(core.T_pos, M(1:3));
    [rdot_TM, ~, ~] = core.H.computeMDDot( ...
        M, core.T_state, r_TM, qy_TM, qz_TM);
    measured_r = [ ...
        truth_obs(1) + noise.D1.r; ...
        truth_obs(5) + noise.D2.r; ...
        truth_obs(9) + noise.T.r];
    measured_rdot = [ ...
        truth_obs(2) + noise.D1.rdot; ...
        truth_obs(6) + noise.D2.rdot; ...
        rdot_TM + noise.T.rdot];

    obs = truth_obs;
    obs(1) = measured_r(1);
    obs(2) = measured_rdot(1);
    obs(3) = truth_obs(3) + noise.D1.qy;
    obs(4) = truth_obs(4) + noise.D1.qz;
    obs(5) = measured_r(2);
    obs(6) = measured_rdot(2);
    obs(7) = truth_obs(7) + noise.D2.qy;
    obs(8) = truth_obs(8) + noise.D2.qz;
    obs(9) = measured_r(3);
    obs(10) = truth_obs(10) + noise.T.qy;
    obs(11) = truth_obs(11) + noise.T.qz;

    source.measurement_step = step_index;
    source.range_source = 'raw_measurement';
    source.los_source = 'raw_measurement';
    if strcmp(mode, 'filtered')
        [r_hat, rdot_hat] = local_filter_preview( ...
            ident_prior, measured_r, measured_rdot);
        obs(1) = r_hat(1);
        obs(2) = rdot_hat(1);
        obs(5) = r_hat(2);
        obs(6) = rdot_hat(2);
        obs(9) = r_hat(3);
        source.range_source = 'kf_posterior';
    end
end

function [r_hat, rdot_hat] = local_filter_preview(ident, measured_r, measured_rdot)
    r_hat = measured_r(:);
    rdot_hat = [measured_rdot(1:2); NaN];
    if isempty(ident) || ~isstruct(ident) || ~isfield(ident, 'ch')
        return;
    end
    for i = 1:3
        if numel(ident.ch) < i || ~ident.ch(i).initialized
            if i == 3
                rdot_hat(i) = 0;
            end
            continue;
        end
        z = [measured_r(i); measured_rdot(i)];
        xi_p = ident.F_xi * ident.ch(i).xi;
        P_p = ident.F_xi * ident.ch(i).P_xi * ident.F_xi' + ident.Q_xi;
        S = ident.H_xi * P_p * ident.H_xi' + ident.R_xi;
        K = (P_p * ident.H_xi') / S;
        xi = xi_p + K * (z - ident.H_xi * xi_p);
        r_hat(i) = xi(1);
        rdot_hat(i) = xi(2);
    end
end

function mode = local_mode(cfg)
    if isfield(cfg, 'actor_obs_mode') && ~isempty(cfg.actor_obs_mode)
        mode = char(cfg.actor_obs_mode);
    else
        mode = 'truth';
    end
    allowed = {'truth','noisy','filtered'};
    if ~any(strcmp(mode, allowed))
        error('stage2_actor_obs_from_abs:BadMode', ...
            'actor_obs_mode须为truth、noisy或filtered，收到%s', mode);
    end
end

