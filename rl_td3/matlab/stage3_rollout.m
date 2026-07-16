function [r_minT, reached, diverged, n_steps] = stage3_rollout(state_abs, t0, cfg_in)
%STAGE3_ROLLOUT  规避段结束后，用固定 PN 一次跑完 stage3，返回真实终端脱靶量。
%
%   [r_minT, reached, diverged, n_steps] = stage3_rollout(state_abs, t0)
%   [r_minT, reached, diverged, n_steps] = stage3_rollout(state_abs, t0, cfg_in)
%
%   输入:
%     state_abs  [18×1] 规避段结束时绝对状态 [M(6); D1(6); D2(6)]
%     t0         当前仿真时刻 (s，从规避段起点算起；仅作时间起点)
%     cfg_in     可选配置（缺省见下方 defaults）
%
%   输出:
%     r_minT     全段最小弹目距离 (m)，即真实终端脱靶量代理
%     reached    是否到达目标附近 (r_MT < reach_radius)
%     diverged   是否因持续远离目标而早停
%     n_steps    实际积分步数
%
%   说明:
%     攻击弹与两防均用 PN；积分在 MATLAB 内一次完成（默认 dt=1e-3），
%     避免 Python↔Engine 逐步往返。精度由 dt 决定，与调用方式无关。

    d.KM = 3; d.KD1 = 4; d.KD2 = 4;
    d.dt = 1e-3;
    d.t_max = 60;
    d.g0 = 9.81;
    d.reach_radius = 50;
    d.tail_time = 0.3;
    d.diverge_time = 2.0;
    d.T = [0, 0, 0];
    if nargin < 3 || isempty(cfg_in); cfg_in = struct(); end
    fn = fieldnames(d);
    for k = 1:numel(fn)
        if ~isfield(cfg_in, fn{k}); cfg_in.(fn{k}) = d.(fn{k}); end
    end

    H = evasion_helpers();
    computeLOS   = H.computeLOS;
    computeMDDot = H.computeMDDot;
    computeMTDot = H.computeMTDot;
    computePN    = H.computePN;
    updateState  = H.updateState;
    saturate     = H.saturate;

    KM = cfg_in.KM; KD1 = cfg_in.KD1; KD2 = cfg_in.KD2;
    dt = cfg_in.dt; t_max = cfg_in.t_max; T = cfg_in.T;
    max_ov_M = 8 * cfg_in.g0;
    max_ov_D = 10 * cfg_in.g0;

    M  = state_abs(1:6)';
    D1 = state_abs(7:12)';
    D2 = state_abs(13:18)';

    t = double(t0);
    n = 0;
    r_minT = inf;
    reached = false; t_reach = NaN;
    diverged = false; t_away = NaN;
    r_MT_prev = inf;

    while (t - t0) < t_max
        n = n + 1;

        [q_theta, q_psi, r_MT] = computeLOS(M(1:3), T);
        [qy_D1M, qz_D1M, r_MD1] = computeLOS(D1(1:3), M(1:3));
        [qy_D2M, qz_D2M, r_MD2] = computeLOS(D2(1:3), M(1:3));

        [~, qd_y1, qd_z1] = computeMDDot(M, D1, r_MD1, qy_D1M, qz_D1M);
        [~, qd_y2, qd_z2] = computeMDDot(M, D2, r_MD2, qy_D2M, qz_D2M);
        [qd_theta, qd_psi] = computeMTDot(M, r_MT, q_theta, q_psi);

        a_M  = saturate(computePN(KM,  M(6),  qd_theta, qd_psi, M(4)),  max_ov_M);
        a_D1 = saturate(computePN(KD1, D1(6), qd_y1,    qd_z1,  D1(4)), max_ov_D);
        a_D2 = saturate(computePN(KD2, D2(6), qd_y2,    qd_z2,  D2(4)), max_ov_D);

        if r_MT < r_minT
            r_minT = r_MT;
        end

        if r_MT < cfg_in.reach_radius && ~reached
            reached = true;
            t_reach = t;
        end
        if reached && t > t_reach + cfg_in.tail_time
            break;
        end

        if ~reached && n >= 2 && r_MT > r_MT_prev
            if isnan(t_away); t_away = t; end
            if t - t_away > cfg_in.diverge_time
                diverged = true;
                break;
            end
        else
            t_away = NaN;
        end
        r_MT_prev = r_MT;

        M  = updateState(M,  a_M,  dt);
        D1 = updateState(D1, a_D1, dt);
        D2 = updateState(D2, a_D2, dt);
        t  = t + dt;
    end

    n_steps = n;
    if ~isfinite(r_minT)
        r_minT = r_MT_prev;
    end
end
