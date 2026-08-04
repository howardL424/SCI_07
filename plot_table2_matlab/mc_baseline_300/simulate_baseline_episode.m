function out = simulate_baseline_episode(M0, D10, D20, mode, cfg)
%SIMULATE_BASELINE_EPISODE 纯 PN / 正弦程序机动 单局闭环仿真（无全轨落盘）。
%
%   out = simulate_baseline_episode(M0, D10, D20, mode, cfg)
%
%   mode : 'pn' | 'sine'
%   cfg  : 可选结构体，缺省字段见 local default_cfg
%
%   测量噪声（可选，默认全关，与旧 MC 行为一致）:
%     cfg.enable_noise   true 时对攻击方制导用测量加噪；防御弹 PN 始终真值
%     cfg.sigma_q/qdot/r/rdot/theta/psi  与参数辨识默认一致（可先经 sigma_scale 缩放）
%     cfg.noise_seed     可复现种子（hash RNG，与 stage2_ident 同族）
%     cfg.noise_hold_dt  噪声采样保持周期，默认 0.1s（对齐 stage2 制导周期）
%     成功判据 / 拦截判定 / r_safe 触发一律用真值几何。
%
%   输出 out 字段：
%     r_MD1_min, r_MD2_min, r_minT, intercepted, evade_ok, strike_ok, both_ok, t_end

    if nargin < 5 || isempty(cfg)
        cfg = struct();
    end
    cfg = default_cfg(cfg, mode);

    T = cfg.T;
    dt = cfg.dt;
    t_max = cfg.t_max;
    r_safe = cfg.r_safe;
    kill_r = cfg.kill_radius;
    strike_ok_th = cfg.strike_ok;
    max_a_M = cfg.max_overload_M;
    max_a_D = cfg.max_overload_D;
    KM = cfg.KM;
    KD1 = cfg.KD1;
    KD2 = cfg.KD2;
    Aw = cfg.Aw;
    ff2 = cfg.ff2;
    cpa_hold = cfg.cpa_hold_steps;
    use_noise = logical(cfg.enable_noise);
    noise_hold = cfg.noise_hold_dt;

    is_sine = strcmpi(mode, 'sine');

    t = 0;
    M = M0(:).';
    D1 = D10(:).';
    D2 = D20(:).';

    r_MD1_min = inf;
    r_MD2_min = inf;
    r_minT = inf;
    intercepted = false;
    penetration_started = false;
    t_start_penetration = 0;
    t_penetrat_end = 1000;
    f1 = 0;
    f2 = 0;
    flag_cpa = 0;
    r_MT_prev = inf;
    stage = 1;

    % 噪声状态：按 noise_hold_dt 重采样并保持（避免 dt=1ms 时等价噪声过强）
    noise = struct('draw', 0, 't_next', -inf, ...
        'n_MT', zeros(1, 4), 'n_MD1', zeros(1, 4), 'n_MD2', zeros(1, 4));

    while t < t_max
        pos_M = M(1:3); theta_M = M(4); V_M = M(6);
        pos_D1 = D1(1:3); theta_D1 = D1(4); V_D1 = D1(6);
        pos_D2 = D2(1:3); theta_D2 = D2(4); V_D2 = D2(6);

        [q_theta, q_psi, r_MT] = computeLOS(pos_M, T);
        [q_y_MD1, q_z_MD1, r_MD1] = computeLOS(pos_D1, pos_M);
        [q_y_MD2, q_z_MD2, r_MD2] = computeLOS(pos_D2, pos_M);

        [r_dot_MD1, q_dot_y_MD1, q_dot_z_MD1] = computeMDDot(M, D1, r_MD1, q_y_MD1, q_z_MD1);
        [r_dot_MD2, q_dot_y_MD2, q_dot_z_MD2] = computeMDDot(M, D2, r_MD2, q_y_MD2, q_z_MD2);
        [q_dot_theta, q_dot_psi] = computeMTDot(M, r_MT, q_theta, q_psi);

        r_MD1_min = min(r_MD1_min, r_MD1);
        r_MD2_min = min(r_MD2_min, r_MD2);
        if r_MT < r_minT
            r_minT = r_MT;
        end

        if min(r_MD1, r_MD2) < kill_r
            intercepted = true;
            break;
        end

        if ~penetration_started && min(r_MD1, r_MD2) <= r_safe
            penetration_started = true;
            t_start_penetration = t;
        end

        % 防御弹 PN：始终真值几何（与 stage2_step 约定一致）
        a_D1_cmd = saturate(computePN(KD1, V_D1, q_dot_y_MD1, q_dot_z_MD1, theta_D1), max_a_D);
        a_D2_cmd = saturate(computePN(KD2, V_D2, q_dot_y_MD2, q_dot_z_MD2, theta_D2), max_a_D);

        % 攻击方制导用测量（可选加噪）；姿态角噪声只进 PN 的 cos(theta) 项
        qdt = q_dot_theta; qdp = q_dot_psi; theta_M_g = theta_M;
        r1_g = r_MD1; rd1_g = r_dot_MD1;
        r2_g = r_MD2; rd2_g = r_dot_MD2;
        if use_noise
            if t >= noise.t_next
                [noise, noise.n_MT]  = sample_channel_noise(noise, cfg);
                [noise, noise.n_MD1] = sample_channel_noise(noise, cfg);
                [noise, noise.n_MD2] = sample_channel_noise(noise, cfg);
                [noise, n_th] = hash_randn(noise, cfg.noise_seed, cfg.sigma_theta);
                noise.n_theta_M = n_th;
                noise.t_next = t + noise_hold;
            end
            % n = [r, rdot, qdy, qdz]
            qdt = q_dot_theta + noise.n_MT(3);
            qdp = q_dot_psi   + noise.n_MT(4);
            r1_g  = r_MD1 + noise.n_MD1(1);
            rd1_g = r_dot_MD1 + noise.n_MD1(2);
            r2_g  = r_MD2 + noise.n_MD2(1);
            rd2_g = r_dot_MD2 + noise.n_MD2(2);
            theta_M_g = theta_M + noise.n_theta_M;
        end

        if ~is_sine
            % 纯 PN：全程对目标比例导引（对齐 main_chunPN_0412_duo）
            a_M_cmd = computePN(KM, V_M, qdt, qdp, theta_M_g);
            stage = 1;
        else
            % 正弦程序机动（对齐 main_zhengxian_0412_duo）
            % 交会时刻估计可用带噪 r/rdot；开环正弦段本身不吃视线测量
            r_dot_s1 = sign(rd1_g) * max(abs(rd1_g), 1e-3);
            r_dot_s2 = sign(rd2_g) * max(abs(rd2_g), 1e-3);
            t_f1 = t - r1_g / r_dot_s1;
            t_f2 = t - r2_g / r_dot_s2;
            if ~penetration_started
                a_M_cmd = computePN(KM, V_M, qdt, qdp, theta_M_g);
                stage = 1;
            elseif t <= max(t_f1, t_f2)
                tau = t - t_start_penetration;
                a_M_cmd = [Aw * cos(2*pi*ff2*tau); Aw * sin(2*pi*ff2*tau)];
                stage = 2;
                t_penetrat_end = t;
            else
                a_M_cmd = computePN(KM, V_M, qdt, qdp, theta_M_g);
                stage = 3;
            end
        end
        a_M_cmd = saturate(a_M_cmd, max_a_M);

        M = updateState(M, a_M_cmd, dt);
        D1 = updateState(D1, a_D1_cmd, dt);
        D2 = updateState(D2, a_D2_cmd, dt);

        if r_dot_MD1 > 0 && f1 == 0
            f1 = 1;
        end
        if r_dot_MD2 > 0 && f2 == 0
            f2 = 1;
        end

        % 过 CPA 判据：双防已交会后（或正弦已回 PN），弹目距离连续增大
        past_evade = (min(f1, f2) == 1);
        if is_sine
            past_evade = past_evade || (stage == 3);
        end
        if past_evade && r_MT > r_MT_prev
            flag_cpa = flag_cpa + 1;
        else
            flag_cpa = 0;
        end
        if flag_cpa >= cpa_hold
            break;
        end
        r_MT_prev = r_MT;

        t = t + dt;
    end

    if ~isfinite(r_minT)
        r_minT = NaN;
    end

    evade_ok = (~intercepted) && (r_MD1_min >= kill_r) && (r_MD2_min >= kill_r);
    strike_ok = (~intercepted) && isfinite(r_minT) && (r_minT <= strike_ok_th);
    both_ok = evade_ok && strike_ok;

    out = struct( ...
        'r_MD1_min', r_MD1_min, ...
        'r_MD2_min', r_MD2_min, ...
        'r_minT', r_minT, ...
        'intercepted', intercepted, ...
        'evade_ok', evade_ok, ...
        'strike_ok', strike_ok, ...
        'both_ok', both_ok, ...
        't_end', t, ...
        'mode', lower(char(mode)));
end

%% ===== 缺省参数 =====
function cfg = default_cfg(cfg, mode)
    if ~isfield(cfg, 'T'), cfg.T = [0, 0, 0]; end
    if ~isfield(cfg, 'dt'), cfg.dt = 0.001; end
    if ~isfield(cfg, 't_max'), cfg.t_max = 35; end
    if ~isfield(cfg, 'r_safe'), cfg.r_safe = 8000; end
    if ~isfield(cfg, 'kill_radius'), cfg.kill_radius = 6.0; end
    if ~isfield(cfg, 'strike_ok'), cfg.strike_ok = 10.0; end
    if ~isfield(cfg, 'max_overload_M'), cfg.max_overload_M = 8 * 9.81; end
    if ~isfield(cfg, 'max_overload_D'), cfg.max_overload_D = 10 * 9.81; end
    if ~isfield(cfg, 'KD1'), cfg.KD1 = 4; end
    if ~isfield(cfg, 'KD2'), cfg.KD2 = 4; end
    if ~isfield(cfg, 'Aw'), cfg.Aw = 54; end
    if ~isfield(cfg, 'ff2'), cfg.ff2 = 0.15; end
    if ~isfield(cfg, 'cpa_hold_steps'), cfg.cpa_hold_steps = 100; end
    if ~isfield(cfg, 'KM') || isempty(cfg.KM)
        if strcmpi(mode, 'sine')
            cfg.KM = 3;   % main_zhengxian_0412_duo
        else
            cfg.KM = 4;   % main_chunPN_0412_duo
        end
    end
    % ---- 测量噪声（默认关闭，旧 MC 行为不变）----
    if ~isfield(cfg, 'enable_noise'), cfg.enable_noise = false; end
    if ~isfield(cfg, 'noise_seed'), cfg.noise_seed = 0; end
    if ~isfield(cfg, 'noise_hold_dt'), cfg.noise_hold_dt = 0.1; end
    sigma_q = 0.1 * pi / 180;
    if ~isfield(cfg, 'sigma_q'), cfg.sigma_q = sigma_q; end
    if ~isfield(cfg, 'sigma_qdot'), cfg.sigma_qdot = 0.001; end
    if ~isfield(cfg, 'sigma_theta'), cfg.sigma_theta = sigma_q; end
    if ~isfield(cfg, 'sigma_psi'), cfg.sigma_psi = sigma_q; end
    if ~isfield(cfg, 'sigma_r'), cfg.sigma_r = 0.0; end
    if ~isfield(cfg, 'sigma_rdot'), cfg.sigma_rdot = 0.0; end
end

%% ===== 测量噪声（与 stage2_ident_step 同族 hash RNG）=====
function [noise, n] = sample_channel_noise(noise, cfg)
% n = [r, rdot, qdy, qdz]
    [noise, n1] = hash_randn(noise, cfg.noise_seed, cfg.sigma_r);
    [noise, n2] = hash_randn(noise, cfg.noise_seed, cfg.sigma_rdot);
    [noise, n3] = hash_randn(noise, cfg.noise_seed, cfg.sigma_qdot);
    [noise, n4] = hash_randn(noise, cfg.noise_seed, cfg.sigma_qdot);
    n = [n1, n2, n3, n4];
end

function [noise, x] = hash_randn(noise, seed, sigma)
    noise.draw = noise.draw + 1;
    u1 = hash01(seed, noise.draw, 1);
    u2 = hash01(seed, noise.draw, 2);
    u1 = max(u1, 1e-12);
    z = sqrt(-2 * log(u1)) * cos(2 * pi * u2);
    x = sigma * z;
end

function u = hash01(seed, draw, lane)
    v = double(seed) * 1.0e-3 + double(draw) * 17.0 + double(lane) * 0.37;
    u = abs(sin(v * 12.9898) * 43758.5453);
    u = u - floor(u);
end

%% ===== 动力学 / 几何（与 table2 制导脚本一致）=====
function [q_y, q_z, r] = computeLOS(pos1, pos2)
    dx = pos2(1) - pos1(1);
    dy = pos2(2) - pos1(2);
    dz = pos2(3) - pos1(3);
    r = max(norm([dx, dy, dz]), 1e-3);
    sin_qy = max(min(dy / r, 1), -1);
    q_y = asin(sin_qy);
    q_z = -atan2(dz, dx);
end

function a_pn = computePN(K, V, q_dot_y, q_dot_z, theta)
    a_pn = [K * V * q_dot_y; -K * V * q_dot_z * cos(theta)];
end

function state_new = updateState(state, a_cmd, dt)
    x = state(1); y = state(2); z = state(3);
    theta = state(4); psi_v = state(5); V = state(6);
    a_y = a_cmd(1); a_z = a_cmd(2);
    if abs(V) < 1e-6
        state_new = state;
        return;
    end
    deriv = @(s) [ ...
        V * cos(s(4)) * cos(s(5));
        V * sin(s(4));
        -V * cos(s(4)) * sin(s(5));
        a_y / V;
        -a_z / (V * cos(s(4)))];
    s = [x; y; z; theta; psi_v];
    k1 = deriv(s);
    k2 = deriv(s + 0.5*dt*k1);
    k3 = deriv(s + 0.5*dt*k2);
    k4 = deriv(s + dt*k3);
    s_new = s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    state_new = [s_new(1:5)', V];
end

function a_sat = saturate(a, limit)
    if any(~isfinite(a))
        a_sat = [0; 0];
        return;
    end
    nrm = norm(a);
    if nrm > limit
        a_sat = a / nrm * limit;
    else
        a_sat = a;
    end
end

function [r_dot, q_dot_y, q_dot_z] = computeMDDot(pos1, pos2, r, q_y, q_z)
    V_M = pos1(6); theta_M = pos1(4); psi_vM = pos1(5);
    V_D = pos2(6); theta_D = pos2(4); psi_vD = pos2(5);

    term_D = V_D * (cos(theta_D) * cos(q_y) * cos(q_z - psi_vD) + sin(theta_D) * sin(q_y));
    term_M = V_M * (cos(theta_M) * cos(q_y) * cos(q_z - psi_vM) + sin(theta_M) * sin(q_y));
    r_dot = -term_D + term_M;

    if abs(r) < 1e-6
        q_dot_y = 0;
    else
        term_D_qy = V_D * (cos(theta_D) * sin(q_y) * cos(q_z - psi_vD) - sin(theta_D) * cos(q_y));
        term_M_qy = V_M * (cos(theta_M) * sin(q_y) * cos(q_z - psi_vM) - sin(theta_M) * cos(q_y));
        q_dot_y = (term_D_qy - term_M_qy) / r;
    end

    cos_qy_safe = max(abs(cos(q_y)), 1e-5);
    r_safe_den = max(abs(r), 1e-3);
    term_D_qz = V_D * cos(theta_D) * sin(q_z - psi_vD);
    term_M_qz = V_M * cos(theta_M) * sin(q_z - psi_vM);
    q_dot_z = (term_D_qz - term_M_qz) / (r_safe_den * cos_qy_safe);
end

function [q_dot_theta, q_dot_psi] = computeMTDot(pos1, r_MT, q_theta, q_psi)
    V_M = pos1(6); theta_M = pos1(4); psi_vM = pos1(5);
    if abs(r_MT) < 1e-6
        q_dot_theta = 0;
        q_dot_psi = 0;
        return;
    end
    term1 = cos(theta_M) * sin(q_theta) * cos(q_psi - psi_vM);
    term2 = -sin(theta_M) * cos(q_theta);
    q_dot_theta = V_M * (term1 + term2) / r_MT;
    if abs(cos(q_theta)) < 1e-6
        q_dot_psi = 0;
    else
        q_dot_psi = (V_M * cos(theta_M) * sin(q_psi - psi_vM)) / (r_MT * cos(q_theta));
    end
end
