function ident = stage2_ident_init(cfg)
%STAGE2_IDENT_INIT  初始化三通道 Kalman + sparse FF-RLS 状态。
%
%   ident = stage2_ident_init(cfg)
%
%   cfg 字段（可由 stage2_core_init 填默认值）:
%     .dt .g0
%     .sigma_r .sigma_rdot .sigma_q .sigma_qdot .sigma_theta .sigma_psi
%     .kf_qj .kf_sigma_a0 .kf_sigma_r .kf_sigma_rdot
%     .rls_lambda .rls_mu .rls_eta .rls_P0 .rls_r0 .rls_a0
%     .noise_seed
%
%   通道: D1/D2 估 [A21;B1;B2;C1]，T 估 [A21;B1;B2]。

    if nargin < 1 || isempty(cfg); cfg = struct(); end

    dt = local_get(cfg, 'dt', 0.1);
    g0 = local_get(cfg, 'g0', 9.81);

    ident = struct();
    ident.dt = dt;
    ident.g0 = g0;
    ident.step_count = 0;
    ident.has_prev = false;
    ident.a_M_sat_prev = [0; 0];
    ident.noise_seed = local_get(cfg, 'noise_seed', 0);
    ident.noise_draw = 0;

    ident.sigma_scale = local_get(cfg, 'sigma_scale', 30.0);
    ident.sigma_r     = local_get(cfg, 'sigma_r', 0.0);
    ident.sigma_rdot  = local_get(cfg, 'sigma_rdot', 0.0);
    ident.sigma_q     = local_get(cfg, 'sigma_q', 3.0 * pi / 180);
    ident.sigma_qdot  = local_get(cfg, 'sigma_qdot', 0.03);
    ident.sigma_theta = local_get(cfg, 'sigma_theta', ident.sigma_q);
    ident.sigma_psi   = local_get(cfg, 'sigma_psi', ident.sigma_q);

    ident.kf_qj       = local_get(cfg, 'kf_qj', 1.0e4);      % jerk intensity
    ident.kf_sigma_a0 = local_get(cfg, 'kf_sigma_a0', 50.0); % m/s^2
    ident.kf_sigma_r  = local_get(cfg, 'kf_sigma_r', 10.0);  % KF 设计标准差，不注入量测
    ident.kf_sigma_rdot = local_get(cfg, 'kf_sigma_rdot', 0.5);

    ident.rls_lambda = local_get(cfg, 'rls_lambda', 0.98);
    ident.rls_mu     = local_get(cfg, 'rls_mu', 1.0e-3);
    ident.rls_eta    = local_get(cfg, 'rls_eta', 0.1);
    ident.rls_P0     = local_get(cfg, 'rls_P0', 1.0e4);
    ident.rls_r0     = local_get(cfg, 'rls_r0', 1.0e4);    % m
    ident.rls_a0     = local_get(cfg, 'rls_a0', 50.0);      % m/s^2
    ident.KD1        = local_get(cfg, 'KD1', 4.0);
    ident.KD2        = local_get(cfg, 'KD2', 4.0);
    ident.max_ov_D   = 10.0 * g0;

    if ~(isscalar(ident.rls_eta) && isfinite(ident.rls_eta) && ident.rls_eta >= 0)
        error('stage2_ident_init:BadEta', 'rls_eta 须为非负有限标量');
    end
    if ~(ident.kf_sigma_r > 0 && ident.kf_sigma_rdot > 0)
        error('stage2_ident_init:BadKfR', ...
            'kf_sigma_r 与 kf_sigma_rdot 须为正数（与实际量测噪声独立）');
    end

    F = [1, dt, 0.5*dt^2;
         0, 1,  dt;
         0, 0,  1];
    Qxi = ident.kf_qj * [ ...
        dt^5/20, dt^4/8, dt^3/6;
        dt^4/8,  dt^3/3, dt^2/2;
        dt^3/6,  dt^2/2, dt ];
    Rxi = diag([ident.kf_sigma_r^2, ident.kf_sigma_rdot^2]);
    Hxi = [1, 0, 0; 0, 1, 0];

    ident.F_xi = F;
    ident.Q_xi = Qxi;
    ident.R_xi = Rxi;
    ident.H_xi = Hxi;

    names = {'D1', 'D2', 'T'};
    nths  = [4, 4, 3];
    ch = struct([]);
    for i = 1:3
        ch(i).name = names{i}; %#ok<AGROW>
        ch(i).n_theta = nths(i);
        ch(i).theta = zeros(nths(i), 1);
        ch(i).P = ident.rls_P0 * eye(nths(i));
        ch(i).xi = zeros(3, 1);
        ch(i).P_xi = diag([ident.kf_sigma_r^2, ident.kf_sigma_rdot^2, ident.kf_sigma_a0^2]);
        ch(i).r_hat_prev = 0;
        ch(i).r_meas_prev = 0;
        ch(i).initialized = false;
    end
    ident.ch = ch;
end

function v = local_get(s, name, default)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default;
    end
end
