function noise = stage2_measurement_noise(cfg, step_index)
%STAGE2_MEASUREMENT_NOISE  按episode种子和拍号生成确定性同源量测噪声。
%
%   noise = stage2_measurement_noise(cfg, step_index)
%
%   每拍固定24个高斯样本，顺序与既有辨识实现一致：
%   D1/D2/T各[r,rdot,qy,qz,qdy,qdz]，再加6个姿态角样本。
%   Actor观测与辨识器调用本函数，可保证同IC、同seed、同拍使用同一噪声。

    if nargin < 2 || isempty(step_index); step_index = 0; end
    if ~(isscalar(step_index) && isfinite(step_index) && step_index >= 0)
        error('stage2_measurement_noise:BadStep', ...
            'step_index须为非负有限标量');
    end
    step_index = round(double(step_index));
    seed = local_get(cfg, 'noise_seed', 0);
    sig = struct( ...
        'r', local_get(cfg, 'sigma_r', 10.0), ...
        'rdot', local_get(cfg, 'sigma_rdot', 0.5), ...
        'qy', local_get(cfg, 'sigma_q', 3.0*pi/180), ...
        'qz', local_get(cfg, 'sigma_q', 3.0*pi/180), ...
        'qdy', local_get(cfg, 'sigma_qdot', 0.03), ...
        'qdz', local_get(cfg, 'sigma_qdot', 0.03));

    draw = step_index * 24;
    [noise.D1, draw] = local_channel(seed, draw, sig);
    [noise.D2, draw] = local_channel(seed, draw, sig);
    [noise.T, draw] = local_channel(seed, draw, sig);
    sigma_theta = local_get(cfg, 'sigma_theta', sig.qy);
    sigma_psi = local_get(cfg, 'sigma_psi', sig.qz);
    [noise.att.theta_M, draw] = local_randn(seed, draw, sigma_theta);
    [noise.att.psi_M, draw] = local_randn(seed, draw, sigma_psi);
    [noise.att.theta_D1, draw] = local_randn(seed, draw, sigma_theta);
    [noise.att.psi_D1, draw] = local_randn(seed, draw, sigma_psi);
    [noise.att.theta_D2, draw] = local_randn(seed, draw, sigma_theta);
    [noise.att.psi_D2, draw] = local_randn(seed, draw, sigma_psi);
    noise.draw_end = draw;
end

function [n, draw] = local_channel(seed, draw, sig)
    names = {'r','rdot','qy','qz','qdy','qdz'};
    n = struct();
    for i = 1:numel(names)
        name = names{i};
        [n.(name), draw] = local_randn(seed, draw, sig.(name));
    end
end

function [x, draw] = local_randn(seed, draw, sigma)
    draw = draw + 1;
    u1 = local_hash01(seed, draw, 1);
    u2 = local_hash01(seed, draw, 2);
    u1 = max(u1, 1e-12);
    z = sqrt(-2 * log(u1)) * cos(2 * pi * u2);
    x = double(sigma) * z;
end

function u = local_hash01(seed, draw, lane)
    v = double(seed) * 1.0e-3 + double(draw) * 17.0 + double(lane) * 0.37;
    u = abs(sin(v * 12.9898) * 43758.5453);
    u = u - floor(u);
end

function value = local_get(s, name, default)
    if isfield(s, name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = default;
    end
end

