function obs_n = normalize_obs_scale(obs)
%NORMALIZE_OBS_SCALE  13 维物理观测 / OBS_SCALE（与 obs_norm.py 一致，不裁剪）。
%
%   obs_n = normalize_obs_scale(obs)
%
%   OBS_SCALE (obs_scale_0714_v1):
%     [3e4, 1.2e3, pi/2, pi, 3e4, 1.2e3, pi/2, pi, 1.5e4, pi/2, pi, 1e3, 1]

    scale = [3.0e4; 1.2e3; pi/2; pi; ...
             3.0e4; 1.2e3; pi/2; pi; ...
             1.5e4; pi/2; pi; 1.0e3; 1.0];
    x = double(obs(:));
    if numel(x) ~= 13
        error('normalize_obs_scale:Dim', '观测须为 13 维，收到 %d', numel(x));
    end
    obs_n = x ./ scale;
end
