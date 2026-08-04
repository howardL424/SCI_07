function ncfg = noisy_ident_cfg(preset, seed, sigma_scale)
%NOISY_IDENT_CFG  Table-2 带噪轨迹复算所用的测量噪声 / KF / RLS 配置。
%
%   ncfg = noisy_ident_cfg()
%   ncfg = noisy_ident_cfg(preset)
%   ncfg = noisy_ident_cfg(preset, seed)
%   ncfg = noisy_ident_cfg(preset, seed, sigma_scale)
%
%   数值与 rl_td3/ident_cfg.py、rl_td3/matlab/stage2_core_init.m 中的参数辨识
%   默认值保持一致，便于两边结果直接对照。
%
%   preset:
%     'plan1'  (默认) 现行辨识默认值：σ_q=σ_θ=σ_ψ=0.1°，σ_q̇=0.001 rad/s，
%                     σ_r=σ_ṙ=0（2026-08-01 起距离通道无噪）
%     'range'         Plan 1 历史占位值：在 'plan1' 基础上 σ_r=10 m、σ_ṙ=1 m/s，
%                     KF 不退化，可用于观察距离量测噪声的影响
%     'clean'         全部 σ=0，用于验证带噪管线在零噪声下退化为解析真值
%
%   seed: 测量噪声可复现种子（hash RNG），默认 0。
%
%   sigma_scale: 所有 σ 的统一倍率，默认 1。KF 的 R 阵由 stage2_ident_init 按
%                同一组 σ 重建，因此滤波器与真实噪声强度保持匹配；kf/rls 超参不缩放。
%
%   返回字段可直接合并进 stage2_core_init 的 cfg。
%
%   See also SIMULATE_TABLE2_NOISY_EPISODE, STAGE2_IDENT_INIT.

if nargin < 1 || isempty(preset)
    preset = 'plan1';
end
if nargin < 2 || isempty(seed)
    seed = 0;
end
if nargin < 3 || isempty(sigma_scale)
    sigma_scale = 1;
end
preset = lower(char(preset));
sigma_scale = double(sigma_scale);
if ~isscalar(sigma_scale) || ~isfinite(sigma_scale) || sigma_scale < 0
    error('noisy_ident_cfg:BadScale', 'sigma_scale 须为非负有限标量');
end

sigma_q = 0.1 * pi / 180;   % 0.1 deg

ncfg = struct();
ncfg.preset      = preset;
ncfg.sigma_q     = sigma_q;
ncfg.sigma_qdot  = 0.001;
ncfg.sigma_theta = sigma_q;
ncfg.sigma_psi   = sigma_q;
ncfg.sigma_r     = 0.0;
ncfg.sigma_rdot  = 0.0;

ncfg.kf_qj       = 1.0e4;
ncfg.kf_sigma_a0 = 50.0;
ncfg.rls_lambda  = 0.98;
ncfg.rls_mu      = 1.0e-3;
ncfg.rls_P0      = 1.0e4;
ncfg.rls_r0      = 1.0e4;
ncfg.rls_a0      = 50.0;
ncfg.noise_seed  = double(seed);

switch preset
    case 'plan1'
        % 已是默认值
    case 'range'
        ncfg.sigma_r    = 10.0;
        ncfg.sigma_rdot = 1.0;
    case 'clean'
        ncfg.sigma_q     = 0.0;
        ncfg.sigma_qdot  = 0.0;
        ncfg.sigma_theta = 0.0;
        ncfg.sigma_psi   = 0.0;
        ncfg.sigma_r     = 0.0;
        ncfg.sigma_rdot  = 0.0;
    otherwise
        error('noisy_ident_cfg:UnknownPreset', ...
            '未知 preset=%s（可选 plan1 / range / clean）', preset);
end

sigma_fields = {'sigma_q', 'sigma_qdot', 'sigma_theta', 'sigma_psi', ...
    'sigma_r', 'sigma_rdot'};
for k = 1:numel(sigma_fields)
    ncfg.(sigma_fields{k}) = sigma_scale * ncfg.(sigma_fields{k});
end
ncfg.sigma_scale = sigma_scale;
end
