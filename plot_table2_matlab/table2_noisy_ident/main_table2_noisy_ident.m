clc; clear; close all;
%% Table-2 初值：纯 TD3 / BC-TD3 在参数辨识测量噪声下的轨迹复算与出图（1 倍噪声）
%
% 噪声强度取自参数辨识默认值（rl_td3/ident_cfg.py、stage2_core_init.m）：
%   σ_q = σ_θ = σ_ψ = 0.1°，σ_q̇ = 0.001 rad/s，σ_r = σ_ṙ = 0
%
% 输出:
%   results/record_table2_<algo>_noisy.mat
%   figures/trajectory_<algo>_noisy.{png,pdf,fig}            三维轨迹 + D1/D2 放大图
%   figures/trajectory_<algo>_truth_vs_noisy.{png,pdf,fig}   与原真值系数轨迹对照
%
% 其他噪声强度见 main_table2_noisy_ident_x3.m，或直接调 run_table2_noisy_ident。
%
% 运行: cd('...\GAIL\plot_table2_matlab\table2_noisy_ident'); main_table2_noisy_ident

addpath(fileparts(mfilename('fullpath')));

opts = struct();
opts.preset      = 'plan1';            % 'plan1' | 'range' | 'clean'
opts.sigma_scale = 1;
opts.noise_seed  = 0;
opts.coeff_mode  = 'analytic_noisy';   % 'analytic_noisy' | 'sparse_ffrls'
opts.obs_noise   = false;              % true 时策略观测同样加噪（Plan 1 默认干净）

run_table2_noisy_ident(opts);
