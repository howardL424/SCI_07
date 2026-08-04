clc; clear; close all;
%% Table-2 初值：纯 TD3 / BC-TD3 在 3 倍参数辨识测量噪声下的轨迹复算与出图
%
% 相对 main_table2_noisy_ident.m 只把所有 σ 放大 3 倍，其余（初值、动力学、
% 三段结构、KF/RLS 超参、种子）完全一致，便于与 1 倍结果逐项对照：
%   σ_q = σ_θ = σ_ψ = 0.3°，σ_q̇ = 0.003 rad/s，σ_r = σ_ṙ = 0
%
% 输出:
%   results/sigma_x3/record_table2_<algo>_noisy.mat
%   figures/sigma_x3/trajectory_<algo>_noisy.{png,pdf,fig}
%   figures/sigma_x3/trajectory_<algo>_truth_vs_noisy.{png,pdf,fig}
%
% 运行: cd('...\GAIL\plot_table2_matlab\table2_noisy_ident'); main_table2_noisy_ident_x3

addpath(fileparts(mfilename('fullpath')));

opts = struct();
opts.preset      = 'plan1';
opts.sigma_scale = 3;
opts.noise_seed  = 0;
opts.coeff_mode  = 'analytic_noisy';
opts.obs_noise   = false;
opts.out_tag     = 'sigma_x3';

run_table2_noisy_ident(opts);
