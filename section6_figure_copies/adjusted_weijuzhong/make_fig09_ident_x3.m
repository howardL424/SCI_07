% make_fig09_ident_x3.m
% Draw fig09: BC-TD3 with online sparse FF-RLS identification under
% 3x measurement noise (same noise preset/seed/scale as fig10, but
% coeff_mode = 'sparse_ffrls' instead of 'analytic_noisy').
%
% Writes fig09.pdf/png next to this script. Original code and figures
% are NOT modified: the original directories are only read (model,
% records, simulation functions); the simulated record stays in memory.
%
% Run:  matlab -batch "cd('.../section6_figure_copies/adjusted'); make_fig09_ident_x3"

this_dir = fileparts(mfilename('fullpath'));
gail = 'd:\LTY\yan\yanerxia\GAIL';
addpath(fullfile(gail, 'plot_table2_matlab', 'table2_noisy_ident'));

% --- Python load workaround (2026-09-26) ---
% pyenv('Version', <venv exe>) currently fails in MATLAB R2024a with
% "Python command requires a supported CPython version" (or crashes
% pyversion.dll). Cause: a broken leftover Python 3.11 registration
% pointing at the nonexistent E:\Python311\python.exe (see `py -0p`).
% Workaround: load the base 3.11 interpreter directly and put the
% venv site-packages on the Python path. td3_load_policy then sees
% Status='Loaded' and skips its own pyenv('Version', venv) call.
pe = pyenv;
if pe.Status ~= "Loaded"
    pyenv('Version', 'E:\Python\Python311\python.exe');
end
venv_sp = 'E:\Python\venvs\rl_pytorch\Lib\site-packages';
sp = py.sys.path;
if count(sp, venv_sp) == 0
    insert(sp, int32(0), venv_sp);
end

%% 1. Simulate BC-TD3 + sparse FF-RLS under 3x noise (in memory only)
ep = struct();
ep.model_path = fullfile(gail, 'rl_td3', 'checkpoints', ...
    'training_300k_20260730_1917', 'best', 'best_model');
ep.prefer_td3bc = true;
ep.algo = 'BC_TD3';
ep.coeff_mode = 'sparse_ffrls';
ep.ident = noisy_ident_cfg('plan1', 0, 3);
ep.obs_noise = false;
ep.py_exe = 'E:\Python\venvs\rl_pytorch\Scripts\python.exe';
ep.verbose = true;

record = simulate_table2_noisy_episode(ep);

%% 2. Axis limits shared with fig10 (x3 noisy TD3 / BC-TD3 records)
pt = fullfile(gail, 'plot_table2_matlab');
rec_td3_n3 = s6_load_record(fullfile(pt, 'table2_noisy_ident', ...
    'results', 'sigma_x3', 'record_table2_td3_noisy.mat'));
rec_bctd3_n3 = s6_load_record(fullfile(pt, 'table2_noisy_ident', ...
    'results', 'sigma_x3', 'record_table2_bctd3_noisy.mat'));
noisy3 = struct('tag', {'td3', 'bctd3'}, 'label', {'TD3', 'BC-TD3'}, ...
    'record', {rec_td3_n3, rec_bctd3_n3});
lims = s6_main_limits(noisy3);

%% 3. Plot (same style as the other adjusted figures)
cfg = struct();
cfg.output_dir = fullfile(this_dir, '..', 'adjusted_centered');
cfg.save_figures = true;
cfg.close_after_save = true;
cfg.save_fig = false;
cfg.dpi = 600;
cfg.view = [132, 21];
cfg.zoom_half_time = 0.20;

m = struct('label', 'BC-TD3', 'record', record, ...
    'main_title', 'Complete engagement (with noise)');
s6_plot_trajectory(m, 'fig09', lims, cfg);

fprintf('FIG09_OK r_min1=%.4f r_min2=%.4f r_minT=%.4f evade=%d strike=%d\n', ...
    record.r_min1, record.r_min2, record.r_minT, ...
    record.evade_ok, record.strike_ok);
