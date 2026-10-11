% plot_section6_adjusted.m
% Redraw section-6 figures with adjusted titles and legend labels.
% Original plotting code and original figures are NOT modified; this
% script only reads the existing record .mat files and writes new
% figures next to this script.
%
% Changes vs. the original figures:
%   fig04/fig05/fig06/fig10: top title centred over the visible content
%   fig05/fig07/fig08: "Pure TD3" -> "TD3"
%   fig04/fig05/fig06/fig09/fig10 legend: HGV -> A, Missile 1 -> D_1,
%                                          Missile 2 -> D_2
%   fig04: clean BC-TD3 record, same style as fig05 (top "BC-TD3",
%          main title "Complete engagement")
%   fig09: 3x-noise + sparse FF-RLS (make_fig09_ident_x3.m)
%   fig10: x3 noisy BC-TD3 record, top title "BC-TD3" (noise bracket
%          removed), main title "Complete engagement (with noise)"
%
% Run:  matlab -batch "cd('.../section6_figure_copies/adjusted'); plot_section6_adjusted"

this_dir = fileparts(mfilename('fullpath'));
gail = 'd:\LTY\yan\yanerxia\GAIL';
pt = fullfile(gail, 'plot_table2_matlab');

rec_pn       = s6_load_record(fullfile(pt, 'record_table2_pn.mat'));
rec_sine     = s6_load_record(fullfile(pt, 'record_table2_sine.mat'));
rec_td3      = s6_load_record(fullfile(pt, 'table2_td3_pure', ...
    'record_table2_td3_pure.mat'));
rec_bctd3    = s6_load_record(fullfile(pt, 'table2_bctd3', ...
    'record_table2_bctd3.mat'));
rec_td3_n3   = s6_load_record(fullfile(pt, 'table2_noisy_ident', ...
    'results', 'sigma_x3', 'record_table2_td3_noisy.mat'));
rec_bctd3_n3 = s6_load_record(fullfile(pt, 'table2_noisy_ident', ...
    'results', 'sigma_x3', 'record_table2_bctd3_noisy.mat'));

% Same 4-method set as the original clean figures (shared axis limits).
clean = struct( ...
    'tag',    {'pn', 'sine', 'td3', 'bctd3'}, ...
    'label',  {'Full-course PN', 'Programmed sine maneuver', ...
               'TD3', 'BC-TD3'}, ...
    'record', {rec_pn, rec_sine, rec_td3, rec_bctd3});
lims_clean = s6_main_limits(clean);

% Same 2-method set as the original x3 noisy run (shared axis limits).
noisy3 = struct( ...
    'tag',    {'td3', 'bctd3'}, ...
    'label',  {'TD3', 'BC-TD3'}, ...
    'record', {rec_td3_n3, rec_bctd3_n3});
lims_noisy3 = s6_main_limits(noisy3);

out_dir = fullfile(this_dir, '..', 'adjusted_centered');

cfg = struct();
cfg.output_dir = out_dir;
cfg.save_figures = true;
cfg.close_after_save = true;
cfg.save_fig = false;
cfg.dpi = 600;
cfg.view = [132, 21];
cfg.zoom_half_time = 0.20;
cfg.safe_distance = 6;
cfg.target_terminal_max = 10;

m = struct('label', 'BC-TD3', 'record', rec_bctd3, ...
    'main_title', 'Complete engagement');
s6_plot_trajectory(m, 'fig04', lims_clean, cfg);

m = struct('label', 'TD3', 'record', rec_td3, ...
    'main_title', 'Complete engagement');
s6_plot_trajectory(m, 'fig05', lims_clean, cfg);

m = struct('label', 'Programmed sine maneuver', 'record', rec_sine, ...
    'main_title', 'Complete engagement');
s6_plot_trajectory(m, 'fig06', lims_clean, cfg);

% fig09 is now the 3x-noise + sparse FF-RLS version, produced by
% make_fig09_ident_x3.m (runs the simulation, then plots). The old
% clean-data placeholder below is kept for reference but disabled so
% re-running this script does not overwrite the identification figure.
% m = struct('label', 'BC-TD3', 'record', rec_bctd3, ...
%     'main_title', 'Complete engagement (with noise)');
% s6_plot_trajectory(m, 'fig09', lims_clean, cfg);

m = struct('label', 'BC-TD3', 'record', rec_bctd3_n3, ...
    'main_title', 'Complete engagement (with noise)');
s6_plot_trajectory(m, 'fig10', lims_noisy3, cfg);

s6_plot_accelerations(clean, 'fig07', cfg);
s6_plot_distances(clean, 'fig08', cfg);

fprintf('adjusted figures written to %s\n', out_dir);
