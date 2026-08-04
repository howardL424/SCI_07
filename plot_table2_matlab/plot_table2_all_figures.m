clc; close all;
%% Four-method Table-2 comparison figures
% Before running this script, generate the four record files once:
%   1) main_chunPN_0412_duo
%   2) main_zhengxian_0412_duo
%   3) table2_td3_pure/main_table2_td3_pure
%   4) table2_bctd3/main_table2_bctd3
%
% Outputs:
%   - four trajectory figures (complete 3-D view + D1/D2 close-ups)
%   - one 2x2 relative-distance comparison figure
%   - one 2x2 acceleration comparison figure

root_dir = fileparts(mfilename('fullpath'));
addpath(root_dir);

cfg = struct();
cfg.output_dir = fullfile(root_dir, 'figures');
cfg.save_figures = true;
cfg.close_after_save = false;
cfg.save_fig = true;
cfg.dpi = 600;

% Paper criteria used by the existing TD3/BC-TD3 evaluation scripts.
cfg.safe_distance = 6;          % minimum safe HGV-missile distance (m)
cfg.target_terminal_max = 10;   % maximum terminal HGV-target distance (m)

% Common 3-D view and closest-approach time window.
cfg.view = [132, 21];           % [azimuth, elevation] in degrees
cfg.zoom_half_time = 0.20;      % seconds around each closest approach

methods = load_table2_records(root_dir);
plot_table2_trajectories(methods, cfg);
plot_table2_distances(methods, cfg);
plot_table2_accelerations(methods, cfg);

fprintf('四方法论文图已输出至:\n  %s\n', cfg.output_dir);
