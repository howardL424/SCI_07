function fig = plot_table2_accelerations(methods, cfg)
%PLOT_TABLE2_ACCELERATIONS Plot HGV acceleration histories in a 2-by-2 layout.
%   The plotted quantities are the two commanded normal-acceleration
%   components a_y and a_z, and their magnitude a_n.

if nargin < 2
    cfg = struct();
end
cfg = apply_defaults(cfg);
if cfg.save_figures && ~isfolder(cfg.output_dir)
    mkdir(cfg.output_dir);
end

[t_lim, a_lim] = common_limits(methods);
fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, 18.2, 13.5], 'Name', 'HGV accelerations');
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

legend_handles = gobjects(1, 3);
first_ax = gobjects(1);
for k = 1:numel(methods)
    ax = nexttile(tl);
    rec = methods(k).record;
    hold(ax, 'on');

    h1 = plot(ax, rec.t, rec.a_M(1, :), '-', ...
        'Color', cfg.colors.ay, 'LineWidth', cfg.line_width);
    h2 = plot(ax, rec.t, rec.a_M(2, :), '--', ...
        'Color', cfg.colors.az, 'LineWidth', cfg.line_width);
    h3 = plot(ax, rec.t, rec.a_n, '-.', ...
        'Color', cfg.colors.an, 'LineWidth', cfg.line_width);

    xlim(ax, t_lim);
    ylim(ax, a_lim);
    xlabel(ax, '$t$ (s)', 'Interpreter', 'latex');
    ylabel(ax, 'Acceleration (m/s$^2$)', 'Interpreter', 'latex');
    title(ax, methods(k).label, 'Interpreter', 'none', ...
        'FontWeight', 'normal');
    format_axes(ax, cfg);

    if k == 1
        first_ax = ax;
        legend_handles = [h1, h2, h3];
    end
end

lgd = legend(first_ax, legend_handles, ...
    {'$a_y$', '$a_z$', '$a_n=\sqrt{a_y^2+a_z^2}$'}, ...
    'Interpreter', 'latex', 'Orientation', 'horizontal', ...
    'Box', 'off', 'FontSize', cfg.legend_size);
lgd.Layout.Tile = 'south';

if cfg.save_figures
    base = fullfile(cfg.output_dir, 'acceleration_comparison_2x2');
    save_figure(fig, base, cfg);
    if cfg.close_after_save
        close(fig);
    end
end
end


function [t_lim, a_lim] = common_limits(methods)
t_max = 0;
a_all = [];
for k = 1:numel(methods)
    rec = methods(k).record;
    t_max = max(t_max, max(rec.t));
    a_all = [a_all, rec.a_M(1, :), rec.a_M(2, :), rec.a_n]; %#ok<AGROW>
end
a_all = a_all(isfinite(a_all));
if isempty(a_all)
    a_lim = [-1, 1];
else
    a_min = min(a_all);
    a_max = max(a_all);
    pad = max(0.06 * (a_max - a_min), 1);
    a_lim = [a_min - pad, a_max + pad];
end
t_lim = [0, 1.01 * t_max];
end


function format_axes(ax, cfg)
set(ax, 'FontName', cfg.font_name, 'FontSize', cfg.axis_size, ...
    'LineWidth', 0.8, 'Box', 'on');
grid(ax, 'off');
end


function cfg = apply_defaults(cfg)
root_dir = fileparts(mfilename('fullpath'));
cfg = set_default(cfg, 'output_dir', fullfile(root_dir, 'figures'));
cfg = set_default(cfg, 'save_figures', true);
cfg = set_default(cfg, 'close_after_save', false);
cfg = set_default(cfg, 'save_fig', true);
cfg = set_default(cfg, 'dpi', 600);
cfg = set_default(cfg, 'font_name', 'Times New Roman');
cfg = set_default(cfg, 'axis_size', 9);
cfg = set_default(cfg, 'legend_size', 8);
cfg = set_default(cfg, 'line_width', 1.25);
cfg.colors = struct('ay', [0.85, 0.15, 0.15], ...
    'az', [0.10, 0.30, 0.85], ...
    'an', [0.10, 0.10, 0.10]);
end


function s = set_default(s, name, value)
if ~isfield(s, name) || isempty(s.(name))
    s.(name) = value;
end
end


function save_figure(fig, base, cfg)
exportgraphics(fig, [base, '.png'], 'Resolution', cfg.dpi);
exportgraphics(fig, [base, '.pdf'], 'ContentType', 'vector');
if cfg.save_fig
    savefig(fig, [base, '.fig']);
end
end
