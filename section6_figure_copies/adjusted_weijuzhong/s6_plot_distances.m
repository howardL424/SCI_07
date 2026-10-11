function fig = s6_plot_distances(methods, out_base, cfg)
%S6_PLOT_DISTANCES Relative distances for four guidance methods.
%   Adapted from plot_table2_matlab/plot_table2_distances.m.
%   Method labels come from methods(k).label ("Pure TD3" -> "TD3"
%   is applied by the caller); output file name is passed in.

cfg = apply_defaults(cfg);
if cfg.save_figures && ~isfolder(cfg.output_dir)
    mkdir(cfg.output_dir);
end

[t_lim, r_lim] = common_limits(methods, cfg);
fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, 18.2, 13.5], 'Name', 'Relative distances');
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

legend_handles = gobjects(1, 5);
first_ax = gobjects(1);
for k = 1:numel(methods)
    ax = nexttile(tl);
    rec = methods(k).record;
    hold(ax, 'on');

    h1 = semilogy(ax, rec.t, positive_only(rec.r_MD1), '--', ...
        'Color', cfg.colors.d1, 'LineWidth', cfg.line_width);
    h2 = semilogy(ax, rec.t, positive_only(rec.r_MD2), '-.', ...
        'Color', cfg.colors.d2, 'LineWidth', cfg.line_width);
    h3 = semilogy(ax, rec.t, positive_only(rec.r_MT), '-', ...
        'Color', cfg.colors.mt, 'LineWidth', cfg.line_width);
    h4 = yline(ax, cfg.safe_distance, ':', ...
        'Color', cfg.colors.safe, 'LineWidth', cfg.threshold_width);
    h5 = yline(ax, cfg.target_terminal_max, '--', ...
        'Color', cfg.colors.terminal, 'LineWidth', cfg.threshold_width);

    xlim(ax, t_lim);
    ylim(ax, r_lim);
    xlabel(ax, '$t$ (s)', 'Interpreter', 'latex');
    ylabel(ax, 'Distance (m)');
    title(ax, methods(k).label, 'Interpreter', 'none', ...
        'FontWeight', 'normal');
    format_axes(ax, cfg);

    if k == 1
        first_ax = ax;
        legend_handles = [h1, h2, h3, h4, h5];
    end
end

lgd = legend(first_ax, legend_handles, ...
    {'$r_{MD1}$', '$r_{MD2}$', '$r_{MT}$', ...
     sprintf('Avoidance limit: %.0f m', cfg.safe_distance), ...
     sprintf('Terminal limit: %.0f m', cfg.target_terminal_max)}, ...
    'Interpreter', 'latex', 'Orientation', 'horizontal', ...
    'Box', 'off', 'FontSize', cfg.legend_size);
lgd.Layout.Tile = 'south';

if cfg.save_figures
    save_figure(fig, fullfile(cfg.output_dir, out_base), cfg);
    if cfg.close_after_save
        close(fig);
    end
end
end


function [t_lim, r_lim] = common_limits(methods, cfg)
t_max = 0;
r_all = [cfg.safe_distance, cfg.target_terminal_max];
for k = 1:numel(methods)
    rec = methods(k).record;
    t_max = max(t_max, max(rec.t));
    r_all = [r_all, rec.r_MD1, rec.r_MD2, rec.r_MT]; %#ok<AGROW>
end
r_all = r_all(isfinite(r_all) & r_all > 0);
r_min = min(r_all);
r_max = max(r_all);
t_lim = [0, 1.01 * t_max];
r_lim = [max(0.5 * r_min, 0.1), 1.15 * r_max];
end


function y = positive_only(y)
y = double(y);
y(~isfinite(y) | y <= 0) = NaN;
end


function format_axes(ax, cfg)
set(ax, 'FontName', cfg.font_name, 'FontSize', cfg.axis_size, ...
    'LineWidth', 0.8, 'Box', 'on', 'YScale', 'log');
grid(ax, 'off');
end


function cfg = apply_defaults(cfg)
cfg = set_default(cfg, 'output_dir', fileparts(mfilename('fullpath')));
cfg = set_default(cfg, 'save_figures', true);
cfg = set_default(cfg, 'close_after_save', false);
cfg = set_default(cfg, 'save_fig', false);
cfg = set_default(cfg, 'dpi', 600);
cfg = set_default(cfg, 'safe_distance', 6);
cfg = set_default(cfg, 'target_terminal_max', 10);
cfg = set_default(cfg, 'font_name', 'Times New Roman');
cfg = set_default(cfg, 'axis_size', 9);
cfg = set_default(cfg, 'legend_size', 8);
cfg = set_default(cfg, 'line_width', 1.25);
cfg = set_default(cfg, 'threshold_width', 1.0);
cfg.colors = struct('d1', [0.10, 0.30, 0.85], ...
    'd2', [0.10, 0.10, 0.10], ...
    'mt', [0.85, 0.15, 0.15], ...
    'safe', [0.55, 0.15, 0.65], ...
    'terminal', [0.45, 0.45, 0.45]);
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
