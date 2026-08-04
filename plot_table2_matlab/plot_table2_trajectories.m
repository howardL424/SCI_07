function figs = plot_table2_trajectories(methods, cfg)
%PLOT_TABLE2_TRAJECTORIES Plot four 3-D trajectory figures with close-ups.
%   Each method gets one figure. The upper panel shows the complete
%   engagement, while the two lower panels show the closest approaches to
%   D1 and D2 in local, encounter-centred coordinates.

if nargin < 2
    cfg = struct();
end
cfg = apply_defaults(cfg);
if cfg.save_figures && ~isfolder(cfg.output_dir)
    mkdir(cfg.output_dir);
end

main_limits = common_main_limits(methods);
figs = gobjects(1, numel(methods));
for k = 1:numel(methods)
    rec = methods(k).record;
    figs(k) = figure('Color', 'w', 'Units', 'centimeters', ...
        'Position', [2, 2, 18.2, 15.5], ...
        'Name', ['Trajectory - ', methods(k).label]);
    tl = tiledlayout(figs(k), 2, 2, 'TileSpacing', 'compact', ...
        'Padding', 'compact');

    ax_main = nexttile(tl, [1, 2]);
    plot_full_trajectory(ax_main, rec, main_limits, cfg);

    ax_d1 = nexttile(tl);
    plot_closest_approach(ax_d1, rec, rec.pos_D1, 'D_1', ...
        cfg.colors.d1, cfg.styles.d1, cfg);

    ax_d2 = nexttile(tl);
    plot_closest_approach(ax_d2, rec, rec.pos_D2, 'D_2', ...
        cfg.colors.d2, cfg.styles.d2, cfg);

    title(tl, methods(k).label, 'FontName', cfg.font_name, ...
        'FontSize', cfg.title_size, 'FontWeight', 'normal', ...
        'Interpreter', 'none');

    if cfg.save_figures
        base = fullfile(cfg.output_dir, ['trajectory_', methods(k).tag]);
        save_figure(figs(k), base, cfg);
        if cfg.close_after_save
            close(figs(k));
        end
    end
end
end


function plot_full_trajectory(ax, rec, lims, cfg)
hold(ax, 'on');
pM = plot3(ax, rec.pos_M(1, :) / 1000, rec.pos_M(3, :) / 1000, ...
    rec.pos_M(2, :) / 1000, '-', 'Color', cfg.colors.m, ...
    'LineWidth', cfg.line_width);
pD1 = plot3(ax, rec.pos_D1(1, :) / 1000, rec.pos_D1(3, :) / 1000, ...
    rec.pos_D1(2, :) / 1000, cfg.styles.d1, ...
    'Color', cfg.colors.d1, 'LineWidth', cfg.line_width);
pD2 = plot3(ax, rec.pos_D2(1, :) / 1000, rec.pos_D2(3, :) / 1000, ...
    rec.pos_D2(2, :) / 1000, cfg.styles.d2, ...
    'Color', cfg.colors.d2, 'LineWidth', cfg.line_width);

target = double(rec.T(1:3));
pT = plot3(ax, target(1) / 1000, target(3) / 1000, ...
    target(2) / 1000, '^', 'Color', cfg.colors.target, ...
    'MarkerFaceColor', cfg.colors.target, 'MarkerSize', 6);

plot3(ax, rec.pos_M(1, 1) / 1000, rec.pos_M(3, 1) / 1000, ...
    rec.pos_M(2, 1) / 1000, 'o', 'Color', cfg.colors.m, ...
    'MarkerFaceColor', 'w', 'MarkerSize', 5, ...
    'HandleVisibility', 'off');
plot3(ax, rec.pos_D1(1, 1) / 1000, rec.pos_D1(3, 1) / 1000, ...
    rec.pos_D1(2, 1) / 1000, 'o', 'Color', cfg.colors.d1, ...
    'MarkerFaceColor', 'w', 'MarkerSize', 5, ...
    'HandleVisibility', 'off');
plot3(ax, rec.pos_D2(1, 1) / 1000, rec.pos_D2(3, 1) / 1000, ...
    rec.pos_D2(2, 1) / 1000, 'o', 'Color', cfg.colors.d2, ...
    'MarkerFaceColor', 'w', 'MarkerSize', 5, ...
    'HandleVisibility', 'off');

xlim(ax, lims(1, :));
ylim(ax, lims(2, :));
zlim(ax, lims(3, :));
daspect(ax, [1, 1, 1]);
view(ax, cfg.view(1), cfg.view(2));
xlabel(ax, '$x$ (km)', 'Interpreter', 'latex');
ylabel(ax, '$z$ (km)', 'Interpreter', 'latex');
zlabel(ax, '$y$ (km)', 'Interpreter', 'latex');
title(ax, 'Complete engagement', 'FontWeight', 'normal');
legend(ax, [pM, pD1, pD2, pT], ...
    {'HGV', 'Missile 1', 'Missile 2', 'Target'}, ...
    'Location', 'best', 'Box', 'off', 'FontSize', cfg.legend_size);
format_axes(ax, cfg);
end


function plot_closest_approach(ax, rec, pos_D, def_name, ...
    def_color, def_style, cfg)
dist = sqrt(sum((rec.pos_M - pos_D).^2, 1));
[r_min, i_min] = min(dist);
idx = find(abs(rec.t - rec.t(i_min)) <= cfg.zoom_half_time);
if numel(idx) < 5
    idx = max(1, i_min - 3):min(numel(rec.t), i_min + 3);
end

centre = 0.5 * (rec.pos_M(:, i_min) + pos_D(:, i_min));
m_local = rec.pos_M(:, idx) - centre;
d_local = pos_D(:, idx) - centre;
m_min = rec.pos_M(:, i_min) - centre;
d_min = pos_D(:, i_min) - centre;

hold(ax, 'on');
plot3(ax, m_local(1, :), m_local(3, :), m_local(2, :), ...
    '-', 'Color', cfg.colors.m, 'LineWidth', cfg.line_width);
plot3(ax, d_local(1, :), d_local(3, :), d_local(2, :), ...
    def_style, 'Color', def_color, 'LineWidth', cfg.line_width);
plot3(ax, [m_min(1), d_min(1)], [m_min(3), d_min(3)], ...
    [m_min(2), d_min(2)], ':', 'Color', cfg.colors.closest, ...
    'LineWidth', 1.1);
plot3(ax, m_min(1), m_min(3), m_min(2), 'o', ...
    'Color', cfg.colors.m, 'MarkerFaceColor', 'w', 'MarkerSize', 5);
plot3(ax, d_min(1), d_min(3), d_min(2), 'o', ...
    'Color', def_color, 'MarkerFaceColor', 'w', 'MarkerSize', 5);

axis(ax, 'equal');
view(ax, cfg.view(1), cfg.view(2));
xlabel(ax, '$\Delta x$ (m)', 'Interpreter', 'latex');
ylabel(ax, '$\Delta z$ (m)', 'Interpreter', 'latex');
zlabel(ax, '$\Delta y$ (m)', 'Interpreter', 'latex');
title(ax, sprintf('%s closest approach: %.2f m', def_name, r_min), ...
    'Interpreter', 'tex', 'FontWeight', 'normal');
format_axes(ax, cfg);
end


function lims = common_main_limits(methods)
xyz = [];
for k = 1:numel(methods)
    rec = methods(k).record;
    target = double(rec.T(1:3));
    xyz = [xyz, rec.pos_M, rec.pos_D1, rec.pos_D2, target(:)]; %#ok<AGROW>
end
plot_xyz = [xyz(1, :); xyz(3, :); xyz(2, :)] / 1000;
mins = min(plot_xyz, [], 2);
maxs = max(plot_xyz, [], 2);
span = maxs - mins;
pad = max(0.05 * span, 0.05);
lims = [mins - pad, maxs + pad];
end


function format_axes(ax, cfg)
set(ax, 'FontName', cfg.font_name, 'FontSize', cfg.axis_size, ...
    'LineWidth', 0.8, 'Box', 'on', 'SortMethod', 'childorder');
grid(ax, 'off');
end


function cfg = apply_defaults(cfg)
root_dir = fileparts(mfilename('fullpath'));
cfg = set_default(cfg, 'output_dir', fullfile(root_dir, 'figures'));
cfg = set_default(cfg, 'save_figures', true);
cfg = set_default(cfg, 'close_after_save', false);
cfg = set_default(cfg, 'save_fig', true);
cfg = set_default(cfg, 'dpi', 600);
cfg = set_default(cfg, 'view', [132, 21]);
cfg = set_default(cfg, 'zoom_half_time', 0.20);
cfg = set_default(cfg, 'font_name', 'Times New Roman');
cfg = set_default(cfg, 'axis_size', 9);
cfg = set_default(cfg, 'legend_size', 8);
cfg = set_default(cfg, 'title_size', 11);
cfg = set_default(cfg, 'line_width', 1.25);
cfg.colors = struct('m', [0.85, 0.15, 0.15], ...
    'd1', [0.10, 0.30, 0.85], ...
    'd2', [0.10, 0.10, 0.10], ...
    'target', [0.10, 0.10, 0.10], ...
    'closest', [0.25, 0.25, 0.25]);
cfg.styles = struct('d1', '--', 'd2', '-.');
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
