function figs = plot_noisy_vs_clean(pairs, cfg)
%PLOT_NOISY_VS_CLEAN  同一策略在真值系数 / 带噪测量下的轨迹对照图。
%
%   figs = plot_noisy_vs_clean(pairs, cfg)
%
%   pairs 为结构体数组，每个元素给出一种算法的两条轨迹:
%     .tag    文件名后缀，如 'td3'
%     .label  图标题，如 'Pure TD3'
%     .clean  解析真值系数下的 record（原 plot_table2_matlab 结果）
%     .noisy  带测量噪声的 record
%
%   每张图上半为完整三维交战，下半两幅为 D1 / D2 最近接近点放大图，
%   放大窗口同时覆盖两条轨迹各自的最近接近时刻。
%
%   See also MAIN_TABLE2_NOISY_IDENT, PLOT_TABLE2_TRAJECTORIES.

if nargin < 2
    cfg = struct();
end
cfg = apply_defaults(cfg);
if cfg.save_figures && ~isfolder(cfg.output_dir)
    mkdir(cfg.output_dir);
end

lims = common_main_limits(pairs);
figs = gobjects(1, numel(pairs));
for k = 1:numel(pairs)
    figs(k) = figure('Color', 'w', 'Units', 'centimeters', ...
        'Position', [2, 2, 18.2, 15.5], ...
        'Name', ['Truth vs noisy - ', pairs(k).label]);
    tl = tiledlayout(figs(k), 2, 2, 'TileSpacing', 'compact', ...
        'Padding', 'compact');

    ax_main = nexttile(tl, [1, 2]);
    plot_full(ax_main, pairs(k), lims, cfg);

    ax1 = nexttile(tl);
    plot_zoom(ax1, pairs(k), 'pos_D1', 'D_1', cfg);

    ax2 = nexttile(tl);
    plot_zoom(ax2, pairs(k), 'pos_D2', 'D_2', cfg);

    title(tl, sprintf('%s: truth coefficients vs noisy measurements', ...
        pairs(k).label), 'FontName', cfg.font_name, ...
        'FontSize', cfg.title_size, 'FontWeight', 'normal', ...
        'Interpreter', 'none');

    if cfg.save_figures
        base = fullfile(cfg.output_dir, ...
            ['trajectory_', pairs(k).tag, '_truth_vs_noisy']);
        save_figure(figs(k), base, cfg);
        if cfg.close_after_save
            close(figs(k));
        end
    end
end
end


function plot_full(ax, pair, lims, cfg)
hold(ax, 'on');
hc = plot3(ax, pair.clean.pos_M(1, :) / 1000, pair.clean.pos_M(3, :) / 1000, ...
    pair.clean.pos_M(2, :) / 1000, '-', 'Color', cfg.colors.clean, ...
    'LineWidth', cfg.line_width);
hn = plot3(ax, pair.noisy.pos_M(1, :) / 1000, pair.noisy.pos_M(3, :) / 1000, ...
    pair.noisy.pos_M(2, :) / 1000, '-', 'Color', cfg.colors.noisy, ...
    'LineWidth', cfg.line_width);

hdc = plot_defenders(ax, pair.clean, cfg.colors.def_clean, '--', cfg);
hdn = plot_defenders(ax, pair.noisy, cfg.colors.def_noisy, '-.', cfg);

target = double(pair.noisy.T(1:3));
ht = plot3(ax, target(1) / 1000, target(3) / 1000, target(2) / 1000, '^', ...
    'Color', cfg.colors.target, 'MarkerFaceColor', cfg.colors.target, ...
    'MarkerSize', 6);

plot3(ax, pair.noisy.pos_M(1, 1) / 1000, pair.noisy.pos_M(3, 1) / 1000, ...
    pair.noisy.pos_M(2, 1) / 1000, 'o', 'Color', cfg.colors.noisy, ...
    'MarkerFaceColor', 'w', 'MarkerSize', 5, 'HandleVisibility', 'off');

xlim(ax, lims(1, :));
ylim(ax, lims(2, :));
zlim(ax, lims(3, :));
daspect(ax, [1, 1, 1]);
view(ax, cfg.view(1), cfg.view(2));
xlabel(ax, '$x$ (km)', 'Interpreter', 'latex');
ylabel(ax, '$z$ (km)', 'Interpreter', 'latex');
zlabel(ax, '$y$ (km)', 'Interpreter', 'latex');
title(ax, 'Complete engagement', 'FontWeight', 'normal');
legend(ax, [hc, hn, hdc, hdn, ht], ...
    {'HGV (truth coeff.)', 'HGV (noisy meas.)', ...
     'Missiles (truth coeff.)', 'Missiles (noisy meas.)', 'Target'}, ...
    'Location', 'best', 'Box', 'off', 'FontSize', cfg.legend_size);
format_axes(ax, cfg);
end


function h = plot_defenders(ax, rec, color, style, cfg)
h = plot3(ax, rec.pos_D1(1, :) / 1000, rec.pos_D1(3, :) / 1000, ...
    rec.pos_D1(2, :) / 1000, style, 'Color', color, ...
    'LineWidth', cfg.line_width * 0.8);
plot3(ax, rec.pos_D2(1, :) / 1000, rec.pos_D2(3, :) / 1000, ...
    rec.pos_D2(2, :) / 1000, style, 'Color', color, ...
    'LineWidth', cfg.line_width * 0.8, 'HandleVisibility', 'off');
end


function plot_zoom(ax, pair, def_field, def_name, cfg)
[rmin_c, tc] = closest_approach(pair.clean, def_field);
[rmin_n, tn] = closest_approach(pair.noisy, def_field);
t_lo = min(tc, tn) - cfg.zoom_half_time;
t_hi = max(tc, tn) + cfg.zoom_half_time;

centre = midpoint_at(pair.noisy, def_field, tn);

hold(ax, 'on');
hc = plot_local(ax, pair.clean, def_field, centre, t_lo, t_hi, ...
    cfg.colors.clean, cfg.colors.def_clean, '--', cfg);
hn = plot_local(ax, pair.noisy, def_field, centre, t_lo, t_hi, ...
    cfg.colors.noisy, cfg.colors.def_noisy, '-.', cfg);

axis(ax, 'equal');
view(ax, cfg.view(1), cfg.view(2));
xlabel(ax, '$\Delta x$ (m)', 'Interpreter', 'latex');
ylabel(ax, '$\Delta z$ (m)', 'Interpreter', 'latex');
zlabel(ax, '$\Delta y$ (m)', 'Interpreter', 'latex');
title(ax, sprintf('%s closest approach: %.2f m (truth) / %.2f m (noisy)', ...
    def_name, rmin_c, rmin_n), 'Interpreter', 'tex', 'FontWeight', 'normal');
legend(ax, [hc, hn], {'truth coeff.', 'noisy meas.'}, ...
    'Location', 'best', 'Box', 'off', 'FontSize', cfg.legend_size);
format_axes(ax, cfg);
end


function h = plot_local(ax, rec, def_field, centre, t_lo, t_hi, ...
    color_m, color_d, style, cfg)
idx = find(rec.t >= t_lo & rec.t <= t_hi);
if numel(idx) < 3
    [~, i_c] = min(abs(rec.t - 0.5 * (t_lo + t_hi)));
    idx = max(1, i_c - 3):min(numel(rec.t), i_c + 3);
end
pos_D = rec.(def_field);
m_local = rec.pos_M(:, idx) - centre;
d_local = pos_D(:, idx) - centre;

h = plot3(ax, m_local(1, :), m_local(3, :), m_local(2, :), '-', ...
    'Color', color_m, 'LineWidth', cfg.line_width);
plot3(ax, d_local(1, :), d_local(3, :), d_local(2, :), style, ...
    'Color', color_d, 'LineWidth', cfg.line_width * 0.8, ...
    'HandleVisibility', 'off');

dist = sqrt(sum((rec.pos_M(:, idx) - pos_D(:, idx)).^2, 1));
[~, j] = min(dist);
plot3(ax, [m_local(1, j), d_local(1, j)], [m_local(3, j), d_local(3, j)], ...
    [m_local(2, j), d_local(2, j)], ':', 'Color', cfg.colors.closest, ...
    'LineWidth', 1.1, 'HandleVisibility', 'off');
plot3(ax, m_local(1, j), m_local(3, j), m_local(2, j), 'o', ...
    'Color', color_m, 'MarkerFaceColor', 'w', 'MarkerSize', 5, ...
    'HandleVisibility', 'off');
end


function [r_min, t_min] = closest_approach(rec, def_field)
dist = sqrt(sum((rec.pos_M - rec.(def_field)).^2, 1));
[r_min, i_min] = min(dist);
t_min = rec.t(i_min);
end


function centre = midpoint_at(rec, def_field, t_at)
[~, i] = min(abs(rec.t - t_at));
pos_D = rec.(def_field);
centre = 0.5 * (rec.pos_M(:, i) + pos_D(:, i));
end


function lims = common_main_limits(pairs)
xyz = [];
for k = 1:numel(pairs)
    for rec = {pairs(k).clean, pairs(k).noisy}
        r = rec{1};
        target = double(r.T(1:3));
        xyz = [xyz, r.pos_M, r.pos_D1, r.pos_D2, target(:)]; %#ok<AGROW>
    end
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
cfg.colors = struct('clean', [0.10, 0.30, 0.85], ...
    'noisy', [0.85, 0.15, 0.15], ...
    'def_clean', [0.45, 0.45, 0.45], ...
    'def_noisy', [0.10, 0.10, 0.10], ...
    'target', [0.10, 0.10, 0.10], ...
    'closest', [0.25, 0.25, 0.25]);
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
