function plot_mc_scatter(metricsPath, opts)
%PLOT_MC_SCATTER 蒙特卡洛评估散点图：1/r_minT vs min(r_MD1,r_MD2)
%
%   plot_mc_scatter()
%   plot_mc_scatter(metricsPath)
%   plot_mc_scatter(metricsPath, opts)
%
% 输入
%   metricsPath  .npz 或 .mat（由 evaluate_td3.py 写出的逐轮指标）
%                默认指向样例 monte_carlo_eval_best_seed0_20260730_150325
%   opts         可选结构体：
%     .failXMode   规避失败点横坐标：'yaxis'(默认，统一贴纵轴) | 'omit'
%     .killR       杀伤半径参考线 (m)，默认 6
%     .markerSize  散点 SizeData，默认 16（x 标记略大）
%     .leftWidth   断轴左侧轴相对宽度，默认 0.28（右侧更宽以展开联合成功）
%     .saveFig     若非空，保存图片到该路径（.png/.fig/.pdf）
%
% 分类（直接读 npz/mat 布尔字段，无需再阈值判断）：
%   联合成功   both_ok
%   仅规避成功 evade_ok & ~strike_ok
%   规避失败   ~evade_ok
%
% 横轴 log(1/r_minT)，在「仅规避成功」与「联合成功」之间断轴；
% 规避失败无有效 r_minT 时统一画在左轴纵轴缘。刻度为 10 的奇次幂。

    if nargin < 1 || isempty(metricsPath)
        here = fileparts(mfilename('fullpath'));
        metricsPath = fullfile(here, '..', 'checkpoints', ...
            'training_260k_20260723_152457', ...
            'monte_carlo_eval_best_seed0_20260730_150325.mat');
    end
    if nargin < 2 || isempty(opts)
        opts = struct();
    end
    failXMode = getfield_default(opts, 'failXMode', 'yaxis');
    killR     = getfield_default(opts, 'killR', 6);
    markerSz  = getfield_default(opts, 'markerSize', 16);
    leftWrel  = getfield_default(opts, 'leftWidth', 0.28);
    saveFig   = getfield_default(opts, 'saveFig', '');

    S = load_mc_metrics(metricsPath);

    r_MD_min = min(S.r_MD1_min(:), S.r_MD2_min(:));
    r_minT   = S.r_minT(:);
    x = 1 ./ r_minT;   % 打击精度倒数；NaN / Inf 保持

    both_ok   = logical(S.both_ok(:));
    evade_ok  = logical(S.evade_ok(:));
    strike_ok = logical(S.strike_ok(:));

    cat_both  = both_ok;
    cat_evade = evade_ok & ~strike_ok;
    cat_fail  = ~evade_ok;

    xEvade = x(cat_evade & isfinite(x) & (x > 0));
    xBoth  = x(cat_both  & isfinite(x) & (x > 0));

    if isempty(xEvade)
        xEvadeMax = 1e-2;
    else
        xEvadeMax = max(xEvade);
    end
    if isempty(xBoth)
        xBothMin = 1;
        xBothMax = 10;
    else
        xBothMin = min(xBoth);
        xBothMax = max(xBoth);
    end

    % 左轴：失败 + 仅规避；右轴：联合成功（断掉中间空白数量级）
    xFail = min([xEvade(:); 1e-4]) / 10;
    if strcmpi(failXMode, 'yaxis')
        x(cat_fail) = xFail;
    elseif strcmpi(failXMode, 'omit')
        x(cat_fail) = NaN;
    else
        error('plot_mc_scatter:BadFailXMode', ...
            'failXMode 应为 yaxis 或 omit，收到: %s', failXMode);
    end

    xLimL = [xFail, 10^ceil(log10(max(xEvadeMax * 1.5, 1e-3)))];
    % 右轴略放宽，保证出现 10^1、10^3 等奇次幂刻度，并拉开蓝色点
    xLimR = [10^floor(log10(xBothMin)), 10^ceil(log10(xBothMax))];
    if xLimR(1) >= xBothMin
        xLimR(1) = 10^(floor(log10(xBothMin)) - 0);  % 保持 decade 下沿
    end
    if mod(round(log10(xLimR(2))), 2) == 0
        xLimR(2) = 10^(log10(xLimR(2)) + 1);  % 抬到下一奇次幂上沿
    end

    yAll = r_MD_min(isfinite(r_MD_min));
    yLim = [0, max(yAll) * 1.08];

    styles = {
        cat_both,  [0.00 0.45 0.74], 'o', '联合成功', 'R';
        cat_evade, [0.85 0.33 0.10], 's', '仅规避成功', 'L';
        cat_fail,  [0.50 0.50 0.50], 'x', '规避失败', 'L';
        };

    fig = figure('Color', 'w', 'Name', 'MC scatter', ...
        'Position', [120 120 920 520]);

    marginL = 0.10;
    marginR = 0.04;
    marginB = 0.14;
    marginT = 0.10;
    gap = 0.028;
    usableW = 1 - marginL - marginR - gap;
    leftW = usableW * leftWrel;
    rightW = usableW * (1 - leftWrel);
    ht = 1 - marginB - marginT;

    axL = axes('Parent', fig, ...
        'Position', [marginL, marginB, leftW, ht]);
    axR = axes('Parent', fig, ...
        'Position', [marginL + leftW + gap, marginB, rightW, ht]);

    hold(axL, 'on'); hold(axR, 'on');
    box(axL, 'on'); box(axR, 'on');
    grid(axL, 'on'); grid(axR, 'on');
    set([axL, axR], 'XScale', 'log', 'FontName', 'Microsoft YaHei', ...
        'FontSize', 11, 'TickLabelInterpreter', 'tex');

    hLeg = gobjects(0);
    legNames = {};
    for i = 1:size(styles, 1)
        msk = styles{i, 1};
        side = styles{i, 5};
        if side == 'L'
            ax = axL;
        else
            ax = axR;
        end
        xx = x(msk);
        yy = r_MD_min(msk);
        ok = isfinite(xx) & (xx > 0) & isfinite(yy);
        nShow = nnz(ok);
        nAll  = nnz(msk);
        if nShow == 0
            continue
        end
        sz = markerSz;
        if strcmp(styles{i, 3}, 'x')
            sz = markerSz * 1.35;
        end
        h = scatter(ax, xx(ok), yy(ok), sz, ...
            'Marker', styles{i, 3}, ...
            'MarkerEdgeColor', styles{i, 2}, ...
            'MarkerFaceColor', styles{i, 2}, ...
            'LineWidth', 0.9, ...
            'DisplayName', sprintf('%s (n=%d)', styles{i, 4}, nAll));
        if strcmp(styles{i, 3}, 'x')
            set(h, 'MarkerFaceColor', 'none', 'LineWidth', 1.2);
        end
        hLeg(end+1) = h; %#ok<AGROW>
        legNames{end+1} = get(h, 'DisplayName'); %#ok<AGROW>
        if nShow < nAll
            fprintf('[plot_mc_scatter] %s: 省略 %d 个非有限横坐标点\n', ...
                styles{i, 4}, nAll - nShow);
        end
    end

    for ax = [axL, axR]
        yline(ax, killR, '--', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
        ylim(ax, yLim);
    end
    text(axL, xLimL(1), killR, sprintf('  KILL_R=%.0fm', killR), ...
        'VerticalAlignment', 'bottom', 'FontSize', 9, ...
        'Color', [0.2 0.2 0.2], 'FontName', 'Microsoft YaHei');

    xlim(axL, xLimL);
    xlim(axR, xLimR);
    set_odd_power_xticks(axL, xLimL);
    set_odd_power_xticks(axR, xLimR);

    % 右轴不重复纵轴刻度数字，保持断轴观感
    set(axR, 'YTickLabel', []);
    ylabel(axL, 'min(r_{MD1,min}, r_{MD2,min})  (m)');

    if strcmpi(failXMode, 'yaxis') && any(cat_fail)
        xline(axL, xFail, ':', 'Color', [0.4 0.4 0.4], 'LineWidth', 0.8, ...
            'Label', '规避失败', 'LabelHorizontalAlignment', 'left', ...
            'HandleVisibility', 'off');
    end

    % 断轴记号
    draw_axis_break(fig, axL, axR);

    % 共用横轴标题 / 总标题
    annotation(fig, 'textbox', [marginL, 0.02, usableW + gap, 0.06], ...
        'String', '1 / r_{minT}  (1/m, log, 断轴)', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
        'FontName', 'Microsoft YaHei', 'FontSize', 11, ...
        'Interpreter', 'tex');
    sgtitle(fig, sprintf('蒙特卡洛结果  N=%d', numel(r_minT)), ...
        'FontName', 'Microsoft YaHei', 'FontSize', 12, 'FontWeight', 'bold');

    legend(axR, hLeg, legNames, 'Location', 'best', ...
        'FontName', 'Microsoft YaHei');

    fprintf(['[plot_mc_scatter] both=%d, evade_only=%d, fail=%d ', ...
        '(nan r_minT=%d); xLimL=[%.2g,%.2g], xLimR=[%.2g,%.2g]\n'], ...
        nnz(cat_both), nnz(cat_evade), nnz(cat_fail), nnz(~isfinite(r_minT)), ...
        xLimL(1), xLimL(2), xLimR(1), xLimR(2));

    if ~isempty(saveFig)
        [~, ~, ext] = fileparts(saveFig);
        if isempty(ext)
            saveFig = [saveFig, '.png'];
        end
        exportgraphics(fig, saveFig, 'Resolution', 200);
        fprintf('[plot_mc_scatter] 已保存 %s\n', saveFig);
    end
end


function set_odd_power_xticks(ax, xLim)
% 横轴主刻度：10 的奇次幂（…, 10^{-3}, 10^{-1}, 10^{1}, 10^{3}, …）
    pLo = floor(log10(xLim(1)));
    pHi = ceil(log10(xLim(2)));
    exps = pLo:pHi;
    exps = exps(mod(exps, 2) ~= 0);  % 奇数（含负奇数）
    ticks = 10.^exps;
    ticks = ticks(ticks >= xLim(1) * (1 - 1e-12) & ticks <= xLim(2) * (1 + 1e-12));
    if isempty(ticks)
        % 区间内无奇次幂时，仍标出两端 decade
        ticks = unique([10^pLo, 10^pHi]);
        exps = round(log10(ticks));
    end
    ax.XTick = ticks;
    ax.XTickLabel = arrayfun(@(e) sprintf('10^{%d}', e), ...
        round(log10(ticks)), 'UniformOutput', false);
    ax.XMinorTick = 'on';
end


function draw_axis_break(fig, axL, axR)
% 在两轴间隙画断轴斜线
    pL = axL.Position;
    pR = axR.Position;
    x0 = pL(1) + pL(3);
    x1 = pR(1);
    y0 = pL(2);
    y1 = pL(2) + pL(4);
    dx = (x1 - x0) * 0.35;
    dy = 0.012;
    % 底端
    annotation(fig, 'line', [x0 - dx, x0 + dx], [y0 - dy, y0 + dy], ...
        'Color', 'k', 'LineWidth', 1.0);
    annotation(fig, 'line', [x1 - dx, x1 + dx], [y0 - dy, y0 + dy], ...
        'Color', 'k', 'LineWidth', 1.0);
    % 顶端
    annotation(fig, 'line', [x0 - dx, x0 + dx], [y1 - dy, y1 + dy], ...
        'Color', 'k', 'LineWidth', 1.0);
    annotation(fig, 'line', [x1 - dx, x1 + dx], [y1 - dy, y1 + dy], ...
        'Color', 'k', 'LineWidth', 1.0);
end


function S = load_mc_metrics(metricsPath)
    metricsPath = char(metricsPath);
    if ~isfile(metricsPath)
        error('plot_mc_scatter:FileNotFound', '找不到文件: %s', metricsPath);
    end
    [~, ~, ext] = fileparts(metricsPath);
    switch lower(ext)
        case '.mat'
            S = load(metricsPath);
        case '.npz'
            S = load_npz_via_python(metricsPath);
        otherwise
            error('plot_mc_scatter:BadExt', '仅支持 .mat / .npz，收到: %s', ext);
    end
    need = {'r_MD1_min', 'r_MD2_min', 'r_minT', 'evade_ok', 'strike_ok', 'both_ok'};
    for i = 1:numel(need)
        if ~isfield(S, need{i})
            error('plot_mc_scatter:MissingField', '缺少字段 %s', need{i});
        end
        S.(need{i}) = double(S.(need{i})(:));
    end
end


function S = load_npz_via_python(npzPath)
% 优先用已配置的 Python；失败则提示先转成 .mat
    npzPath = char(npzPath);
    matPath = [npzPath(1:end-4), '.mat'];
    try
        np = py.importlib.import_module('numpy');
        d = np.load(npzPath);
        keyList = cell(d.files);
        S = struct();
        for i = 1:numel(keyList)
            k = char(keyList{i});
            raw = d{k};  % NpzFile.__getitem__
            S.(k) = double(raw);
        end
        try %#ok<TRYNC>
            d.close();
        end
    catch ME
        error('plot_mc_scatter:NpzLoadFailed', ...
            ['无法用 MATLAB 的 Python 接口读 npz。\n', ...
             '可先转 mat（推荐）：\n', ...
             '  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe -c ', ...
             '"from scipy.io import savemat; import numpy as np; ', ...
             'd=np.load(r''%s''); savemat(r''%s'', {k: d[k] for k in d.files})"\n', ...
             '原因: %s'], ...
            npzPath, matPath, ME.message);
    end
end


function v = getfield_default(s, name, default)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default;
    end
end
