function [tau, wE_used, cand] = solve_warmstart(M0, D10, D20, cfg, wE_list)
%SOLVE_WARMSTART 为单条样本搜索可行的 warm-start 轨迹 (轻量 wE 扫描)。
%   [tau, wE_used] = solve_warmstart(M0, D10, D20, cfg, wE_list)
%   阶段A 仅需"物理可行"初值: 不被拦截、双防脱靶量 >= d* (是否命中目标不作硬性
%   要求, 留给阶段B 用式22 的 wT 项优化)。由于式47 的规避强度由增益 w/wE 决定
%   (w 固定), 此处按 wE 由大到小 (机动由弱到强) 扫描, 取第一个满足可行条件者
%   (机动最省、且最靠近目标); 若无可行解, 取综合评分最高的尽力解。
%   d* 固定为 cfg.dstar (默认 20)。
%
%   说明: 这是 warm-start 的初值生成, 不是最终最优解; 阶段B 伪谱 NLP 会以此为
%   起点继续优化 [gamma, wT, wE] (w 固定 1e4, d* 固定 20)。

    if nargin < 4 || isempty(cfg); cfg = struct(); end
    if nargin < 5 || isempty(wE_list); wE_list = [1440 700 400 250 150 100 60]; end
    if ~isfield(cfg, 'dstar'); cfg.dstar = 20; end
    dstar = cfg.dstar;

    nW = numel(wE_list);
    cand = struct('wE', cell(1, nW), 'tau', cell(1, nW), ...
                  'feasible', cell(1, nW), 'reach', cell(1, nW), 'score', cell(1, nW));

    bestScore = -inf; bestIdx = 0;
    for i = 1:nW
        c = cfg; c.wE = wE_list(i);
        tau_i = simulate_warmstart_trajectory(M0, D10, D20, c);
        minr = min(tau_i.r_min1, tau_i.r_min2);
        feasible = ~tau_i.intercepted && (minr >= dstar);

        % 综合评分: 优先不被拦截 -> 规避充分 -> 命中目标(次要, r_minT 越小越好)
        score = 1e6 * double(~tau_i.intercepted) ...
              + 100 * min(minr, 3 * dstar) ...
              + 10 * double(tau_i.reached) ...
              - 0.001 * tau_i.r_minT;

        cand(i).wE = wE_list(i); cand(i).tau = tau_i;
        cand(i).feasible = feasible; cand(i).reach = tau_i.reached;
        cand(i).score = score;

        if score > bestScore
            bestScore = score; bestIdx = i;
        end
    end

    % 选择优先级: (规避+达标) 中最大 wE > (规避) 中最大 wE > 评分最高尽力解
    feas = [cand.feasible]; rch = [cand.reach];
    pick = find(feas & rch, 1);          % wE 降序, 第一个即最大
    if isempty(pick); pick = find(feas, 1); end
    if isempty(pick); pick = bestIdx; end

    tau = cand(pick).tau;
    wE_used = cand(pick).wE;
end
