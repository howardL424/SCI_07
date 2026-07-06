%TEST_STAGE3_GUIDANCE_COMPARISON 对比 stage3 使用 PN vs 式47（多组增益）。
%   仅改动 stage3；stage1 仍为 PN，stage2 仍为式47。
%   增益匹配依据：PN 等效增益 KM·V ≈ 3×300 = 900 m/s²，
%   式47 目标项增益 ≈ (wT/wE)·tgo；tgo_stage3 ≈ 10s → 需 wT/wE ≈ 90。
%   若匹配变体 reached% ≥ 94%，则额外输出能量对比。
%
%   用法:
%     test_stage3_guidance_comparison          % 20条
%     test_stage3_guidance_comparison(50)      % 50条
%     out = test_stage3_guidance_comparison(50, true);  % 50条 + 保存图

function out = test_stage3_guidance_comparison(nRows, saveFig)
if nargin < 1 || isempty(nRows); nRows = 20; end
if nargin < 2 || isempty(saveFig); saveFig = false; end

scriptDir = fileparts(mfilename('fullpath'));
expertRoot = fileparts(scriptDir);
addpath(expertRoot);
setup_expert_path();
[~, projectRoot] = project_paths();

samples_file = fullfile(projectRoot, 'samples_clean.mat');
if ~isfile(samples_file)
    error('未找到 %s，请先生成 samples_clean.mat', samples_file);
end
S = load(samples_file, 'samples_clean');
sc = S.samples_clean;
N = min(nRows, size(sc, 1));
fprintf('加载 samples_clean: %d 行，本次测试前 %d 条\n', size(sc,1), N);

% ---- 增益说明 ----
% PN:            KM·V ≈ 3×300 = 900 m/s²
% 式47目标项增益 ≈ (wT/wE)·tgo（tgo_stage3 ≈ 10s）
% 各变体 wT/wE 与对应等效增益：
%   optimal(原始):       1e3/1440 ≈ 0.69  → tgo10 ≈ 6.9
%   optimal_boost(上轮): 1e5/1440 ≈ 69    → tgo10 ≈ 690
%   matched_90:          9000/100  = 90   → tgo10 ≈ 900  (与PN匹配)
%   matched_150:         1.5e4/100 = 150  → tgo10 ≈ 1500 (适度超调)
variants = { ...
    struct('name', 'pn',           'stage3_guidance','pn',      'stage3_wT',[], 'stage3_wE',[]), ...
    struct('name', 'opt_raw',      'stage3_guidance','optimal', 'stage3_wT',[], 'stage3_wE',[]), ...
    struct('name', 'opt_boost',    'stage3_guidance','optimal', 'stage3_wT',1e5,'stage3_wE',[]), ...
    struct('name', 'matched_90',   'stage3_guidance','optimal', 'stage3_wT',9e3,'stage3_wE',100), ...
    struct('name', 'matched_150',  'stage3_guidance','optimal', 'stage3_wT',1.5e4,'stage3_wE',100) ...
};
nVar = numel(variants);

baseCfg = struct('w',1e4,'gamma',0,'wE',1440,'wT',1e3,'dstar',20, ...
                 'dt',1e-3,'t_max',60,'r_safe',8000);

results(N*nVar) = struct('row',0,'variant','', ...
    'reached',false,'diverged',false,'intercepted',false,'success',false, ...
    'r_min1',NaN,'r_min2',NaN,'r_minT',NaN, ...
    'energy_total',NaN,'energy_stage1',NaN,'energy_stage2',NaN,'energy_stage3',NaN, ...
    'tau',[]);

idx = 0;
for i = 1:N
    [M0, D10, D20] = reconstruct_geometry(sc(i,:));
    for v = 1:nVar
        idx = idx + 1;
        cfg = baseCfg;
        cfg.stage3_guidance = variants{v}.stage3_guidance;
        cfg.stage3_wT       = variants{v}.stage3_wT;
        cfg.stage3_wE       = variants{v}.stage3_wE;

        tau = simulate_warmstart_trajectory(M0, D10, D20, cfg);

        results(idx).row        = i;
        results(idx).variant    = variants{v}.name;
        results(idx).reached    = tau.reached;
        results(idx).diverged   = tau.diverged;
        results(idx).intercepted= tau.intercepted;
        results(idx).success    = tau.success;
        results(idx).r_min1     = tau.r_min1;
        results(idx).r_min2     = tau.r_min2;
        results(idx).r_minT     = tau.r_minT;
        results(idx).energy_total  = tau.energy_total;
        results(idx).energy_stage1 = tau.energy_stage1;
        results(idx).energy_stage2 = tau.energy_stage2;
        results(idx).energy_stage3 = tau.energy_stage3;
        results(idx).tau        = tau;
    end
    if mod(i,10)==0
        fprintf('  已完成 %d / %d 条样本\n', i, N);
    end
end

fprintf('\n========== stage3 制导律对比 (N=%d) ==========\n', N);
fprintf('%-14s | wT/wE比 | reached%% | diverged%% | r_minT_med |  r_min1_med | r_min2_med\n','variant');
fprintf('%s\n', repmat('-', 1, 86));

summary = struct();
pn_reached_pct = NaN;
for v = 1:nVar
    name = variants{v}.name;
    mask = strcmp({results.variant}, name);
    R    = results(mask);

    pct  = @(x) 100*mean([R.(x)]);
    med  = @(fld) median([R.(fld)]);

    wT3 = variants{v}.stage3_wT; if isempty(wT3); wT3 = baseCfg.wT; end
    wE3 = variants{v}.stage3_wE; if isempty(wE3); wE3 = baseCfg.wE; end
    ratio_str = '';
    if strcmp(variants{v}.stage3_guidance,'optimal')
        ratio_str = sprintf('%5.0f',wT3/wE3);
    else
        ratio_str = '  PN ';
    end

    summary.(name) = struct( ...
        'reached_pct',   pct('reached'), ...
        'diverged_pct',  pct('diverged'), ...
        'intercepted_pct',pct('intercepted'), ...
        'r_minT_med',    med('r_minT'), ...
        'r_min1_med',    med('r_min1'), ...
        'r_min2_med',    med('r_min2'), ...
        'energy_total_med',  med('energy_total'), ...
        'energy_stage3_med', med('energy_stage3'));

    fprintf('%-14s | %7s | %7.1f%% | %9.1f%% | %10.1f | %11.1f | %10.1f\n', ...
        name, ratio_str, summary.(name).reached_pct, summary.(name).diverged_pct, ...
        summary.(name).r_minT_med, summary.(name).r_min1_med, summary.(name).r_min2_med);

    if strcmp(name,'pn'); pn_reached_pct = summary.(name).reached_pct; end
end

% 规避一致性检查
pn_mask  = strcmp({results.variant},'pn');
opt_mask = strcmp({results.variant},'matched_90');
r1_diff  = abs([results(pn_mask).r_min1] - [results(opt_mask).r_min1]);
r2_diff  = abs([results(pn_mask).r_min2] - [results(opt_mask).r_min2]);
fprintf('\n规避一致性 (pn vs matched_90): max|r_min1差|=%.2f m, max|r_min2差|=%.2f m\n',...
    max(r1_diff), max(r2_diff));

% 判断是否有变体达到 94% reached
fprintf('\n');
reached_vec = zeros(1,nVar);
varNames = cellfun(@(v) v.name, variants, 'UniformOutput', false);
for v = 1:nVar; reached_vec(v) = summary.(variants{v}.name).reached_pct; end
good_idx = find(reached_vec >= 94 & ~strcmp(varNames,'pn'));

% 始终输出所有变体能量对比（便于分析）
fprintf('====== 能量对比（所有变体，中位数，以 pn 归一化）======\n');
fprintf('%-14s | reached%% | energy_total  | energy_stage3 | e_s3占比%% | vs_pn_total%% | vs_pn_s3%%\n','variant');
fprintf('%s\n', repmat('-', 1, 90));
pn_et = summary.pn.energy_total_med;
pn_e3 = summary.pn.energy_stage3_med;
for v = 1:nVar
    name = variants{v}.name;
    et = summary.(name).energy_total_med;
    e3 = summary.(name).energy_stage3_med;
    fprintf('%-14s | %8.1f%% | %13.2f | %13.2f | %9.1f%% | %+12.1f%% | %+9.1f%%\n', ...
        name, summary.(name).reached_pct, et, e3, 100*e3/et, 100*(et/pn_et-1), 100*(e3/pn_e3-1));
end

if ~isempty(good_idx)
    fprintf('\n达到 reached≥94%% 的式47变体: ');
    for vi = good_idx; fprintf('%s ', variants{vi}.name); end
    fprintf('\n');
else
    fprintf('\n没有式47变体达到 reached≥94%%。\n');
end

% 轨迹图：r_MT(t)，选3条代表样本
plot_rows = unique(round(linspace(1, N, min(3,N))));
fig = figure('Name','stage3 guidance comparison','Position',[60 60 1300 420]);
colors = lines(nVar);
for k = 1:numel(plot_rows)
    subplot(1,numel(plot_rows),k); hold on; grid on;
    for v = 1:nVar
        m = strcmp({results.variant},variants{v}.name) & ([results.row]==plot_rows(k));
        tau = results(m).tau;
        plot(tau.t, tau.r_MT,'Color',colors(v,:),'LineWidth',1.2,...
            'DisplayName',variants{v}.name);
    end
    xlabel('t (s)'); ylabel('r_{MT} (m)');
    title(sprintf('样本 row %d', plot_rows(k)));
    if k==1; legend('Location','best','FontSize',7); end
end
sgtitle(sprintf('stage3制导律对比 (N=%d)  PN reached=%.0f%%', N, pn_reached_pct));

if saveFig
    outDir = fullfile(expertRoot,'output');
    if ~isfolder(outDir); mkdir(outDir); end
    figPath = fullfile(outDir,'stage3_guidance_comparison.png');
    saveas(fig,figPath);
    fprintf('图已保存: %s\n', figPath);
end

out = struct('results',results,'summary',summary,'variants',{variants},'N',N);
end
