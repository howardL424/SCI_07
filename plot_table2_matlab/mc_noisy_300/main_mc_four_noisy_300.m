%% 四方法带测量噪声蒙特卡洛（不带辨识）
%  方法: 纯 PN / 正弦机动 / 纯 TD3 / BC-TD3
%  初态: samples_clean_test.mat 前 N 条（与无噪 MC / evaluate_td3 同序）
%  噪声: 参数辨识默认 σ × sigma_scale；TD3/BC-TD3 走 analytic_noisy（无 RLS）
%  判据: evade r_MD≥6m；strike r_minT≤10m；both = 两者同时成立
%
%  用法:
%    main_mc_four_noisy_300                      % 默认 N=300, σ×30, 四方法
%    main_mc_four_noisy_300('N', 2)              % 烟雾
%    main_mc_four_noisy_300('sigma_scale', 3)    % 3 倍噪声
%    main_mc_four_noisy_300('methods', {'pn','sine'})  % 只跑基线
%    main_mc_four_noisy_300('methods', {'td3','bctd3'}) % 只跑 RL

function main_mc_four_noisy_300(varargin)
    this_dir = fileparts(mfilename('fullpath'));
    plot_root = fileparts(this_dir);
    gail_root = fileparts(plot_root);
    mc_base = fullfile(plot_root, 'mc_baseline_300');
    addpath(this_dir);
    addpath(mc_base);
    addpath(fullfile(gail_root, 'expert_trajectory'));

    p = inputParser;
    addParameter(p, 'N', 300, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'methods', {'pn', 'sine', 'td3', 'bctd3'}, ...
        @(c) iscell(c) || ischar(c) || isstring(c));
    addParameter(p, 'samples', fullfile(gail_root, 'samples_clean_test.mat'), @ischar);
    % addParameter(p, 'sigma_scale', 1.0, @isnumeric);   % 原默认：1×
    addParameter(p, 'sigma_scale', 30.0, @isnumeric);     % 2026-08-06：改为 30×
    addParameter(p, 'noise_seed', 0, @isnumeric);
    addParameter(p, 'seed', 0, @isnumeric);
    addParameter(p, 'progress_every', 10, @isnumeric);
    addParameter(p, 'py_exe', 'E:\Python\venvs\rl_pytorch\Scripts\python.exe', @ischar);
    addParameter(p, 'skip_pn_sine', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'skip_rl', false, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});
    opt = p.Results;

    if ischar(opt.methods) || isstring(opt.methods)
        methods = {char(opt.methods)};
    else
        methods = cellfun(@(c) lower(char(c)), opt.methods, 'UniformOutput', false);
    end

    out_tag = sprintf('sigma_x%g', opt.sigma_scale);
    out_dir = fullfile(this_dir, 'results', out_tag);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    log_path = fullfile(out_dir, sprintf('mc_four_noisy_N%d_%s.log', ...
        opt.N, datestr(now, 'yyyymmdd_HHMMSS'))); %#ok<TNOW1,DATST>
    diary(log_path);
    cleanup = onCleanup(@() diary('off')); %#ok<NASGU>

    fprintf('==== 四方法带噪 MC（不带辨识）====\n');
    fprintf('N=%d  σ×%g  methods={%s}\n', opt.N, opt.sigma_scale, strjoin(methods, ', '));
    fprintf('samples=%s\n', opt.samples);
    fprintf('out=%s\n\n', out_dir);

    summary = {};

    %% 1) PN / 正弦（MATLAB，enable_noise=true）
    base_methods = intersect(methods, {'pn', 'sine'}, 'stable');
    if ~isempty(base_methods) && ~opt.skip_pn_sine
        main_mc_pn_sine_300( ...
            'N', opt.N, ...
            'methods', base_methods, ...
            'samples', opt.samples, ...
            'seed', opt.seed, ...
            'enable_noise', true, ...
            'sigma_scale', opt.sigma_scale, ...
            'noise_seed', opt.noise_seed, ...
            'progress_every', opt.progress_every, ...
            'out_dir', out_dir);
        summary = [summary, load_latest_base_stats(out_dir, base_methods)]; %#ok<AGROW>
    end

    %% 2) TD3 / BC-TD3（Python，analytic_noisy）
    rl_methods = intersect(methods, {'td3', 'bctd3'}, 'stable');
    if ~isempty(rl_methods) && ~opt.skip_rl
        if numel(rl_methods) == 2
            algo = 'both';
        else
            algo = rl_methods{1};
        end
        py_script = fullfile(this_dir, 'run_mc_rl_noisy.py');
        cmd = sprintf(['"%s" "%s" --algo %s --episodes %d --sigma-scale %g ' ...
            '--noise-seed %d --seed %d --samples "%s" --out-dir "%s" ' ...
            '--progress-every %d'], ...
            opt.py_exe, py_script, algo, opt.N, opt.sigma_scale, ...
            opt.noise_seed, opt.seed, opt.samples, out_dir, opt.progress_every);
        fprintf('\n调用 Python RL MC:\n%s\n\n', cmd);
        status = system(cmd);
        if status ~= 0
            error('main_mc_four_noisy_300:PyFail', 'Python RL MC 失败，exit=%d', status);
        end
        summary = [summary, load_latest_rl_stats(out_dir, rl_methods)]; %#ok<AGROW>
    end

    %% 3) 总表
    if ~isempty(summary)
        fprintf('\n==================== 四方法汇总（σ×%g）====================\n', ...
            opt.sigma_scale);
        fprintf('%-8s  %6s  %12s  %12s  %12s\n', ...
            '方法', 'N', '双规避成功', '打击成功', '联合成功');
        for k = 1:numel(summary)
            s = summary{k};
            fprintf('%-8s  %6d  %5.1f%%         %5.1f%%         %5.1f%%\n', ...
                upper(s.mode), s.N, ...
                100 * s.rate_evade, 100 * s.rate_strike, 100 * s.rate_both);
        end
        fprintf('===========================================================\n');
    end
    fprintf('\n日志: %s\n', log_path);
end


function rows = load_latest_base_stats(out_dir, methods)
    rows = {};
    d = dir(fullfile(out_dir, 'mc_*_noisy_*.mat'));
    if isempty(d)
        return;
    end
    [~, ord] = sort([d.datenum], 'descend');
    d = d(ord);
    seen = containers.Map();
    for i = 1:numel(d)
        S = load(fullfile(d(i).folder, d(i).name), 'result');
        if ~isfield(S, 'result') || ~isfield(S.result, 'mode')
            continue;
        end
        mode = lower(char(S.result.mode));
        if ~ismember(mode, methods) || seen.isKey(mode)
            continue;
        end
        seen(mode) = true;
        st = S.result.stats;
        rows{end+1} = struct( ...
            'mode', mode, ...
            'N', st.N, ...
            'rate_evade', st.rate_evade, ...
            'rate_strike', st.rate_strike, ...
            'rate_both', st.rate_both); %#ok<AGROW>
        if seen.Count == numel(methods)
            break;
        end
    end
end


function rows = load_latest_rl_stats(out_dir, methods)
    rows = {};
    d = dir(fullfile(out_dir, 'summary_*_x*.json'));
    if isempty(d)
        return;
    end
    [~, ord] = sort([d.datenum], 'descend');
    d = d(ord);
    seen = containers.Map();
    for i = 1:numel(d)
        txt = fileread(fullfile(d(i).folder, d(i).name));
        j = jsondecode(txt);
        if ~isfield(j, 'algo')
            continue;
        end
        mode = lower(char(j.algo));
        if ~ismember(mode, methods) || seen.isKey(mode)
            continue;
        end
        seen(mode) = true;
        rows{end+1} = struct( ...
            'mode', mode, ...
            'N', j.n, ...
            'rate_evade', j.rate_evade, ...
            'rate_strike', j.rate_strike, ...
            'rate_both', j.rate_both); %#ok<AGROW>
        if seen.Count == numel(methods)
            break;
        end
    end
end
