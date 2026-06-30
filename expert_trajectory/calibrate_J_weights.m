function out = calibrate_J_weights(nRows, doSolve)
%CALIBRATE_J_WEIGHTS 标定 J 三项固定系数 s_w/s_wE/s_wT (消除贴边界)。
%   out = calibrate_J_weights(nRows, doSolve)
%   流程:
%     1) 对 samples_clean 前 nRows 行跑阶段A, 筛"规避成功且含规避段"的轨迹;
%     2) 在 warm-start θ=[0;wT0;wE0] 处实测三项原始量级:
%        A=(miss1-d*)^2+(miss2-d*)^2 (规避), E=∫‖u‖²dt (能量), Tt=missT^2 (目标);
%     3) 按"三项加权贡献平衡 + 规避项加成 k_avoid"算建议 s_w/s_wE/s_wT;
%     4) doSolve=true 时用新系数对前 1~2 条正常求解, 校验 θ 内点 / miss≈d* / 耗时。
%   返回 out: 含逐条原始量级、参考量级、建议系数、验证解。

    if nargin < 1 || isempty(nRows);   nRows = 6;    end
    if nargin < 2 || isempty(doSolve); doSolve = true; end

    setup_expert_path();
    [~, projectRoot] = project_paths();
    add_casadi_path();

    % ---------------- 标定常数 ----------------
    C       = 1e5;          % 三项加权后的公共目标量级
    k_avoid = 2;            % 规避项温和加成 (把 miss 钉在 d* 而非塌到下限)
    dstar   = 20;
    dfloor  = 6;
    A_floor = 2 * (dstar - dfloor)^2;   % 规避项"塌到下限"的惩罚基准 = 392

    % ---------------- 阶段A + 实测原始量级 ----------------
    samples_file = fullfile(projectRoot, 'samples_clean.mat');
    S  = load(samples_file, 'samples_clean');
    sc = S.samples_clean;
    valid = find(any(sc ~= 0, 2));
    nRows = min(nRows, numel(valid));

    sel = struct('row', {}, 'tau', {}, 'A', {}, 'E', {}, 'Tt', {}, ...
                 'miss1', {}, 'miss2', {}, 'missT', {});
    fprintf('==== 标定测量 (阶段A warm θ 处的三项原始量级) ====\n');
    fprintf('%5s | %8s %8s %8s | %8s %8s %8s\n', ...
            'row', 'A', 'E', 'Tt', 'miss1', 'miss2', 'missT');
    for k = 1:nRows
        row = sc(valid(k), :);
        [M0, D10, D20] = reconstruct_geometry(row);
        tau = solve_warmstart(M0, D10, D20);
        if ~(tau.success && any(tau.stage == 2))
            fprintf('%5d | (跳过: 非规避成功或无规避段)\n', valid(k));
            continue;
        end
        r = solve_ss_trajectory(tau, struct('report_terms_only', true, ...
                                             'dstar', dstar, 'dfloor', dfloor));
        if ~r.ok; continue; end
        tr = r.terms_raw;
        j = numel(sel) + 1;
        sel(j).row = valid(k); sel(j).tau = tau;
        sel(j).A = tr.A; sel(j).E = tr.E; sel(j).Tt = tr.Tt;
        sel(j).miss1 = tr.miss1; sel(j).miss2 = tr.miss2; sel(j).missT = tr.missT;
        fprintf('%5d | %8.3g %8.3g %8.3g | %8.2f %8.2f %8.2f\n', ...
                valid(k), tr.A, tr.E, tr.Tt, tr.miss1, tr.miss2, tr.missT);
    end
    if isempty(sel)
        error('calibrate_J_weights:noTraj', '前 %d 行无可用规避成功轨迹', nRows);
    end

    % ---------------- 参考量级 (中位数, 抗离群) ----------------
    E_ref  = median([sel.E]);
    Tt_ref = median([sel.Tt]);

    % ---------------- 平衡式 ----------------
    %   规避项 = (s_w/4)*A,  能量项 = (s_wE/2)*E,  目标项 = (s_wT/2)*Tt
    %   令三项加权贡献均 ~C, 规避项额外 ×k_avoid:
    s_w  = round2sig(k_avoid * 4 * C / A_floor, 2);
    s_wE = round2sig(2 * C / E_ref,  2);
    s_wT = round2sig(2 * C / Tt_ref, 2);

    fprintf('\n==== 参考量级 (中位数) ====\n');
    fprintf('E_ref = %.4g,  Tt_ref = %.4g,  A_floor = %g (固定)\n', E_ref, Tt_ref, A_floor);
    fprintf('\n==== 建议固定系数 (C=%.0e, k_avoid=%g) ====\n', C, k_avoid);
    fprintf('s_w  = %.4g   (旧 1e4)\n', s_w);
    fprintf('s_wE = %.4g   (旧 1.44e3)\n', s_wE);
    fprintf('s_wT = %.4g   (旧 1e3)\n', s_wT);
    % 校核: 用新系数在 A_floor/E_ref/Tt_ref 处的加权贡献
    fprintf('校核加权贡献: 规避(@floor)=%.3g  能量=%.3g  目标=%.3g\n', ...
            0.25*s_w*A_floor, 0.5*s_wE*E_ref, 0.5*s_wT*Tt_ref);

    out = struct('sel', sel, 'E_ref', E_ref, 'Tt_ref', Tt_ref, ...
                 'A_floor', A_floor, 'C', C, 'k_avoid', k_avoid, ...
                 's_w', s_w, 's_wE', s_wE, 's_wT', s_wT);

    % ---------------- 验证求解 (新系数, 前 1~2 条) ----------------
    if doSolve
        nV = min(2, numel(sel));
        cfg = struct('s_w', s_w, 's_wE', s_wE, 's_wT', s_wT, ...
                     'dstar', dstar, 'dfloor', dfloor);
        fprintf('\n==== 验证求解 (新系数, %d 条) ====\n', nV);
        out.verify = struct('row', {}, 'theta_first', {}, 'theta_last', {}, ...
                            'miss', {}, 'terms', {}, 'sec', {}, 'ok', {});
        for v = 1:nV
            tic;
            r = solve_ss_trajectory(sel(v).tau, cfg);
            sec = toc;
            out.verify(v).row = sel(v).row; out.verify(v).ok = r.ok; out.verify(v).sec = sec;
            fprintf('-- row %d --  ok=%d  耗时=%.1fs\n', sel(v).row, r.ok, sec);
            if isfield(r, 'theta') && ~isempty(r.theta)
                th1 = r.theta(:, 1); thE = r.theta(:, end);
                out.verify(v).theta_first = th1; out.verify(v).theta_last = thE;
                fprintf('   θ首段 [γ wT wE] = [% .3f  %8.1f  %8.1f]\n', th1(1), th1(2), th1(3));
                fprintf('   θ末段 [γ wT wE] = [% .3f  %8.1f  %8.1f]\n', thE(1), thE(2), thE(3));
                fprintf('   边界检查 γ∈(-3,3) wT∈(1,1e5) wE∈(10,1e4)\n');
            end
            if isfield(r, 'miss')
                out.verify(v).miss = r.miss;
                fprintf('   miss = [%.2f  %.2f  zT=%.2f]  (目标 miss≈%d)\n', ...
                        r.miss(1), r.miss(2), r.miss(3), dstar);
            end
            if isfield(r, 'terms')
                out.verify(v).terms = r.terms;
                t = r.terms;
                fprintf('   加权三项: 规避=%.3g 能量=%.3g 目标=%.3g\n', ...
                        t.avoid, t.energy, t.target);
            end
        end
    end
end

% -------- 取 n 位有效数字 --------
function y = round2sig(x, n)
    if x == 0; y = 0; return; end
    d = n - ceil(log10(abs(x)));
    y = round(x * 10^d) / 10^d;
end

% -------- 定位并添加 CasADi 路径 --------
function add_casadi_path()
    import casadi.*  %#ok<NSTIMP>
    if exist('casadi.Opti', 'class'); return; end
    casadiRoot = 'E:\matlab\casadi-3.7.2-windows64-matlab2018b';
    toolboxRoot = fullfile(matlabroot, 'toolbox', 'casadi');
    if isfolder(toolboxRoot) && isfile(fullfile(toolboxRoot, 'casadiMEX.mexw64'))
        addpath(toolboxRoot);
    elseif isfolder(casadiRoot) && isfile(fullfile(casadiRoot, 'casadiMEX.mexw64'))
        addpath(casadiRoot);
    else
        error('calibrate_J_weights:MissingCasADi', '未找到 CasADi: %s', casadiRoot);
    end
end
