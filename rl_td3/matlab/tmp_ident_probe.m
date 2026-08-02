function tmp_ident_probe()
%TMP_IDENT_PROBE  临时探针：θ_true 时变程度 + 回归量条件数。
    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir); addpath(fullfile(this_dir,'..','..'));
    addpath(fullfile(this_dir,'..','..','expert_trajectory'));

    cfg = struct('coeff_mode','sparse_ffrls','ident_apply_cmd',false, ...
        'sigma_r',0,'sigma_rdot',0,'noise_seed',0);
    core = stage2_core_init(cfg);

    M  = [8000, 0, 2000, 0.05, 0.02, 800];
    D1 = [3000, 500, 1200, 0, 0, 900];
    D2 = [3000, -500, 1200, 0, 0, 900];
    sabs = [M(:); D1(:); D2(:)];
    an = [0; 0; 0];
    Tev = 6; dt = 0.1; n = round(Tev/dt);

    ident = [];
    TH = nan(n, 4); PHI = nan(n, 4); RDD_T = nan(n,1); RDD_H = nan(n,1);
    t = 0;
    for k = 1:n
        [~, sabs_n, ~, fact, ident] = stage2_step(core, sabs, t, Tev, an, ident);
        d = fact.ident_diag;
        if isfield(d,'theta_true')
            TH(k,:) = d.theta_true{1}(:)';       % D1 通道 [A21;B1;B2;C1]
            RDD_T(k) = d.rddot_true(1);
            RDD_H(k) = d.rddot_hat(1);
            PHI(k,:) = [d.r_hat(1), fact.a_M_sat(1), fact.a_M_sat(2), 1];
        end
        sabs = sabs_n; t = t + dt;
        if fact.terminated; break; end
    end
    ok = ~isnan(TH(:,1));
    TH = TH(ok,:); PHI = PHI(ok,:); RDD_T = RDD_T(ok); RDD_H = RDD_H(ok);

    fprintf('步数 = %d\n\n', size(TH,1));
    names = {'A21','B1','B2','C1'};
    fprintf('--- theta_true(D1) 时变程度 ---\n');
    for j = 1:4
        v = TH(:,j);
        fprintf('  %-4s  min=%11.4g  max=%11.4g  mean=%11.4g  std=%10.4g  std/|mean|=%8.3f\n', ...
            names{j}, min(v), max(v), mean(v), std(v), std(v)/max(abs(mean(v)),1e-12));
    end
    % 相邻步相对变化
    fprintf('\n--- 相邻步 |Δθ|/|θ| 中位数（λ=0.98 记忆约 50 步）---\n');
    for j = 1:4
        dv = abs(diff(TH(:,j))) ./ max(abs(TH(1:end-1,j)), 1e-12);
        fprintf('  %-4s  median=%.4f   50步累计变化倍数=%.3g\n', names{j}, ...
            median(dv), max(abs(TH(:,j)))/max(min(abs(TH(:,j))),1e-12));
    end

    fprintf('\n--- 回归量 phi 条件数（50 步滑窗，缩放前/后）---\n');
    S = diag([1e4, 50, 50, 1]);
    w = min(50, size(PHI,1));
    Praw = PHI(end-w+1:end,:);
    Psc  = (S \ Praw')';
    fprintf('  cond(phi_raw)   = %.4g\n', cond(Praw));
    fprintf('  cond(phi_scaled)= %.4g\n', cond(Psc));
    G = Psc' * Psc / w;
    fprintf('  Gram 特征值     = %s\n', mat2str(sort(eig(G))', 4));

    fprintf('\n--- KF r̈ 跟踪（σ=0 时）---\n');
    v = ~isnan(RDD_T);
    fprintf('  RMSE(rddot_hat - rddot_true) = %.4g\n', sqrt(mean((RDD_H(v)-RDD_T(v)).^2)));
    fprintf('  std(rddot_true) = %.4g   range=[%.4g, %.4g]\n', ...
        std(RDD_T(v)), min(RDD_T(v)), max(RDD_T(v)));

    fprintf('\n--- 用 theta_true 复现 rddot：不同时序配对 ---\n');
    fprintf('  基准 std(rddot_true) = %.4g（RMSE 需远小于它模型才有效）\n', std(RDD_T(~isnan(RDD_T))));
    N = size(TH,1);
    idx = 3:N;   % 保证 k-1 可用且 rddot_true 有效
    idx = idx(~isnan(RDD_T(idx)));

    pa = sum(PHI(idx,:)   .* TH(idx,:),   2);        % 同拍 phi(k)·θ(k)
    pb = sum(PHI(idx-1,:) .* TH(idx,:),   2);        % 实现：phi(k-1)·θ(k)
    pc = sum(PHI(idx-1,:) .* TH(idx-1,:), 2);        % 全滞后 phi(k-1)·θ(k-1)
    pd = 0.5*(pa + pc);                              % 中点平均
    y  = RDD_T(idx);
    rmse = @(p) sqrt(mean((p - y).^2));
    fprintf('  (a) phi(k)  ·theta(k)   RMSE=%8.4g   corr=%.3f\n', rmse(pa), corr(pa,y));
    fprintf('  (b) phi(k-1)·theta(k)   RMSE=%8.4g   corr=%.3f   <== 当前实现\n', rmse(pb), corr(pb,y));
    fprintf('  (c) phi(k-1)·theta(k-1) RMSE=%8.4g   corr=%.3f\n', rmse(pc), corr(pc,y));
    fprintf('  (d) 中点平均            RMSE=%8.4g   corr=%.3f\n', rmse(pd), corr(pd,y));

    fprintf('\n--- 各项贡献量级（同拍分解）---\n');
    contrib = PHI(idx,:) .* TH(idx,:);
    nm = {'A21*r','B1*ay','B2*az','C1'};
    for j = 1:4
        fprintf('  %-6s  mean=%10.4g  std=%10.4g  max|.|=%10.4g\n', ...
            nm{j}, mean(contrib(:,j)), std(contrib(:,j)), max(abs(contrib(:,j))));
    end

    fprintf('\n--- 若用最小二乘直接拟合（理论最好情况）---\n');
    Xs = (diag([1e4,50,50,1]) \ PHI(idx-1,:)')';
    th_ls = Xs \ y;
    fprintf('  batch LS 残差 RMSE = %.4g  (vs std(y)=%.4g)  R^2=%.3f\n', ...
        sqrt(mean((Xs*th_ls - y).^2)), std(y), 1 - mean((Xs*th_ls-y).^2)/var(y));
end
