% d = load('expert_lib_1000.mat', 'results');
d = load('expert_lib_p2.mat', 'results');
r = d.results;

fprintf('=== 批处理统计 ===\n')
fprintf('  阶段A成功: %d / %d\n', r.stats.nA_succ, r.N)
fprintf('  阶段B: ok=%d  debug降级=%d  fail=%d\n', r.stats.nB_ok, r.stats.nB_dbg, r.stats.nB_fail)
fprintf('  (s,a)总行数: %d  均每条%.1f步\n', size(r.expert_sa,1), ...
        size(r.expert_sa,1)/max(r.stats.nB_ok+r.stats.nB_dbg,1))
fprintf('  用时: %.1f min\n', r.stats.elapsed_s/60)

% state 13列范围
labels = {'r_MD1','rdot_MD1','qy_D1','qz_D1','r_MD2','rdot_MD2','qy_D2','qz_D2',...
          'r_MT','qy_MT','qz_MT','V_M','tgo_norm'};
fprintf('\n=== state 各列范围 ===\n')
for c = 1:13
    fprintf('  col%02d %-10s [%9.2f, %9.2f]\n', c, labels{c}, ...
            min(r.expert_sa(:,c)), max(r.expert_sa(:,c)))
end

% action 4列统计
act = {'w','gamma','wT','wE'};
fprintf('\n=== action 各列统计 ===\n')
for c = 1:4
    col = r.expert_sa(:, 13+c);
    fprintf('  %-6s  min=%9.2f  max=%9.2f  mean=%9.2f std=%9.2f\n', act{c}, min(col), max(col), mean(col), std(col))
end