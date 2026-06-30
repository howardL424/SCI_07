function check_expert_sa(r, name)
    sa = r.expert_sa;
    fprintf('\n===== %s expert_sa =====\n', name)
    fprintf('  size = %d x %d  (应为 N x 17)\n', size(sa,1), size(sa,2))
    assert(size(sa,2) == 17, '列数不对')

    state_labels = {'r_MD1','rdot_MD1','qy_D1','qz_D1','r_MD2','rdot_MD2', ...
                    'qy_D2','qz_D2','r_MT','qy_MT','qz_MT','V_M','tgo_norm'};
    act_labels   = {'w','gamma','wT','wE'};

    fprintf('\n--- state ---\n')
    for c = 1:13
        col = sa(:,c);
        fprintf('  %2d %-10s  min=%10.3f  max=%10.3f  NaN=%d\n', ...
            c, state_labels{c}, min(col), max(col), sum(isnan(col)))
    end

    fprintf('\n--- action ---\n')
    for c = 1:4
        col = sa(:,13+c);
        fprintf('  %-6s  min=%12.3f  max=%12.3f  mean=%12.3f  NaN=%d\n', ...
            act_labels{c}, min(col), max(col), mean(col), sum(isnan(col)))
    end

    % 快速 sanity check
    assert(all(sa(:,14) == 1e4), 'w 应恒为 1e4')
    assert(all(sa(:,13) >= 0), 'tgo_norm 不应为负')
    assert(max(sa(:,13)) <= 1.001, 'tgo_norm 不应明显大于 1')
    assert(all(sa(:,12) > 0), 'V_M 应 > 0')
    fprintf('  基本 sanity check 通过\n')
end

% check_expert_sa(r2, 'P2')
check_expert_sa(results, 'P3')