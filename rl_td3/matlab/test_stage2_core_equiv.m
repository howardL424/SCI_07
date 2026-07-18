function ok = test_stage2_core_equiv()
%TEST_STAGE2_CORE_EQUIV  env_step 与预初始化 stage2_step 逐步等价冒烟。
%
%   ok = test_stage2_core_equiv()
%   用固定初态 + 固定动作序列对比 obs/sabs/reward/terminated。

    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);
    addpath(fullfile(this_dir, '..', '..'));
    addpath(fullfile(this_dir, '..', '..', 'expert_trajectory'));

    % 简单初态：攻弹在远处，两防在侧前方
    M  = [8000, 0, 2000, 0, 0, 800];
    D1 = [6000, 500, 1500, 0, 0, 700];
    D2 = [6000, -500, 1500, 0, 0, 700];
    sabs0 = [M(:); D1(:); D2(:)];
    Tev = 2.0;
    actions = [ ...
        0.0, 0.0, 0.0; ...
        0.5, -0.3, 0.2; ...
       -0.8, 0.9, -0.5; ...
        0.2, 0.1, 0.0 ...
    ];

    core = stage2_core_init(struct());
    sabs_a = sabs0;
    sabs_b = sabs0;
    t_a = 0; t_b = 0;
    tol = 1e-9;
    ok = true;

    for i = 1:size(actions, 1)
        an = actions(i, :)';

        [obs1, s1, r1, term1, info1] = env_step(sabs_a, t_a, Tev, 0.0, an, struct());
        [obs2, s2, t_new, fact] = stage2_step(core, sabs_b, t_b, Tev, an);
        r2 = stage2_process_reward(core, fact);

        err_obs = max(abs(obs1(:) - obs2(:)));
        err_s   = max(abs(s1(:) - s2(:)));
        err_r   = abs(r1 - r2);
        same_term = (logical(term1) == fact.terminated);
        same_int  = (logical(info1(4)) == fact.intercepted);
        same_s2   = (logical(info1(5)) == fact.stage2_done);

        if err_obs > tol || err_s > tol || err_r > tol || ~same_term || ~same_int || ~same_s2
            fprintf('[FAIL] step %d obs=%.3g sabs=%.3g r=%.3g term=%d/%d int=%d/%d s2=%d/%d\n', ...
                i, err_obs, err_s, err_r, term1, fact.terminated, ...
                info1(4), fact.intercepted, info1(5), fact.stage2_done);
            ok = false;
            break;
        end

        sabs_a = s1; t_a = t_a + 0.1;
        sabs_b = s2; t_b = t_new;
        if term1
            break;
        end
    end

    % 动作映射往返
    [an, ~] = phys_to_action_norm(1.5, 3e3, 5e2);
    [g, wt, we] = action_norm_to_phys(an);
    [an2, ~] = phys_to_action_norm(g, wt, we);
    if max(abs(an(:) - an2(:))) > 1e-12
        fprintf('[FAIL] action map roundtrip\n');
        ok = false;
    end

    meta = transition_schema_meta();
    if ~strcmp(meta.schema_version, 'phys_transition_v1')
        fprintf('[FAIL] schema_version\n');
        ok = false;
    end

    if ok
        fprintf('[OK] stage2_core ↔ env_step 等价 + schema meta\n');
    end
end
