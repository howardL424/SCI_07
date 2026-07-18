function reward = stage2_process_reward(core, fact)
%STAGE2_PROCESS_REWARD  由物理终止事实计算过程奖励（在线 env / 旧 aligned 用）。
%
%   物理 transition 层不写死 -50；本函数供在线 env_step 与 rebuild 投影。
%   拦截: -intercept_pen；否则 0.05*(R2+R3) 软饱和 log 距离项。

    cfg = core.cfg;
    if fact.intercepted
        reward = -cfg.intercept_pen;
        return;
    end
    R2 = soft_sat_log_evade(fact.r_MD1, cfg.dstar, cfg.kill_radius);
    R3 = soft_sat_log_evade(fact.r_MD2, cfg.dstar, cfg.kill_radius);
    reward = cfg.proc_coef * (R2 + R3);
end

function r = soft_sat_log_evade(r_AD, dstar, kill_radius)
    if r_AD <= kill_radius
        r = 0.0;
    else
        r_eff = min(r_AD, dstar);
        r = log10(max(r_eff - 5.0, 1e-6));
    end
end
