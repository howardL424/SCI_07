function [gamma, wT, wE] = action_norm_to_phys(action_norm)
%ACTION_NORM_TO_PHYS  TD3 归一化动作 → 物理权重（先假定已在 [-1,1]）。
%
%   [gamma, wT, wE] = action_norm_to_phys(action_norm)
%
%     gamma = a(1) * 3             → [-3, 3]
%     wT    = 10^(a(2)*1.5 + 3.5) → [1e2, 1e5]
%     wE    = 10^(a(3) + 3)        → [1e2, 1e4]

    a = double(action_norm(:));
    gamma = a(1) * 3.0;
    wT    = 10^(a(2) * 1.5 + 3.5);
    wE    = 10^(a(3) + 3.0);
end
