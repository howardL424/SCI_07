function [an, clipped] = phys_to_action_norm(gamma, wT, wE)
%PHYS_TO_ACTION_NORM  物理权重 → TD3 归一化动作，并裁剪到 [-1,1]^3。
%
%   [an, clipped] = phys_to_action_norm(gamma, wT, wE)
%
%   映射（与 evasion_env.norm_action / env_step 反归一化成对）:
%     gamma_n = gamma / 3
%     wT_n    = (log10(clip(wT,[1e2,1e5])) - 3.5) / 1.5
%     wE_n    = log10(clip(wE,[1e2,1e4])) - 3.0
%   再裁剪到 [-1,1]。clipped=true 表示发生过物理盒或归一化裁剪。

    gn0  = double(gamma) / 3.0;
    wTc  = min(max(double(wT), 1e2), 1e5);
    wEc  = min(max(double(wE), 1e2), 1e4);
    wTn0 = (log10(wTc) - 3.5) / 1.5;
    wEn0 = log10(wEc) - 3.0;
    gn   = min(max(gn0,  -1.0), 1.0);
    wTn  = min(max(wTn0, -1.0), 1.0);
    wEn  = min(max(wEn0, -1.0), 1.0);
    an = [gn; wTn; wEn];
    clipped = (abs(gn - gn0) > 1e-12) || (abs(wTn - wTn0) > 1e-12) || ...
              (abs(wEn - wEn0) > 1e-12) || (wTc ~= double(wT)) || (wEc ~= double(wE));
end
