function [an, clipped] = phys_to_action_norm(gamma, wT, wE, version, w)
%PHYS_TO_ACTION_NORM  物理权重 → 归一化动作，并裁剪到 [-1,1]^3。
%
%   [an, clipped] = phys_to_action_norm(gamma, wT, wE)
%   [an, clipped] = phys_to_action_norm(gamma, wT, wE, version)
%   [an, clipped] = phys_to_action_norm(gamma, wT, wE, version, w)
%
%   version:
%     'td3_box_log_v1'   — [γ_n, wT_n, wE_n]（旧）
%     'ratio_log_box_v1' — [ρE_n, ρT_n, γ_n]，ρE=w/wE，ρT=wT/wE（默认）
%
%   比值盒（w=1e4 固定）:
%     ρE∈[1,100]，  aρE = log10(ρE)-1
%     ρT∈[1e-2,1e3]，aρT = (log10(ρT)-0.5)/2.5
%     γ∈[-3,3]，    aγ  = γ/3

    if nargin < 4 || isempty(version)
        meta = transition_schema_meta();
        version = meta.action_mapping_version;
    end
    if nargin < 5 || isempty(w)
        w = 1e4;
    end

    version = char(version);
    switch version
        case 'td3_box_log_v1'
            [an, clipped] = local_td3_box(gamma, wT, wE);
        case 'ratio_log_box_v1'
            [an, clipped] = local_ratio(gamma, wT, wE, w);
        otherwise
            error('phys_to_action_norm:UnknownVersion', ...
                '未知 action_mapping_version=%s', version);
    end
end

function [an, clipped] = local_td3_box(gamma, wT, wE)
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

function [an, clipped] = local_ratio(gamma, wT, wE, w)
    wEc = max(double(wE), 1e-12);
    rhoE0 = double(w) / wEc;
    rhoT0 = double(wT) / wEc;
    rhoE = min(max(rhoE0, 1.0), 100.0);
    rhoT = min(max(rhoT0, 1e-2), 1e3);
    ae0 = log10(rhoE) - 1.0;
    at0 = (log10(rhoT) - 0.5) / 2.5;
    g0  = double(gamma) / 3.0;
    ae = min(max(ae0, -1.0), 1.0);
    at = min(max(at0, -1.0), 1.0);
    g  = min(max(g0,  -1.0), 1.0);
    an = [ae; at; g];
    clipped = (abs(ae - ae0) > 1e-12) || (abs(at - at0) > 1e-12) || ...
              (abs(g - g0) > 1e-12) || (abs(rhoE - rhoE0) > 1e-12) || ...
              (abs(rhoT - rhoT0) > 1e-12);
end
