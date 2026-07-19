function [gamma, wT, wE] = action_norm_to_phys(action_norm, version, w)
%ACTION_NORM_TO_PHYS  归一化动作 → 物理权重（先假定已在 [-1,1]）。
%
%   [gamma, wT, wE] = action_norm_to_phys(action_norm)
%   [gamma, wT, wE] = action_norm_to_phys(action_norm, version)
%   [gamma, wT, wE] = action_norm_to_phys(action_norm, version, w)
%
%   td3_box_log_v1:
%     gamma = a(1)*3;  wT=10^(a(2)*1.5+3.5);  wE=10^(a(3)+3)
%   ratio_log_box_v1（默认）:
%     ρE=10^(a(1)+1); ρT=10^(a(2)*2.5+0.5); gamma=a(3)*3
%     wE=w/ρE; wT=ρT*wE

    if nargin < 2 || isempty(version)
        meta = transition_schema_meta();
        version = meta.action_mapping_version;
    end
    if nargin < 3 || isempty(w)
        w = 1e4;
    end

    a = double(action_norm(:));
    version = char(version);
    switch version
        case 'td3_box_log_v1'
            gamma = a(1) * 3.0;
            wT    = 10^(a(2) * 1.5 + 3.5);
            wE    = 10^(a(3) + 3.0);
        case 'ratio_log_box_v1'
            rhoE  = 10^(a(1) + 1.0);
            rhoT  = 10^(a(2) * 2.5 + 0.5);
            gamma = a(3) * 3.0;
            wE    = double(w) / max(rhoE, 1e-12);
            wT    = rhoT * wE;
        otherwise
            error('action_norm_to_phys:UnknownVersion', ...
                '未知 action_mapping_version=%s', version);
    end
end
