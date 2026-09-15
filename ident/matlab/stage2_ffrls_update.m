function [theta, P] = stage2_ffrls_update( ...
        theta, P, phi, y, lambda, eta, p_trace_max)
%STAGE2_FFRLS_UPDATE  一步缩放域 FF-RLS、稀疏收缩与物理投影。
%
%   [theta, P] = stage2_ffrls_update(theta, P, phi, y, ...
%       lambda, eta, p_trace_max)
%
%   theta 与 phi 均位于式(56)的缩放域。软阈值只作用于第2个及
%   后续分量；第1个分量 A21 不收缩，但投影到非负域。

    theta = double(theta(:));
    phi = double(phi(:));
    P = double(P);

    denom = lambda + phi' * P * phi;
    K = (P * phi) / denom;
    theta = theta + K * (double(y) - phi' * theta);
    P = (1 / lambda) * (P - K * phi' * P);
    P = 0.5 * (P + P');
    P = local_cap_covariance_trace(P, p_trace_max);

    if eta > 0 && numel(theta) >= 2
        idx = 2:numel(theta);
        theta(idx) = sign(theta(idx)) .* max(abs(theta(idx)) - eta, 0);
    end
    theta(1) = max(theta(1), 0);
end

function P = local_cap_covariance_trace(P, p_trace_max)
    trP = real(trace(P));
    if isfinite(p_trace_max) && trP > p_trace_max
        P = P * (p_trace_max / trP);
        P = 0.5 * (P + P');
    end
end
