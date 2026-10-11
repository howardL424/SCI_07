function [obs16, source] = stage2_actor_obs16_from_abs( ...
        core, state_abs, t, Tev, ident_prior)
%STAGE2_ACTOR_OBS16_FROM_ABS  16D  truth 观测包装器（新增，不改 13D 路径）。
%   obs16 = [obs13(来自 stage2_actor_obs_from_abs), Q1, Q2, QT]。
%   首轮仅允许 actor_obs_mode=truth；noisy/filtered 下 Q 必须由 Actor 实际
%   可获得的量测/滤波状态计算，当前拒绝（避免把解析真值拼入含噪观测）。

    if nargin < 5; ident_prior = []; end
    [obs13, source] = stage2_actor_obs_from_abs(core, state_abs, t, Tev, ident_prior);
    if ~strcmp(char(source.mode), 'truth')
        error('stage2_actor_obs16_from_abs:NoisyMode', ...
            ['16D Q 当前仅支持 truth 模式，收到 %s；', ...
             'noisy/filtered 的 Q 口径待定义，拒绝拼接真值 Q。'], char(source.mode));
    end
    [Q1, Q2, QT] = stage2_qmag_from_abs(core, state_abs);
    obs16 = [obs13(:); Q1; Q2; QT];
    source.obs_schema = 'rel13_qmag16_v1';
    source.obs_dim = 16;
end
