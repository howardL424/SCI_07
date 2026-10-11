function [Q1, Q2, QT] = stage2_qmag_from_abs(core, state_abs)
%STAGE2_QMAG_FROM_ABS  由绝对态解析计算三个方位变化率幅值（16D 扩展用）。
%   与 stage2_step.m / computeOptimalCmd_duo_target 的 Q 定义完全一致：
%     Q1 = sqrt(qd_y1^2  + qd_z1^2  * cos(qy_D1M)^2)
%     Q2 = sqrt(qd_y2^2  + qd_z2^2  * cos(qy_D2M)^2)
%     QT = sqrt(qd_yTM^2 + qd_zTM^2 * cos(qy_TM)^2)   % T->M 几何口径
%   使用 computeMDDot 解析结果，不做差分；返回原始 Q（不做 Q*tgo 钳制）。
%   本文件为新增一致性校验口径，不修改任何现有 13D 函数。

    H = core.H;
    M  = state_abs(1:6)';
    D1 = state_abs(7:12)';
    D2 = state_abs(13:18)';
    T_pos   = core.T_pos;
    T_state = core.T_state;

    [qy_D1M, qz_D1M, r_MD1] = H.computeLOS(D1(1:3), M(1:3));
    [~, qd_y1, qd_z1] = H.computeMDDot(M, D1, r_MD1, qy_D1M, qz_D1M);
    Q1 = sqrt(qd_y1^2 + qd_z1^2 * cos(qy_D1M)^2);

    [qy_D2M, qz_D2M, r_MD2] = H.computeLOS(D2(1:3), M(1:3));
    [~, qd_y2, qd_z2] = H.computeMDDot(M, D2, r_MD2, qy_D2M, qz_D2M);
    Q2 = sqrt(qd_y2^2 + qd_z2^2 * cos(qy_D2M)^2);

    [qy_TM, qz_TM, r_TM] = H.computeLOS(T_pos, M(1:3));
    [~, qd_yTM, qd_zTM] = H.computeMDDot(M, T_state, r_TM, qy_TM, qz_TM);
    QT = sqrt(qd_yTM^2 + qd_zTM^2 * cos(qy_TM)^2);
end
