function [M0, D10, D20, info] = reconstruct_geometry(row)
%RECONSTRUCT_GEOMETRY 由 samples_clean 的一行(14列)重建各飞行器初态。
%   [M0, D10, D20] = reconstruct_geometry(row) 输入 1x14 样本向量, 输出攻击弹
%   M0、防御弹 D10/D20 的初始状态 [x, y, z, theta, psi_v, V] (1x6)。
%   info 额外返回重建过程中的中间量 (rAD0, beta, r_AD10, r_AD20 等)。
%
%   坐标系: y 轴指天, x 轴大致指向目标; 目标 T 固定于原点 [0,0,0],
%   防御弹 D1 初始位于原点 (论文设定防御弹自目标附近发射)。
%
%   samples_clean 14 列映射 (见 lhsyangbenshengcheng_beta_0.m / Spec §3.2):
%     1: eta_y_A10   2: eta_z_A10   3: eta_y_A20   4: eta_z_A20
%     5: q_y_MD10(qy10)
%     6: eta_y_D10   7: eta_z_D10   8: eta_y_D20   9: eta_z_D20
%    10: rAD0       11: beta
%    12: q_z_MD20(qz20)  13: q_z_MD10(qz10)  14: q_y_MD20(qy20)
%
%   距离 (Spec §3.2 / 论文 §4.1): r_AD10 = rAD0 + beta, r_AD20 = rAD0 - beta。

    eta_y_M10 = row(1);  eta_z_M10 = row(2);
    % row(3) eta_y_M20, row(4) eta_z_M20 为攻击弹相对 D2 的速度前置角, 几何重建不直接用
    q_y_MD10  = row(5);
    eta_y_D10 = row(6);  eta_z_D10 = row(7);
    eta_y_D20 = row(8);  eta_z_D20 = row(9);
    rAD0      = row(10);
    beta      = row(11);
    q_z_MD20  = row(12);
    q_z_MD10  = row(13);
    q_y_MD20  = row(14);

    r_AD10 = rAD0 + beta;     % 攻-D1 初始距离
    r_AD20 = rAD0 - beta;     % 攻-D2 初始距离

    % 攻击弹弹道角 (由 D1 视线几何反推)
    theta_M0 = eta_y_M10 - q_y_MD10;
    psi_VM0  = eta_z_M10 + q_z_MD10 + pi;

    % 位置: D1 置于原点; M 由 D1->M 视线 (q_y_MD10,q_z_MD10) 与距离 r_AD10 确定
    pos_D10 = [0; 0; 0];
    pos_M0  = [r_AD10*cos(q_y_MD10)*cos(q_z_MD10);
               r_AD10*sin(q_y_MD10);
               r_AD10*cos(q_y_MD10)*sin(-q_z_MD10)];
    % D2 由 D2->M 视线 (q_y_MD20,q_z_MD20) 与距离 r_AD20 反推
    pos_D20 = [pos_M0(1) - r_AD20*cos(q_y_MD20)*cos(q_z_MD20);
               pos_M0(2) - r_AD20*sin(q_y_MD20);
               pos_M0(3) - r_AD20*cos(q_y_MD20)*sin(-q_z_MD20)];

    % 防御弹弹道角
    theta_D10 = eta_y_D10 + q_y_MD10;
    theta_D20 = eta_y_D20 + q_y_MD20;
    psi_VD10  = eta_z_D10 + q_z_MD10;
    psi_VD20  = eta_z_D20 + q_z_MD20;

    V_A = 500; V_D = 600;
    M0  = [pos_M0(1),  pos_M0(2),  pos_M0(3),  theta_M0,  psi_VM0,  V_A];
    D10 = [pos_D10(1), pos_D10(2), pos_D10(3), theta_D10, psi_VD10, V_D];
    D20 = [pos_D20(1), pos_D20(2), pos_D20(3), theta_D20, psi_VD20, V_D];

    info = struct('rAD0', rAD0, 'beta', beta, 'r_AD10', r_AD10, 'r_AD20', r_AD20, ...
                  'q_y_MD10', q_y_MD10, 'q_z_MD10', q_z_MD10, ...
                  'q_y_MD20', q_y_MD20, 'q_z_MD20', q_z_MD20);
end
