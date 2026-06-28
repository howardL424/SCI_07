function zT = casadi_target_zem(M)
%CASADI_TARGET_ZEM 给定攻击弹状态 M(6), 计算其对原点静止目标的零控脱靶量(ZEM)。
%   zT = casadi_target_zem(M)  M=[x,y,z,theta,psi_v,V] (SX/MX 或 double)
%   用于式22 目标打击项 z_T(t_fT): 沿当前态外推到对目标的预测脱靶量, 越小越
%   表示规避末端越"对准"目标。坐标系/公式与式47 目标通道一致 (T->M 视线)。

    import casadi.*
    posx = M(1); posy = M(2); posz = M(3);
    thetaM = M(4); psiM = M(5); VM = M(6);

    r = sqrt(posx^2 + posy^2 + posz^2 + 1e-6);
    q_y = asin(posy / r);
    q_z = -atan2(posz, posx);

    % 目标静止 (V_D=0): r_dot 与 LOS 角速率 (computeMDDot, term_D=0)
    rdot = VM * (cos(thetaM)*cos(q_y)*cos(q_z - psiM) + sin(thetaM)*sin(q_y));
    term_M_qy = VM * (cos(thetaM)*sin(q_y)*cos(q_z - psiM) - sin(thetaM)*cos(q_y));
    qd_y = -term_M_qy / r;
    cos_qy = sqrt(cos(q_y)^2 + 1e-10);
    qd_z = (-VM * cos(thetaM) * sin(q_z - psiM)) / (r * cos_qy);

    Q = sqrt(qd_y^2 + qd_z^2 * cos(q_y)^2 + 1e-10);
    tgo = -r / rdot;                       % t_fT - t (rdot<0 接近 => tgo>0)
    Qt = if_else(Q*tgo > 30, 30, if_else(Q*tgo < -30, -30, Q*tgo));

    zT = cosh(Qt) * r + sinh(Qt) / Q * rdot;   % 预测对目标脱靶量
end
