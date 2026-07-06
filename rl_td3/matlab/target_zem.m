function zT = target_zem(M)
%TARGET_ZEM  计算攻击弹对原点静止目标的零控脱靶量(ZEM)。
%
%   zT = target_zem(M)
%
%   输入 M = [x,y,z,theta,psi_v,V]，坐标和角度定义与
%   casadi_target_zem / 式47 目标通道保持一致。zT 越小，表示当前
%   状态在零控外推下越对准目标。

    M = M(:).';
    posx = M(1); posy = M(2); posz = M(3);
    thetaM = M(4); psiM = M(5); VM = M(6);

    r = sqrt(posx^2 + posy^2 + posz^2 + 1e-6);
    q_y = asin(max(min(posy / r, 1.0), -1.0));
    q_z = -atan2(posz, posx);

    rdot = VM * (cos(thetaM)*cos(q_y)*cos(q_z - psiM) + sin(thetaM)*sin(q_y));
    term_M_qy = VM * (cos(thetaM)*sin(q_y)*cos(q_z - psiM) - sin(thetaM)*cos(q_y));
    qd_y = -term_M_qy / r;
    cos_qy = sqrt(cos(q_y)^2 + 1e-10);
    qd_z = (-VM * cos(thetaM) * sin(q_z - psiM)) / (r * cos_qy);

    Q = sqrt(qd_y^2 + qd_z^2 * cos(q_y)^2 + 1e-10);
    if rdot < -1e-3
        tgo = -r / rdot;
    else
        % 非接近或接近率过小时，给出保守的大 tgo，避免奖励出现奇异值。
        tgo = r / 1e-3;
    end

    Qt = max(min(Q * tgo, 30), -30);
    zT = cosh(Qt) * r + sinh(Qt) / Q * rdot;
    zT = abs(double(zT));
end
