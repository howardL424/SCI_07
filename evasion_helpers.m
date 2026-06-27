function H = evasion_helpers()
%EVASION_HELPERS 返回双防规避/目标打击场景的数值计算函数句柄集合。
%   H = evasion_helpers() 返回一个结构体，字段为各核心函数的句柄，供阶段A
%   (RK4 warm-start 仿真) 以及阶段B 的数值-符号对拍复用。这些函数与
%   main_changjing3_0421_LM.m / main_jiaBP3_0421_duo.m 中的同名局部函数一致。
%
%   坐标系: y 轴指天, x 轴大致指向目标 (非北天东)。
%   状态向量约定: state = [x, y, z, theta, psi_v, V]
%     theta 为弹道倾角, psi_v 为弹道偏角 (从 x 轴正向逆时针为正)。

    H = struct();
    H.computeLOS           = @computeLOS;
    H.calc_Phi12           = @calc_Phi12;
    H.computeBC            = @computeBC;
    H.computeOptimalCmd    = @computeOptimalCmd;
    H.computeOptimalCmd_duo= @computeOptimalCmd_duo;
    H.computeOptimalCmd_duo_target = @computeOptimalCmd_duo_target;
    H.computePN            = @computePN;
    H.updateState          = @updateState;
    H.saturate             = @saturate;
    H.computeMDDot         = @computeMDDot;
    H.computeMTDot         = @computeMTDot;
end

% 函数1: 计算视线角(参考论文图1)
function [q_y, q_z, r] = computeLOS(pos1, pos2)
    % 从 pos1 的角度看 pos2
    dx = pos2(1) - pos1(1);
    dy = pos2(2) - pos1(2);
    dz = pos2(3) - pos1(3);

    r = max(norm([dx, dy, dz]), 1e-3);   % 强制距离下限, 避免除以0

    sin_qy = max(min(dy / r, 1), -1);     % 俯仰视线角 (向上为正)
    q_y = asin(sin_qy);
    q_z = -atan2(dz, dx);                 % 偏航视线角 (从 x 轴正向逆时针为正)
end

% 函数2: 计算交叉项积分 Phi_12 (式34 交叉耦合积分)
function Phi_12 = calc_Phi12(Q_dot1, Q_dot2, t_go1, t_go2)
    Tmin = min(t_go1, t_go2);  sum_tgo = t_go1 + t_go2;
    delta_t = abs(t_go1 - t_go2);
    Q1_tgo1 = Q_dot1 * t_go1;
    Q2_tgo2 = Q_dot2 * t_go2;
    eps_limit = 1e-6;

    if Q_dot1 < eps_limit && Q_dot2 < eps_limit
        Phi_12 = t_go1 * t_go2 * Tmin - sum_tgo * Tmin^2 / 2 + Tmin^3 / 3;
    elseif abs(Q_dot1 - Q_dot2) < eps_limit
        Q_dot = (Q_dot1 + Q_dot2) / 2;
        term1 = exp(Q_dot * sum_tgo) - exp(Q_dot * (sum_tgo - 2*Tmin)) / (2*Q_dot);
        term2 = 2 * Tmin * cosh(Q_dot*delta_t);
        term3 = exp(-Q_dot * (sum_tgo - 2*Tmin)) - exp(-Q_dot * sum_tgo) / (2*Q_dot);
        Phi_12 = (1 / (4 * Q_dot^2)) * (term1 - term2 + term3);
    elseif Q_dot1 < eps_limit
        term1 = ((t_go1 - Tmin) * cosh(Q_dot2 * (t_go2 - Tmin)) - t_go1 * cosh(Q2_tgo2)) / (Q_dot2^2);
        term2 = (sinh(Q2_tgo2) - sinh(Q_dot2 * (t_go2 - Tmin))) / (Q_dot2^3);
        Phi_12 = term1 + term2;
    elseif Q_dot2 < eps_limit
        term1 = ((t_go2 - Tmin) * cosh(Q_dot1 * (t_go1 - Tmin)) - t_go2 * cosh(Q1_tgo1)) / (Q_dot1^2);
        term2 = (sinh(Q1_tgo1) - sinh(Q_dot1 * (t_go1 - Tmin))) / (Q_dot1^3);
        Phi_12 = term1 + term2;
    else
        sumQ = Q_dot1 + Q_dot2;
        diffQ = Q_dot1 - Q_dot2;
        exp_11 = exp(Q1_tgo1); exp_22 = exp(Q2_tgo2);
        exp_sumT = exp(sumQ*Tmin); exp_diffT = exp(diffQ*Tmin);
        part1 = (exp_11 * exp_22 - exp_11 * exp_22 / exp_sumT) / sumQ;
        part2 = (exp_11 / exp_22 - exp_11 / exp_22 / exp_diffT) / diffQ;
        part3 = (exp_22 / exp_11 * exp_diffT - 1 / exp_11 * exp_22) / diffQ;
        part4 = (exp_sumT / exp_11 / exp_22 - 1 / (exp_11 * exp_22)) / sumQ;
        Phi_12 = (1 / (4 * Q_dot1 * Q_dot2)) * (part1 - part2 - part3 + part4);
    end

    if isnan(Phi_12) || isinf(Phi_12)
        Phi_12 = 0;
    end
end

% 函数3: 计算中间变量 B1,B2,C1 (式12)
function [B1, B2, C1] = computeBC(q_y, q_z, theta_M, psi_vM, theta_D, psi_vD, a_yD, a_zD)
    B1 = sin(q_y)*cos(theta_M) - cos(q_y)*sin(theta_M)*cos(q_z - psi_vM);
    B2 = -cos(q_y)*sin(q_z - psi_vM);
    term1 = cos(q_y)*sin(q_z - psi_vD)*a_zD;
    term2 = (-cos(q_y)*sin(theta_D)*cos(q_z - psi_vD) + sin(q_y)*cos(theta_D))*a_yD;
    C1 = term1 - term2;
end

% 函数4.0: 计算最优突防制导指令(单弹)
function a_M = computeOptimalCmd(q_y, q_z, q_dot_y, q_dot_z, r, r_dot, t, t_f, ...
                                  theta_M, psi_vM, theta_D, psi_vD, a_D, a, b, r_star)
    Q_dot = sqrt(q_dot_y^2 + q_dot_z^2 * (cos(q_y))^2);
    t_go = t_f - t;
    if Q_dot * t_go > 100
        Q_dot = 100 / t_go;
    end

    [B1, B2, C1] = computeBC(q_y, q_z, theta_M, psi_vM, theta_D, psi_vD, a_D(1), a_D(2));
    v_r = -C1;

    if Q_dot < 1e-6
        Omega = [1, t_go; 0, 1];
        z = [1, 0] * Omega * [r; r_dot];
        M = 0.5 * t_go^2 * v_r;
        KG = (a * (B1^2 + B2^2) / b) * (t_go^3 / 3);
        common_gain = (a / b) * t_go;
        a_M = -common_gain * (z - M - r_star) / (1 + KG + eps) * [B1; B2];
    else
        exp_pos = exp(Q_dot * t_go);
        exp_neg = exp(-Q_dot * t_go);
        Omega = [(exp_pos+exp_neg)/2, (exp_pos-exp_neg)/(2*Q_dot);
                 Q_dot*(exp_pos-exp_neg)/2, (exp_pos+exp_neg)/2];
        z = [1, 0] * Omega * [r; r_dot];
        M = (exp(Q_dot*t_go) + exp(-Q_dot*t_go) - 2) / (2*Q_dot^2) * v_r;
        K = a * (B1^2 + B2^2) / (4 * b * Q_dot^2);
        G = (exp(2*Q_dot*t_go) - exp(-2*Q_dot*t_go))/(2*Q_dot) - 2*t_go;
        common_gain = a * (exp(Q_dot*t_go) - exp(-Q_dot*t_go)) / (2 * b * Q_dot);
        a_M = -common_gain * (z - M - r_star) / (1 + K*G + eps) * [B1; B2];
    end
end

% 函数4.5: 计算最优突防制导指令(双弹, 式34)
function a_M = computeOptimalCmd_duo(q_y1, q_z1, q_dot_y1, q_dot_z1, r1, r_dot1, t, t_f1, ...
                                 q_y2, q_z2, q_dot_y2, q_dot_z2, r2, r_dot2, t_f2, ...
                                  theta_M, psi_vM, theta_D1, psi_vD1, theta_D2, psi_vD2, ...
                                  a_D1, a_D2, a1, a2, wE, r_star)
    % 拦截弹 D1
    Q_dot1 = sqrt(q_dot_y1^2 + q_dot_z1^2 * (cos(q_y1))^2);
    t_go1 = t_f1 - t;
    if Q_dot1 * t_go1 > 100
        Q_dot1 = 100 / t_go1;
    end
    [B11, B21, C11] = computeBC(q_y1, q_z1, theta_M, psi_vM, theta_D1, psi_vD1, a_D1(1), a_D1(2));
    v_r1 = -C11;
    b1_vec = [B11, B21]';

    % 拦截弹 D2
    Q_dot2 = sqrt(q_dot_y2^2 + q_dot_z2^2 * (cos(q_y2))^2);
    t_go2 = t_f2 - t;
    if Q_dot2 * t_go2 > 100
        Q_dot2 = 100 / t_go2;
    end
    [B12, B22, C12] = computeBC(q_y2, q_z2, theta_M, psi_vM, theta_D2, psi_vD2, a_D2(1), a_D2(2));
    v_r2 = -C12;
    b2_vec = [B12, B22]';

    Q1_tgo1 = Q_dot1 * t_go1;
    Q2_tgo2 = Q_dot2 * t_go2;

    if Q_dot1 < 1e-6 && Q_dot2 < 1e-6
        Omega1 = [1, t_go1; 0, 1]; Omega2 = [1, t_go2; 0, 1];
        phi1 = t_go1; phi2 = t_go2;
        Phi_1 = t_go1^3 / 3; Phi_2 = t_go2^3 / 3;
        N_1 = 0.5 * t_go1^2 * v_r1; N_2 = 0.5 * t_go2^2 * v_r2;
    elseif Q_dot1 < 1e-6
        Omega1 = [1, t_go1; 0, 1];
        Omega2 = [cosh(Q2_tgo2), sinh(Q2_tgo2) / Q_dot2;
                  Q_dot2 * sinh(Q2_tgo2), cosh(Q2_tgo2)];
        phi1 = t_go1; phi2 = sinh(Q2_tgo2) / Q_dot2;
        Phi_1 = t_go1^3 / 3; Phi_2 = (sinh(2 * Q2_tgo2) / Q_dot2 - 2 * t_go2) / (4 * Q_dot2^2);
        N_1 = 0.5 * t_go1^2 * v_r1; N_2 = (2 * cosh(Q2_tgo2) - 2) / (2 * Q_dot2^2) * v_r2;
    elseif Q_dot2 < 1e-6
        Omega1 = [cosh(Q1_tgo1), sinh(Q1_tgo1) / Q_dot1;
                  Q_dot1 * sinh(Q1_tgo1), cosh(Q1_tgo1)];
        Omega2 = [1, t_go2; 0, 1];
        phi1 = sinh(Q1_tgo1) / Q_dot1; phi2 = sinh(Q2_tgo2) / Q_dot2;
        Phi_1 = (sinh(2 * Q1_tgo1) / Q_dot1 - 2 * t_go1) / (4 * Q_dot1^2); Phi_2 = t_go2^3 / 3;
        N_1 = (2 * cosh(Q1_tgo1) - 2) / (2 * Q_dot1^2) * v_r1; N_2 = 0.5 * t_go2^2 * v_r2;
    else
        Omega1 = [cosh(Q1_tgo1), sinh(Q1_tgo1) / Q_dot1;
                  Q_dot1 * sinh(Q1_tgo1), cosh(Q1_tgo1)];
        Omega2 = [cosh(Q2_tgo2), sinh(Q2_tgo2) / Q_dot2;
                  Q_dot2 * sinh(Q2_tgo2), cosh(Q2_tgo2)];
        phi1 = sinh(Q1_tgo1) / Q_dot1;
        phi2 = sinh(Q2_tgo2) / Q_dot2;
        Phi_1 = (sinh(2 * Q1_tgo1) / Q_dot1 - 2 * t_go1) / (4 * Q_dot1^2);
        N_1 = (2 * cosh(Q1_tgo1) - 2) / (2 * Q_dot1^2) * v_r1;
        Phi_2 = (sinh(2 * Q2_tgo2) / Q_dot2 - 2 * t_go2) / (4 * Q_dot2^2);
        N_2 = (2 * cosh(Q2_tgo2) - 2) / (2 * Q_dot2^2) * v_r2;
    end

    z1 = [1, 0] * Omega1 * [r1; r_dot1]; z2 = [1, 0] * Omega2 * [r2; r_dot2];
    Phi_12 = calc_Phi12(Q_dot1, Q_dot2, t_go1, t_go2);
    G_11 = a1 * (B11^2 + B21^2) / wE;
    G_22 = a2 * (B12^2 + B22^2) / wE;
    G_12 = a2 * (b1_vec' * b2_vec) / wE;
    G_21 = a1 * (b1_vec' * b2_vec) / wE;
    Delta_Z_1 = z1 - N_1 - r_star; Delta_Z_2 = z2 - N_2 - r_star;
    Delta_det = (1 + G_11 * Phi_1) * (1 + G_22 * Phi_2) - G_12 * G_21 * Phi_12^2;
    I_1 = double(t <= t_f1); I_2 = double(t <= t_f2);
    term1 = (a1 / wE) * phi1 * ((1 + G_22 * Phi_2) * Delta_Z_1 - G_12 * Phi_12 * Delta_Z_2) * I_1 .* b1_vec;
    term2 = (a2 / wE) * phi2 * (-G_21 * Phi_12 * Delta_Z_1 + (1 + G_11 * Phi_1) * Delta_Z_2) * I_2 .* b2_vec;
    a_M = (term1 + term2) .* (-1 / Delta_det);
end

% 函数4.7: 计算最优突防+目标打击一体化制导指令(式47, 三通道 D1/D2/目标 耦合)
function a_M = computeOptimalCmd_duo_target( ...
        q_y1, q_z1, qd_y1, qd_z1, r1, rdot1, t, tf1, ...
        q_y2, q_z2, qd_y2, qd_z2, r2, rdot2, tf2, ...
        q_th, q_ps, qd_th, qd_ps, r_AT, rdot_AT, tfT, ...
        theta_M, psi_vM, theta_D1, psi_vD1, theta_D2, psi_vD2, ...
        a_D1, a_D2, w1, w2, wT, wE, r_star)
    % w1, w2: 双防规避项权重 (a1, a2); wT: 目标打击项权重; r_star: 期望规避脱靶量 d*
    % 三个通道: 1 -> 防御弹D1, 2 -> 防御弹D2, T -> 目标 (目标静止, v_rT=0, d_T*=0)

    % ----- 通道1 (D1) -----
    Q1 = sqrt(qd_y1^2 + qd_z1^2 * cos(q_y1)^2);
    tgo1 = tf1 - t;
    if Q1 * tgo1 > 100; Q1 = 100 / tgo1; end
    [B11, B21, C11] = computeBC(q_y1, q_z1, theta_M, psi_vM, theta_D1, psi_vD1, a_D1(1), a_D1(2));
    v_r1 = -C11; b1 = [B11; B21];

    % ----- 通道2 (D2) -----
    Q2 = sqrt(qd_y2^2 + qd_z2^2 * cos(q_y2)^2);
    tgo2 = tf2 - t;
    if Q2 * tgo2 > 100; Q2 = 100 / tgo2; end
    [B12, B22, C12] = computeBC(q_y2, q_z2, theta_M, psi_vM, theta_D2, psi_vD2, a_D2(1), a_D2(2));
    v_r2 = -C12; b2 = [B12; B22];

    % ----- 通道T (目标, 攻-目标视线; 目标静止故加速度取0 -> C1T=0) -----
    QT = sqrt(qd_th^2 + qd_ps^2 * cos(q_th)^2);
    tgoT = tfT - t;
    if QT * tgoT > 100; QT = 100 / tgoT; end
    [B1T, B2T, ~] = computeBC(q_th, q_ps, theta_M, psi_vM, 0, 0, 0, 0);
    bT = [B1T; B2T];

    % ----- 各通道零控脱靶量 z、varphi、自积分 Phi、附加项 N (式16-21,33) -----
    [z1, phi1, Phi1, N1] = chan_quantities(Q1, tgo1, r1, rdot1, v_r1);
    [z2, phi2, Phi2, N2] = chan_quantities(Q2, tgo2, r2, rdot2, v_r2);
    [zT, phiT, PhiT, NT] = chan_quantities(QT, tgoT, r_AT, rdot_AT, 0);

    % ----- 交叉项积分 (式34,35,36) -----
    Phi12 = calc_Phi12(Q1, Q2, tgo1, tgo2);
    Phi1T = calc_Phi12(Q1, QT, tgo1, tgoT);
    Phi2T = calc_Phi12(Q2, QT, tgo2, tgoT);

    % ----- 自/交叉耦合增益 (式38-40), G_ij = w_j (b_i·b_j)/wE -----
    d11 = b1' * b1; d22 = b2' * b2; dTT = bT' * bT;
    d12 = b1' * b2; d1T = b1' * bT; d2T = b2' * bT;
    G11 = w1 * d11 / wE; G22 = w2 * d22 / wE; GTT = wT * dTT / wE;
    G12 = w2 * d12 / wE; G21 = w1 * d12 / wE;
    G1T = wT * d1T / wE; GT1 = w1 * d1T / wE;
    G2T = wT * d2T / wE; GT2 = w2 * d2T / wE;

    % ----- 右端项 Z (式41); 目标期望脱靶量为 0 -----
    Z1 = z1 - N1 - r_star;
    Z2 = z2 - N2 - r_star;
    ZT = zT - NT - 0;
    Zvec = [Z1; Z2; ZT];

    % ----- 3x3 耦合矩阵 (式42) 与克莱姆法则 (式43-46) -----
    A = [1 + G11*Phi1,  G12*Phi12,    G1T*Phi1T;
         G21*Phi12,     1 + G22*Phi2, G2T*Phi2T;
         GT1*Phi1T,     GT2*Phi2T,    1 + GTT*PhiT];
    detA = det(A);
    if abs(detA) < 1e-12; detA = sign(detA + eps) * 1e-12; end
    Delta1 = det([Zvec, A(:,2), A(:,3)]);
    Delta2 = det([A(:,1), Zvec, A(:,3)]);
    DeltaT = det([A(:,1), A(:,2), Zvec]);

    % ----- 最优制导指令 (式47) -----
    I1 = double(t <= tf1); I2 = double(t <= tf2);   % 目标项始终作用
    a_M = -(1 / (wE * detA)) * ( ...
            w1 * Delta1 * I1 * (phi1 * b1) + ...
            w2 * Delta2 * I2 * (phi2 * b2) + ...
            wT * DeltaT *      (phiT * bT) );

    if any(isnan(a_M)) || any(isinf(a_M))
        a_M = [0; 0];
    end
end

% 辅助: 单通道零控脱靶量及相关积分量 (Q->0 用泰勒极限)
function [z, phi, Phi, N] = chan_quantities(Q, tgo, r, rdot, v_r)
    if Q < 1e-6
        z   = r + tgo * rdot;
        phi = tgo;
        Phi = tgo^3 / 3;
        N   = 0.5 * tgo^2 * v_r;
    else
        Qt = Q * tgo;
        z   = cosh(Qt) * r + sinh(Qt) / Q * rdot;
        phi = sinh(Qt) / Q;
        Phi = (sinh(2*Qt) / Q - 2 * tgo) / (4 * Q^2);
        N   = (2 * cosh(Qt) - 2) / (2 * Q^2) * v_r;
    end
end

% 函数5: 计算比例导引指令(式9/53)
function a_pn = computePN(K, V, q_dot_y, q_dot_z, theta)
    a_pn = [K * V * q_dot_y; -K * V * q_dot_z * cos(theta)];
end

% 函数6: 状态更新(龙格-库塔法, 基于式5)
function state_new = updateState(state, a_cmd, dt)
    theta = state(4); psi_v = state(5); V = state(6);
    a_y = a_cmd(1); a_z = a_cmd(2);

    if abs(V) < 1e-6
        state_new = state;
        warning('V is too small for deriv');
    else
        deriv = @(s) [ ...
            V * cos(s(4)) * cos(s(5));
            V * sin(s(4));
            -V * cos(s(4)) * sin(s(5));
            a_y / V;
            -a_z / (V * cos(s(4)));
        ];
        s = [state(1); state(2); state(3); theta; psi_v];
        k1 = deriv(s);
        k2 = deriv(s + 0.5*dt*k1);
        k3 = deriv(s + 0.5*dt*k2);
        k4 = deriv(s + dt*k3);
        s_new = s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        state_new = [s_new(1:5)', V];
    end
end

% 辅助函数: 过载饱和
function a_sat = saturate(a, limit)
    if any(isnan(a)) || any(isinf(a))
        a_sat = [0; 0];
        return;
    end
    norm_a = norm(a);
    if norm_a > limit
        a_sat = a / norm_a * limit;
    else
        a_sat = a;
    end
end

% 函数7: 攻击弹-防御弹间视线角速率、相对速度(式1)
function [r_dot, q_dot_y, q_dot_z] = computeMDDot(pos1, pos2, r, q_y, q_z)
    V_M = pos1(6); theta_M = pos1(4); psi_vM = pos1(5);
    V_D = pos2(6); theta_D = pos2(4); psi_vD = pos2(5);

    term_D = V_D * (cos(theta_D) * cos(q_y) * cos(q_z - psi_vD) + sin(theta_D) * sin(q_y));
    term_M = V_M * (cos(theta_M) * cos(q_y) * cos(q_z - psi_vM) + sin(theta_M) * sin(q_y));
    r_dot = -term_D + term_M;

    if abs(r) < 1e-6
        q_dot_y = 0;
    else
        term_D_qy = V_D * (cos(theta_D) * sin(q_y) * cos(q_z - psi_vD) - sin(theta_D) * cos(q_y));
        term_M_qy = V_M * (cos(theta_M) * sin(q_y) * cos(q_z - psi_vM) - sin(theta_M) * cos(q_y));
        q_dot_y = (term_D_qy - term_M_qy) / r;
    end

    cos_qy_safe = max(abs(cos(q_y)), 1e-5);
    r_safe = max(abs(r), 1e-3);
    term_D_qz = V_D * cos(theta_D) * sin(q_z - psi_vD);
    term_M_qz = V_M * cos(theta_M) * sin(q_z - psi_vM);
    q_dot_z = (term_D_qz - term_M_qz) / (r_safe * cos_qy_safe);
end

% 函数8: 攻击弹-目标间视线角速率(式2)
function [q_dot_theta, q_dot_psi] = computeMTDot(pos1, r_MT, q_theta, q_psi)
    V_M = pos1(6); theta_M = pos1(4); psi_vM = pos1(5);

    if abs(r_MT) < 1e-6
        q_dot_theta = 0;
        q_dot_psi = 0;
        warning('r_MT is too small for computing q_dot');
    else
        term1 = cos(theta_M) * sin(q_theta) * cos(q_psi - psi_vM);
        term2 = -sin(theta_M) * cos(q_theta);
        q_dot_theta = V_M * (term1 + term2) / r_MT;
        if abs(cos(q_theta)) < 1e-6
            q_dot_psi = 0;
        else
            q_dot_psi = (V_M * cos(theta_M) * sin(q_psi - psi_vM)) / (r_MT * cos(q_theta));
        end
    end
end
