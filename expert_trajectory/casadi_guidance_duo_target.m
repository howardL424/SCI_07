function uA = casadi_guidance_duo_target( ...
        q_y1, q_z1, qd_y1, qd_z1, r1, rdot1, t, tf1, ...
        q_y2, q_z2, qd_y2, qd_z2, r2, rdot2, tf2, ...
        q_th, q_ps, qd_th, qd_ps, r_AT, rdot_AT, tfT, ...
        theta_M, psi_vM, theta_D1, psi_vD1, theta_D2, psi_vD2, ...
        aD1y, aD1z, aD2y, aD2z, w1, w2, wT, wE, r_star)
%CASADI_GUIDANCE_DUO_TARGET 式47 制导指令的 CasADi 光滑无分支符号实现。
%   uA = casadi_guidance_duo_target(...) 与数值版 computeOptimalCmd_duo_target
%   逐项对应, 但把非光滑项替换为 CasADi 支持的光滑近似:
%     - Q = sqrt(.. + epsQ)  保证恒 > 0, 去掉 Q<1e-6 / Q1==Q2 等分支;
%     - 可去奇点 (sinh x)/x, (cosh x-1)/x^2, ((sinh x)/x-1)/x^2, (e^{dT}-1)/d
%       小参数时切到泰勒级数 (if_else), 避免相消误差;
%     - 指示函数 I(t<=tf) -> 0.5*(1+tanh(t_go/epsI)) 的 sigmoid 平滑。
%   入参与数值版相同 (防御弹过载以分量 aDiy,aDiz 传入)。返回 2x1 SX/MX。
%
%   说明: 该函数所有运算均为 CasADi 可微算子, 可直接嵌入伪谱 NLP 的动力学,
%   亦可经 casadi.Function 封装后与数值版逐点对拍验证一致性。

    import casadi.*
    epsQ = 1e-10;     % Q 正则项 (Q^2 下限)
    epsI = 0.02;      % 指示函数 sigmoid 平滑尺度(s)

    % ---------- 三个通道的基本量 ----------
    [z1, phi1, Phi1, N1, B11, B21, Q1, tgo1] = chan_sym( ...
        q_y1, q_z1, qd_y1, qd_z1, r1, rdot1, t, tf1, ...
        theta_M, psi_vM, theta_D1, psi_vD1, aD1y, aD1z, epsQ);
    [z2, phi2, Phi2, N2, B12, B22, Q2, tgo2] = chan_sym( ...
        q_y2, q_z2, qd_y2, qd_z2, r2, rdot2, t, tf2, ...
        theta_M, psi_vM, theta_D2, psi_vD2, aD2y, aD2z, epsQ);
    % 目标通道: 静止物体, 过载 = 0 (=> v_rT = 0)
    [zT, phiT, PhiT, NT, B1T, B2T, QT, tgoT] = chan_sym( ...
        q_th, q_ps, qd_th, qd_ps, r_AT, rdot_AT, t, tfT, ...
        theta_M, psi_vM, 0, 0, 0, 0, epsQ);

    b1 = [B11; B21]; b2 = [B12; B22]; bT = [B1T; B2T];

    % ---------- 交叉项积分 (式34,35,36) ----------
    Phi12 = phi12_sym(Q1, Q2, tgo1, tgo2);
    Phi1T = phi12_sym(Q1, QT, tgo1, tgoT);
    Phi2T = phi12_sym(Q2, QT, tgo2, tgoT);

    % ---------- 自/交叉耦合增益 (式38-40) ----------
    d11 = b1.' * b1; d22 = b2.' * b2; dTT = bT.' * bT;
    d12 = b1.' * b2; d1T = b1.' * bT; d2T = b2.' * bT;
    G11 = w1 * d11 / wE; G22 = w2 * d22 / wE; GTT = wT * dTT / wE;
    G12 = w2 * d12 / wE; G21 = w1 * d12 / wE;
    G1T = wT * d1T / wE; GT1 = w1 * d1T / wE;
    G2T = wT * d2T / wE; GT2 = w2 * d2T / wE;

    % ---------- 右端项 Z (式41); 目标期望脱靶量 0 ----------
    Z1 = z1 - N1 - r_star;
    Z2 = z2 - N2 - r_star;
    ZT = zT - NT;
    Zvec = [Z1; Z2; ZT];

    % ---------- 3x3 耦合矩阵与克莱姆法则 (式42-46) ----------
    A = [1 + G11*Phi1,  G12*Phi12,    G1T*Phi1T;
         G21*Phi12,     1 + G22*Phi2, G2T*Phi2T;
         GT1*Phi1T,     GT2*Phi2T,    1 + GTT*PhiT];
    detA   = det3(A);
    Delta1 = det3([Zvec, A(:,2), A(:,3)]);
    Delta2 = det3([A(:,1), Zvec, A(:,3)]);
    DeltaT = det3([A(:,1), A(:,2), Zvec]);

    % ---------- 指示函数平滑 (式47); 目标项始终作用 ----------
    I1 = 0.5 * (1 + tanh((tf1 - t) / epsI));
    I2 = 0.5 * (1 + tanh((tf2 - t) / epsI));

    uA = -(1 / (wE * detA)) * ( ...
            w1 * Delta1 * I1 * (phi1 * b1) + ...
            w2 * Delta2 * I2 * (phi2 * b2) + ...
            wT * DeltaT *      (phiT * bT) );
end

% ======================================================================
% 单通道: 计算 z, phi, Phi, N, B1, B2, Q, tgo (式12,16-21,33 的光滑版)
function [z, phi, Phi, N, B1, B2, Q, tgo] = chan_sym( ...
        q_y, q_z, qd_y, qd_z, r, rdot, t, tf, ...
        theta_M, psi_vM, theta_D, psi_vD, aDy, aDz, epsQ)

    Q   = sqrt(qd_y^2 + qd_z^2 * cos(q_y)^2 + epsQ);
    tgo = tf - t;
    Qt  = clampval(Q * tgo, 30);    % 仅钳制双曲参数(防 r_dot->0 溢出), 有效区恒等

    % B1,B2,C1 (式12)
    B1 = sin(q_y)*cos(theta_M) - cos(q_y)*sin(theta_M)*cos(q_z - psi_vM);
    B2 = -cos(q_y)*sin(q_z - psi_vM);
    term1 = cos(q_y)*sin(q_z - psi_vD)*aDz;
    term2 = (-cos(q_y)*sin(theta_D)*cos(q_z - psi_vD) + sin(q_y)*cos(theta_D))*aDy;
    C1 = term1 - term2;
    v_r = -C1;

    % 借助光滑可去奇点函数 (有效区 Qt=Q*tgo, 钳制不激活)
    z   = cosh(Qt) * r + tgo * sinhc(Qt) * rdot;
    phi = tgo * sinhc(Qt);
    Phi = 2 * tgo^3 * d2func(2 * Qt);     % = tgo^3/3 当 Qt->0
    N   = tgo^2 * c2func(Qt) * v_r;       % = 0.5 tgo^2 v_r 当 Qt->0
end

% ======================================================================
% 交叉项积分 Phi_12 (式34) 的统一光滑公式 (用 expdiv 消去 Q1-Q2 奇点)
function P = phi12_sym(Q1, Q2, tgo1, tgo2)
    Tmin  = smin(tgo1, tgo2);
    sumQ  = Q1 + Q2;
    diffQ = Q1 - Q2;
    Qt1 = clampval(Q1 * tgo1, 30); Qt2 = clampval(Q2 * tgo2, 30);

    part1 = exp( Qt1 + Qt2) * expdiv(-sumQ,  Tmin);
    part2 = exp( Qt1 - Qt2) * expdiv(-diffQ, Tmin);
    part3 = exp( Qt2 - Qt1) * expdiv( diffQ, Tmin);
    part4 = exp(-(Qt1 + Qt2)) * expdiv(sumQ, Tmin);

    P = (1 / (4 * Q1 * Q2)) * (part1 - part2 - part3 + part4);
end

% ---------- 光滑辅助函数 ----------
% sinhc(x) = sinh(x)/x
function y = sinhc(x)
    import casadi.*
    s = 1 + x.^2/6 + x.^4/120 + x.^6/5040;   % 小参数泰勒
    y = if_else(abs(x) < 1e-3, s, sinh(x) ./ x);
end

% c2(x) = (cosh(x)-1)/x^2  -> 1/2 当 x->0
function y = c2func(x)
    import casadi.*
    s = 1/2 + x.^2/24 + x.^4/720 + x.^6/40320;
    y = if_else(abs(x) < 1e-2, s, (cosh(x) - 1) ./ x.^2);
end

% d2(x) = (sinh(x)/x - 1)/x^2  -> 1/6 当 x->0
function y = d2func(x)
    import casadi.*
    s = 1/6 + x.^2/120 + x.^4/5040 + x.^6/362880;
    y = if_else(abs(x) < 1e-2, s, (sinh(x) ./ x - 1) ./ x.^2);
end

% expdiv(d,T) = (exp(d*T)-1)/d  -> T 当 d->0 (指数参数光滑钳制防溢出)
function y = expdiv(d, T)
    import casadi.*
    s = T + d.*T.^2/2 + d.^2.*T.^3/6 + d.^3.*T.^4/24;
    u = clampval(d .* T, 60);
    y = if_else(abs(d) < 1e-4, s, (exp(u) - 1) ./ d);
end

% 硬钳制到 [-L, L] (有效区恒等, 仅极端饱和)
function y = clampval(x, L)
    import casadi.*
    y = if_else(x > L, L, if_else(x < -L, -L, x));
end

% 光滑 min
function m = smin(a, b)
    m = 0.5 * (a + b - sqrt((a - b)^2 + 1e-9));
end

% 3x3 行列式
function d = det3(A)
    d = A(1,1)*(A(2,2)*A(3,3) - A(2,3)*A(3,2)) ...
      - A(1,2)*(A(2,1)*A(3,3) - A(2,3)*A(3,1)) ...
      + A(1,3)*(A(2,1)*A(3,2) - A(2,2)*A(3,1));
end
