function [Xdot, aM, aD1, aD2] = casadi_dynamics(X, theta, t, params)
%CASADI_DYNAMICS 规避段闭环动力学的 CasADi 符号实现 (供伪谱配点)。
%   [Xdot, aM, aD1, aD2] = casadi_dynamics(X, theta, t, params)
%   第2~4输出为攻击弹/防御弹法向加速度指令 (供能量项与过载约束)。
%   X     : 18x1 全状态 [M(6); D1(6); D2(6)], 每段 = [x,y,z,theta,psi_v,V]
%   theta : 3x1 时变决策权重 [gamma, wT, wE] (w 固定于 params.w)
%   t     : 当前时刻 (标量, 配点处已知)
%   params: 结构体, 字段 w(=1e4), KD1, KD2, r_star(=d*)
%
%   攻击弹用式47 一体化最优制导 (casadi_guidance_duo_target);
%   防御弹 D1/D2 用比例导引 (式9/53) 跟踪攻击弹; 运动学为式5。
%   所有算子均为 CasADi 可微, 可直接嵌入 NLP。

    import casadi.*

    w = params.w; KD1 = params.KD1; KD2 = params.KD2; r_star = params.r_star;
    if isfield(params,'amax_M'); amax_M = params.amax_M; else; amax_M = 8*9.81; end
    if isfield(params,'amax_D'); amax_D = params.amax_D; else; amax_D = 10*9.81; end

    M  = X(1:6);  D1 = X(7:12);  D2 = X(13:18);
    gamma = theta(1); wT = theta(2); wE = theta(3);

    % 规避项权重 (式23): a1+a2 = w
    a1 = w / (1 + exp(-gamma));
    a2 = w * exp(-gamma) / (1 + exp(-gamma));

    Tstate = [0; 0; 0; 0; 0; 0];   % 原点静止目标

    % ---------- 视线几何 ----------
    [q_y_MD1, q_z_MD1, r_MD1] = los(D1(1:3), M(1:3));
    [q_y_MD2, q_z_MD2, r_MD2] = los(D2(1:3), M(1:3));
    [q_y_TM,  q_z_TM,  r_TM ] = los(Tstate(1:3), M(1:3));

    [r_dot_MD1, qd_y1, qd_z1]      = mddot(M, D1, r_MD1, q_y_MD1, q_z_MD1);
    [r_dot_MD2, qd_y2, qd_z2]      = mddot(M, D2, r_MD2, q_y_MD2, q_z_MD2);
    [r_dot_TM,  qd_y_TM, qd_z_TM]  = mddot(M, Tstate, r_TM, q_y_TM, q_z_TM);

    % ---------- 终端时刻 (式48/49); 闭合速度恒为负, 不需符号截断 ----------
    tf1 = t - r_MD1 / r_dot_MD1;
    tf2 = t - r_MD2 / r_dot_MD2;
    tfT = t - r_TM  / r_dot_TM;

    % ---------- 防御弹 PN 过载 ----------
    aD1 = pn(KD1, D1(6), qd_y1, qd_z1, D1(4));
    aD2 = pn(KD2, D2(6), qd_y2, qd_z2, D2(4));

    % ---------- 攻击弹式47 一体化制导 ----------
    aM = casadi_guidance_duo_target( ...
        q_y_MD1, q_z_MD1, qd_y1, qd_z1, r_MD1, r_dot_MD1, t, tf1, ...
        q_y_MD2, q_z_MD2, qd_y2, qd_z2, r_MD2, r_dot_MD2, tf2, ...
        q_y_TM, q_z_TM, qd_y_TM, qd_z_TM, r_TM, r_dot_TM, tfT, ...
        M(4), M(5), D1(4), D1(5), D2(4), D2(5), ...
        aD1(1), aD1(2), aD2(1), aD2(2), a1, a2, wT, wE, r_star);

    % ---------- 光滑过载饱和 (贴合 warm-start 物理) ----------
    aM  = satf(aM,  amax_M);
    aD1 = satf(aD1, amax_D);
    aD2 = satf(aD2, amax_D);

    % ---------- 运动学 (式5) ----------
    Xdot = [kin(M, aM); kin(D1, aD1); kin(D2, aD2)];
end

% 光滑过载饱和: 保方向, ||a||<=amax (a<amax 时近似恒等)
function as = satf(a, amax)
    s = sqrt(a(1)^2 + a(2)^2 + 1e-9);
    smax = 0.5 * (s + amax + sqrt((s - amax)^2 + (0.05*amax)^2));
    as = a * (amax / smax);
end

% ---------- 视线角 (computeLOS 符号版) ----------
function [q_y, q_z, r] = los(p1, p2)
    dx = p2(1) - p1(1);
    dy = p2(2) - p1(2);
    dz = p2(3) - p1(3);
    r  = sqrt(dx^2 + dy^2 + dz^2 + 1e-6);
    q_y = asin(dy / r);
    q_z = -atan2(dz, dx);
end

% ---------- 相对运动 (computeMDDot 符号版) ----------
function [r_dot, q_dot_y, q_dot_z] = mddot(p1, p2, r, q_y, q_z)
    V_M = p1(6); theta_M = p1(4); psi_vM = p1(5);
    V_D = p2(6); theta_D = p2(4); psi_vD = p2(5);

    term_D = V_D * (cos(theta_D)*cos(q_y)*cos(q_z - psi_vD) + sin(theta_D)*sin(q_y));
    term_M = V_M * (cos(theta_M)*cos(q_y)*cos(q_z - psi_vM) + sin(theta_M)*sin(q_y));
    r_dot = -term_D + term_M;

    r_safe = sqrt(r^2 + 1e-6);
    term_D_qy = V_D * (cos(theta_D)*sin(q_y)*cos(q_z - psi_vD) - sin(theta_D)*cos(q_y));
    term_M_qy = V_M * (cos(theta_M)*sin(q_y)*cos(q_z - psi_vM) - sin(theta_M)*cos(q_y));
    q_dot_y = (term_D_qy - term_M_qy) / r_safe;

    cos_qy_safe = sqrt(cos(q_y)^2 + 1e-10);
    term_D_qz = V_D * cos(theta_D) * sin(q_z - psi_vD);
    term_M_qz = V_M * cos(theta_M) * sin(q_z - psi_vM);
    q_dot_z = (term_D_qz - term_M_qz) / (r_safe * cos_qy_safe);
end

% ---------- 比例导引 (computePN 符号版) ----------
function a = pn(K, V, q_dot_y, q_dot_z, theta)
    a = [K * V * q_dot_y; -K * V * q_dot_z * cos(theta)];
end

% ---------- 运动学导数 (式5) ----------
function d = kin(s, a)
    theta = s(4); psi_v = s(5); V = s(6);
    d = [V * cos(theta) * cos(psi_v);
         V * sin(theta);
        -V * cos(theta) * sin(psi_v);
         a(1) / V;
        -a(2) / (V * cos(theta));
         0];
end
