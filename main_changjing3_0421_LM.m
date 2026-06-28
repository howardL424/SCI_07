clc; clear;
%% 导弹摆脱距离可控的最优突防制导律 - 场景3 复现 (坐标系y轴指天，x轴 ...
% 可能指向目标，不是北天东，下同)

% 角度定义：θ为弹道倾角，ψ_v为弹道偏角（从x轴正向逆时针为正）

%% 1. 初始化参数 (按论文4.1节场景3)

% 攻击弹初始状态 [x, y, z, θ_M, ψ_v_M, V_M]
M0 = [10000, 6000, 3500, deg2rad(-20), deg2rad(155), 500]; % 地面坐标系，两个角分别弹道倾角、偏角
% M0 = [10000, 6000, 3500, deg2rad(-20), deg2rad(155), 0];% 静止测试

% 防御弹初始状态 [x, y, z, θ_D, ψ_v_D, V_D]
% D0 = [1000, 0, 0, deg2rad(40), deg2rad(-20), 600];
D10 = [1000, 0, 0, deg2rad(40), deg2rad(-20), 600];
D20 = [-1000, 0, 0, deg2rad(40), deg2rad(-20), 600];% 暂定
% D20 = [-10000, 0, 0, deg2rad(40), deg2rad(-20), 600];% 暂定
% % gemini提供
% D10 = [1500, 3000, 0, deg2rad(40), deg2rad(-10), 600];
% D20 = [500, -1000, 0, deg2rad(40), deg2rad(-30), 600];% 暂定

% 目标固定位置 [x, y, z]
T = [0, 0, 0];

% 制导参数
KM = 3;      % 攻击弹比例导引系数
% KD = 4;     % 防御弹比例导引系数
KD1 = 4;     % 防御弹1比例导引系数
KD2 = 4;     
% a = 10;     % 突防制导权重
a = 10^4;    % 突防规避权重和
wE = 1.44 * 1000;   % 控制能量权重
% r_min = 50;% 初始化最小规避距离
r_min_T = 5000;% 初始化最小弹目距离
r_star = 59.667962;% 期望零控脱靶量（制导律参数）
gamma = 0.091976;% 暂定，此时a1=a2

% 仿真参数
dt = 0.001;         % 积分步长
% t_max = 15;         % 最大仿真时间
t_max = 30;
r_safe = 8000;      % 突防触发距离
% r_safe = 10000;
% max_overload = 8 * 9.81; % 最大过载
% max_overload = 180 * 9.81;r_MT
max_overload_M = 8 * 9.81; % 攻击弹最大过载
max_overload_D = 10 * 9.81; % 防御弹最大过载

% 场景2固定参数（取消）
% t_f_fixed = 10.50;  % 固定终端时间(从突防开始)
% t_f_fixed = 10.50;  % 固定终端时间(从t=0开始)
penetration_started = false;
% t_start_penetration = 0;

% 数据记录
% D1
record.t = [];
record.r_MT = []; record.r_MD1 = []; 
% record.z = [];
record.a_M = []; record.a_D1 = [];
record.pos_M = []; record.pos_D1 = [];
record.q_MD1 = []; 
% record.q_DM = []; record.q_dot_DM_prev = [0; 0];
record.q_MT = []; 
record.t_f1 = [];
    % D2
    % record.t = [record.t, t];
    % record.r_MT = [record.r_MT, r_MT];
    record.r_MD2 = [];
    % record.a_M = [record.a_M, a_M_cmd];
    record.a_D2 = [];
    % record.pos_M = [record.pos_M, pos_M'];
    record.pos_D2 = [];
    record.q_MD2 = [];
    % record.q_DM = [record.q_DM, [q_y_DM; q_z_DM]];
    % record.q_MT = [record.q_MT, [q_theta; q_psi]];
    % record.t_f = [record.t_f, t_f];% 记录突防预估结束时刻tf变化
    record.t_f2 = [];

%% 2. 核心计算函数

% 函数1: 计算视线角(参考论文图1)
function [q_y, q_z, r] = computeLOS(pos1, pos2)
    % pos1, pos2，从pos1的角度出发: 
    dx = pos2(1) - pos1(1); 
    dy = pos2(2) - pos1(2);   
    dz = pos2(3) - pos1(3); 
    
    r_vec = [dx,dy,dz];
    r = max(norm(r_vec), 1e-3);% 强制距离下限，避免除以0
    
    % 俯仰视线角q_y: 与x-z平面的夹角，向上为正
    sin_qy = dy / r;
    sin_qy = max(min(sin_qy, 1), -1);% 防asin超限
    q_y = asin(sin_qy);
    
    % 偏航视线角q_z: 在x-z平面内，从x轴正方向逆时针到投影向量的角度
    % 注意：MATLAB的atan2(y,x)返回从正x轴逆时针的角度
    % 我们需要从x轴正向逆时针，所以q_z = -atan2(dz, dx)
    q_z = -atan2(dz, dx);
end


% 函数2: 计算交叉项积分Phi_12(式27)
function Phi_12 = calc_Phi12(Q_dot1, Q_dot2, t_go1, t_go2)
    % 输入参数:
    % Q1, Q2 : 攻击弹相对于两枚防御弹的 Q_dot
    % tgo1, tgo2 : 攻击弹相对于两枚防御弹的剩余飞行时间 (t_f - t)
    
    % 1. 一些共同量
    Tmin = min(t_go1, t_go2);  sum_tgo = t_go1 + t_go2;
    delta_t = abs(t_go1 - t_go2); % 两者剩余时间差
    Q1_tgo1 = Q_dot1 * t_go1;
    Q2_tgo2 = Q_dot2 * t_go2;
    
    % 2. 数值稳定性阈值 (防止分母为0)
    eps_limit = 1e-6; 

    if Q_dot1 < eps_limit && Q_dot2 < eps_limit
        % --- 情况二: Q1 约等于 Q2 且接近0  ---
        Phi_12 = t_go1 * t_go2 * Tmin - sum_tgo * Tmin^2 / 2 + Tmin^3 / 3;
    elseif abs(Q_dot1 - Q_dot2) < eps_limit
        % --- 情况一: Q1 约等于 Q2 但非0  ---
        Q_dot = (Q_dot1 + Q_dot2) / 2; % 取平均值
        
        term1 = exp(Q_dot * sum_tgo) - exp(Q_dot * (sum_tgo - 2*Tmin)) / (2*Q_dot);
        term2 = 2 * Tmin * cosh(Q_dot*delta_t);% cosh偶函数
        term3 = exp(-Q_dot * (sum_tgo - 2*Tmin)) - exp(-Q_dot * sum_tgo) / (2*Q_dot);
        Phi_12 = (1 / (4 * Q_dot^2)) * (term1 - term2 + term3);
    elseif Q_dot1 < eps_limit
        % --- 情况三-Q1: Q1,Q2 仅有一个接近0  ---
        term1 = ((t_go1 - Tmin) * cosh(Q_dot2 * (t_go2 - Tmin)) - t_go1 * cosh(Q2_tgo2)) / (Q_dot2^2);
        term2 = (sinh(Q2_tgo2) - sinh(Q_dot2 * (t_go2 - Tmin))) / (Q_dot2^3);
        Phi_12 = term1 + term2;
    elseif Q_dot2 < eps_limit
        % --- 情况三-Q2: Q1,Q2 仅有一个接近0  ---
        term1 = ((t_go2 - Tmin) * cosh(Q_dot1 * (t_go1 - Tmin)) - t_go2 * cosh(Q1_tgo1)) / (Q_dot1^2);
        term2 = (sinh(Q1_tgo1) - sinh(Q_dot1 * (t_go1 - Tmin))) / (Q_dot1^3);
        Phi_12 = term1 + term2;
    else
        % --- 正常情况 ---
        % 提前计算量
        sumQ = Q_dot1 + Q_dot2;
        diffQ = Q_dot1 - Q_dot2;
        exp_11 = exp(Q1_tgo1); exp_22 = exp(Q2_tgo2);
        exp_sumT = exp(sumQ*Tmin); exp_diffT = exp(diffQ*Tmin);
        
        % 计算四个积分分量
        % part1 = (exp(Q1_tgo1 + Q_dot2*t_go2) - exp(Q1_tgo1 + Q_dot2*t_go2 - sumQ*Tmin)) / sumQ;
        % part2 = (exp(Q1_tgo1 - Q_dot2*t_go2) - exp(Q1_tgo1 - Q_dot2*t_go2 - diffQ*Tmin)) / diffQ;
        % part3 = (exp(-Q1_tgo1 + Q_dot2*t_go2 + diffQ*Tmin) - exp(-Q1_tgo1 + Q_dot2*t_go2)) / diffQ;
        % part4 = (exp(-Q1_tgo1 - Q_dot2*t_go2 + sumQ*Tmin) - exp(-Q1_tgo1 - Q_dot2*t_go2)) / sumQ;
        part1 = (exp_11 * exp_22 - exp_11 * exp_22 / exp_sumT) / sumQ;
        part2 = (exp_11 / exp_22 - exp_11 / exp_22 / exp_diffT) / diffQ;
        part3 = (exp_22 / exp_11 * exp_diffT - 1 / exp_11 * exp_22) / diffQ;
        part4 = (exp_sumT / exp_11 / exp_22 - 1 / (exp_11 * exp_22)) / sumQ;
        
        Phi_12 = (1 / (4 * Q_dot1 * Q_dot2)) * (part1 - part2 - part3 + part4);
    end
    
    % 3. 安全检查 (防止因异常输入产生非数)
    if isnan(Phi_12) || isinf(Phi_12)
        Phi_12 = 0;
    end
end

% 函数3: 计算中间变量B1,B2,C1(式12)(在函数4用)(只针对单个拦截弹与攻击弹间)
function [B1, B2, C1] = computeBC(q_y, q_z, theta_M, psi_vM, theta_D, psi_vD, a_yD, a_zD)
        % a_yD1 = a_D1(1); a_zD1 = a_D1(2);
        B1 = sin(q_y)*cos(theta_M) - cos(q_y)*sin(theta_M)*cos(q_z - psi_vM);
        % B2 = cos(q_y)*sin(q_z - psi_vM);
        B2 = -cos(q_y)*sin(q_z - psi_vM);
        
        % C1计算，注意符号：v_r = -C1? 根据式(12)和上下文推导(ps: 这里是生成时没识别到v_r表达式，后已补上）
        term1 = cos(q_y)*sin(q_z - psi_vD)*a_zD;
        term2 = (-cos(q_y)*sin(theta_D)*cos(q_z - psi_vD) + sin(q_y)*cos(theta_D))*a_yD;
        C1 = term1 - term2;
end

% 函数4.0: 计算最优突防制导指令(单弹)
function a_M = computeOptimalCmd(q_y, q_z, q_dot_y, q_dot_z, r, r_dot, t, t_f, ...
                                  theta_M, psi_vM, theta_D, psi_vD, a_D, a, b, r_star)
    Q_dot = sqrt(q_dot_y^2 + q_dot_z^2 * (cos(q_y))^2);
    t_go = t_f - t;
    
    % --- 【关键修复：防爆护栏】 ---
    % t_go = max(t_go, 1e-3); % 1. 防止 t_go 出现负数或 0
    if Q_dot * t_go > 100
        Q_dot = 100 / t_go; % 2. 限制乘积最大为 100（exp(100)完全在安全范围内），杜绝 Inf
    end
    
    % 提前计算B1, B2
    [B1, B2, C1] = computeBC(q_y, q_z, theta_M, psi_vM, theta_D, psi_vD, a_D(1), a_D(2));
    v_r = -C1; % 数学推导中 -v_r = C1，这一步极其关键
    
    if Q_dot < 1e-6
        % ===== 当视线旋转极慢时，采用泰勒展开极限，防止 0/0 崩溃 =====
        Omega =[1, t_go; 0, 1];
        z = [1, 0] * Omega * [r; r_dot];
        
        M = 0.5 * t_go^2 * v_r; % 正确的数学极限，不能设为0！
        
        % K 和 G 须合并计算极限
        KG = (a * (B1^2 + B2^2) / b) * (t_go^3 / 3);
        common_gain = (a / b) * t_go;
        
        % a_M = common_gain * (z - M - r_star) / (1 + KG + eps) *[B1; B2];% gemini指出遗漏负号
        a_M = -common_gain * (z - M - r_star) / (1 + KG + eps) *[B1; B2];
    else
        % ===== 正常公式计算 =====
        % Omega = computeOmega(Q_dot, t_go);
        exp_pos = exp(Q_dot * t_go);
        exp_neg = exp(-Q_dot * t_go);
        Omega = [(exp_pos+exp_neg)/2, (exp_pos-exp_neg)/(2*Q_dot);
                 Q_dot*(exp_pos-exp_neg)/2, (exp_pos+exp_neg)/2];
        z = [1, 0] * Omega *[r; r_dot];
        
        M = (exp(Q_dot*t_go) + exp(-Q_dot*t_go) - 2) / (2*Q_dot^2) * v_r;
        K = a * (B1^2 + B2^2) / (4 * b * Q_dot^2);
        G = (exp(2*Q_dot*t_go) - exp(-2*Q_dot*t_go))/(2*Q_dot) - 2*t_go;
        common_gain = a * (exp(Q_dot*t_go) - exp(-Q_dot*t_go)) / (2 * b * Q_dot);
        
        % a_M = common_gain * (z - M - r_star) / (1 + K*G + eps) * [B1; B2];
        a_M = -common_gain * (z - M - r_star) / (1 + K*G + eps) * [B1; B2];
    end

end

% 函数4.5: 计算最优突防制导指令(多弹)
% function a_M = computeOptimalCmd(q_y, q_z, q_dot_y, q_dot_z, r, r_dot, t, t_f, ...
%                                   theta_M, psi_vM, theta_D, psi_vD, a_D, a, b, r_star, ...
%                                   v_r)
function a_M = computeOptimalCmd_duo(q_y1, q_z1, q_dot_y1, q_dot_z1, r1, r_dot1, t, t_f1, ...
                                 q_y2, q_z2, q_dot_y2, q_dot_z2, r2, r_dot2, t_f2, ...
                                  theta_M, psi_vM, theta_D1, psi_vD1, theta_D2, psi_vD2, ...
                                  a_D1, a_D2, a1, a2, wE, r_star)
   
    % % ===== 提前计算一些量 =====
    % Q1_tgo1 = Q_dot1 * t_go1;
    % Q2_tgo2 = Q_dot2 * t_go2;

    % 拦截弹D1
    Q_dot1 = sqrt(q_dot_y1^2 + q_dot_z1^2 * (cos(q_y1))^2);% 恒非负
    t_go1 = t_f1 - t;  
    % 修复
    % t_go1 = max(t_go1, 1e-3); % 1. 防止 t_go 出现负数或 0
    if Q_dot1 * t_go1 > 100
        Q_dot1 = 100 / t_go1; % 2. 限制乘积最大为 100（exp(100)完全在安全范围内），杜绝 Inf
    end
    
    % 计算B1, B2
    % [B1, B2, C1] = computeBC(q_y, q_z, theta_M, psi_vM, theta_D, psi_vD, a_D(1), a_D(2));
    [B11, B21, C11] = computeBC(q_y1, q_z1, theta_M, psi_vM, theta_D1, psi_vD1, a_D1(1), a_D1(2));
    v_r1 = -C11; % 数学推导中 -v_ri = C1i，这一步极其关键
    b1_vec = [B11, B21]';

    % 拦截弹D2
    Q_dot2 = sqrt(q_dot_y2^2 + q_dot_z2^2 * (cos(q_y2))^2);% 恒非负
    t_go2 = t_f2 - t;
    
    % 修复
    % t_go2 = max(t_go2, 1e-3); % 1. 防止 t_go 出现负数或 0
    if Q_dot2 * t_go2 > 100
        Q_dot2 = 100 / t_go2; % 2. 限制乘积最大为 100（exp(100)完全在安全范围内），杜绝 Inf
    end
    
    % 计算B1, B2
    % [B1, B2, C1] = computeBC(q_y, q_z, theta_M, psi_vM, theta_D, psi_vD, a_D(1), a_D(2));
    [B12, B22, C12] = computeBC(q_y2, q_z2, theta_M, psi_vM, theta_D2, psi_vD2, a_D2(1), a_D2(2));
    v_r2 = -C12; % 数学推导中 -v_ri = C1i，这一步极其关键
    b2_vec = [B12, B22]';

    % ===== 提前计算一些量 =====
    Q1_tgo1 = Q_dot1 * t_go1;
    Q2_tgo2 = Q_dot2 * t_go2;
    
    % 判断Q_dot是否接近0
    if Q_dot1 < 1e-6 && Q_dot2 < 1e-6
        % ===== 当视线旋转极慢时，采用泰勒展开极限，防止 0/0 崩溃 =====
        Omega1 =[1, t_go1; 0, 1]; Omega2 =[1, t_go2; 0, 1];
        % z1 = [1, 0] * Omega1 * [r1; r_dot1]; z2 = [1, 0] * Omega2 * [r2; r_dot2];
        
        phi1 = t_go1; phi2 = t_go2;% varphi
        Phi_1 = t_go1^3 / 3; Phi_2 = t_go2^3 / 3;
        N_1 = 0.5 * t_go1^2 * v_r1; N_2 = 0.5 * t_go2^2 * v_r2;
        
    elseif Q_dot1 < 1e-6
        Omega1 =[1, t_go1; 0, 1]; 
        Omega2 =[cosh(Q2_tgo2), sinh(Q2_tgo2) / Q_dot2;
        Q_dot2 * sinh(Q2_tgo2), cosh(Q2_tgo2)];
        % z1 = [1, 0] * Omega1 * [r1; r_dot1]; z2 = [1, 0] * Omega2 * [r2; r_dot2];
        
        phi1 = t_go1; phi2 = sinh(Q2_tgo2) / Q_dot2;% varphi
        Phi_1 = t_go1^3 / 3; Phi_2 = (sinh(2 * Q2_tgo2) / Q_dot2 - 2 * t_go2) / (4 * Q_dot2^2);
        N_1 = 0.5 * t_go1^2 * v_r1; N_2 = (2 * cosh(Q2_tgo2) - 2) / (2 * Q_dot2^2) * v_r2;
    elseif Q_dot2 < 1e-6
        Omega1 =[cosh(Q1_tgo1), sinh(Q1_tgo1) / Q_dot1;
        Q_dot1 * sinh(Q1_tgo1), cosh(Q1_tgo1)]; 
        Omega2 =[1, t_go2; 0, 1];
        % z1 = [1, 0] * Omega1 * [r1; r_dot1]; z2 = [1, 0] * Omega2 * [r2; r_dot2];
        
        phi1 = sinh(Q1_tgo1) / Q_dot1; phi2 = sinh(Q2_tgo2) / Q_dot2;% varphi
        Phi_1 = (sinh(2 * Q1_tgo1) / Q_dot1 - 2 * t_go1) / (4 * Q_dot1^2); Phi_2 = t_go2^3 / 3;
        N_1 = (2 * cosh(Q1_tgo1) - 2) / (2 * Q_dot1^2) * v_r1; N_2 = 0.5 * t_go2^2 * v_r2;
    else
        % ===== 正常公式计算 =====
        Omega1 =[cosh(Q1_tgo1), sinh(Q1_tgo1) / Q_dot1;
        Q_dot1 * sinh(Q1_tgo1), cosh(Q1_tgo1)];
        % z1 = [1, 0] * Omega1 * [r1; r_dot1];
        Omega2 =[cosh(Q2_tgo2), sinh(Q2_tgo2) / Q_dot2;
        Q_dot2 * sinh(Q2_tgo2), cosh(Q2_tgo2)];
        % z2 = [1, 0] * Omega2 * [r2; r_dot2];
        % z = [1, 0] * Omega *[r; r_dot];
        % 式14
        phi1 = sinh(Q1_tgo1) / Q_dot1;% varphi
        phi2 = sinh(Q2_tgo2) / Q_dot2;% （差点错了）sinh已经包括分母2了
        % 式26
        Phi_1 = (sinh(2 * Q1_tgo1) / Q_dot1 - 2 * t_go1) / (4 * Q_dot1^2);
        N_1 = (2 * cosh(Q1_tgo1) - 2) / (2 * Q_dot1^2) * v_r1;
        Phi_2 = (sinh(2 * Q2_tgo2) / Q_dot2 - 2 * t_go2) / (4 * Q_dot2^2);
        N_2 = (2 * cosh(Q2_tgo2) - 2) / (2 * Q_dot2^2) * v_r2;
        
        end
     % 计算最优制导指令(公共项)
     z1 = [1, 0] * Omega1 * [r1; r_dot1]; z2 = [1, 0] * Omega2 * [r2; r_dot2];% 零控脱靶量
     Phi_12 = calc_Phi12(Q_dot1, Q_dot2, t_go1, t_go2);% 交叉项积分
     % 自耦合增益
     G_11 = a1 * (B11^2 + B21^2) / wE;
     G_22 = a2 * (B12^2 + B22^2) / wE; 
     % 交叉耦合增益
     % b_1' * b_2 实现向量内积计算
     G_12 = a2 * (b1_vec' * b2_vec) / wE;
     G_21 = a1 * (b1_vec' * b2_vec) / wE;
     Delta_Z_1 = z1 - N_1 - r_star; Delta_Z_2 = z2 - N_2 - r_star;
     Delta_det = (1 + G_11 * Phi_1) * (1 + G_22 * Phi_2) - G_12 * G_21 * Phi_12^2;
     I_1 = double(t <= t_f1); I_2 = double(t <= t_f2);
     % 式34
     term1 = (a1 / wE) * phi1 * ((1 + G_22 * Phi_2) * Delta_Z_1 - G_12 * Phi_12 * Delta_Z_2) * I_1 .* b1_vec;
     term2 = (a2 / wE) * phi2 * (-G_21 * Phi_12 * Delta_Z_1 + (1 + G_11 * Phi_1) * Delta_Z_2) * I_2 .* b2_vec;
     % 合并得到最终的二维控制量矩阵 u_A^*(t)
     % a_M = -(1 / Delta_det) .* (term1 + term2);
     a_M = (term1 + term2) .* (-1 / Delta_det);

end

% 函数5: 计算比例导引指令(式9,39)
function a_pn = computePN(K, V, q_dot_y, q_dot_z, theta)
    a_pn = [K * V * q_dot_y; -K * V * q_dot_z * cos(theta)];
end

% 函数6: 状态更新(龙格-库塔法，基于式4)
function state_new = updateState(state, a_cmd, dt)
    % state: [x, y, z, θ, ψ_v, V]
    % a_cmd: [a_y; a_z] 法向加速度
    
    x = state(1); y = state(2); z = state(3);
    theta = state(4); psi_v = state(5); V = state(6);
    a_y = a_cmd(1); a_z = a_cmd(2);
    
    % 检查V是否为0，不为0则积分
    if abs(V) < 1e-6
        state_new = state;% 状态不变，不更新状态
        warning('V is too small for deriv');
    else
        % 状态导数函数(严格按式4)
        deriv = @(s) [                    % s = [x, y, z, θ, ψ_v] (V不变)
            % V * cos(s(4)) * sin(s(5));    % dx/dt 
            V * cos(s(4)) * cos(s(5));
            V * sin(s(4));                % dy/dt 
            -V * cos(s(4)) * sin(s(5));   % dz/dt
            a_y / V;                      % dθ/dt
            -a_z / (V * cos(s(4)));       % dψ_v/dt
        ];
        
        s = [x; y; z; theta; psi_v];
        k1 = deriv(s);
        k2 = deriv(s + 0.5*dt*k1);
        k3 = deriv(s + 0.5*dt*k2);
        k4 = deriv(s + dt*k3);
        
        s_new = s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        
        % 保持V不变
        state_new = [s_new(1:5)', V];
    end
    
    
end

% 辅助函数: 过载饱和
function a_sat = saturate(a, limit)
    % 【关键修复：NaN / Inf 免疫】
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

% 函数7: 计算攻击弹-防御弹间相互视线角速率、相对速度
function [r_dot, q_dot_y, q_dot_z] = computeMDDot(pos1, pos2, r, q_y, q_z)
    % 参数传递
    V_M = pos1(6); theta_M = pos1(4); psi_vM = pos1(5);
    V_D = pos2(6); theta_D = pos2(4); psi_vD = pos2(5);
    
    % 计算 r_dot (式1第一行)
    term_D = V_D * (cos(theta_D) * cos(q_y) * cos(q_z - psi_vD) + sin(theta_D) * sin(q_y));
    
    term_M = V_M * (cos(theta_M) * cos(q_y) * cos(q_z - psi_vM) + sin(theta_M) * sin(q_y));
    
    r_dot = -term_D + term_M;  % 注意公式前面的负号

    if abs(r) < 1e-6
        q_dot_y = 0;
    else
        term_D_qy = V_D * (cos(theta_D) * sin(q_y) * cos(q_z - psi_vD) + -sin(theta_D) * cos(q_y));
        term_M_qy = V_M * (cos(theta_M) * sin(q_y) * cos(q_z - psi_vM) + -sin(theta_M) * cos(q_y));
        
        q_dot_y = (term_D_qy - term_M_qy) / r;
    end

    % 将原来的直接相除，改为使用带下限的安全分母
    cos_qy_safe = max(abs(cos(q_y)), 1e-5); % 限制分母极小值，防止 Inf 和 NaN
    r_safe = max(abs(r), 1e-3);
    
    term_D_qz = V_D * cos(theta_D) * sin(q_z - psi_vD);
    term_M_qz = V_M * cos(theta_M) * sin(q_z - psi_vM);
    
    q_dot_z = (term_D_qz - term_M_qz) / (r_safe * cos_qy_safe);
end

% 函数8: 计算攻击弹-目标间相互视线角速率、距离
function [q_dot_theta, q_dot_psi] = computeMTDot(pos1, r_MT, q_theta, q_psi)
    % 计算攻击弹-目标视线角变化率 (基于论文式2)
    % 输入：
    %   V_M, theta_M, psi_vM: 攻击弹速度大小、弹道倾角、弹道偏角
    %   r_MT: 攻击弹与目标的距离
    %   q_theta, q_psi: 攻击弹看目标的俯仰、偏航方向视线角
    % 输出：
    %   q_dot_theta: 俯仰视线角变化率
    %   q_dot_psi: 偏航视线角变化率

    % 参数传递
    V_M = pos1(6); theta_M = pos1(4); psi_vM = pos1(5);
    
    if abs(r_MT) < 1e-6
        q_dot_theta = 0;
        q_dot_psi = 0;
        warning('r_MT is too small for computing q_dot');
    else
        % 计算 q_dot_theta (式2第二行)
        term1 = cos(theta_M) * sin(q_theta) * cos(q_psi - psi_vM);
        term2 = -sin(theta_M) * cos(q_theta);
        q_dot_theta = V_M * (term1 + term2) / r_MT;
        
        % 计算 q_dot_psi (式2第三行)
        if abs(cos(q_theta)) < 1e-6
            q_dot_psi = 0;
            warning('cos(q_theta) is near zero, setting q_dot_psi to 0');
        else
            q_dot_psi = (V_M * cos(theta_M) * sin(q_psi - psi_vM)) / (r_MT * cos(q_theta));
        end
    end
end


%% 3. 主仿真循环
t = 0;
M = M0; D1 = D10; D2 = D20;  % 当前状态
intercepted = false;
flag = 0;% 初始化弹目距离增大标志
f1 = 0;% 计数标志，表示未进入过r_dot_MD1>0状态
f2 = 0;
idx_pan = 0; % 判断是否先用单弹规避以及先对第几个规避标志
flag_pan_dan = false;% 记录是否第一次进入特殊单弹模式标志
% flag_pan_dan1 = 0;% 记录是否第一次进入单弹1模式标志
% flag_pan_dan2 = 0;
load('bpnet_0421_5035lm.mat');% 15维


% a1 = a / (1+exp(-gamma));% D1规避项权重系数
% a2 = a * exp(-gamma) / (1+exp(-gamma));

t_penetrat_end = 1000;% 初始化为一个大值

% while t < t_penetrat_end + 5 && t < t_max
while t < t_max % 截止点为经过弹目距离最小点或最大时间1000s
    % 3.1 提取当前状态
    pos_M = M(1:3); theta_M = M(4); psi_vM = M(5); V_M = M(6);
    % pos_D = D(1:3); theta_D = D(4); psi_vD = D(5); V_D = D(6);
    pos_D1 = D1(1:3); theta_D1 = D1(4); psi_vD1 = D1(5); V_D1 = D1(6);
    pos_D2 = D2(1:3); theta_D2 = D2(4); psi_vD2 = D2(5); V_D2 = D2(6);
    
    % 3.2 计算视线几何
    % 攻击弹-目标
    [q_theta, q_psi, r_MT] = computeLOS(pos_M, T);
    % 攻击弹-防御弹（从防御弹视角出发）  
    % [q_y_MD, q_z_MD, r_MD_vec, r_MD] = computeLOS(pos_M, pos_D);
    % [q_y_MD, q_z_MD, r_MD] = computeLOS(pos_D, pos_M);
    [q_y_MD1, q_z_MD1, r_MD1] = computeLOS(pos_D1, pos_M);
    [q_y_MD2, q_z_MD2, r_MD2] = computeLOS(pos_D2, pos_M);
    % % 防御弹-攻击弹
    % [q_y_DM, q_z_DM, ~] = computeLOS(pos_M, pos_D);
    
    % 3.3 计算视线角变化率(对qy, qz求导的解析式)
    % [r_dot_MD, q_dot_y_MD, q_dot_z_MD] = computeMDDot(M, D, r_MD, q_y_MD, q_z_MD);
    [r_dot_MD1, q_dot_y_MD1, q_dot_z_MD1] = computeMDDot(M, D1, r_MD1, q_y_MD1, q_z_MD1);% 计算攻击弹-防御弹1视线角变化率、相对距离变化率
    [r_dot_MD2, q_dot_y_MD2, q_dot_z_MD2] = computeMDDot(M, D2, r_MD2, q_y_MD2, q_z_MD2);
    % [~, q_dot_y_DM, q_dot_z_DM] = computeMDDot(D, M, r_MD, q_y_DM, q_z_DM);
    [q_dot_theta, q_dot_psi] = computeMTDot(M, r_MT, q_theta, q_psi);% 计算攻击弹-目标视线角变化率
    % q_dot_DM = [q_dot_y_DM, q_dot_z_DM];% 攻击弹M视角
    q_dot_MT = [q_dot_theta, q_dot_psi];
    q_dot_MD1 = [q_dot_y_MD1, q_dot_z_MD1];% 防御弹1视线角速度向量
    q_dot_MD2 = [q_dot_y_MD2, q_dot_z_MD2];

    % 3.4 突防阶段判断+启动时刻赋值
    if ~penetration_started && min(r_MD1,r_MD2) <= r_safe
        penetration_started = true;
        t_start_penetration = t;
        fprintf('突防开始: t=%.3fs, r_MD=%.2fm\n', t, min(r_MD1,r_MD2));
        % 求此时突防状态初值
        % 样本向量[eta_yA10, eta_zA10, eta_yA20, eta_zA20, qy10, eta_yD10, eta_zD10, eta_yD20, eta_zD20, 
        % d*, gamma, beta, qz20, qz10,
        % qy20]% 0420:少个rAD0（注：此处1对应先进入突防的那个拦截弹）
        % X_tu0 = zeros(1,13);
        X_tu0 = zeros(13,1);
        if r_MD1 <= r_MD2 
            X_tu0(1) = M(4) + q_y_MD1; X_tu0(2) = M(5) - q_z_MD1 - pi;
            X_tu0(3) = M(4) + q_y_MD2; X_tu0(4) = M(5) - q_z_MD2 - pi;
            X_tu0(5) = q_y_MD1;
            X_tu0(6) = D1(4) - q_y_MD1; X_tu0(7) = D1(5) - q_z_MD1;
            X_tu0(8) = D2(4) - q_y_MD2; X_tu0(9) = D2(5) - q_z_MD2;
            % X_tu0(10) = (r_MD1 + r_MD2) / 2;% 中间距离
            X_tu0(10) = r_MD2 - r_MD1;% beta，一定为正或0
            % X_tu0(11) = q_z_MD2; X_tu0(12) = q_z_MD1;
            % X_tu0(13) = q_y_MD2;
            % X_tu0(11) = r_MD2 - r_MD1;% beta，一定为正或0
            X_tu0(11) = q_z_MD2; X_tu0(12) = q_z_MD1;
            X_tu0(13) = q_y_MD2;
        else % 调换D1, D2
            X_tu0(1) = M(4) + q_y_MD2; X_tu0(2) = M(5) - q_z_MD2 - pi;
            X_tu0(3) = M(4) + q_y_MD1; X_tu0(4) = M(5) - q_z_MD1 - pi;
            X_tu0(5) = q_y_MD2;
            X_tu0(6) = D2(4) - q_y_MD2; X_tu0(7) = D2(5) - q_z_MD2;
            X_tu0(8) = D1(4) - q_y_MD1; X_tu0(9) = D1(5) - q_z_MD1;
            % X_tu0(10) = (r_MD1 + r_MD2) / 2;% 中间距离
            X_tu0(10) = r_MD1 - r_MD2;% beta，一定为正
            % X_tu0(11) = (r_MD1 - r_MD2) / 2;% beta
            X_tu0(11) = q_z_MD1; X_tu0(12) = q_z_MD2;
            X_tu0(13) = q_y_MD1;
            % X_tu0(12) = q_z_MD1; X_tu0(13) = q_z_MD2;
            % X_tu0(14) = q_y_MD1;
        end
        tic
        [r_star, gamma] = LMfanjie(net, X_tu0);% 调用LM反解训好的代理模型得制导参数
        toc
        % t_ji = timeit(@LMfanjie);% 函数计时
        % fprintf('高精度运行时长：%.8f 秒\n', t_ji);
        a1 = a / (1+exp(-gamma));% D1规避项权重系数
        a2 = a * exp(-gamma) / (1+exp(-gamma));
        % r_star_dan = LMfanjie_dan(net, X_tu0);% 突防初始调用LM单弹反解一次
        % a1_dan = a; a2_dan = a;
    end
    
    % 3.5 计算防御弹1、2指令(始终比例导引)
    % a_D_cmd = computePN(KD, V_D, q_dot_MD(1), q_dot_MD(2), theta_D);
    a_D1_cmd = computePN(KD1, V_D1, q_dot_MD1(1), q_dot_MD1(2), theta_D1);
    a_D2_cmd = computePN(KD2, V_D2, q_dot_MD2(1), q_dot_MD2(2), theta_D2);
    % a_D_cmd = [0;0];% 无控测试
    
        
    % 3.6 计算攻击弹指令
    t_f1 = t - r_MD1/r_dot_MD1;% 变化突防终端时刻
    t_f2 = t - r_MD2/r_dot_MD2;
    if ~penetration_started
        % 阶段1: 攻击目标(比例导引)v_r
        a_M_cmd = computePN(KM, V_M, q_dot_MT(1), q_dot_MT(2), theta_M);
        % a_M_cmd = [0;0];% 无控测试
        stage = 1;
    % elseif penetration_started && t <= (t_start_penetration + t_f_fixed)
    % elseif penetration_started && t <= t_f
    elseif penetration_started && t <= max(t_f1,t_f2)% 两弹中至少一弹还在突防
        % 阶段2: 最优突防(固定t_f)
        t_go1 = t_f1 - t;% D1预估剩余时间
        t_go2 = t_f2 - t;% D2预估剩余时间
       
    % if abs(t_go1 - t_go2) > 4
    if abs(r_MD1 - r_MD2 ) > 3000
        [~, idx_pan] = min([t_go1, t_go2]);% 突防启动距离相差较大时选剩余时间最小的那个先做单弹规避
        if ~flag_pan_dan == true
            flag_pan_dan = true;
            % r_star_dan = LMfanjie_dan(net_dan, X_tu0_1);% 突防初始调用LM单弹反解一次
            % a1_dan = a; a2_dan = a;
            if idx_pan == 1
                X_tu0(1) = M(4) + q_y_MD1; X_tu0(2) = M(5) - q_z_MD1 - pi;
                X_tu0(3) = q_y_MD1;
                X_tu0(4) = D1(4) - q_y_MD1; X_tu0(5) = D1(5) - q_z_MD1;
                % X_tu0(10) = (r_MD1 + r_MD2) / 2;% 中间距离
                X_tu0(6) = r_MD1;
                r_star_dan1 = LMfanjie_dan(net_dan, X_tu0_1);% 突防初始调用LM单弹反解一次
            end
            if idx_pan == 2
                X_tu0(1) = M(4) + q_y_MD2; X_tu0(2) = M(5) - q_z_MD2 - pi;
                X_tu0(3) = q_y_MD2;
                X_tu0(4) = D2(4) - q_y_MD2; X_tu0(5) = D2(5) - q_z_MD2;
                % X_tu0(10) = (r_MD1 + r_MD2) / 2;% 中间距离
                X_tu0(6) = r_MD2;
                r_star_dan2 = LMfanjie_dan(net_dan, X_tu0);% 突防初始调用LM单弹反解一次
            end
        end
        if idx_pan == 1
            % a1_dan = a;
            a_M_cmd = computeOptimalCmd(q_y_MD1, q_z_MD1, q_dot_MD1(1), q_dot_MD1(2), ...
                                    r_MD1, r_dot_MD1, t, t_f1, ...
                                    theta_M, psi_vM, theta_D1, psi_vD1, ...
                                    a_D1_cmd, a, wE, r_star_dan1);
        end
        if idx_pan == 2
            % a2_dan = a;
            a_M_cmd = computeOptimalCmd(q_y_MD2, q_z_MD2, q_dot_MD2(1), q_dot_MD2(2), ...
                                    r_MD2, r_dot_MD2, t, t_f2, ...
                                    theta_M, psi_vM, theta_D2, psi_vD2, ...
                                    a_D2_cmd, a, wE, r_star_dan2);
        end
    elseif min(t_go1, t_go2) > 0.002
        % 【关键修正2：末端奇异性保护】
        % 两弹突防影响都考虑

        % 正常调用MD视角最优控制律对制导指令进行修改
        a_M_cmd = computeOptimalCmd_duo(q_y_MD1, q_z_MD1, q_dot_MD1(1), q_dot_MD1(2), ...
                                    r_MD1, r_dot_MD1, t, t_f1, ...
                                    q_y_MD2, q_z_MD2, q_dot_MD2(1), q_dot_MD2(2), ...
                                    r_MD2, r_dot_MD2, t_f2, ...
                                    theta_M, psi_vM, theta_D1, psi_vD1, theta_D2, psi_vD2,...
                                    a_D1_cmd, a_D2_cmd, a1, a2, wE, r_star);% 多弹
    elseif t_go1 > 0.002
        % % 求此时单弹突防状态初值
        % % 样本向量[eta_yA0, eta_zA0, qy0, eta_yD0, eta_zD0, r0]
        % if ~flag_pan_dan1 == true
        %     flag_pan_dan1 = false;
       
        %     r_star_dan1 = LMfanjie_dan(net, X_tu0);% 突防初始调用LM单弹反解一次
        % end
        a_M_cmd = computeOptimalCmd(q_y_MD1, q_z_MD1, q_dot_MD1(1), q_dot_MD1(2), ...
                                    r_MD1, r_dot_MD1, t, t_f1, ...
                                    theta_M, psi_vM, theta_D1, psi_vD1, ...
                                    a_D1_cmd, a, wE, r_star_dan);
    elseif t_go2 > 0.002
        % if ~flag_pan_dan2 == true
        %     flag_pan_dan2 = false;
        %     % a1_dan = a; a2_dan = a;
        %     X_tu0(1) = M(4) + q_y_MD2; X_tu0(2) = M(5) - q_z_MD2 - pi;
        %     X_tu0(3) = q_y_MD2;
        %     X_tu0(4) = D2(4) - q_y_MD2; X_tu0(5) = D2(5) - q_z_MD2;
        %     % X_tu0(10) = (r_MD1 + r_MD2) / 2;% 中间距离
        %     X_tu0(6) = r_MD2;
        %     r_star_dan2 = LMfanjie_dan(net_dan, X_tu0);% 突防初始调用LM单弹反解一次
        % end
        a_M_cmd = computeOptimalCmd(q_y_MD2, q_z_MD2, q_dot_MD2(1), q_dot_MD2(2), ...
                                    r_MD2, r_dot_MD2, t, t_f2, ...
                                    theta_M, psi_vM, theta_D2, psi_vD2, ...
                                    a_D2_cmd, a, wE, r_star);
    end
        stage = 2;
        t_penetrat_end = t;% 记录突防最终结束时刻
    else
        % 阶段3: 恢复攻击目标
        a_M_cmd = computePN(KM, V_M, q_dot_MT(1), q_dot_MT(2), theta_M);
        stage = 3;
    end
    
    % 3.7 过载饱和
    % a_M_cmd = saturate(a_M_cmd, max_overload);
    a_M_cmd = saturate(a_M_cmd, max_overload_M);
    % a_D_cmd = saturate(a_D_cmd, max_overload_D);
    a_D1_cmd = saturate(a_D1_cmd, max_overload_D);
    a_D2_cmd = saturate(a_D2_cmd, max_overload_D);
    
    % 3.8 状态更新
    M = updateState(M, a_M_cmd, dt);
    % D = updateState(D, a_D_cmd, dt);
    D1 = updateState(D1, a_D1_cmd, dt);
    D2 = updateState(D2, a_D2_cmd, dt);
    
    % 3.9 记录数据
    % D1
    record.t = [record.t, t];
    record.r_MT = [record.r_MT, r_MT];
    record.r_MD1 = [record.r_MD1, r_MD1];
    record.a_M = [record.a_M, a_M_cmd];
    record.a_D1 = [record.a_D1, a_D1_cmd];
    record.pos_M = [record.pos_M, pos_M'];
    record.pos_D1 = [record.pos_D1, pos_D1'];
    record.q_MD1 = [record.q_MD1, [q_y_MD1; q_z_MD1]];
    % record.q_DM = [record.q_DM, [q_y_DM; q_z_DM]];
    record.q_MT = [record.q_MT, [q_theta; q_psi]];
    % record.t_f = [record.t_f, t_f];% 记录突防预估结束时刻tf变化
    record.t_f1 = [record.t_f1, t_f1];
    % D2
    % record.t = [record.t, t];
    % record.r_MT = [record.r_MT, r_MT];
    record.r_MD2 = [record.r_MD2, r_MD2];
    % record.a_M = [record.a_M, a_M_cmd];
    record.a_D2 = [record.a_D2, a_D2_cmd];
    % record.pos_M = [record.pos_M, pos_M'];
    record.pos_D2 = [record.pos_D2, pos_D2'];
    record.q_MD2 = [record.q_MD2, [q_y_MD2; q_z_MD2]];
    % record.q_DM = [record.q_DM, [q_y_DM; q_z_DM]];
    % record.q_MT = [record.q_MT, [q_theta; q_psi]];
    % record.t_f = [record.t_f, t_f];% 记录突防预估结束时刻tf变化
    record.t_f2 = [record.t_f2, t_f2];
    
         
    % 3.10 拦截判断
    % if min(r_MD1, r_MD2) < 20 && ~intercepted
    if min(r_MD1, r_MD2) < 6 && ~intercepted
    % if r_MD < 0.6 && ~intercepted
        intercepted = true;
        [~, idx_D] = min([r_MD1, r_MD2]);% 第几个防御弹拦截
        fprintf('拦截成功: t=%.3fs, r_MD=%.2fm，由第%d枚防御弹\n', t, min(r_MD1, r_MD2), idx_D);
    end

    % 3.11 D1突防成功判断
    if r_dot_MD1 > 0 && f1 == 0
        % stage = 3;% 表示突防成功，5s后结束仿真
        t_penetrat_end1 = t;% 记录突防最终结束时刻
        % fprintf('突防成功: r_MD=%.4fm, r*=%.4f m\n', r_MD, r_star);
        % r_min = r_MD;
        f1 = 1;% f置为1
    end
    % 3.11 D2突防成功判断
    if r_dot_MD2 > 0 && f2 == 0
        % stage = 3;% 表示突防成功，5s后结束仿真
        t_penetrat_end2 = t;% 记录突防最终结束时刻
        % fprintf('突防成功: r_MD=%.4fm, r*=%.4f m\n', r_MD, r_star);
        % r_min = r_MD;
        f2 = 1;% f置为1
    end
    
    % 实时抓取本轮仿真的弹目最小距离与时刻
    if r_MT < r_min_T
        r_min_T = r_MT;
        t_Target = t;
    end
    % 3.12 目标打击结束判断
    if stage == 3 && record.r_MT(end) > record.r_MT(end-1)
        flag = flag + 1;% 突防成功后弹目距离增大标志
    end
    if flag == 100
        fprintf('已经过目标距离最近点: t=%.3fs, r_MT_min=%.2fm\n', t_Target, r_min_T);
        break;
    end

    % 3.13 时间推进
    t = t + dt;
end

%% 4. 结果分析与绘图

% save('record.pos_M.mat');% 保存规避者三维轨迹
% save('record.pos_M.mat');% 保存规避者三维轨迹
% save('record.pos_M.mat');% 保存规避者三维轨迹

%  规避能耗计算(kJ)
    [~, idx_tf_start] = min(abs(record.t - t_start_penetration));
    [~, idx_tf_end] = min(abs(record.t - t_penetrat_end));
    % idx_evade = (1 + t_start_penetration/dt): 1 :(1 + t_penetrat_end/dt);% 规避段索引
    idx_evade = idx_tf_start : idx_tf_end;% 规避段索引
    % a_cmd = vecnorm(record.a_M(idx_evade),2,1);
    a_cmd = sqrt(record.a_M(1,idx_evade).^2 + record.a_M(2,idx_evade).^2);
    Energy = 0.5 * trapz(a_cmd.^2) * dt / 1000;% 梯形积分（单位kJ）
fprintf('\n仿真结束。D1最小弹间距离 r_min1 = %.4f m, D2最小弹间距离 r_min2 = %.4f m\n', min(record.r_MD1), min(record.r_MD2));
fprintf('规避阶段机动能量消耗 Energy = %.4f kJ\n', Energy);
if penetration_started
    % t_f_global = t_start_penetration + t_f_fixed;
    % t_f_global = t_f_fixed;
    % t_f_global = t_f;
    % [~, idx_tf] = min(abs(record.t - t_f_global));
    % D1
    % [~, idx_tf1] = min(abs(record.t - t_penetrat_end));
    [~, idx_tf1] = min(abs(record.t - t_penetrat_end1));
    % [~, idx_tf_rmin1] = min(abs(record.r_MD1- min(record.r_MD1)));% 记录最小弹间距离时刻
    [~, idx_tf_rmin1] = min(record.r_MD1);
    % z_at_tf1 = record.z1(idx_tf1);
    r_at_tf1 = record.r_MD1(idx_tf1);
    t_rmin1 = (idx_tf_rmin1 - 1) * dt;% 最小距离时刻
    % fprintf('\n在实时终端时刻 t_f = %.3f s (全局: %.3f s):\n', t_f, t_f_global);
    % fprintf('\n仿真结束。在D1最小弹间距离时刻 t_rmin1 = %.3f s, 与D1最小弹间距离 r_min1 = %.4f m\n', t_rmin1, min(record.r_MD1));
    fprintf('\n两弹突防最终结束时刻 t_penetrat_end = %.3f s \n', t_penetrat_end);
    fprintf('\nD1预估突防终端时刻: %.3f s, D2预估突防终端时刻: %.3f s:\n', t_penetrat_end1, t_penetrat_end2);
    % fprintf('  零控脱靶量 z(t_f) = %.4f m (期望规避脱靶量 r* = %.1f m)\n', z_at_tf, r_min);
    fprintf('结束与D1突防时刻实际与D1弹间距离 r1(t_f) = %.4f m\n', r_at_tf1);

end

% 绘图1: 三维轨迹
figure('Name', '3-D Evasion Trajectory');
plot3(record.pos_M(1,:), record.pos_M(3,:), record.pos_M(2,:), 'k-', 'LineWidth',1.2); hold on;
plot3(record.pos_D1(1,:), record.pos_D1(3,:), record.pos_D1(2,:), 'r--', 'LineWidth',1.2);
plot3(record.pos_D2(1,:), record.pos_D2(3,:), record.pos_D2(2,:), 'b-.', 'LineWidth',1.2);
plot3(T(1), T(3), T(2), 'k^', 'MarkerSize',8, 'MarkerFaceColor','k');
plot3(M0(1), M0(3), M0(2), 'ko', 'MarkerSize',8, 'MarkerFaceColor','k');
plot3(D10(1), D10(3), D10(2), 'ro', 'MarkerSize',8, 'MarkerFaceColor','r');
plot3(D20(1), D20(3), D20(2), 'bo', 'MarkerSize',8, 'MarkerFaceColor','b');
xlabel('x/m'); ylabel('z/m'); zlabel('y/m');
legend('Evader A', 'D1', 'D2','T','A0','D10','D20'); grid on;
title('3-D Evasion Trajectory');
grid on; view(3);
set(gca, 'SortMethod', 'childorder');  % 修复线型的关键设置！


% 绘图4: 攻击弹、防御弹加速度
figure('Name', 'Accelerations of Evader A');
% subplot(2,1,1);
% plot(record.t, vecnorm(record.a_M,2,1)/9.81, 'k-');
% xlabel('时间(s)'); ylabel('总过载(g)'); title('规避者过载'); grid on;
% % subplot(2,1,2);
plot(record.t, record.a_M(1,:), 'b-', 'LineWidth',1.2); hold on;
plot(record.t, record.a_M(2,:), 'r--', 'LineWidth',1.2);
xlabel('t/s'); ylabel('Acceleration/(m/s^2)');
legend('a_{yA}', 'a_{zA}'); title('Accelerations of Evader A'); grid on;
set(gca, 'SortMethod', 'childorder');  % 修复线型的关键设置！