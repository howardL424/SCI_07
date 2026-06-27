clc; clear;
pi = 3.14159265358979;
%% 训练样本生成 (坐标系y轴指天，x轴 ...
% 可能指向目标，不是北天东，下同)

% 角度定义：θ为弹道倾角，ψ_v为弹道偏角（从x轴正向逆时针为正）

%% 1. 初始化参数 (按论文4.1节场景3)
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
r_star = 50;% 期望零控脱靶量初值（制导律参数）
gamma = 0;% 给初值，此时a1=a2

% 仿真参数
dt = 0.001;         % 积分步长
% t_max = 135;% 最大截止仿真时间
t_max = 60;
max_overload = 8 * 9.81; % 最大过载
max_overload_M = 8 * 9.81; % 攻击弹最大过载
max_overload_D = 10 * 9.81; % 防御弹最大过载

penetration_started = false;

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
        B1 = sin(q_y)*cos(theta_M) - cos(q_y)*sin(theta_M)*cos(q_z - psi_vM);
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
        exp_pos = exp(Q_dot * t_go);
        exp_neg = exp(-Q_dot * t_go);
        Omega = [(exp_pos+exp_neg)/2, (exp_pos-exp_neg)/(2*Q_dot);
                 Q_dot*(exp_pos-exp_neg)/2, (exp_pos+exp_neg)/2];
        z = [1, 0] * Omega *[r; r_dot];
        
        M = (exp(Q_dot*t_go) + exp(-Q_dot*t_go) - 2) / (2*Q_dot^2) * v_r;
        K = a * (B1^2 + B2^2) / (4 * b * Q_dot^2);
        G = (exp(2*Q_dot*t_go) - exp(-2*Q_dot*t_go))/(2*Q_dot) - 2*t_go;
        common_gain = a * (exp(Q_dot*t_go) - exp(-Q_dot*t_go)) / (2 * b * Q_dot);
        
        a_M = -common_gain * (z - M - r_star) / (1 + K*G + eps) * [B1; B2];
    end

end

% 函数4.5: 计算最优突防制导指令(多弹)
function a_M = computeOptimalCmd_duo(q_y1, q_z1, q_dot_y1, q_dot_z1, r1, r_dot1, t, t_f1, ...
                                 q_y2, q_z2, q_dot_y2, q_dot_z2, r2, r_dot2, t_f2, ...
                                  theta_M, psi_vM, theta_D1, psi_vD1, theta_D2, psi_vD2, ...
                                  a_D1, a_D2, a1, a2, wE, r_star)
   

    % 拦截弹D1
    Q_dot1 = sqrt(q_dot_y1^2 + q_dot_z1^2 * (cos(q_y1))^2);% 恒非负
    t_go1 = t_f1 - t;  
    % 修复
    % t_go1 = max(t_go1, 1e-3); % 1. 防止 t_go 出现负数或 0
    if Q_dot1 * t_go1 > 100
        Q_dot1 = 100 / t_go1; % 2. 限制乘积最大为 100（exp(100)完全在安全范围内），杜绝 Inf
    end
    
    % 计算B1, B2
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
        
        phi1 = t_go1; phi2 = t_go2;% varphi
        Phi_1 = t_go1^3 / 3; Phi_2 = t_go2^3 / 3;
        N_1 = 0.5 * t_go1^2 * v_r1; N_2 = 0.5 * t_go2^2 * v_r2;
        
    elseif Q_dot1 < 1e-6
        Omega1 =[1, t_go1; 0, 1]; 
        Omega2 =[cosh(Q2_tgo2), sinh(Q2_tgo2) / Q_dot2;
        Q_dot2 * sinh(Q2_tgo2), cosh(Q2_tgo2)];
        
        phi1 = t_go1; phi2 = sinh(Q2_tgo2) / Q_dot2;% varphi
        Phi_1 = t_go1^3 / 3; Phi_2 = (sinh(2 * Q2_tgo2) / Q_dot2 - 2 * t_go2) / (4 * Q_dot2^2);
        N_1 = 0.5 * t_go1^2 * v_r1; N_2 = (2 * cosh(Q2_tgo2) - 2) / (2 * Q_dot2^2) * v_r2;
    elseif Q_dot2 < 1e-6
        Omega1 =[cosh(Q1_tgo1), sinh(Q1_tgo1) / Q_dot1;
        Q_dot1 * sinh(Q1_tgo1), cosh(Q1_tgo1)]; 
        Omega2 =[1, t_go2; 0, 1];
        
        phi1 = sinh(Q1_tgo1) / Q_dot1; phi2 = sinh(Q2_tgo2) / Q_dot2;% varphi
        Phi_1 = (sinh(2 * Q1_tgo1) / Q_dot1 - 2 * t_go1) / (4 * Q_dot1^2); Phi_2 = t_go2^3 / 3;
        N_1 = (2 * cosh(Q1_tgo1) - 2) / (2 * Q_dot1^2) * v_r1; N_2 = 0.5 * t_go2^2 * v_r2;
    else
        % ===== 正常公式计算 =====

        Omega1 =[cosh(Q1_tgo1), sinh(Q1_tgo1) / Q_dot1;
        Q_dot1 * sinh(Q1_tgo1), cosh(Q1_tgo1)];
        Omega2 =[cosh(Q2_tgo2), sinh(Q2_tgo2) / Q_dot2;
        Q_dot2 * sinh(Q2_tgo2), cosh(Q2_tgo2)];
        % 式14
        phi1 = sinh(Q1_tgo1) / Q_dot1;% varphi
        phi2 = sinh(Q2_tgo2) / Q_dot2;% （差点错了）sinh已经包括分母2了
        % 式26
        Phi_1 = (sinh(2 * Q1_tgo1) / Q_dot1 - 2 * t_go1) / (4 * Q_dot1^2);
        N_1 = (2 * cosh(Q1_tgo1) - 2) / (2 * Q_dot1^2) * v_r1;
        Phi_2 = (sinh(2 * Q2_tgo2) / Q_dot2 - 2 * t_go2) / (4 * Q_dot2^2);
        N_2 = (2 * cosh(Q2_tgo2) - 2) / (2 * Q_dot2^2) * v_r2;
        
        % Phi_12 = calc_Phi12(Q_dot1, Q_dot2, t_go1, t_go2);% 交叉项积分
        
        
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


%% 3. 主仿真循环

% 3.0 导入样本生成仿真初始参数
load('samples_clean_0421_181195.mat')
Lan = 0;% 记录拦截成功次数
Tu = 0;% 记录突防成功次数
% flag_pan = 0;% 记录进入过特殊单弹模式次数
flag_pan1 = 0; flag_pan2 = 0;
% flag_pan_duo = 0;% 记录进入过双弹模式的次数
N_jian = 1;
j = 1;
% Tu_i_rec = zeros(3,1e5);% 记录突防成功索引、摆脱距离与r*（虽然后两者应该不相近也没事）
% Tu_i_rec = zeros(4, 1.2*1e5);% 记录突防成功索引、摆脱距离与r*
Tu_i_rec = zeros(4, 181000);% 记录突防成功索引、摆脱距离与r*

samples_rmin = zeros(181000,2);% 预分配记录摆脱距离的向量

% for i = 1:(1.16*1e5 / N_jian) % 二进制数可能无法准确表示1.16造成浮点误差（1.16e5本身没问题但有除就不一定了，最好写整数）
for i = 1:(181000 / N_jian)
% ... 初始化参数 ...
% is_valid_sample = true; % 初始化样本有效标志
pan = false;% 判断本次是否进入特殊单弹模式标志
pan1 = false; pan2 = false;% 判断本次结尾是否进入对1或2单弹模式标志

t = 0;% 初始化每一轮时间
% r_min = 10000;% 初始化摆脱距离
r_min1 = 10000;% 初始化D1摆脱距离
r_min2 = 10000;% 初始化D2摆脱距离

% 样本向量[eta_yM10, eta_zM10, eta_yM20, eta_zM20, qy10, eta_yD10, eta_zD10, eta_yD20, eta_zD20, 
%  gamma, rAD0, beta, qy20]
eta_y_M10 = samples_clean(i*N_jian, 1); eta_z_M10 = samples_clean(i*N_jian, 2);
eta_y_M20 = samples_clean(i*N_jian, 3); eta_z_M20 = samples_clean(i*N_jian, 4);
q_y_MD10 = samples_clean(i*N_jian, 5); 
eta_y_D10 = samples_clean(i*N_jian, 6); eta_z_D10 = samples_clean(i*N_jian, 7);
eta_y_D20 = samples_clean(i*N_jian, 8); eta_z_D20 = samples_clean(i*N_jian, 9);
% r_star = samples_clean(i*N_jian, 10);
gamma = samples_clean(i*N_jian, 10);
rAD0 = samples_clean(i*N_jian, 11);% 两防御者突防初始距离中间值
beta = samples_clean(i*N_jian, 12); 
q_z_MD20 = samples_clean(i*N_jian, 13); q_z_MD10 = samples_clean(i*N_jian, 14);
q_y_MD20 = samples_clean(i*N_jian, 15);

% q_z_MD0 = 0;% 暂定!!(gemini说没问题）
% q_z_MD10 = 0;% 令qz10为0，推qz20

% 攻击弹弹道角（初始）计算
theta_M0 = eta_y_M10 - q_y_MD10;  % 攻击弹弹道倾角θ_M，单位rad，唯一
psi_VM0 = eta_z_M10 + q_z_MD10 + pi;      % 攻击弹弹道偏角ψ_VM，单位rad

r10 = 8000; r20 = 8000 + beta;% 假定第1枚防御弹先到

% pos_D0 = [0, 0, 0]';% 设防御弹突防初始位置为原点
pos_D10 = [0, 0, 0]';% 设防御弹D1突防初始位置为原点
pos_M0 = [r10*cos(q_y_MD10)*cos(q_z_MD10), r10*sin(q_y_MD10), r10*cos(q_y_MD10)*sin(-q_z_MD10)];% 攻击弹突防初始位置
pos_D20 = [pos_M0(1) - r20*cos(q_y_MD20)*cos(q_z_MD20);
          pos_M0(2) - r20*sin(q_y_MD20);
          pos_M0(3) - r20*cos(q_y_MD20)*sin(-q_z_MD20)];
    
% 防御弹弹道角（初始）计算
theta_D10 = eta_y_D10 + q_y_MD10;          % 防御弹D1弹道倾角θ_D，单位rad
theta_D20 = eta_y_D20 + q_y_MD20;
psi_VD10 = eta_z_D10 + q_z_MD10;           % 防御弹D1弹道偏角ψ_VD，单位rad
psi_VD20 = eta_z_D20 + q_z_MD20;

M0 = [pos_M0(1), pos_M0(2), pos_M0(3), theta_M0, psi_VM0, 500];
D10 = [pos_D10(1), pos_D10(2), pos_D10(3), theta_D10, psi_VD10, 600];
D20 = [pos_D20(1), pos_D20(2), pos_D20(3), theta_D20, psi_VD20, 600];
M = M0; D1 = D10; D2 = D20; % 推出的M、D1、D2初始状态
a1 = a / (1+exp(-gamma));% D1规避项权重系数
a2 = a * exp(-gamma) / (1+exp(-gamma));

intercepted = false;
penetration_started = true;% 样本生成只考虑突防

% 每次样本生成技术计数标志初始化
f1 = 0;% 计数标志，表示未进入过r_dot_MD1>0状态
f2 = 0;
success_flag = false;% 完全规避成功标志
d = 0;% 计数标志，表示未被拦截
% t_penetrat_end = 80;% 初始化为一个大值
t_penetrat_end1 = 80; t_penetrat_end2 = 80;
% flag_pan = 0;% 记录进入特殊单弹模式次数
idx_pan = 0; % 判断是否先用单弹规避以及先对第几个规避标志

while t < t_max
    % 3.1 提取当前状态
    pos_M = M(1:3); theta_M = M(4); psi_vM = M(5); V_M = M(6);
    pos_D1 = D1(1:3); theta_D1 = D1(4); psi_vD1 = D1(5); V_D1 = D1(6);
    pos_D2 = D2(1:3); theta_D2 = D2(4); psi_vD2 = D2(5); V_D2 = D2(6);
    
    % 3.2 计算视线几何
    % 攻击弹-防御弹1、2（从防御弹视角出发）  
    [q_y_MD1, q_z_MD1, r_MD1] = computeLOS(pos_D1, pos_M);
    [q_y_MD2, q_z_MD2, r_MD2] = computeLOS(pos_D2, pos_M);

    
    % 3.3 计算视线角变化率(对qy, qz求导的解析式)
    [r_dot_MD1, q_dot_y_MD1, q_dot_z_MD1] = computeMDDot(M, D1, r_MD1, q_y_MD1, q_z_MD1);% 计算攻击弹-防御弹1视线角变化率、相对距离变化率
    [r_dot_MD2, q_dot_y_MD2, q_dot_z_MD2] = computeMDDot(M, D2, r_MD2, q_y_MD2, q_z_MD2);
    % [q_dot_theta, q_dot_psi] = computeMTDot(M, r_MT, q_theta, q_psi);% 计算攻击弹-目标视线角变化率
    % q_dot_DM = [q_dot_y_DM, q_dot_z_DM];% 攻击弹M视角
    q_dot_MD1 = [q_dot_y_MD1, q_dot_z_MD1];% 防御弹1视线角速度向量
    q_dot_MD2 = [q_dot_y_MD2, q_dot_z_MD2];

    % % 3.4 突防阶段判断
    %     fprintf('突防开始: t=%.3fs, r_MD=%.2fm\n', t, min(r_MD1,r_MD2));
    
    % 3.5 计算防御弹1、2指令(始终比例导引)
    a_D1_cmd = computePN(KD1, V_D1, q_dot_MD1(1), q_dot_MD1(2), theta_D1);
    a_D2_cmd = computePN(KD2, V_D2, q_dot_MD2(1), q_dot_MD2(2), theta_D2);
    % a_D_cmd = [0;0];% 无控测试
    
    
    % 3.6 计算攻击弹指令
    % t_f = t - r_MD/r_dot_MD;% 变化剩余时间
    % t_f = t - r_MD/r_dot_MD;% 变化突防终端时刻
    r_dot_safe1 = sign(r_dot_MD1) * max(abs(r_dot_MD1), 1e-3);
    r_dot_safe2 = sign(r_dot_MD2) * max(abs(r_dot_MD2), 1e-3);
    t_f1 = t - r_MD1/r_dot_safe1;% 变化突防终端时刻
    % t_f2 = t - r_MD2/r_dot_safe1;% 0413训练时的错误代码（0414注）
    t_f2 = t - r_MD2/r_dot_safe2;
    

    % 阶段2: 最优突防(固定t_f)
        t_go1 = t_f1 - t;% D1预估剩余时间
        t_go2 = t_f2 - t;% D2预估剩余时间
    % if abs(t_go1 - t_go2) > 6
    %     [~, idx_pan] = min([t_go1, t_go2]);% 判断剩余时间相差较大时选哪个先做单弹规避
    %         pan = true;% 已进入特殊单弹模式
    if min(t_go1, t_go2) > 0.002
        % 【关键修正2：末端奇异性保护】
        % 正常调用MD视角最优控制律对制导指令进行修改
        a_M_cmd = computeOptimalCmd_duo(q_y_MD1, q_z_MD1, q_dot_MD1(1), q_dot_MD1(2), ...
                                    r_MD1, r_dot_MD1, t, t_f1, ...
                                    q_y_MD2, q_z_MD2, q_dot_MD2(1), q_dot_MD2(2), ...
                                    r_MD2, r_dot_MD2, t_f2, ...
                                    theta_M, psi_vM, theta_D1, psi_vD1, theta_D2, psi_vD2,...
                                    a_D1_cmd, a_D2_cmd, a1, a2, wE, r_star);% 多弹
    elseif t_go1 > 0.002
        % 判断是否进入过对1单弹算法
        if ~pan1
            pan1 = true;% 已进入特殊单弹模式
            flag_pan1 = flag_pan1 + 1;
        end
        a_M_cmd = computeOptimalCmd(q_y_MD1, q_z_MD1, q_dot_MD1(1), q_dot_MD1(2), ...
                                    r_MD1, r_dot_MD1, t, t_f1, ...
                                    theta_M, psi_vM, theta_D1, psi_vD1, ...
                                    a_D1_cmd, a, wE, r_star);
    elseif t_go2 > 0.002 % 需要加条件，否则负的和小于0.002s的t_go也会进来）
        % 判断是否进入过对2单弹算法
        if ~pan2
            pan2 = true;% 已进入特殊单弹模式
            flag_pan2 = flag_pan2 + 1;
        end
        a_M_cmd = computeOptimalCmd(q_y_MD2, q_z_MD2, q_dot_MD2(1), q_dot_MD2(2), ...
                                    r_MD2, r_dot_MD2, t, t_f2, ...
                                    theta_M, psi_vM, theta_D2, psi_vD2, ...
                                    a_D2_cmd, a, wE, r_star);
    end
        % stage = 2;    
        % t_penetrat_end = t;% 记录突防最终结束时刻
        
    % 3.7 过载饱和
    a_M_cmd = saturate(a_M_cmd, max_overload_M);
    a_D1_cmd = saturate(a_D1_cmd, max_overload_D);
    a_D2_cmd = saturate(a_D2_cmd, max_overload_D);
    
    % 3.8 状态更新
    M = updateState(M, a_M_cmd, dt);
    D1 = updateState(D1, a_D1_cmd, dt);
    D2 = updateState(D2, a_D2_cmd, dt);
    
    
    % 3.10 拦截判断
    if min(r_MD1, r_MD2) <= 6 && ~intercepted
        intercepted = true;
        % fprintf('拦截成功: t=%.3fs, r_MD=%.2fm\n', t, r_MD);
        d = 1;% d置为1
        % break;% 跳出while t<tmax 循环
    end
    
    
    % 3.11 D1突防成功判断
    if r_dot_MD1 > 0 && f1 == 0
        % stage = 3;% 表示突防成功，5s后结束仿真
        t_penetrat_end1 = t;% 记录突防最终结束时刻
        % fprintf('突防成功: r_MD=%.4fm, r*=%.4f m\n', r_MD, r_star);
        f1 = 1;% f置为1
    end
    % 3.11 D2突防成功判断
    if r_dot_MD2 > 0 && f2 == 0
        % stage = 3;% 表示突防成功，5s后结束仿真
        t_penetrat_end2 = t;% 记录突防最终结束时刻
        % fprintf('突防成功: r_MD=%.4fm, r*=%.4f m\n', r_MD, r_star);
        f2 = 1;% f置为1
    end

    % 实时抓取本轮仿真的绝对物理最小距离
    if r_MD1 < r_min1
        r_min1 = r_MD1;
    end
    if r_MD2 < r_min2
        r_min2 = r_MD2;
    end
    
    % 如果两弹都规避成功，再运行一小段时间后退出
    if f1 == 1 && f2 == 1
        if ~success_flag
            success_time = t;
            success_flag = true;
        end
        if t > success_time + 2.0  % 成功后再运行2秒
            break;
        end
    end


    % 3.12 时间推进
    t = t + dt;
end

% 记录规避脱靶量（最小距离）
samples_rmin(i, 1) = r_min1;
samples_rmin(i, 2) = r_min2;

% 记录拦截成功次数
if d == 1
    Lan = Lan + 1;
end

% 记录突防成功次数与索引数
if min(f1, f2) == 1 && d == 0
    Tu = Tu + 1;
    Tu_i_rec(1,j) = i * N_jian;% 突防成功索引数
    Tu_i_rec(2,j) = r_min1;% 突防成功时距离
    Tu_i_rec(3,j) = r_min2;% 突防成功时距离
    Tu_i_rec(4,j) = r_star;% 突防成功r*
    j = j + 1;
end

end