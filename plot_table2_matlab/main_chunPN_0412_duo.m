clc; clear;
this_dir = fileparts(mfilename('fullpath'));
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
% D0 = [1000, 0, 0, deg2rad(40), deg2rad(-20), 1500];
% D0 = [1000, 0, 0, deg2rad(40), deg2rad(-20), 0];
% 目标固定位置 [x, y, z]
T = [0, 0, 0];

% 制导参数
KM = 4;      % 攻击弹比例导引系数
% KD = 4;     % 防御弹比例导引系数
KD1 = 4;     % 防御弹1比例导引系数
KD2 = 4;     
% a = 10;     % 突防制导权重
a = 10^4;    % 突防规避权重和
wE = 1.44 * 1000;   % 控制能量权重
% r_min = 50;% 初始化最小规避距离
r_min_T = 5000;% 初始化最小弹目距离
r_star = 50;% 期望零控脱靶量（制导律参数）
gamma = 0;% 暂定，此时a1=a2

% 仿真参数
dt = 0.001;         % 积分步长
% t_max = 15;         % 最大仿真时间
t_max = 35;
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

% a1 = a / (1+exp(-gamma));% D1规避项权重系数
% a2 = a * exp(-gamma) / (1+exp(-gamma));

t_penetrat_end = 1000;% 初始化为一个大值
t_penetrat_end1 = 150; t_penetrat_end2 = 150;% 各弹规避结束时刻初始化为一个大值

while t < t_penetrat_end + 1 && t < t_max
% while t < t_max % 截止点为经过弹目距离最小点或最大时间1000s
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

    % 3.4 突防阶段判断
    if ~penetration_started && min(r_MD1,r_MD2) <= r_safe
        penetration_started = true;
        t_start_penetration = t;
        fprintf('突防开始: t=%.3fs, r_MD=%.2fm\n', t, min(r_MD1,r_MD2));
    end
    
    % 3.5 计算防御弹1、2指令(始终比例导引)
    % a_D_cmd = computePN(KD, V_D, q_dot_MD(1), q_dot_MD(2), theta_D);
    a_D1_cmd = computePN(KD1, V_D1, q_dot_MD1(1), q_dot_MD1(2), theta_D1);
    a_D2_cmd = computePN(KD2, V_D2, q_dot_MD2(1), q_dot_MD2(2), theta_D2);
    % a_D_cmd = [0;0];% 无控测试
    
    % a_M_cmd = computePN(KM, V_M, q_dot_MT(1), q_dot_MT(2), theta_M);
    
    % 3.6 计算攻击弹指令
    a_M_cmd = computePN(KM, V_M, q_dot_MT(1), q_dot_MT(2), theta_M);
    % t_f = t - r_MD/r_dot_MD;% 变化剩余时间
    % t_f = t - r_MD/r_dot_MD;% 变化突防终端时刻
    t_f1 = t - r_MD1/r_dot_MD1;% 变化突防终端时刻
    t_f2 = t - r_MD2/r_dot_MD2;
    % if ~penetration_started
    %     % 阶段1: 攻击目标(比例导引)v_r
    %     a_M_cmd = computePN(KM, V_M, q_dot_MT(1), q_dot_MT(2), theta_M);
    %     % a_M_cmd = [0;0];% 无控测试
    %     stage = 1;
    % % elseif penetration_started && t <= (t_start_penetration + t_f_fixed)
    % % elseif penetration_started && t <= t_f
    % elseif penetration_started && t <= max(t_f1,t_f2)% 两弹中至少一弹还在突防
    %     % 阶段2: 最优突防(固定t_f)
    %     % current_t_f = t_start_penetration + t_f_fixed;
    %     % current_t_f = t_f_fixed;
    % 
    %     % % 计算vr
    %     % % 式(6): 地面坐标系到视线坐标系的转移矩阵 L(q_y, q_z)（防御弹视角）
    %     % L_q_D = [cos(q_z_MD)*cos(q_y_MD), sin(q_y_MD), -cos(q_y_MD)*sin(q_z_MD); 
    %     %         -sin(q_y_MD)*cos(q_z_MD), cos(q_y_MD), sin(q_y_MD)*sin(q_z_MD); 
    %     %         sin(q_z_MD), 0, cos(q_z_MD)];
    %     % 
    %     % % 式(7): 地面坐标系到弹道坐标系的转移矩阵 L(theta, psi_V)
    %     % L_theta_D = [cos(theta_D)*cos(psi_vD), sin(theta_D), -cos(theta_D)*sin(psi_vD); 
    %     %             -sin(theta_D)*cos(psi_vD), cos(theta_D), sin(theta_D)*sin(psi_vD); 
    %     %             sin(psi_vD), 0, cos(psi_vD)];
    %     % v_r = [1,0,0] * (L_q * L_theta_D' * [0; a_M_cmd]);% 正交坐标变换矩阵，其逆等于其转置
    %     % v_r = [1,0,0] * (L_q_D * L_theta_D' * [0; a_D_cmd(1); a_D_cmd(2)]);
    %     % a_M_cmd = computeOptimalCmd(q_y_MD, q_z_MD, q_dot_MD(1), q_dot_MD(2), ...
    %     %                             r_MD, r_dot_MD, t, current_t_f, ...
    %     %                             theta_M, psi_vM, theta_D, psi_vD, ...
    %     %                             a_D_cmd, a, b, r_star, ...
    %     %                             v_r);
    %     t_go1 = t_f1 - t;% D1预估剩余时间
    %     t_go2 = t_f2 - t;% D2预估剩余时间
    % if min(t_go1, t_go2) > 0.002
    %     % 【关键修正2：末端奇异性保护】
    %     % 两弹突防影响都考虑
    % 
    %     % 正常调用MD视角最优控制律对制导指令进行修改
    %     a_M_cmd = computeOptimalCmd_duo(q_y_MD1, q_z_MD1, q_dot_MD1(1), q_dot_MD1(2), ...
    %                                 r_MD1, r_dot_MD1, t, t_f1, ...
    %                                 q_y_MD2, q_z_MD2, q_dot_MD2(1), q_dot_MD2(2), ...
    %                                 r_MD2, r_dot_MD2, t_f2, ...
    %                                 theta_M, psi_vM, theta_D1, psi_vD1, theta_D2, psi_vD2,...
    %                                 a_D1_cmd, a_D2_cmd, a1, a2, wE, r_star);% 多弹
    % elseif t_go1 > 0.002
    %     a_M_cmd = computeOptimalCmd(q_y_MD1, q_z_MD1, q_dot_MD1(1), q_dot_MD1(2), ...
    %                                 r_MD1, r_dot_MD1, t, t_f1, ...
    %                                 theta_M, psi_vM, theta_D1, psi_vD1, ...
    %                                 a_D1_cmd, a1, wE, r_star);
    % elseif t_go2 > 0.002
    %     a_M_cmd = computeOptimalCmd(q_y_MD2, q_z_MD2, q_dot_MD2(1), q_dot_MD2(2), ...
    %                                 r_MD2, r_dot_MD2, t, t_f2, ...
    %                                 theta_M, psi_vM, theta_D2, psi_vD2, ...
    %                                 a_D2_cmd, a2, wE, r_star);
    % end
    %     stage = 2;
    %     t_penetrat_end = t;% 记录突防最终结束时刻
    % else
    %     % 阶段3: 恢复攻击目标
    %     a_M_cmd = computePN(KM, V_M, q_dot_MT(1), q_dot_MT(2), theta_M);
    %     stage = 3;
    % end
    
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
    
    % 计算并记录零控脱靶量z(t)

    % current_t_f = t_f_fixed;
    %     t_go = current_t_f - t;
    %     Q_dot = sqrt(q_dot_MD(1)^2 + q_dot_MD(2)^2 * (cos(q_y_MD))^2);
    %     Omega = computeOmega(Q_dot, t_go);
    %     z_t = [1, 0] * Omega * [r_MD; r_dot_MD];
    %     record.z = [record.z, z_t];

    % % 记录零控脱靶量
    % if stage == 2
    %     % current_t_f = t_start_penetration + t_f_fixed;
    %     % current_t_f = t_f_fixed;
    %     % current_t_f = t_f;
    %     t_go = t_f - t;
    %     Q_dot = sqrt(q_dot_MD(1)^2 + q_dot_MD(2)^2 * (cos(q_y_MD))^2);
    %     Omega = computeOmega(Q_dot, t_go);
    %     z_t = [1, 0] * Omega * [r_MD; r_dot_MD];
    %     record.z = [record.z, z_t];
    % else
    %     record.z = [record.z, NaN];
    % end
    
    % 3.10 拦截判断
    % if min(r_MD1, r_MD2) < 20 && ~intercepted
    if min(r_MD1, r_MD2) < 6 && ~intercepted
    % if r_MD < 0.6 && ~intercepted
        intercepted = true;
        [~, idx_D] = min([r_MD1, r_MD2]);% 第几个防御弹拦截
        fprintf('拦截成功: t=%.3fs, r_MD=%.2fm，由第%d枚防御弹\n', t, min(r_MD1, r_MD2), idx_D);
        break;
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
    
    t_penetrat_end = max(t_penetrat_end1, t_penetrat_end2);% 规避最终结束时刻

    % 实时抓取本轮仿真的弹目最小距离与时刻
    if r_MT < r_min_T
        r_min_T = r_MT;
        t_Target = t;
    end
    % 3.12 目标打击结束判断
    if min(f1, f2) == 1 && record.r_MT(end) > record.r_MT(end-1)
        flag = flag + 1;% 突防成功后弹目距离增大标志
    end
    if flag == 1000
        fprintf('已经过目标距离最近点: t=%.3fs, r_MT_min=%.2fm\n', t_Target, r_min_T);
        break;
    end

    % 3.13 时间推进
    t = t + dt;
end

%% 4. 结果分析与绘图
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
    fprintf('\n仿真结束。在D1最小弹间距离时刻 t_rmin1 = %.3f s, 与D1最小弹间距离 r_min1 = %.4f m\n', t_rmin1, min(record.r_MD1));
    fprintf('\n两弹突防最终结束时刻 t_penetrat_end = %.3f s \n', t_penetrat_end);
    fprintf('\nD1预估突防终端时刻: %.3f s, D2预估突防终端时刻: %.3f s:\n', t_penetrat_end1, t_penetrat_end2);
    % fprintf('  零控脱靶量 z(t_f) = %.4f m (期望规避脱靶量 r* = %.1f m)\n', z_at_tf, r_min);
    fprintf('结束与D1突防时刻实际与D1弹间距离 r1(t_f) = %.4f m\n', r_at_tf1);
    % % D2
    % [~, idx_tf2] = min(abs(record.t - t_penetrat_end));
    % [~, idx_tf_rmin2] = min(abs(record.r_MD- min(record.r_MD)));% 记录最小弹间距离时刻
    % % z_at_tf2 = record.z2(idx_tf2);
    % r_at_tf2 = record.r_MD(idx_tf2);
    % t_rmin2 = (idx_tf_rmin2 - 1) * dt;% 最小距离时刻
    % % fprintf('\n在实时终端时刻 t_f = %.3f s (全局: %.3f s):\n', t_f, t_f_global);
    % fprintf('\n仿真结束。在D2最小弹间距离时刻 t_rmin2 = %.3f s, 最小弹间距离 r_min2 = %.4f m\n', t_rmin2, min(record.r_MD2));
    % fprintf('\n在D2突防最终结束时刻 t_penetrat_end2 = %.3f s (预估突防终端时刻: %.3f s):\n', t_penetrat_end2, t_f2);
    % % fprintf('  零控脱靶量 z(t_f) = %.4f m (期望规避脱靶量 r* = %.1f m)\n', z_at_tf, r_min);
    % fprintf('  结束与D2突防时刻实际与D1弹间距离 r2(t_f) = %.4f m\n', r_at_tf);
end

% 保存统一 record，供四方法论文图脚本读取。
record.M0 = M0;
record.D10 = D10;
record.D20 = D20;
record.T = T;
record.r_min1 = min(record.r_MD1);
record.r_min2 = min(record.r_MD2);
record.r_minT = min(record.r_MT);
record.r_terminal = record.r_MT(end);
record.Energy = Energy;
record.intercepted = intercepted;
record.algo = 'PN';
if penetration_started
    record.t_pen0 = t_start_penetration;
    record.t_pen_end = t_penetrat_end;
end
out_mat = fullfile(this_dir, 'record_table2_pn.mat');
save(out_mat, 'record');
fprintf('已保存统一绘图数据: %s\n', out_mat);

% 绘图1: 三维轨迹
figure('Name', '3-D Evasion Trajectory');
% plot3(record.pos_M(1,:), record.pos_M(3,:), record.pos_M(2,:), 'k-', 'LineWidth',1.0); hold on;
% plot3(record.pos_D1(1,:), record.pos_D1(3,:), record.pos_D1(2,:), 'r-', 'LineWidth',1.0);
% plot3(record.pos_D2(1,:), record.pos_D2(3,:), record.pos_D2(2,:), 'b-', 'LineWidth',1.0);
plot3(record.pos_M(1,:), record.pos_M(3,:), record.pos_M(2,:), 'k-', 'LineWidth',1.2); hold on;
plot3(record.pos_D1(1,:), record.pos_D1(3,:), record.pos_D1(2,:), 'r--', 'LineWidth',1.2);
plot3(record.pos_D2(1,:), record.pos_D2(3,:), record.pos_D2(2,:), 'b-.', 'LineWidth',1.2);
plot3(T(1), T(3), T(2), 'k^', 'MarkerSize',8, 'MarkerFaceColor','k');
plot3(M0(1), M0(3), M0(2), 'ko', 'MarkerSize',8, 'MarkerFaceColor','k');
plot3(D10(1), D10(3), D10(2), 'ro', 'MarkerSize',8, 'MarkerFaceColor','r');
plot3(D20(1), D20(3), D20(2), 'bo', 'MarkerSize',8, 'MarkerFaceColor','b');
xlabel('x/m'); ylabel('z/m'); zlabel('y/m');
plot3(record.pos_M(1, idx_tf_rmin1), record.pos_M(3, idx_tf_rmin1), record.pos_M(2, idx_tf_rmin1), 'ko', 'MarkerSize',10);
legend('Evader A', 'D1', 'D2','T','A0','D10','D20','Collision Point');
title('3-D Evasion Trajectory');
grid on; view(3);
set(gca, 'SortMethod', 'childorder');  % 修复线型的关键设置！

% idx_D

% % 绘图3.1: 弹间距离D1
% figure('Name', '弹间距离r_MD1');
% plot(record.t, record.r_MD1, 'k-', 'LineWidth',1.5); hold on;
% yline(20, 'r--', 'LineWidth',1.5);
% if penetration_started
%     xline(t_start_penetration, 'g--', 'LineWidth',1);
%     % xline(t_f_global, 'b--', 'LineWidth',1);
%     xline(t_penetrat_end, 'b--', 'LineWidth',1);
%     plot(record.t(idx_tf1), record.r_MD1(idx_tf1), 'bo', 'MarkerSize',10);
% end
% xlabel('时间(s)'); ylabel('弹间距离r_{MD1}(m)'); title('攻击弹-防御弹1距离');
% legend('r_{MD1}', '拦截阈值20m', '突防开始', '突防结束时刻', 't_{penetrat-end}时刻');
% grid on;
% 
% % 绘图3.2: 弹间距离D2
% figure('Name', '弹间距离r_MD2');
% plot(record.t, record.r_MD2, 'k-', 'LineWidth',1.5); hold on;
% yline(20, 'r--', 'LineWidth',1.5);
% if penetration_started
%     xline(t_start_penetration, 'g--', 'LineWidth',1);
%     % xline(t_f_global, 'b--', 'LineWidth',1);
%     xline(t_penetrat_end, 'b--', 'LineWidth',1);
%     % plot(record.t(idx_tf2), record.r_MD1(idx_tf2), 'bo', 'MarkerSize',10);
% end
% xlabel('时间(s)'); ylabel('弹间距离r_{MD2}(m)'); title('攻击弹-防御弹2距离');
% legend('r_{MD2}', '拦截阈值20m', '突防开始', '突防结束时刻');
% grid on;

% 绘图4: 攻击弹、防御弹加速度
% 绘图4: 攻击弹、防御弹加速度
figure('Name', 'Accelerations of Evader A');
% subplot(2,1,1);
% plot(record.t, vecnorm(record.a_M,2,1)/9.81, 'k-');
% xlabel('时间(s)'); ylabel('总过载(g)'); title('规避者过载'); grid on;
% % subplot(2,1,2);
% plot(record.t, record.a_M(1,:), 'b-'); hold on;
% plot(record.t, record.a_M(2,:), 'r-');
plot(record.t, record.a_M(1,:), 'b-', 'LineWidth',1.2); hold on;
plot(record.t, record.a_M(2,:), 'r--', 'LineWidth',1.2);
xlabel('t/s'); ylabel('Acceleration/(m/s^2)');
legend('a_{yA}', 'a_{zA}'); title('Accelerations of Evader A'); grid on;
set(gca, 'SortMethod', 'childorder');  % 修复线型的关键设置！
% figure('Name', '规避者与两防御者加速度');
% subplot(2,1,1);
% plot(record.t, vecnorm(record.a_M,2,1)/9.81, 'k-');
% xlabel('时间(s)'); ylabel('总过载(g)'); title('规避者过载'); grid on;
% % subplot(2,1,2);
% % plot(record.t, record.a_M(1,:), 'b-'); hold on;
% % plot(record.t, record.a_M(2,:), 'r-');
% % xlabel('时间(s)'); ylabel('加速度(m/s^2)');
% % legend('a_{y_M}', 'a_{z_M}'); title('攻击弹加速度分量'); grid on;
% subplot(2,1,2);
% plot(record.t, vecnorm(record.a_D1,2,1)/9.81, 'r-'); hold on;
% plot(record.t, vecnorm(record.a_D2,2,1)/9.81, 'b-');
% legend('a_{D1}','a_{D2}');
% xlabel('时间(s)'); ylabel('总过载(g)'); title('两防御者过载'); grid on;