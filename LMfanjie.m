% clear; clc;
function [r_star, gamma] = LMfanjie(net, X_tu0)


%% ================= 1. 初始化环境与数据 =================

% --- 1.1 加载/定义你的 BP 网络 (请替换为你自己的网络) ---
% 这里我先模拟一个随机网络供你测试代码，实际使用时请 load 你的 net
% load('bpnet_0421_5035lm.mat');% 15维
% load('X_tu0.mat');% 除制导参数外的突防初始参数向量

% --- 1.2 已知条件 (请修改为你的实际数据) ---
X_input = X_tu0;              % 给定的 X (除制导参数外突防场景3初始条件)

% target_rmin = [25; 25];         % 目标输出 [rmin1_target, rmin2_target]
% target_rmin = [30, 30];% 行向量
target_rmin = [15; 15]';

% --- 1.3 物理约束与先验信息 (非常重要，请根据实际情况修改!) ---
% 参数边界 (越紧越好)
lb = [15, -2];  % [d*_min, gamma_min]
ub = [70, 2];  % [d*_max, gamma_max]
% ub = [120, 2.5];  % [d*_max, gamma_max]

% 正则化设置
% x_prior = [mean(target_rmin)+5, -0.5]; % 先验猜测值 (你认为最可能的 r* 和 gamma)
x_prior = [mean(target_rmin), -1];
lambda = 0.01;         % 正则化系数 (建议范围: 1e-4 ~ 1e-1)

% --- 1.4 多起点设置 ---
num_starts = 3;       % 跑多少个不同的初始点 (建议 20-50)
% num_starts = 15;
num_clusters = 2;      % 聚成几类 (通常 2-3 类即可)

%% ================= 2. 多起点 LM 优化 =================
% fprintf('>>> 开始 %d 次多起点优化...\n', num_starts);

% 存储结果
all_solutions = zeros(num_starts, 2); % 每一行是一个解 [r*, gamma]
all_residuals = zeros(num_starts, 1); % 对应的残差

% 使用拉丁超立方采样 (LHS) 生成分布均匀的初始值
X_lhs = lhsdesign(num_starts, 2); % 在 [0,1] 空间采样
x0_all = lb + (ub - lb) .* X_lhs; % 映射到实际参数范围

% 配置优化选项
options = optimoptions('lsqnonlin', ...
    'Algorithm', 'levenberg-marquardt', ...
    'Display', 'off', ... % 关闭单次迭代显示，避免刷屏
    'MaxFunctionEvaluations', 500, ...
    'TolX', 1e-6, ...
    'TolFun', 1e-6);

% 循环跑所有起点
for i = 1:num_starts
    x0 = x0_all(i, :);
    
    % 定义带参数的匿名函数
    objFun = @(x) residual_regularized(x, net, X_input, target_rmin, lambda, x_prior);
    
    % 运行优化
    [x_sol, resnorm] = lsqnonlin(objFun, x0, lb, ub, options);
    
    % 保存结果
    all_solutions(i, :) = x_sol;
    all_residuals(i) = resnorm;
    
    % fprintf('  起点 %d/%d 完成. 残差: %.4e\n', i, num_starts, resnorm);
end

%% ================= 3. 聚类分析 =================
% fprintf('\n>>> 开始 双弹 K-Means 聚类分析 (聚为 %d 类)...\n', num_clusters);

% 对解进行聚类
% [idx, C] = kmeans(all_solutions, num_clusters, 'Replicates', 5);
[idx, ~] = kmeans(all_solutions, num_clusters, 'Replicates', 5);

% 找出最优的聚类 (残差最小的那个类)
best_cluster_idx = -1;
best_cluster_mean_res = inf;

for k = 1:num_clusters
    cluster_members = (idx == k);
    mean_res = mean(all_residuals(cluster_members));
    
    fprintf('  聚类 %d: 包含 %d 个解, 平均残差: %.4e\n', k, sum(cluster_members), mean_res);
    
    if mean_res < best_cluster_mean_res
        best_cluster_mean_res = mean_res;
        best_cluster_idx = k;
    end
end

% 在最优聚类中选择残差最小的那个作为最终解
best_in_cluster = (idx == best_cluster_idx);
[~, min_res_idx_in_cluster] = min(all_residuals(best_in_cluster));

% 找到原始索引
original_indices = find(best_in_cluster);
final_sol_idx = original_indices(min_res_idx_in_cluster);

r_star = all_solutions(final_sol_idx, 1);
gamma = all_solutions(final_sol_idx, 2);
% final_residual = all_residuals(final_sol_idx);

% r_star = final_r_star; gamma = final_gamma;

end
% %% ================= 4. 结果输出与可视化 =================
% fprintf('\n=========================================\n');
% fprintf('>>> 优化完成！最终结果:\n');
% fprintf('  求解得到 d*    = %.6f\n', final_r_star);
% fprintf('  求解得到 gamma = %.6f\n', final_gamma);
% fprintf('  最终残差       = %.4e\n', final_residual);
% fprintf('  来自聚类 %d\n', best_cluster_idx);
% fprintf('=========================================\n');
%% ================= 5. 残差函数定义 (嵌套在主脚本内) =================
function F = residual_regularized(x, net, X, target, lambda, x_prior)
    % x(1) = r*, x(2) = gamma
    
    % 1. 构造网络输入 (假设输入格式是 [X; d*; gamma])
    % 注意：请根据你训练网络时的输入格式调整这里的拼接方式！
    % nn_input = [X; x(1); x(2)]; 
    nn_input = zeros(15, 1);
    % nn_input = zeros(16, 1);
    for j = 1:9
        nn_input(j) = X(j);
    end
    nn_input(10) = x(1); nn_input(11) = x(2);
    for j = 12:15
    % for j = 12:16
        nn_input(j) = X(j-2);
    end
    
    % 2. 前向传播
    % 使用 net() 而不是 sim() 以兼容新版本 MATLAB
    nn_output = net(nn_input); 
    % nn_output = predict(net, nn_input');
    % nn_output = double(predict(net, nn_input'));% dpl需要输入为行向量/矩阵
    % nn_output_log = double(predict(net, nn_input'));
    % 使用dpl网络时对输出数据进行log(x+1)还原
    % nn_output = exp(nn_output_log) - 1;
    
    
    % 3. 数据残差 (Data Fidelity)
    res_data = nn_output - target;
    
    % 4. 正则化残差 (Tikhonov Regularization)
    % 惩罚偏离先验值的解
    res_reg = sqrt(lambda) * (x- x_prior);
    % res_reg = sqrt(lambda) * (x' - x_prior');
    
    % 5. 组合残差 (LM 会自动计算这个向量的平方和)
    F = [res_data; res_reg];
end

