clc; clear;
%% 拉丁超立方采样 (LHS) 核心代码
% 样本规模 N=1e3，变量维数 dim=16

% N = 2 * 1e5;
N = 1.4e3;
% dim = 14;
dim = 14;

% 1. 生成 [0, 1] 之间的标准 LHS 样本
% 使用 'correlation' 准则可以最大程度降低变量间的相关性，优化采样分布
X_raw = lhsdesign(N, dim, 'criterion', 'correlation', 'iterations', 10);

% 初始化样本矩阵
samples = zeros(N, dim);
samples_clean = zeros(N, dim);

%% 2. 映射至目标分布
% 样本向量[eta_yA10, eta_zA10, eta_yA20, eta_zA20, qy10, eta_yD10, eta_zD10, eta_yD20, eta_zD20, 
%  rAD0, beta, qz20,
% qz10, qy20]
% --- 正态分布变量 (变量 1-4，A相关速度前置角) ---
% 目标分布为 N(0, 6^2)，即 mu = 0, sigma = 6
mu = 0;
% sigma = 6;
sigma = deg2rad(6);
lower_bound = deg2rad(-20); upper_bound = deg2rad(20);

% 截断正态分布映射
% 计算边界对应的正态分布累积概率 (CDF)
cdf_lower = normcdf(lower_bound, mu, sigma);
cdf_upper = normcdf(upper_bound, mu, sigma);

% 将原本 [0,1] 的 LHS 采样空间，线性缩放到截断后的概率区间 [cdf_lower, cdf_upper]
X_scaled = cdf_lower + X_raw(:, 1:4) * (cdf_upper - cdf_lower);
% X_scaled = cdf_lower + X_raw(:, 1:3) * (cdf_upper - cdf_lower);

% 通过逆累积分布函数映射回实际物理量，保证结果严格在 [-20°, 20°] 内且满足正态分布形状
samples(:, 1:4) = norminv(X_scaled, mu, sigma);

% samples(:, 1:2) = norminv(X_raw(:, 1:2), mu, sigma);

% --- 均匀分布变量 (变量 5-13) ---
% 这里的 ranges 矩阵定义了各变量的 [下界, 上界]
% 请根据实际物理背景替换以下数值
ranges = [
     
     % 5, 10;   % 开始突防距离r0/km
     deg2rad(20), deg2rad(70);  % qy10
     % deg2rad(20), deg2rad(70);  % qy20
     deg2rad(-10), deg2rad(10);   % eta_yD10
     deg2rad(-10), deg2rad(10); % eta_zD10
     deg2rad(-10), deg2rad(10);   % eta_yD20
     deg2rad(-10), deg2rad(10); % eta_zD20
     % 20, 120;    % d*/m, 即r_star
     % -3, 3; % gamma
     6500, 13500;   % 开始突防距离中间值r0/m
     % -1.5, 1.5 % beta
     -1500, 1500 % beta/m
     deg2rad(-25), deg2rad(25);  % qz20
];
for i = 5:(dim-2)
    lb = ranges(i-4, 1);
    ub = ranges(i-4, 2);
    % 线性变换：X = lb + (ub - lb) * U
    samples(:, i) = lb + (ub - lb) * X_raw(:, i);
end
% for i = 1:N
%     samples(i, 4) = samples(i, 2);% 由于qzi=0, eta_zA20=eta_zA10
% end
% qy20（放在了最后）
samples(:, dim) = samples(:, 3) - (samples(:, 1) - samples(:, 5));

% qz10
samples(:, dim-1) = samples(:, 4) + samples(:, 12) - samples(:, 2);

j = 0;
for i = 1:N
    if samples(i, dim) <= deg2rad(70) && samples(i, dim) >= deg2rad(20) && samples(i, dim-1) <= deg2rad(25) && samples(i, dim-1) >= deg2rad(-25)
        j = j + 1;
        samples_clean(j, :) = samples(i, :);% 将符合qy20、qz10要求的样本做记录
        % j = j + 1;
    end
end 

% save('samples.mat');
save('samples_clean.mat');

% %% Z-score标准化
% % 对矩阵 X_actual 进行 Z-score 标准化
% % Z 是标准化后的数据，大小为 N x 7，每列均值为0，标准差为1
% % mu 是一个 1x7 的行向量，保存了原始数据每一列的均值
% % sigma 是一个 1x7 的行向量，保存了原始数据每一列的标准差
% [Z, mu, sigma] = zscore(samples);
% 
% % 现在 Z 就可以作为BP神经网络的输入进行训练了
% % 一定要将 mu 和 sigma 保存下来，例如存为 .mat 文件
% save('normalization_params.mat', 'mu', 'sigma');

%% 3. 统计特征验证
fprintf('变量均值:\n'); disp(mean(samples));
fprintf('变量标准差:\n'); disp(std(samples));
% fprintf('归一化后变量标准差:\n'); disp(std(Z));