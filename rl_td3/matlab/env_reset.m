function [obs, state_abs, Tev, r_MT0, ok] = env_reset(row, cfg_in)
%ENV_RESET  初始化一个 episode：重建几何 → warm-start → 规避段初态 → 13维 obs。
%
%   [obs, state_abs, Tev, r_MT0, ok] = env_reset(row)
%   [obs, state_abs, Tev, r_MT0, ok] = env_reset(row, cfg_in)
%
%   输入:
%     row    [1×14] samples_clean 中的一行 (物理单位，弧度/米)
%     cfg_in  可选配置 struct，字段同 simulate_warmstart_trajectory 默认值
%
%   输出:
%     obs        [13×1 double]  GAIL 状态向量
%       1  r_MD1    攻-D1 距离 (m)
%       2  rdot_MD1 攻-D1 接近率 (m/s，正=远离)
%       3  qy_D1M   LOS 高低角 D1→M (rad)
%       4  qz_D1M   LOS 方位角 D1→M (rad)
%       5  r_MD2    攻-D2 距离 (m)
%       6  rdot_MD2 攻-D2 接近率 (m/s)
%       7  qy_D2M   LOS 高低角 D2→M (rad)
%       8  qz_D2M   LOS 方位角 D2→M (rad)
%       9  r_MT     攻-目标距离 (m)
%      10  qy_MT    M→T 高低角 (rad)
%      11  qz_MT    M→T 方位角 (rad)
%      12  V_M      攻弹速度 (m/s)
%      13  tgo_norm 归一化剩余规避时间 =1 (初始)
%     state_abs  [18×1 double]  绝对状态 [M(6); D1(6); D2(6)]，供 env_step 使用
%     Tev        规避段估计时长 (s)
%     r_MT0      规避段开始时攻-目标距离 (m)，用于 reward 归一化
%     ok         逻辑，是否找到有效规避段 (false 时 Python 端应重新采样)

    % 添加项目路径
    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);                           % rl_td3/matlab/
    addpath(fullfile(this_dir, '..', '..'));      % GAIL 根目录 (evasion_helpers)
    addpath(fullfile(this_dir, '..', '..', 'expert_trajectory'));  % reconstruct_geometry 等

    if nargin < 2 || isempty(cfg_in); cfg_in = struct(); end

    % ---- 重建初始几何 ----
    [M0, D10, D20] = reconstruct_geometry(row(:)');   % 确保行向量

    % ---- 阶段A warm-start ----
    tau = simulate_warmstart_trajectory(M0, D10, D20, cfg_in);

    % ---- 找规避段（stage == 2）起点 ----
    idx2 = find(tau.stage == 2);
    if isempty(idx2)
        obs = zeros(13, 1); state_abs = zeros(18, 1);
        Tev = 10; r_MT0 = 1e4; ok = false;
        return;
    end
    ok  = true;
    i0  = idx2(1);
    i_e = idx2(end);

    M  = tau.M(:, i0)';   % 1×6
    D1 = tau.D1(:, i0)';
    D2 = tau.D2(:, i0)';

    Tev  = max(tau.t(i_e) - tau.t(i0), 1.0);   % 防零除
    T_pos = [0, 0, 0];

    % ---- 计算初始 13维 obs ----
    H = evasion_helpers();
    [qy_D1M, qz_D1M, r_MD1] = H.computeLOS(D1(1:3), M(1:3));
    [qy_D2M, qz_D2M, r_MD2] = H.computeLOS(D2(1:3), M(1:3));
    [qy_MT,  qz_MT,  r_MT]  = H.computeLOS(M(1:3),  T_pos);
    [rdot_MD1, ~, ~]         = H.computeMDDot(M, D1, r_MD1, qy_D1M, qz_D1M);
    [rdot_MD2, ~, ~]         = H.computeMDDot(M, D2, r_MD2, qy_D2M, qz_D2M);

    obs = [r_MD1; rdot_MD1; qy_D1M; qz_D1M;
           r_MD2; rdot_MD2; qy_D2M; qz_D2M;
           r_MT;  qy_MT;   qz_MT;  M(6);  1.0];   % 13×1, tgo_norm=1

    r_MT0     = r_MT;
    state_abs = [M(:); D1(:); D2(:)];   % 18×1
end
