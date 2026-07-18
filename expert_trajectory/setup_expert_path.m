function setup_expert_path()
%SETUP_EXPERT_PATH 将专家轨迹模块与项目根目录加入 MATLAB 路径。
%   调用后可直接使用 generate_ps_expert_library 等函数,
%   以及上级目录的 evasion_helpers.m、rl_td3/matlab 共享动力学核心
%   (TD3 环境与专家提取共用)。

    [expertRoot, projectRoot] = project_paths();
    addpath(expertRoot);
    addpath(projectRoot);
    addpath(fullfile(projectRoot, 'rl_td3', 'matlab'));
end
