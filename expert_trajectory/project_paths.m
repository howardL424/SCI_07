function [expertRoot, projectRoot] = project_paths()
%PROJECT_PATHS 返回专家轨迹模块目录与 GAIL 项目根目录。
%   [expertRoot, projectRoot] = project_paths()
%   expertRoot  : expert_trajectory/  (本模块)
%   projectRoot : GAIL/             (上级目录, 含 evasion_helpers.m 等共享代码)
%
%   归档专家库路径见 expert_lib_1000_file()。

    expertRoot  = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(expertRoot);
end
