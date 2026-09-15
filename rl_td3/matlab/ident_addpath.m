function ident_addpath()
%IDENT_ADDPATH  将 ident/matlab 加入 MATLAB 路径。
    this_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(this_dir, '..', '..', 'ident', 'matlab'));
end
