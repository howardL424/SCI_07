addpath(fileparts(fileparts(mfilename('fullpath'))));  % expert_trajectory/
setup_expert_path();

opts.run_stageB       = true;
opts.plot             = false;
opts.save             = true;
% opts.save_file        = 'expert_lib_1000.mat';   % 原：先生成于 expert_trajectory/, 再手动移至 E 盘
opts.save_file        = fullfile('E:\matlab\result_1010', 'expert_lib_1000.mat');
% opts.checkpoint_every = 50;
opts.checkpoint_every = 200;

% Inf = 使用当前初态 mat 全部有效样本 (不限前 1000 条)
% 初态由 generate_ps_expert_library 读取 samples_clean_kuoda_train.mat
% 还原 samples_clean.mat 时，改 generate_ps_expert_library.m 里被注释的 samples_file
r1000 = generate_ps_expert_library(Inf, opts);
disp(r1000.stats)
fprintf('处理样本: %d 条, 入库专家: %d 条 (ok=%d + debug=%d)\n', ...
    r1000.N, r1000.stats.nB_ok + r1000.stats.nB_dbg, ...
    r1000.stats.nB_ok, r1000.stats.nB_dbg)
fprintf('expert_sa: %d x %d\n', size(r1000.expert_sa,1), size(r1000.expert_sa,2))
