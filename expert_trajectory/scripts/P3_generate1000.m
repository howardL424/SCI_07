addpath(fileparts(fileparts(mfilename('fullpath'))));  % expert_trajectory/
setup_expert_path();

opts.run_stageB       = true;
opts.plot             = false;
opts.save             = true;
opts.save_file        = 'expert_lib_1000.mat';   % 先生成于 expert_trajectory/, 再手动移至 E 盘归档目录
opts.checkpoint_every = 50;

% Inf = 使用 samples_clean.mat 全部有效样本 (不限前 1000 条)
r1000 = generate_ps_expert_library(Inf, opts);
disp(r1000.stats)
fprintf('处理样本: %d 条, 入库专家: %d 条 (ok=%d + debug=%d)\n', ...
    r1000.N, r1000.stats.nB_ok + r1000.stats.nB_dbg, ...
    r1000.stats.nB_ok, r1000.stats.nB_dbg)
fprintf('expert_sa: %d x %d\n', size(r1000.expert_sa,1), size(r1000.expert_sa,2))
