addpath(fileparts(fileparts(mfilename('fullpath'))));  % expert_trajectory/
setup_expert_path();

opts.run_stageB       = true;
opts.plot             = false;
opts.save             = true;
opts.save_file        = 'expert_lib_1000.mat';   % 先生成于 expert_trajectory/, 再手动移至 E 盘归档目录
opts.checkpoint_every = 50;

r1000 = generate_ps_expert_library(1000, opts);
disp(r1000.stats)
fprintf('expert_sa: %d x %d\n', size(r1000.expert_sa,1), size(r1000.expert_sa,2))
