addpath(fileparts(fileparts(mfilename('fullpath'))));  % expert_trajectory/
setup_expert_path();

opts.run_stageB       = true;
opts.plot             = false;
opts.save             = true;
% opts.save_file        = 'expert_lib_p2.mat';   % 原：保存于 expert_trajectory/
% opts.save_file        = fullfile('E:\matlab\result_1010', 'expert_lib_p2.mat');
opts.save_file        = fullfile('E:\matlab\result_1010', 'p2_trial100', 'expert_lib_p2.mat');
opts.checkpoint_every = 20;

r100 = generate_ps_expert_library(100, opts);
disp(r100.stats)
fprintf('expert_sa: %d x %d\n', size(r100.expert_sa,1), size(r100.expert_sa,2))
