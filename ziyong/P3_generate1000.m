cd('D:\LTY\yan\yanerxia\GAIL')

opts.run_stageB       = true;
opts.plot             = false;
opts.save             = true;
opts.save_file        = 'expert_lib_1000.mat';
opts.checkpoint_every = 50;   % 每 50 条保存一次，共 20 个 checkpoint

r1000 = generate_ps_expert_library(1000, opts);
disp(r1000.stats)
fprintf('expert_sa: %d x %d\n', size(r1000.expert_sa,1), size(r1000.expert_sa,2))