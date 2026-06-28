cd('D:\LTY\yan\yanerxia\GAIL')

opts.run_stageB       = true;
opts.plot             = false;
opts.save             = true;
opts.save_file        = 'expert_lib_p2.mat';
opts.checkpoint_every = 20;   % 每 20 条保存一次中间文件

r100 = generate_ps_expert_library(100, opts);
disp(r100.stats)
fprintf('expert_sa: %d x %d\n', size(r100.expert_sa,1), size(r100.expert_sa,2))