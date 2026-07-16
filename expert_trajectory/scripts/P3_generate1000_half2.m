addpath(fileparts(fileparts(mfilename('fullpath'))));  % expert_trajectory/
setup_expert_path();
[expertRoot, projectRoot] = project_paths();
S = load(fullfile(projectRoot, 'samples_clean.mat'), 'samples_clean');
nValid = nnz(any(S.samples_clean ~= 0, 2));
midPos = floor(nValid / 2);
if nValid < 2
    error('P3_generate1000_half2:NoValidSamples', 'samples_clean 有效行不足，无法执行后半批。');
end

% ===== 运行模式 =====
% 'fresh'  : 从头跑 (忽略已有 checkpoint)
% 'resume' : 从最新 checkpoint 续跑 (按修改时间取 expert_lib_1000_half2_ckpt*.mat 最新)
RUN_MODE = 'resume';
% RUN_MODE = 'fresh';

opts.run_stageB       = true;
opts.plot             = false;
opts.save             = true;
opts.save_file        = 'expert_lib_1000_half2.mat';
opts.checkpoint_every = 50;
opts.valid_pos        = (midPos + 1):nValid;

if strcmpi(RUN_MODE, 'resume')
    ckpts = dir(fullfile(expertRoot, 'expert_lib_1000_half2_ckpt*.mat'));
    if isempty(ckpts)
        error('P3_generate1000_half2:NoCheckpoint', ...
            '未找到 checkpoint, 请将 RUN_MODE 改为 ''fresh'' 或检查文件路径');
    end
    [~, idx] = max([ckpts.datenum]);
    ckpt_file = fullfile(expertRoot, ckpts(idx).name);
    opts.resume_ckpt = ckpt_file;
    fprintf('续跑模式: 从 %s 继续\n', ckpt_file);
elseif strcmpi(RUN_MODE, 'fresh')
    fprintf('从头跑模式: 忽略已有 checkpoint\n');
else
    error('P3_generate1000_half2:BadRunMode', ...
        'RUN_MODE 必须是 ''fresh'' 或 ''resume''');
end

r_half2 = generate_ps_expert_library(Inf, opts);
disp(r_half2.stats)
fprintf('有效样本总数: %d, 后半批范围: %d:%d\n', nValid, midPos + 1, nValid);
fprintf('处理样本: %d 条, 入库专家: %d 条 (ok=%d + debug=%d)\n', ...
    r_half2.N, r_half2.stats.nB_ok + r_half2.stats.nB_dbg, ...
    r_half2.stats.nB_ok, r_half2.stats.nB_dbg);
fprintf('expert_sa: %d x %d\n', size(r_half2.expert_sa,1), size(r_half2.expert_sa,2));
