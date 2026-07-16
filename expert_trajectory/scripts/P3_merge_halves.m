addpath(fileparts(fileparts(mfilename('fullpath'))));  % expert_trajectory/
setup_expert_path();
[expertRoot, projectRoot] = project_paths();

f1 = fullfile(expertRoot, 'expert_lib_1000_half1.mat');
f2 = fullfile(expertRoot, 'expert_lib_1000_half2.mat');
out = fullfile(expertRoot, 'expert_lib_1000.mat');

if ~isfile(f1)
    error('P3_merge_halves:MissingHalf1', '未找到 %s', f1);
end
if ~isfile(f2)
    error('P3_merge_halves:MissingHalf2', '未找到 %s', f2);
end

S1 = load(f1, 'results');  r1 = S1.results;
S2 = load(f2, 'results');  r2 = S2.results;

if any(ismember(r1.valid_pos, r2.valid_pos))
    error('P3_merge_halves:Overlap', '两批 valid_pos 有重叠，不能合并');
end

sc = load(fullfile(projectRoot, 'samples_clean.mat'), 'samples_clean');
nValid = nnz(any(sc.samples_clean ~= 0, 2));
% 统一为行向量，避免两批 orientation 不一致导致 horzcat 失败
vp1 = r1.valid_pos(:).'; vp2 = r2.valid_pos(:).';
vi1 = r1.valid_idx(:).'; vi2 = r2.valid_idx(:).';
nMerged = numel(unique([vp1, vp2]));
if nMerged ~= nValid
    warning('P3_merge_halves:CountMismatch', ...
        '合并后样本数 (%d) != 当前 samples_clean 有效行 (%d)，请确认两批基于同一份样本', ...
        nMerged, nValid);
end

tr = [r1.trajectories, r2.trajectories];
[~, ord] = sort([tr.valid_pos]);
tr = tr(ord);

if isempty(r1.expert_sa) && isempty(r2.expert_sa)
    expert_sa = zeros(0, 17);
elseif isempty(r1.expert_sa)
    expert_sa = r2.expert_sa;
elseif isempty(r2.expert_sa)
    expert_sa = r1.expert_sa;
else
    expert_sa = vertcat(r1.expert_sa, r2.expert_sa);
end

results = struct();
results.trajectories = tr;
results.expert_sa = expert_sa;
results.valid_pos = [vp1, vp2];
results.valid_idx = [vi1, vi2];
results.N = r1.N + r2.N;
results.stats = struct( ...
    'nA_succ',   r1.stats.nA_succ   + r2.stats.nA_succ, ...
    'nA_int',    r1.stats.nA_int    + r2.stats.nA_int, ...
    'nB_ok',     r1.stats.nB_ok     + r2.stats.nB_ok, ...
    'nB_dbg',    r1.stats.nB_dbg    + r2.stats.nB_dbg, ...
    'nB_fail',   r1.stats.nB_fail   + r2.stats.nB_fail, ...
    'elapsed_s', r1.stats.elapsed_s + r2.stats.elapsed_s);

save(out, 'results', '-v7.3');
fprintf('已合并 -> %s\n', out);
fprintf('轨迹条数: %d, expert_sa: %d x %d\n', ...
    results.N, size(results.expert_sa, 1), size(results.expert_sa, 2));
fprintf('入库专家: ok=%d, debug=%d, fail=%d\n', ...
    results.stats.nB_ok, results.stats.nB_dbg, results.stats.nB_fail);
