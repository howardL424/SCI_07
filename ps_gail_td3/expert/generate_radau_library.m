function results = generate_radau_library(N, opts)
%generate_radau_library - Generate a resumable direct acceleration library
%   RESULTS = generate_radau_library(N,OPTS) solves valid LHS sample rows.
%   See also calibrate_radau_weights, extract_accel_transitions

arguments
    N (1,1) double {mustBePositive} = 5
    opts (1,1) struct = struct()
end
ps_expert_setup(); branch=fileparts(fileparts(mfilename('fullpath')));
d=struct('samples_file',fullfile(fileparts(branch),'samples_clean.mat'), ...
    'save_file',fullfile(branch,'data','radau_expert_lib.mat'), ...
    'calibration_file',fullfile(branch,'data','radau_calibration.mat'), ...
    'valid_pos',[],'checkpoint_every',5,'resume',false,'cfg',struct());
names=fieldnames(d);
for k=1:numel(names)
    if ~isfield(opts,names{k}); opts.(names{k})=d.(names{k}); end
end
out=ps_output_path(opts.save_file); ckpt=[out(1:end-4),'_ckpt.mat'];
S=load(opts.samples_file,'samples_clean'); rows=S.samples_clean;
valid=find(vecnorm(rows,2,2)>1e-12);
pos=opts.valid_pos; if isempty(pos); pos=1:min(N,numel(valid)); end
pos=pos(1:min(N,numel(pos)));
C=load(opts.calibration_file,'calibration'); cfg=opts.cfg;
for name={'s_w','s_wE','s_wT','Jscale'}
    cfg.(name{1})=C.calibration.(name{1});
end
cfg.calibration_sha=ps_sha256(opts.calibration_file);
meta=ps_transition_meta(); meta.samples_sha=ps_sha256(opts.samples_file);
assert(strcmp(C.calibration.samples_sha,meta.samples_sha), ...
    'Calibration was measured on a different initial condition pool');
meta.samples_rows=size(rows,1); meta.scores=cfg;
meta.ipopt=struct('max_iter',300,'tol',1e-3,'hessian','limited-memory');
results=struct('meta',meta,'valid_pos',pos,'trajectories',{{}}, ...
    'completed',0,'stats',struct());
if opts.resume && isfile(ckpt)
    old=load(ckpt,'results_ckpt'); results=old.results_ckpt;
    assert(isequal(results.valid_pos,pos) && ...
        strcmp(results.meta.samples_sha,meta.samples_sha) && ...
        isequaln(results.meta.scores,meta.scores),'Checkpoint config mismatch');
end
for k=results.completed+1:numel(pos)
    row_id=valid(pos(k));
    [~,x0,Tev,~,ok,warm]=ps_env_reset(rows(row_id,:),struct());
    entry=struct('row_idx',row_id-1,'valid_pos',pos(k),'ok_use',false, ...
        'x0',x0,'Tev',Tev,'stageB',struct(),'transitions',struct());
    if ok
        res=solve_radau_accel(x0,Tev,warm,cfg); entry.stageB=res;
        if res.ok_use
            entry.transitions=extract_accel_transitions(res,struct());
            entry.ok_use=true;
        end
    end
    results.trajectories{k}=entry; results.completed=k;
    fprintf('[%d/%d] row=%d accepted=%d\n',k,numel(pos),row_id,entry.ok_use);
    if opts.checkpoint_every>0 && mod(k,opts.checkpoint_every)==0
        results_ckpt=results; save(ckpt,'results_ckpt','-v7');
    end
end
accepted=cellfun(@(e) e.ok_use,results.trajectories);
seconds=cellfun(@(e) get_seconds(e),results.trajectories);
results.stats=struct('accepted',sum(accepted),'attempted',numel(pos), ...
    'usable_rate',mean(accepted),'solve_seconds',seconds);
save(out,'results','-v7');
end

function s=get_seconds(e)
s=NaN;
if isfield(e.stageB,'solve_seconds'); s=e.stageB.solve_seconds; end
end
