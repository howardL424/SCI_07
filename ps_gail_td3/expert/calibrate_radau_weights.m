function calibration = calibrate_radau_weights(samples_file, valid_pos, out)
%calibrate_radau_weights - Measure warm-state term scales and balance them
%   CALIBRATION = calibrate_radau_weights(FILE,POS,OUT) measures 5-10 rows.
%   See also solve_radau_accel, generate_radau_library

arguments
    samples_file (1,:) char
    valid_pos (1,:) double {mustBeInteger,mustBePositive}
    out (1,:) char
end
ps_expert_setup(); out=ps_output_path(out);
S=load(samples_file,'samples_clean');
valid=find(vecnorm(S.samples_clean,2,2)>1e-12);
assert(numel(valid_pos)>=5 && numel(valid_pos)<=10);
raw=nan(3,numel(valid_pos)); misses=raw;
for k=1:numel(valid_pos)
    row=S.samples_clean(valid(valid_pos(k)),:);
    [~,x0,Tev,~,ok,warm]=ps_env_reset(row,struct());
    assert(ok,'Calibration sample has no evasion window');
    r=solve_radau_accel(x0,Tev,warm,struct('report_terms_only',true));
    raw(:,k)=r.terms; misses(:,k)=r.miss;
end
% U starts at zero, so energy is measured from warm state derivatives.
% These derivative-derived physical accelerations are calibration probes,
% never supervision labels or controls for the solver/replay buffer.
energy=zeros(1,numel(valid_pos));
for k=1:numel(valid_pos)
    [~,~,~,~,~,warm]=ps_env_reset(S.samples_clean(valid(valid_pos(k)),:),struct());
    idx=find(warm.stage==2); tt=warm.t(idx);
    ay=gradient(warm.M(4,idx),tt).*warm.M(6,idx);
    az=-gradient(warm.M(5,idx),tt).*warm.M(6,idx).*cos(warm.M(4,idx));
    energy(k)=0.5*trapz(tt,ay.^2+az.^2);
end
raw(2,:)=energy;
assert(all(isfinite(raw(:))) && all(raw(:)>=0));
scales=max(median(raw,2),1e-8);
weights=1e5./scales; % Each median weighted term equals 1e5.
calibration=struct('s_w',weights(1),'s_wE',weights(2),'s_wT',weights(3), ...
    'Jscale',3e5,'raw_terms',raw,'weighted_terms',raw.*weights, ...
    'misses',misses,'valid_pos',valid_pos,'samples_sha',ps_sha256(samples_file), ...
    'method','median_warm_state_terms_1e5_each','meta',ps_transition_meta());
save(out,'calibration','-v7'); disp(calibration);
end
