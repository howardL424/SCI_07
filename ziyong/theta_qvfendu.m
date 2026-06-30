d = load('expert_lib_1000.mat', 'results');
r = d.results;
gamma_all=[]; wT_all=[]; wE_all=[];
for k=1:numel(results.trajectories)
    tr=results.trajectories(k);
    if tr.ok_use && ~isempty(tr.stageB) && ~isempty(tr.stageB.theta_K)
        tk=tr.stageB.theta_K;
        gamma_all=[gamma_all,tk(1,:)];
        wT_all=[wT_all,tk(2,:)];
        wE_all=[wE_all,tk(3,:)];
    end
end
fprintf('gamma: std=%.3f\nwT: std=%.1f\nwE: std=%.1f\n', ...
    std(gamma_all), std(wT_all), std(wE_all))
% 建议: std(wT)>1000, std(wE)>100