function lims = s6_main_limits(methods)
%S6_MAIN_LIMITS Common 3-D axis limits over a set of method records.
%   Same logic as common_main_limits in
%   plot_table2_matlab/plot_table2_trajectories.m.

xyz = [];
for k = 1:numel(methods)
    rec = methods(k).record;
    target = double(rec.T(1:3));
    xyz = [xyz, rec.pos_M, rec.pos_D1, rec.pos_D2, target(:)]; %#ok<AGROW>
end
plot_xyz = [xyz(1, :); xyz(3, :); xyz(2, :)] / 1000;
mins = min(plot_xyz, [], 2);
maxs = max(plot_xyz, [], 2);
span = maxs - mins;
pad = max(0.05 * span, 0.05);
lims = [mins - pad, maxs + pad];
end
