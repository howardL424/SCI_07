%PLOT_TRAINING_REWARD_EP_REW_MEAN Plot TD3 vs BC-TD3 training reward curves.
%   Uses TensorBoard rollout/ep_rew_mean only (no std shade, no markers).
%
%   Run from anywhere:
%     matlab -batch "cd('.../rl_td3/plots'); plot_training_reward_ep_rew_mean"

this_dir = fileparts(mfilename('fullpath'));
data_path = fullfile(this_dir, 'ep_rew_mean_300k_20260730.mat');
out_dir = fullfile(this_dir, '..', 'out_pictures');
if ~isfolder(out_dir)
    mkdir(out_dir);
end

S = load(data_path);

fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, 16, 10], ...
    'Name', 'Training reward of TD3 / BC-TD3');
ax = axes(fig);
hold(ax, 'on');

% Plot in thousands of steps so tick labels read 0..300 with unit "k"
h1 = plot(ax, S.td3_steps / 1000, S.td3_rew, '-', ...
    'Color', [0.13, 0.55, 0.13], 'LineWidth', 1.6);
h2 = plot(ax, S.bctd3_steps / 1000, S.bctd3_rew, '-', ...
    'Color', [0.84, 0.15, 0.16], 'LineWidth', 1.6);

xlabel(ax, 'Timesteps (k)');
ylabel(ax, 'Reward');
title(ax, 'Training reward of TD3 / BC-TD3', 'FontWeight', 'normal');
xlim(ax, [0, 300]);
xticks(ax, 0:50:300);
grid(ax, 'off');
box(ax, 'on');
ax.TickDir = 'in';
ax.LineWidth = 1.0;
ax.XAxis.Exponent = 0;

legend(ax, [h1, h2], {'Reward-TD3', 'Reward-BC-TD3'}, ...
    'Location', 'southeast', 'Box', 'on');

base = fullfile(out_dir, 'training_reward_td3_vs_bctd3_ep_rew_mean');
exportgraphics(fig, [base, '.png'], 'Resolution', 300);
exportgraphics(fig, [base, '.pdf'], 'ContentType', 'vector');
savefig(fig, [base, '.fig']);
fprintf('Saved: %s.png\n', base);
fprintf('Saved: %s.pdf\n', base);
fprintf('Saved: %s.fig\n', base);
close(fig);
