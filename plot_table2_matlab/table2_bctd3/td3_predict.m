function action = td3_predict(bridge, obs_norm)
%TD3_PREDICT  用已加载的 Td3PolicyBridge 对归一化观测做推理。
%
%   action = td3_predict(bridge, obs_norm)   % action: 3×1 ∈[-1,1]

    x = double(obs_norm(:));
    if numel(x) ~= 13
        error('td3_predict:Dim', '观测须为 13 维，收到 %d', numel(x));
    end

    % 行向量 float32 → numpy
    obs_py = py.numpy.asarray(x.', pyargs('dtype', py.numpy.float32));
    act_py = bridge.predict(obs_py);
    action = double(py.array.array('d', py.numpy.nditer(act_py)));
    action = action(:);
    if numel(action) ~= 3
        error('td3_predict:ActDim', '动作须为 3 维，收到 %d', numel(action));
    end
    action = max(min(action, 1), -1);
end
