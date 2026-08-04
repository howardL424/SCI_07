function bridge = td3_load_policy(model_path, py_exe, device, prefer_td3bc)
%TD3_LOAD_POLICY  经 MATLAB py 接口加载 BC-TD3 / TD3 策略桥接对象。
%
%   bridge = td3_load_policy(model_path)
%   bridge = td3_load_policy(model_path, py_exe)
%   bridge = td3_load_policy(model_path, py_exe, device)
%   bridge = td3_load_policy(model_path, py_exe, device, prefer_td3bc)
%
%   model_path   : .../best/best_model 或 .../best/best_model.zip
%   py_exe       : 可选，默认 E:\Python\venvs\rl_pytorch\Scripts\python.exe
%   device       : 'cpu' | 'cuda'，默认 'cpu'
%   prefer_td3bc : 可选，默认 true（BC-TD3）；纯 TD3 基线请传 false

    if nargin < 2 || isempty(py_exe)
        py_exe = 'E:\Python\venvs\rl_pytorch\Scripts\python.exe';
    end
    if nargin < 3 || isempty(device)
        device = 'cpu';
    end
    if nargin < 4 || isempty(prefer_td3bc)
        prefer_td3bc = true;
    end

    this_dir  = fileparts(mfilename('fullpath'));   % .../plot_table2_matlab/table2_bctd3
    plot_root = fileparts(this_dir);                 % .../plot_table2_matlab
    gail_root = fileparts(plot_root);                % .../GAIL
    rl_td3    = fullfile(gail_root, 'rl_td3');

    % 指向训练用 venv（若尚未配置）
    try
        pe = pyenv;
        if pe.Status ~= "Loaded"
            if exist(py_exe, 'file')
                pyenv('Version', py_exe);
            end
        end
    catch ME
        warning('td3_load_policy:PyEnv', 'pyenv 配置失败: %s', ME.message);
    end

    % 本目录含 td3_policy_bridge；rl_td3 含 td3_bc
    sp = py.sys.path;
    if count(sp, this_dir) == 0
        insert(sp, int32(0), this_dir);
    end
    if count(sp, rl_td3) == 0
        insert(sp, int32(0), rl_td3);
    end

    mod = py.importlib.import_module('td3_policy_bridge');
    try
        mod = py.importlib.reload(mod);
    catch
        % reload 失败时沿用已导入模块
    end

    model_path = char(model_path);
    if endsWith(model_path, '.zip')
        model_path = model_path(1:end-4);
    end
    if ~exist([model_path '.zip'], 'file')
        error('td3_load_policy:MissingModel', '找不到模型: %s.zip', model_path);
    end

    bridge = mod.Td3PolicyBridge(model_path, ...
        pyargs('device', char(device), 'prefer_td3bc', logical(prefer_td3bc)));
end
