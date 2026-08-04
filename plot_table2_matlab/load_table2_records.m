function methods = load_table2_records(root_dir)
%LOAD_TABLE2_RECORDS Load and standardize the four Table-2 trajectories.
%   METHODS = LOAD_TABLE2_RECORDS(ROOT_DIR) reads records produced by the
%   PN, programmed-sine, pure-TD3 and BC-TD3 simulation scripts.

if nargin < 1 || isempty(root_dir)
    root_dir = fileparts(mfilename('fullpath'));
end

defs = struct( ...
    'tag',   {'pn', 'sine', 'td3', 'bctd3'}, ...
    'label', {'Full-course PN', 'Programmed sine maneuver', ...
              'Pure TD3', 'BC-TD3'}, ...
    'file',  {fullfile(root_dir, 'record_table2_pn.mat'), ...
              fullfile(root_dir, 'record_table2_sine.mat'), ...
              fullfile(root_dir, 'table2_td3_pure', ...
                       'record_table2_td3_pure.mat'), ...
              fullfile(root_dir, 'table2_bctd3', ...
                       'record_table2_bctd3.mat')});

missing = {defs(~arrayfun(@(d) isfile(d.file), defs)).file};
if ~isempty(missing)
    msg = sprintf('  %s\n', missing{:});
    error('load_table2_records:MissingData', [ ...
        '缺少下列轨迹数据文件:\n%s\n' ...
        '请依次运行:\n' ...
        '  main_chunPN_0412_duo\n' ...
        '  main_zhengxian_0412_duo\n' ...
        '  table2_td3_pure/main_table2_td3_pure\n' ...
        '  table2_bctd3/main_table2_bctd3'], msg);
end

methods = repmat(struct('tag', '', 'label', '', 'file', '', ...
    'record', struct()), 1, numel(defs));
for k = 1:numel(defs)
    data = load(defs(k).file);
    if ~isfield(data, 'record') || ~isstruct(data.record)
        error('load_table2_records:BadFile', ...
            '文件不含结构体 record: %s', defs(k).file);
    end

    methods(k).tag = defs(k).tag;
    methods(k).label = defs(k).label;
    methods(k).file = defs(k).file;
    methods(k).record = standardize_record(data.record, defs(k).file);
end
end


function record = standardize_record(record, source_file)
required = {'t', 'pos_M', 'pos_D1', 'pos_D2', 'a_M'};
for k = 1:numel(required)
    if ~isfield(record, required{k})
        error('load_table2_records:MissingField', ...
            'record 缺少字段 %s: %s', required{k}, source_file);
    end
end

record.t = double(record.t(:).');
record.pos_M = as_3_by_n(record.pos_M, 'pos_M', source_file);
record.pos_D1 = as_3_by_n(record.pos_D1, 'pos_D1', source_file);
record.pos_D2 = as_3_by_n(record.pos_D2, 'pos_D2', source_file);

n = min([numel(record.t), size(record.pos_M, 2), ...
    size(record.pos_D1, 2), size(record.pos_D2, 2)]);
if n < 2
    error('load_table2_records:TooShort', ...
        '轨迹有效点不足 2 个: %s', source_file);
end

record.t = record.t(1:n);
record.pos_M = record.pos_M(:, 1:n);
record.pos_D1 = record.pos_D1(:, 1:n);
record.pos_D2 = record.pos_D2(:, 1:n);

a_M = double(record.a_M);
if size(a_M, 1) ~= 2 && size(a_M, 2) == 2
    a_M = a_M.';
end
if size(a_M, 1) < 2
    error('load_table2_records:BadAcceleration', ...
        'a_M 必须至少含 a_y、a_z 两行: %s', source_file);
end
record.a_M = fit_columns(a_M(1:2, :), n, NaN);

if isfield(record, 'T') && numel(record.T) >= 3
    target = double(record.T(1:3));
    target = target(:);
else
    target = zeros(3, 1);
    record.T = target.';
end

record.r_MD1 = get_or_derive_distance(record, 'r_MD1', ...
    record.pos_M, record.pos_D1, n);
record.r_MD2 = get_or_derive_distance(record, 'r_MD2', ...
    record.pos_M, record.pos_D2, n);
target_path = repmat(target, 1, n);
record.r_MT = get_or_derive_distance(record, 'r_MT', ...
    record.pos_M, target_path, n);

record.a_n = hypot(record.a_M(1, :), record.a_M(2, :));
end


function value = as_3_by_n(value, field_name, source_file)
value = double(value);
if size(value, 1) ~= 3 && size(value, 2) == 3
    value = value.';
end
if size(value, 1) ~= 3
    error('load_table2_records:BadPosition', ...
        '%s 必须为 3xN 或 Nx3: %s', field_name, source_file);
end
end


function value = get_or_derive_distance(record, field_name, p1, p2, n)
if isfield(record, field_name) && ~isempty(record.(field_name))
    value = record.(field_name);
    value = double(value(:).');
    value = fit_columns(value, n, NaN);
else
    value = sqrt(sum((p1 - p2).^2, 1));
end

bad = ~isfinite(value);
if any(bad)
    derived = sqrt(sum((p1 - p2).^2, 1));
    value(bad) = derived(bad);
end
end


function value = fit_columns(value, n, fill_value)
if size(value, 2) >= n
    value = value(:, 1:n);
else
    value(:, end + 1:n) = fill_value;
end
end
